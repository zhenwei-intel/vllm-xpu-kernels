#include <sycl/sycl.hpp>

#include <algorithm>

#include "dispatch_utils.h"
#include "ops.h"
#include "utils.h"

namespace {

constexpr int64_t GGML_TYPE_Q4_0 = 2;
constexpr int64_t GGML_TYPE_Q8_0 = 8;
constexpr int64_t QK_QX_0 = 32;
constexpr int64_t Q4_0_TYPE_SIZE = 18;
constexpr int64_t Q8_0_TYPE_SIZE = 34;

struct alignas(2) block_q4_0 {
  uint16_t d;
  uint8_t qs[QK_QX_0 / 2];
};

struct alignas(2) block_q8_0 {
  uint16_t d;
  int8_t qs[QK_QX_0];
};

static_assert(sizeof(block_q4_0) == Q4_0_TYPE_SIZE);
static_assert(sizeof(block_q8_0) == Q8_0_TYPE_SIZE);

inline float half_bits_to_float(uint16_t bits) {
  return static_cast<float>(sycl::bit_cast<sycl::half>(bits));
}

inline int64_t ggml_type_size(int64_t type) {
  switch (type) {
    case GGML_TYPE_Q4_0:
      return Q4_0_TYPE_SIZE;
    case GGML_TYPE_Q8_0:
      return Q8_0_TYPE_SIZE;
    default:
      TORCH_CHECK(false, "Unsupported GGUF quant type on XPU: ", type);
      return 0;
  }
}

inline void check_gguf_weight(
    const torch::Tensor& W, int64_t type, int64_t rows, int64_t cols) {
  CHECK_DEVICE(W);
  CHECK_CONTIGUOUS(W);
  TORCH_CHECK(W.dim() == 2, "GGUF weight must be a 2D tensor.");
  TORCH_CHECK(W.scalar_type() == torch::kUInt8, "GGUF weight must be uint8.");
  TORCH_CHECK(rows > 0 && cols > 0, "GGUF weight shape must be positive.");
  TORCH_CHECK(cols % QK_QX_0 == 0, "GGUF columns must be divisible by 32.");
  TORCH_CHECK(W.size(0) == rows, "GGUF weight row count mismatch.");
  TORCH_CHECK(
      W.size(1) == (cols / QK_QX_0) * ggml_type_size(type),
      "GGUF packed weight width does not match quant type and logical columns.");
}

template <typename scalar_t>
inline float dequantize_qx_0_value(
    const uint8_t* row_ptr, int64_t col, int64_t type) {
  int64_t block_idx = col / QK_QX_0;
  int64_t block_offset = col % QK_QX_0;
  if (type == GGML_TYPE_Q8_0) {
    auto block = reinterpret_cast<const block_q8_0*>(
        row_ptr + block_idx * sizeof(block_q8_0));
    return half_bits_to_float(block->d) * static_cast<float>(block->qs[block_offset]);
  }

  auto block = reinterpret_cast<const block_q4_0*>(
      row_ptr + block_idx * sizeof(block_q4_0));
  uint8_t packed = block->qs[block_offset % (QK_QX_0 / 2)];
  int value = block_offset < (QK_QX_0 / 2) ? (packed & 0xF) : (packed >> 4);
  return half_bits_to_float(block->d) * static_cast<float>(value - 8);
}

template <typename scalar_t>
class ggml_dequantize_kernel {
 public:
  ggml_dequantize_kernel(
      scalar_t* out,
      const uint8_t* weight,
      int64_t packed_cols,
      int64_t rows,
      int64_t cols,
      int64_t type)
      : out_(out),
        weight_(weight),
        packed_cols_(packed_cols),
        rows_(rows),
        cols_(cols),
        type_(type) {}

  void operator()(sycl::nd_item<1> item) const {
    int64_t idx = item.get_global_linear_id();
    int64_t stride = item.get_global_range(0);
    for (; idx < rows_ * cols_; idx += stride) {
      int64_t row = idx / cols_;
      int64_t col = idx % cols_;
      const uint8_t* row_ptr = weight_ + row * packed_cols_;
      out_[idx] = static_cast<scalar_t>(
          dequantize_qx_0_value<scalar_t>(row_ptr, col, type_));
    }
  }

 private:
  scalar_t* out_;
  const uint8_t* weight_;
  int64_t packed_cols_;
  int64_t rows_;
  int64_t cols_;
  int64_t type_;
};

template <typename scalar_t>
class ggml_mul_mat_kernel {
 public:
  ggml_mul_mat_kernel(
      scalar_t* out,
      const scalar_t* x,
      const uint8_t* weight,
      int64_t x_row_stride,
      int64_t out_row_stride,
      int64_t packed_cols,
      int64_t batch,
      int64_t rows,
      int64_t cols,
      int64_t type)
      : out_(out),
        x_(x),
        weight_(weight),
        x_row_stride_(x_row_stride),
        out_row_stride_(out_row_stride),
        packed_cols_(packed_cols),
        batch_(batch),
        rows_(rows),
        cols_(cols),
        type_(type) {}

  void operator()(sycl::nd_item<1> item) const {
    int64_t idx = item.get_global_linear_id();
    int64_t stride = item.get_global_range(0);
    for (; idx < batch_ * rows_; idx += stride) {
      int64_t batch_idx = idx / rows_;
      int64_t row = idx % rows_;
      const scalar_t* x_row = x_ + batch_idx * x_row_stride_;
      const uint8_t* w_row = weight_ + row * packed_cols_;
      float acc = 0.0f;
      for (int64_t col = 0; col < cols_; ++col) {
        acc += static_cast<float>(x_row[col]) *
               dequantize_qx_0_value<scalar_t>(w_row, col, type_);
      }
      out_[batch_idx * out_row_stride_ + row] = static_cast<scalar_t>(acc);
    }
  }

 private:
  scalar_t* out_;
  const scalar_t* x_;
  const uint8_t* weight_;
  int64_t x_row_stride_;
  int64_t out_row_stride_;
  int64_t packed_cols_;
  int64_t batch_;
  int64_t rows_;
  int64_t cols_;
  int64_t type_;
};

inline void check_gguf_mm_args(
    const torch::Tensor& W, const torch::Tensor& X, int64_t type, int64_t row) {
  CHECK_DEVICE(X);
  CHECK_CONTIGUOUS(X);
  TORCH_CHECK(X.dim() == 2, "GGUF matmul input must be a 2D tensor.");
  TORCH_CHECK(row > 0, "row must be positive.");
  TORCH_CHECK(
      X.scalar_type() == torch::kFloat16 || X.scalar_type() == torch::kBFloat16 ||
          X.scalar_type() == torch::kFloat32,
      "GGUF matmul input must be float16, bfloat16, or float32.");
  TORCH_CHECK(W.device() == X.device(), "W and X must be on the same XPU device.");
  check_gguf_weight(W, type, row, X.size(1));
}

template <typename LaunchFn>
inline void launch_1d(int64_t total_work, LaunchFn&& launch_fn) {
  constexpr int64_t threads = 256;
  int64_t groups = (total_work + threads - 1) / threads;
  sycl::range<1> grid(std::max<int64_t>(groups, 1));
  sycl::range<1> block(threads);
  auto& queue = vllm::xpu::vllmGetQueue();
  queue.submit([&](sycl::handler& cgh) {
    launch_fn(cgh, sycl::nd_range<1>(grid * block, block));
  });
}

}  // namespace

torch::Tensor ggml_dequantize(
    torch::Tensor W,
    int64_t type,
    int64_t m,
    int64_t n,
    std::optional<c10::ScalarType> const& dtype) {
  check_gguf_weight(W, type, m, n);
  const at::DeviceGuard device_guard(W.device());
  const auto out_dtype = dtype.value_or(torch::kHalf);
  TORCH_CHECK(
      out_dtype == torch::kHalf || out_dtype == torch::kBFloat16 ||
          out_dtype == torch::kFloat,
      "GGUF dequantize output must be float16, bfloat16, or float32.");
  auto out = torch::empty({m, n}, W.options().dtype(out_dtype));
  VLLM_DISPATCH_FLOATING_TYPES(
      out.scalar_type(), "ggml_dequantize_xpu", [&] {
        using sycl_t = vllm::xpu::SyclTypeTrait<scalar_t>::Type;
        launch_1d(m * n, [&](sycl::handler& cgh, sycl::nd_range<1> range) {
          cgh.parallel_for(
              range,
              ggml_dequantize_kernel<sycl_t>(
                  reinterpret_cast<sycl_t*>(out.data_ptr<scalar_t>()),
                  W.data_ptr<uint8_t>(),
                  W.size(1),
                  m,
                  n,
                  type));
        });
      });
  return out;
}

static torch::Tensor ggml_mul_mat_qx_0(
    torch::Tensor W, torch::Tensor X, int64_t type, int64_t row) {
  check_gguf_mm_args(W, X, type, row);
  const at::DeviceGuard device_guard(X.device());
  auto out = torch::empty({X.size(0), row}, X.options());
  VLLM_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_mul_mat_qx_0", [&] {
    using sycl_t = vllm::xpu::SyclTypeTrait<scalar_t>::Type;
    launch_1d(X.size(0) * row, [&](sycl::handler& cgh, sycl::nd_range<1> range) {
      cgh.parallel_for(
          range,
          ggml_mul_mat_kernel<sycl_t>(
              reinterpret_cast<sycl_t*>(out.data_ptr<scalar_t>()),
              reinterpret_cast<const sycl_t*>(X.data_ptr<scalar_t>()),
              W.data_ptr<uint8_t>(),
              X.stride(0),
              out.stride(0),
              W.size(1),
              X.size(0),
              row,
              X.size(1),
              type));
    });
  });
  return out;
}

torch::Tensor
ggml_mul_mat_vec_a8(torch::Tensor W, torch::Tensor X, int64_t type, int64_t row) {
  return ggml_mul_mat_qx_0(W, X, type, row);
}

torch::Tensor
ggml_mul_mat_a8(torch::Tensor W, torch::Tensor X, int64_t type, int64_t row) {
  return ggml_mul_mat_qx_0(W, X, type, row);
}
