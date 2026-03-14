#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/xpu/XPUContext.h>
#include <sycl/sycl.hpp>

#include <cstdint>

#include "dispatch_utils.h"
#include "ops.h"
#include "utils.h"

namespace {

constexpr int kQK4_0 = 32;
constexpr int kQK4_1 = 32;
constexpr int kQK5_0 = 32;
constexpr int kQK5_1 = 32;
constexpr int kQK8_0 = 32;
constexpr int kQK_K = 256;
constexpr int kKScaleSize = 12;

enum class GGUFQuantType : int64_t {
  kQ4_0 = 2,
  kQ4_1 = 3,
  kQ5_0 = 6,
  kQ5_1 = 7,
  kQ8_0 = 8,
  kQ2_K = 10,
  kQ3_K = 11,
  kQ4_K = 12,
  kQ5_K = 13,
  kQ6_K = 14,
};

struct GGUFTypeTraits {
  int values_per_block;
  int bytes_per_block;
  const char* name;
};

constexpr GGUFTypeTraits get_type_traits(GGUFQuantType type) {
  switch (type) {
    case GGUFQuantType::kQ4_0:
      return {kQK4_0, 2 + kQK4_0 / 2, "q4_0"};
    case GGUFQuantType::kQ4_1:
      return {kQK4_1, 2 * 2 + kQK4_1 / 2, "q4_1"};
    case GGUFQuantType::kQ5_0:
      return {kQK5_0, 2 + 4 + kQK5_0 / 2, "q5_0"};
    case GGUFQuantType::kQ5_1:
      return {kQK5_1, 2 * 2 + 4 + kQK5_1 / 2, "q5_1"};
    case GGUFQuantType::kQ8_0:
      return {kQK8_0, 2 + kQK8_0, "q8_0"};
    case GGUFQuantType::kQ2_K:
      return {kQK_K, 2 * 2 + kQK_K / 16 + kQK_K / 4, "q2_K"};
    case GGUFQuantType::kQ3_K:
      return {kQK_K, 2 + kQK_K / 4 + kQK_K / 8 + 12, "q3_K"};
    case GGUFQuantType::kQ4_K:
      return {kQK_K, 2 * 2 + kKScaleSize + kQK_K / 2, "q4_K"};
    case GGUFQuantType::kQ5_K:
      return {kQK_K, 2 * 2 + kKScaleSize + kQK_K / 8 + kQK_K / 2, "q5_K"};
    case GGUFQuantType::kQ6_K:
      return {kQK_K, 2 + kQK_K / 16 + 3 * kQK_K / 4, "q6_K"};
  }

  return {0, 0, "unknown"};
}

inline GGUFQuantType parse_quant_type(int64_t ggml_type) {
  switch (ggml_type) {
    case 2:
      return GGUFQuantType::kQ4_0;
    case 3:
      return GGUFQuantType::kQ4_1;
    case 6:
      return GGUFQuantType::kQ5_0;
    case 7:
      return GGUFQuantType::kQ5_1;
    case 8:
      return GGUFQuantType::kQ8_0;
    case 10:
      return GGUFQuantType::kQ2_K;
    case 11:
      return GGUFQuantType::kQ3_K;
    case 12:
      return GGUFQuantType::kQ4_K;
    case 13:
      return GGUFQuantType::kQ5_K;
    case 14:
      return GGUFQuantType::kQ6_K;
    default:
      TORCH_CHECK(false, "Unsupported GGUF ggml_type: ", ggml_type);
  }
}

inline uint16_t load_u16_le(const uint8_t* ptr) {
  return static_cast<uint16_t>(ptr[0]) |
         (static_cast<uint16_t>(ptr[1]) << 8);
}

inline uint32_t load_u32_le(const uint8_t* ptr) {
  return static_cast<uint32_t>(ptr[0]) |
         (static_cast<uint32_t>(ptr[1]) << 8) |
         (static_cast<uint32_t>(ptr[2]) << 16) |
         (static_cast<uint32_t>(ptr[3]) << 24);
}

inline float load_f16_le(const uint8_t* ptr) {
  return static_cast<float>(sycl::bit_cast<sycl::half>(load_u16_le(ptr)));
}

template <typename dst_t>
inline void write_value(dst_t* dst, int index, float value) {
  dst[index] = static_cast<dst_t>(value);
}

template <typename dst_t>
inline void dequantize_block_q4_0(const uint8_t* src, dst_t* dst) {
  const float d = load_f16_le(src);
  const uint8_t* qs = src + 2;

  for (int i = 0; i < kQK4_0 / 2; ++i) {
    const uint8_t q = qs[i];
    write_value(dst, i, d * (static_cast<float>(q & 0x0F) - 8.0f));
    write_value(dst, i + 16, d * (static_cast<float>(q >> 4) - 8.0f));
  }
}

template <typename dst_t>
inline void dequantize_block_q4_1(const uint8_t* src, dst_t* dst) {
  const float d = load_f16_le(src);
  const float m = load_f16_le(src + 2);
  const uint8_t* qs = src + 4;

  for (int i = 0; i < kQK4_1 / 2; ++i) {
    const uint8_t q = qs[i];
    write_value(dst, i, sycl::fma(static_cast<float>(q & 0x0F), d, m));
    write_value(dst, i + 16, sycl::fma(static_cast<float>(q >> 4), d, m));
  }
}

template <typename dst_t>
inline void dequantize_block_q5_0(const uint8_t* src, dst_t* dst) {
  const float d = load_f16_le(src);
  const uint32_t qh = load_u32_le(src + 2);
  const uint8_t* qs = src + 6;

  for (int i = 0; i < kQK5_0 / 2; ++i) {
    const uint8_t q = qs[i];
    const int xh0 = ((qh >> i) << 4) & 0x10;
    const int xh1 = ((qh >> (i + 12))) & 0x10;
    write_value(dst, i, d * (static_cast<float>((q & 0x0F) | xh0) - 16.0f));
    write_value(dst, i + 16, d * (static_cast<float>((q >> 4) | xh1) - 16.0f));
  }
}

template <typename dst_t>
inline void dequantize_block_q5_1(const uint8_t* src, dst_t* dst) {
  const float d = load_f16_le(src);
  const float m = load_f16_le(src + 2);
  const uint32_t qh = load_u32_le(src + 4);
  const uint8_t* qs = src + 8;

  for (int i = 0; i < kQK5_1 / 2; ++i) {
    const uint8_t q = qs[i];
    const int xh0 = ((qh >> i) << 4) & 0x10;
    const int xh1 = ((qh >> (i + 12))) & 0x10;
    write_value(
        dst,
        i,
        sycl::fma(static_cast<float>((q & 0x0F) | xh0), d, m));
    write_value(
        dst,
        i + 16,
        sycl::fma(static_cast<float>((q >> 4) | xh1), d, m));
  }
}

template <typename dst_t>
inline void dequantize_block_q8_0(const uint8_t* src, dst_t* dst) {
  const float d = load_f16_le(src);
  const int8_t* qs = reinterpret_cast<const int8_t*>(src + 2);

  for (int i = 0; i < kQK8_0; ++i) {
    write_value(dst, i, d * static_cast<float>(qs[i]));
  }
}

inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

template <typename dst_t>
inline void dequantize_block_q2_k(const uint8_t* src, dst_t* dst) {
  const uint8_t* scales = src;
  const uint8_t* qs = src + kQK_K / 16;
  const float dall = load_f16_le(src + kQK_K / 16 + kQK_K / 4);
  const float dmin = load_f16_le(src + kQK_K / 16 + kQK_K / 4 + 2);

  for (int n = 0; n < 2; ++n) {
    const uint8_t* q_block = qs + 32 * n;
    dst_t* y = dst + 128 * n;
    for (int l = 0; l < 32; ++l) {
      const int is = 8 * n + l / 16;
      const uint8_t q = q_block[l];
      write_value(
          y,
          l,
          dall * static_cast<float>(scales[is] & 0x0F) *
                  static_cast<float>((q >> 0) & 0x3) -
              dmin * static_cast<float>(scales[is] >> 4));
      write_value(
          y,
          l + 32,
          dall * static_cast<float>(scales[is + 2] & 0x0F) *
                  static_cast<float>((q >> 2) & 0x3) -
              dmin * static_cast<float>(scales[is + 2] >> 4));
      write_value(
          y,
          l + 64,
          dall * static_cast<float>(scales[is + 4] & 0x0F) *
                  static_cast<float>((q >> 4) & 0x3) -
              dmin * static_cast<float>(scales[is + 4] >> 4));
      write_value(
          y,
          l + 96,
          dall * static_cast<float>(scales[is + 6] & 0x0F) *
                  static_cast<float>((q >> 6) & 0x3) -
              dmin * static_cast<float>(scales[is + 6] >> 4));
    }
  }
}

inline uint8_t decode_q3_scale_byte(const uint8_t* scales, int is) {
  if (is < 4) {
    return (scales[is] & 0x0F) | (((scales[is + 8] >> 0) & 0x03) << 4);
  }
  if (is < 8) {
    return (scales[is] & 0x0F) | (((scales[is + 4] >> 2) & 0x03) << 4);
  }
  if (is < 12) {
    return (scales[is - 8] >> 4) | (((scales[is] >> 4) & 0x03) << 4);
  }
  return (scales[is - 8] >> 4) | (((scales[is - 4] >> 6) & 0x03) << 4);
}

template <typename dst_t>
inline void dequantize_block_q3_k(const uint8_t* src, dst_t* dst) {
  const uint8_t* hmask = src;
  const uint8_t* qs = src + kQK_K / 8;
  const uint8_t* scales = src + kQK_K / 8 + kQK_K / 4;
  const float d = load_f16_le(src + kQK_K / 8 + kQK_K / 4 + 12);

  for (int n = 0; n < 2; ++n) {
    const uint8_t* q = qs + 32 * n;
    const uint8_t* hm = hmask + 32 * n;
    for (int j = 0; j < 4; ++j) {
      dst_t* y = dst + 128 * n + 32 * j;
      const uint8_t m = static_cast<uint8_t>(1u << (4 * n + j));
      const int shift = 2 * j;
      for (int is0 = 0; is0 < 2; ++is0) {
        const int is = 8 * n + 2 * j + is0;
        const float dl = d * (static_cast<float>(decode_q3_scale_byte(scales, is)) - 32.0f);
        for (int l = 16 * is0; l < 16 * is0 + 16; ++l) {
          const int ql = (q[l] >> shift) & 0x03;
          const int sign = (hm[l] & m) ? 0 : 4;
          write_value(y, l, dl * static_cast<float>(ql - sign));
        }
      }
    }
  }
}

template <typename dst_t>
inline void dequantize_q4_k_common(
    dst_t* dst,
    const uint8_t* qs,
    float dall,
    float dmin,
    const uint8_t* scales,
    int il,
    int ir) {
  const int is = 2 * il;
  uint8_t sc;
  uint8_t m;
  get_scale_min_k4(is, scales, sc, m);
  const float d1 = dall * static_cast<float>(sc);
  const float m1 = dmin * static_cast<float>(m);

  get_scale_min_k4(is + 1, scales, sc, m);
  const float d2 = dall * static_cast<float>(sc);
  const float m2 = dmin * static_cast<float>(m);

  const uint8_t* q = qs + 32 * il + 4 * ir;
  for (int l = 0; l < 4; ++l) {
    write_value(dst, l, d1 * static_cast<float>(q[l] & 0x0F) - m1);
    write_value(dst, l + 32, d2 * static_cast<float>(q[l] >> 4) - m2);
  }
}

template <typename dst_t>
inline void dequantize_block_q4_k(const uint8_t* src, dst_t* dst) {
  const float dall = load_f16_le(src);
  const float dmin = load_f16_le(src + 2);
  const uint8_t* scales = src + 4;
  const uint8_t* qs = src + 4 + kKScaleSize;

  for (int il = 0; il < 4; ++il) {
    for (int ir = 0; ir < 8; ++ir) {
      dequantize_q4_k_common(
          dst + 64 * il + 4 * ir, qs, dall, dmin, scales, il, ir);
    }
  }
}

template <typename dst_t>
inline void dequantize_block_q5_k(const uint8_t* src, dst_t* dst) {
  const float dall = load_f16_le(src);
  const float dmin = load_f16_le(src + 2);
  const uint8_t* scales = src + 4;
  const uint8_t* qh = src + 4 + kKScaleSize;
  const uint8_t* qs = src + 4 + kKScaleSize + kQK_K / 8;

  for (int il = 0; il < 4; ++il) {
    const int is = 2 * il;
    uint8_t sc;
    uint8_t m;
    get_scale_min_k4(is, scales, sc, m);
    const float d1 = dall * static_cast<float>(sc);
    const float m1 = dmin * static_cast<float>(m);
    get_scale_min_k4(is + 1, scales, sc, m);
    const float d2 = dall * static_cast<float>(sc);
    const float m2 = dmin * static_cast<float>(m);

    for (int ir = 0; ir < 16; ++ir) {
      dst_t* y = dst + 64 * il + 2 * ir;
      const uint8_t* ql = qs + 32 * il + 2 * ir;
      const uint8_t* qh_block = qh + 2 * ir;
      uint8_t hm = static_cast<uint8_t>(1u << (2 * il));
      write_value(
          y,
          0,
          d1 * static_cast<float>((ql[0] & 0x0F) + ((qh_block[0] & hm) ? 16 : 0)) - m1);
      write_value(
          y,
          1,
          d1 * static_cast<float>((ql[1] & 0x0F) + ((qh_block[1] & hm) ? 16 : 0)) - m1);
      hm <<= 1;
      write_value(
          y,
          32,
          d2 * static_cast<float>((ql[0] >> 4) + ((qh_block[0] & hm) ? 16 : 0)) - m2);
      write_value(
          y,
          33,
          d2 * static_cast<float>((ql[1] >> 4) + ((qh_block[1] & hm) ? 16 : 0)) - m2);
    }
  }
}

template <typename dst_t>
inline void dequantize_block_q6_k(const uint8_t* src, dst_t* dst) {
  const uint8_t* ql = src;
  const uint8_t* qh = src + kQK_K / 2;
  const int8_t* scales = reinterpret_cast<const int8_t*>(src + kQK_K / 2 + kQK_K / 4);
  const float d = load_f16_le(src + kQK_K / 2 + kQK_K / 4 + kQK_K / 16);

  for (int ip = 0; ip < 2; ++ip) {
    for (int il = 0; il < 32; ++il) {
      dst_t* y = dst + 128 * ip + il;
      const int is = 8 * ip + il / 16;
      const uint8_t qh_byte = qh[32 * ip + il];
      const uint8_t* ql_ptr = ql + 64 * ip + il;
      const int8_t* sc = scales + is;

      write_value(
          y,
          0,
          d * static_cast<float>(sc[0]) *
              static_cast<float>(static_cast<int8_t>((ql_ptr[0] & 0x0F) | (((qh_byte >> 0) & 0x03) << 4)) - 32));
      write_value(
          y,
          32,
          d * static_cast<float>(sc[2]) *
              static_cast<float>(static_cast<int8_t>((ql_ptr[32] & 0x0F) | (((qh_byte >> 2) & 0x03) << 4)) - 32));
      write_value(
          y,
          64,
          d * static_cast<float>(sc[4]) *
              static_cast<float>(static_cast<int8_t>((ql_ptr[0] >> 4) | (((qh_byte >> 4) & 0x03) << 4)) - 32));
      write_value(
          y,
          96,
          d * static_cast<float>(sc[6]) *
              static_cast<float>(static_cast<int8_t>((ql_ptr[32] >> 4) | (((qh_byte >> 6) & 0x03) << 4)) - 32));
    }
  }
}

template <typename dst_t, GGUFQuantType qtype>
inline void dequantize_block(const uint8_t* src, dst_t* dst) {
  if constexpr (qtype == GGUFQuantType::kQ4_0) {
    dequantize_block_q4_0(src, dst);
  } else if constexpr (qtype == GGUFQuantType::kQ4_1) {
    dequantize_block_q4_1(src, dst);
  } else if constexpr (qtype == GGUFQuantType::kQ5_0) {
    dequantize_block_q5_0(src, dst);
  } else if constexpr (qtype == GGUFQuantType::kQ5_1) {
    dequantize_block_q5_1(src, dst);
  } else if constexpr (qtype == GGUFQuantType::kQ8_0) {
    dequantize_block_q8_0(src, dst);
  } else if constexpr (qtype == GGUFQuantType::kQ2_K) {
    dequantize_block_q2_k(src, dst);
  } else if constexpr (qtype == GGUFQuantType::kQ3_K) {
    dequantize_block_q3_k(src, dst);
  } else if constexpr (qtype == GGUFQuantType::kQ4_K) {
    dequantize_block_q4_k(src, dst);
  } else if constexpr (qtype == GGUFQuantType::kQ5_K) {
    dequantize_block_q5_k(src, dst);
  } else if constexpr (qtype == GGUFQuantType::kQ6_K) {
    dequantize_block_q6_k(src, dst);
  }
}

template <typename dst_t, GGUFQuantType qtype>
class gguf_dequantize_kernel {
 public:
  gguf_dequantize_kernel(
      dst_t* out,
      const uint8_t* input,
      int64_t num_rows,
      int64_t num_blocks,
      int64_t input_row_stride,
      int64_t out_row_stride)
      : out_(out),
        input_(input),
        num_rows_(num_rows),
        num_blocks_(num_blocks),
        input_row_stride_(input_row_stride),
        out_row_stride_(out_row_stride) {}

  void operator()(sycl::nd_item<1> item) const {
    const auto traits = get_type_traits(qtype);
    const int64_t total_blocks = num_rows_ * num_blocks_;
    const int64_t global_id = item.get_global_linear_id();
    const int64_t global_stride = item.get_global_range(0);

    for (int64_t linear_idx = global_id; linear_idx < total_blocks;
         linear_idx += global_stride) {
      const int64_t row = linear_idx / num_blocks_;
      const int64_t block = linear_idx % num_blocks_;
      const uint8_t* src = input_ + row * input_row_stride_ +
                           block * traits.bytes_per_block;
      dst_t* dst = out_ + row * out_row_stride_ +
                   block * traits.values_per_block;
      dequantize_block<dst_t, qtype>(src, dst);
    }
  }

 private:
  dst_t* out_;
  const uint8_t* input_;
  int64_t num_rows_;
  int64_t num_blocks_;
  int64_t input_row_stride_;
  int64_t out_row_stride_;
};

template <typename dst_t, GGUFQuantType qtype>
void launch_gguf_dequantize(
    torch::Tensor& out,
    torch::Tensor const& input,
    int64_t num_blocks) {
  const int64_t num_rows = input.size(0);
  const int64_t total_blocks = num_rows * num_blocks;
  constexpr int64_t threads = 256;
  const int64_t global = ((total_blocks + threads - 1) / threads) * threads;

  auto& queue = vllm::xpu::vllmGetQueue();
  queue.submit([&](sycl::handler& cgh) {
    auto kernel = gguf_dequantize_kernel<dst_t, qtype>(
        out.data_ptr<dst_t>(),
        input.data_ptr<uint8_t>(),
        num_rows,
        num_blocks,
        input.stride(0),
        out.stride(0));
    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(threads)), kernel);
  });
}

#define VLLM_GGUF_DEQUANT_CASE(QTYPE, BODY) \
  case GGUFQuantType::QTYPE: {               \
    constexpr GGUFQuantType qtype = GGUFQuantType::QTYPE; \
    BODY();                                  \
    break;                                   \
  }

}  // namespace

void gguf_dequantize(
    torch::Tensor& out,
    torch::Tensor const& input,
    int64_t ggml_type) {
  TORCH_CHECK(input.scalar_type() == at::kByte, "input must have dtype torch.uint8");
  TORCH_CHECK(input.dim() == 2, "input must be a 2D tensor of packed GGUF rows");
  TORCH_CHECK(out.dim() == 2, "out must be a 2D tensor");
  TORCH_CHECK(input.stride(-1) == 1, "input rows must be contiguous");
  TORCH_CHECK(out.stride(-1) == 1, "output rows must be contiguous");
  TORCH_CHECK(input.size(0) == out.size(0), "input and out must have the same number of rows");

  const GGUFQuantType quant_type = parse_quant_type(ggml_type);
  const auto traits = get_type_traits(quant_type);

  TORCH_CHECK(
      input.size(1) % traits.bytes_per_block == 0,
      "input.shape[1]=",
      input.size(1),
      " is not divisible by GGUF block size ",
      traits.bytes_per_block,
      " for ",
      traits.name);

  const int64_t num_blocks = input.size(1) / traits.bytes_per_block;
  TORCH_CHECK(
      out.size(1) == num_blocks * traits.values_per_block,
      "out.shape[1]=",
      out.size(1),
      " does not match expected dequantized row width ",
      num_blocks * traits.values_per_block,
      " for ",
      traits.name);

  if (num_blocks == 0 || input.size(0) == 0) {
    return;
  }

  at::Device cur_device = at::Device(at::kXPU, at::xpu::current_device());
  at::DeviceGuard device_guard(cur_device);

  VLLM_DISPATCH_FLOATING_TYPES(out.scalar_type(), "gguf_dequantize_out_type", [&] {
    switch (quant_type) {
      VLLM_GGUF_DEQUANT_CASE(kQ4_0, [&] { launch_gguf_dequantize<scalar_t, qtype>(out, input, num_blocks); });
      VLLM_GGUF_DEQUANT_CASE(kQ4_1, [&] { launch_gguf_dequantize<scalar_t, qtype>(out, input, num_blocks); });
      VLLM_GGUF_DEQUANT_CASE(kQ5_0, [&] { launch_gguf_dequantize<scalar_t, qtype>(out, input, num_blocks); });
      VLLM_GGUF_DEQUANT_CASE(kQ5_1, [&] { launch_gguf_dequantize<scalar_t, qtype>(out, input, num_blocks); });
      VLLM_GGUF_DEQUANT_CASE(kQ8_0, [&] { launch_gguf_dequantize<scalar_t, qtype>(out, input, num_blocks); });
      VLLM_GGUF_DEQUANT_CASE(kQ2_K, [&] { launch_gguf_dequantize<scalar_t, qtype>(out, input, num_blocks); });
      VLLM_GGUF_DEQUANT_CASE(kQ3_K, [&] { launch_gguf_dequantize<scalar_t, qtype>(out, input, num_blocks); });
      VLLM_GGUF_DEQUANT_CASE(kQ4_K, [&] { launch_gguf_dequantize<scalar_t, qtype>(out, input, num_blocks); });
      VLLM_GGUF_DEQUANT_CASE(kQ5_K, [&] { launch_gguf_dequantize<scalar_t, qtype>(out, input, num_blocks); });
      VLLM_GGUF_DEQUANT_CASE(kQ6_K, [&] { launch_gguf_dequantize<scalar_t, qtype>(out, input, num_blocks); });
    }
  });
}