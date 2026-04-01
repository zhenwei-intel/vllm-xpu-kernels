#pragma once

#include "dispatch_utils.h"
#include "utils.h"

#include <cstdint>
#include <sycl/sycl.hpp>

namespace vllm {
namespace ggml {

constexpr int64_t GGML_TYPE_Q4_0 = 2;
constexpr int64_t GGML_TYPE_Q5_0 = 6;
constexpr int64_t GGML_TYPE_Q8_0 = 8;
constexpr int64_t GGML_TYPE_Q2_K = 10;
constexpr int64_t GGML_TYPE_Q3_K = 11;
constexpr int64_t GGML_TYPE_Q4_K = 12;
constexpr int64_t GGML_TYPE_Q5_K = 13;
constexpr int64_t GGML_TYPE_Q6_K = 14;
constexpr int64_t QK4_0 = 32;
constexpr int64_t QK5_0 = 32;
constexpr int64_t QK8_0 = 32;
constexpr int64_t QK_K = 256;
constexpr int64_t K_SCALE_SIZE = 12;

struct block_q4_0 {
  sycl::half d;
  uint8_t qs[QK4_0 / 2];
};

static_assert(sizeof(block_q4_0) == 18, "Unexpected Q4_0 block size");

struct block_q5_0 {
  sycl::half d;
  uint8_t qh[4];
  uint8_t qs[QK5_0 / 2];
};

static_assert(sizeof(block_q5_0) == 22, "Unexpected Q5_0 block size");

struct block_q8_0 {
  sycl::half d;
  int8_t qs[QK8_0];
};

static_assert(sizeof(block_q8_0) == 34, "Unexpected Q8_0 block size");

struct block_q2_K {
  uint8_t scales[QK_K / 16];
  uint8_t qs[QK_K / 4];
  sycl::half dm[2];
};

static_assert(sizeof(block_q2_K) == 84, "Unexpected Q2_K block size");

struct block_q3_K {
  uint8_t hmask[QK_K / 8];
  uint8_t qs[QK_K / 4];
  uint8_t scales[K_SCALE_SIZE];
  sycl::half d;
};

static_assert(sizeof(block_q3_K) == 110, "Unexpected Q3_K block size");

struct block_q4_K {
  sycl::half dm[2];
  uint8_t scales[3 * QK_K / 64];
  uint8_t qs[QK_K / 2];
};

static_assert(sizeof(block_q4_K) == 144, "Unexpected Q4_K block size");

struct block_q5_K {
  sycl::half dm[2];
  uint8_t scales[K_SCALE_SIZE];
  uint8_t qh[QK_K / 8];
  uint8_t qs[QK_K / 2];
};

static_assert(sizeof(block_q5_K) == 176, "Unexpected Q5_K block size");

struct block_q6_K {
  uint8_t ql[QK_K / 2];
  uint8_t qh[QK_K / 4];
  int8_t scales[QK_K / 16];
  sycl::half d;
};

static_assert(sizeof(block_q6_K) == 210, "Unexpected Q6_K block size");

inline uint32_t load_u32_le(const uint8_t* bytes) {
  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8) |
         (static_cast<uint32_t>(bytes[2]) << 16) |
         (static_cast<uint32_t>(bytes[3]) << 24);
}

inline int unpack_q3_scale(const uint8_t* scales, int index) {
  if (index < 4) {
    return (scales[index] & 0x0F) | (((scales[index + 8] >> 0) & 0x03) << 4);
  }
  if (index < 8) {
    return (scales[index] & 0x0F) | (((scales[index + 4] >> 2) & 0x03) << 4);
  }
  if (index < 12) {
    return (scales[index - 8] >> 4) |
           (((scales[index] >> 4) & 0x03) << 4);
  }
  return (scales[index - 8] >> 4) |
         (((scales[index - 4] >> 6) & 0x03) << 4);
}

inline void get_scale_min_k4(int index, const uint8_t* scales, uint8_t& d, uint8_t& m) {
  if (index < 4) {
    d = scales[index] & 0x3F;
    m = scales[index + 4] & 0x3F;
    return;
  }

  d = (scales[index + 4] & 0x0F) | ((scales[index - 4] >> 6) << 4);
  m = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
}

template <typename scalar_t>
class ggml_dequantize_q4_0_kernel {
 public:
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;

  ggml_dequantize_q4_0_kernel(
      const block_q4_0* blocks, sycl_t* out, int64_t numel)
      : blocks_(blocks), out_(out), numel_(numel) {}

  void operator()(sycl::id<1> index) const {
    const int64_t i = index[0];
    if (i >= numel_) {
      return;
    }

    const int64_t block_index = i / QK4_0;
    const int64_t block_offset = i % QK4_0;
    const block_q4_0& block = blocks_[block_index];
    const bool is_high_half = block_offset >= (QK4_0 / 2);
    const int64_t quant_index =
        is_high_half ? (block_offset - QK4_0 / 2) : block_offset;
    const uint8_t packed = block.qs[quant_index];
    const int quant = is_high_half ? (packed >> 4) : (packed & 0x0F);
    const float value =
        (static_cast<float>(quant) - 8.0f) * static_cast<float>(block.d);
    out_[i] = static_cast<sycl_t>(value);
  }

 private:
  const block_q4_0* blocks_;
  sycl_t* out_;
  int64_t numel_;
};

template <typename scalar_t>
class ggml_dequantize_q5_0_kernel {
 public:
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;

  ggml_dequantize_q5_0_kernel(
      const block_q5_0* blocks, sycl_t* out, int64_t numel)
      : blocks_(blocks), out_(out), numel_(numel) {}

  void operator()(sycl::id<1> index) const {
    const int64_t i = index[0];
    if (i >= numel_) {
      return;
    }

    const int64_t block_index = i / QK5_0;
    const int64_t block_offset = i % QK5_0;
    const block_q5_0& block = blocks_[block_index];
    const bool is_high_half = block_offset >= (QK5_0 / 2);
    const int64_t quant_index =
        is_high_half ? (block_offset - QK5_0 / 2) : block_offset;
    const uint8_t packed = block.qs[quant_index];
    const uint32_t qh = load_u32_le(block.qh);
    const int xh = is_high_half ? ((qh >> (quant_index + 12)) & 0x10)
                                : (((qh >> quant_index) << 4) & 0x10);
    const int base_quant = is_high_half ? (packed >> 4) : (packed & 0x0F);
    const float value = (static_cast<float>(base_quant | xh) - 16.0f) *
                        static_cast<float>(block.d);
    out_[i] = static_cast<sycl_t>(value);
  }

 private:
  const block_q5_0* blocks_;
  sycl_t* out_;
  int64_t numel_;
};

template <typename scalar_t>
class ggml_dequantize_q8_0_kernel {
 public:
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;

  ggml_dequantize_q8_0_kernel(
      const block_q8_0* blocks, sycl_t* out, int64_t numel)
      : blocks_(blocks), out_(out), numel_(numel) {}

  void operator()(sycl::id<1> index) const {
    const int64_t i = index[0];
    if (i >= numel_) {
      return;
    }

    const int64_t block_index = i / QK8_0;
    const int64_t block_offset = i % QK8_0;
    const block_q8_0& block = blocks_[block_index];
    const float value = static_cast<float>(block.qs[block_offset]) *
                        static_cast<float>(block.d);
    out_[i] = static_cast<sycl_t>(value);
  }

 private:
  const block_q8_0* blocks_;
  sycl_t* out_;
  int64_t numel_;
};

template <typename scalar_t>
class ggml_dequantize_q2_K_kernel {
 public:
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;

  ggml_dequantize_q2_K_kernel(
      const block_q2_K* blocks, sycl_t* out, int64_t numel)
      : blocks_(blocks), out_(out), numel_(numel) {}

  void operator()(sycl::id<1> index) const {
    const int64_t i = index[0];
    if (i >= numel_) {
      return;
    }

    const int64_t block_index = i / QK_K;
    const int64_t block_offset = i % QK_K;
    const block_q2_K& block = blocks_[block_index];
    const int64_t n = block_offset / 128;
    const int64_t offset_in_half = block_offset % 128;
    const int64_t lane = offset_in_half % 32;
    const int64_t segment = offset_in_half / 32;
    const int64_t scale_index = 8 * n + lane / 16 + 2 * segment;
    const uint8_t packed = block.qs[32 * n + lane];
    const int quant = (packed >> (2 * segment)) & 0x03;
    const uint8_t scale = block.scales[scale_index];
    const float d = static_cast<float>(block.dm[0]);
    const float dmin = static_cast<float>(block.dm[1]);
    const float value = d * static_cast<float>(scale & 0x0F) * quant -
                        dmin * static_cast<float>(scale >> 4);
    out_[i] = static_cast<sycl_t>(value);
  }

 private:
  const block_q2_K* blocks_;
  sycl_t* out_;
  int64_t numel_;
};

template <typename scalar_t>
class ggml_dequantize_q3_K_kernel {
 public:
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;

  ggml_dequantize_q3_K_kernel(
      const block_q3_K* blocks, sycl_t* out, int64_t numel)
      : blocks_(blocks), out_(out), numel_(numel) {}

  void operator()(sycl::id<1> index) const {
    const int64_t i = index[0];
    if (i >= numel_) {
      return;
    }

    const int64_t block_index = i / QK_K;
    const int64_t block_offset = i % QK_K;
    const block_q3_K& block = blocks_[block_index];
    const int64_t n = block_offset / 128;
    const int64_t offset_in_half = block_offset % 128;
    const int64_t j = offset_in_half / 32;
    const int64_t lane = offset_in_half % 32;
    const int64_t scale_index = 8 * n + 2 * j + lane / 16;
    const int scale = unpack_q3_scale(block.scales, scale_index) - 32;
    const uint8_t ql = (block.qs[32 * n + lane] >> (2 * j)) & 0x03;
    const uint8_t mask = 1u << (4 * n + j);
    const int quant = static_cast<int>(ql) - ((block.hmask[lane] & mask) ? 0 : 4);
    const float value = static_cast<float>(block.d) * static_cast<float>(scale) *
                        static_cast<float>(quant);
    out_[i] = static_cast<sycl_t>(value);
  }

 private:
  const block_q3_K* blocks_;
  sycl_t* out_;
  int64_t numel_;
};

template <typename scalar_t>
class ggml_dequantize_q4_K_kernel {
 public:
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;

  ggml_dequantize_q4_K_kernel(
      const block_q4_K* blocks, sycl_t* out, int64_t numel)
      : blocks_(blocks), out_(out), numel_(numel) {}

  void operator()(sycl::id<1> index) const {
    const int64_t i = index[0];
    if (i >= numel_) {
      return;
    }

    const int64_t block_index = i / QK_K;
    const int64_t block_offset = i % QK_K;
    const block_q4_K& block = blocks_[block_index];
    const int64_t group = block_offset / 32;
    const int64_t offset_in_group = block_offset % 32;
    const int64_t chunk = group / 2;
    uint8_t scale;
    uint8_t min;
    get_scale_min_k4(static_cast<int>(group), block.scales, scale, min);
    const uint8_t packed = block.qs[32 * chunk + offset_in_group];
    const int quant = (group % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
    const float value = static_cast<float>(block.dm[0]) *
                            static_cast<float>(scale) * static_cast<float>(quant) -
                        static_cast<float>(block.dm[1]) * static_cast<float>(min);
    out_[i] = static_cast<sycl_t>(value);
  }

 private:
  const block_q4_K* blocks_;
  sycl_t* out_;
  int64_t numel_;
};

template <typename scalar_t>
class ggml_dequantize_q5_K_kernel {
 public:
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;

  ggml_dequantize_q5_K_kernel(
      const block_q5_K* blocks, sycl_t* out, int64_t numel)
      : blocks_(blocks), out_(out), numel_(numel) {}

  void operator()(sycl::id<1> index) const {
    const int64_t i = index[0];
    if (i >= numel_) {
      return;
    }

    const int64_t block_index = i / QK_K;
    const int64_t block_offset = i % QK_K;
    const block_q5_K& block = blocks_[block_index];
    const int64_t group = block_offset / 32;
    const int64_t offset_in_group = block_offset % 32;
    const int64_t chunk = group / 2;
    uint8_t scale;
    uint8_t min;
    get_scale_min_k4(static_cast<int>(group), block.scales, scale, min);
    const uint8_t packed = block.qs[32 * chunk + offset_in_group];
    const int ql = (group % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
    const int qh = ((block.qh[offset_in_group] >> group) & 0x01) << 4;
    const float value = static_cast<float>(block.dm[0]) *
                            static_cast<float>(scale) *
                            static_cast<float>(ql | qh) -
                        static_cast<float>(block.dm[1]) * static_cast<float>(min);
    out_[i] = static_cast<sycl_t>(value);
  }

 private:
  const block_q5_K* blocks_;
  sycl_t* out_;
  int64_t numel_;
};

template <typename scalar_t>
class ggml_dequantize_q6_K_kernel {
 public:
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;

  ggml_dequantize_q6_K_kernel(
      const block_q6_K* blocks, sycl_t* out, int64_t numel)
      : blocks_(blocks), out_(out), numel_(numel) {}

  void operator()(sycl::id<1> index) const {
    const int64_t i = index[0];
    if (i >= numel_) {
      return;
    }

    const int64_t block_index = i / QK_K;
    const int64_t block_offset = i % QK_K;
    const block_q6_K& block = blocks_[block_index];
    const int64_t n = block_offset / 128;
    const int64_t offset_in_half = block_offset % 128;
    const int64_t segment = offset_in_half / 32;
    const int64_t lane = offset_in_half % 32;
    const int64_t scale_index = 8 * n + lane / 16 + 2 * segment;
    const int64_t ql_index = 64 * n + ((segment & 1) ? 32 : 0) + lane;
    const uint8_t ql = block.ql[ql_index];
    const int low = segment >= 2 ? (ql >> 4) : (ql & 0x0F);
    const int high = (block.qh[32 * n + lane] >> (2 * segment)) & 0x03;
    const int quant = (low | (high << 4)) - 32;
    const float value = static_cast<float>(block.d) *
                        static_cast<float>(block.scales[scale_index]) *
                        static_cast<float>(quant);
    out_[i] = static_cast<sycl_t>(value);
  }

 private:
  const block_q6_K* blocks_;
  sycl_t* out_;
  int64_t numel_;
};

inline int64_t get_expected_nbytes(int64_t type, int64_t numel) {
  switch (type) {
    case GGML_TYPE_Q4_0:
      return (numel / QK4_0) * static_cast<int64_t>(sizeof(block_q4_0));
    case GGML_TYPE_Q5_0:
      return (numel / QK5_0) * static_cast<int64_t>(sizeof(block_q5_0));
    case GGML_TYPE_Q8_0:
      return (numel / QK8_0) * static_cast<int64_t>(sizeof(block_q8_0));
    case GGML_TYPE_Q2_K:
      return (numel / QK_K) * static_cast<int64_t>(sizeof(block_q2_K));
    case GGML_TYPE_Q3_K:
      return (numel / QK_K) * static_cast<int64_t>(sizeof(block_q3_K));
    case GGML_TYPE_Q4_K:
      return (numel / QK_K) * static_cast<int64_t>(sizeof(block_q4_K));
    case GGML_TYPE_Q5_K:
      return (numel / QK_K) * static_cast<int64_t>(sizeof(block_q5_K));
    case GGML_TYPE_Q6_K:
      return (numel / QK_K) * static_cast<int64_t>(sizeof(block_q6_K));
    default:
      return -1;
  }
}

inline int64_t get_quant_block_size(int64_t type) {
  switch (type) {
    case GGML_TYPE_Q4_0:
      return QK4_0;
    case GGML_TYPE_Q5_0:
      return QK5_0;
    case GGML_TYPE_Q8_0:
      return QK8_0;
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
      return QK_K;
    default:
      return -1;
  }
}

inline const char* ggml_type_name(int64_t type) {
  switch (type) {
    case GGML_TYPE_Q4_0:
      return "Q4_0";
    case GGML_TYPE_Q5_0:
      return "Q5_0";
    case GGML_TYPE_Q8_0:
      return "Q8_0";
    case GGML_TYPE_Q2_K:
      return "Q2_K";
    case GGML_TYPE_Q3_K:
      return "Q3_K";
    case GGML_TYPE_Q4_K:
      return "Q4_K";
    case GGML_TYPE_Q5_K:
      return "Q5_K";
    case GGML_TYPE_Q6_K:
      return "Q6_K";
    default:
      return "unknown";
  }
}

}  // namespace ggml
}  // namespace vllm