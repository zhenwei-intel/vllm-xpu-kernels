#include <sycl/sycl.hpp>

#include <algorithm>
#include "utils.h"
#include "dispatch_utils.h"
#include "ggml-common.h"

namespace vllm {

template <typename scalar_t>
static void quantize_q8_1(const scalar_t* __restrict__ x,
                                     void* __restrict__ vy, const int kx,
                                     const int kx_padded,
                                     const sycl::nd_item<3> &item_ct1) {
  const auto ix = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);
  if (ix >= kx_padded) {
    return;
  }
  const auto iy = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                  item_ct1.get_local_id(1);
  const int i_padded = iy * kx_padded + ix;

  block_q8_1* y = (block_q8_1*)vy;

  const int ib = i_padded / QK8_1;   // block index
  const int iqs = i_padded % QK8_1;  // quant index

  const float xi = ix < kx ? static_cast<float>(x[iy * kx + ix]) : 0.0f;
  float amax = sycl::fabs((float)xi);
  float sum = xi;

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, VLLM_SHFL_XOR_SYNC_WIDTH(amax, mask, 32));
    sum += VLLM_SHFL_XOR_SYNC_WIDTH(sum, mask, 32);
  }

  const float d = amax / 127;
  const int8_t q = amax == 0.0f ? 0 : sycl::round(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }

  y[ib].ds.x() = sycl::vec<float, 1>(d)
                     .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
  y[ib].ds.y() = sycl::vec<float, 1>(sum)
                     .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

template <typename scalar_t>
static void quantize_row_q8_1_cuda(const scalar_t* x, void* vy, const int kx,
                                   const int ky, dpct::queue_ptr stream) {
  const int64_t kx_padded = (kx + 512 - 1) / 512 * 512;
  const int block_num_x =
      (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  constexpr int MAX_BLOCK_SIZE = 65535;
  for (int off = 0; off < ky; off += MAX_BLOCK_SIZE) {
    const int num_blocks_y = std::min(ky, off + MAX_BLOCK_SIZE) - off;
    const dpct::dim3 num_blocks(block_num_x, num_blocks_y, 1);
    const dpct::dim3 block_size(CUDA_DEQUANTIZE_BLOCK_SIZE, 1, 1);
        stream->submit([&](sycl::handler& cgh) {
            auto x_off_kx_ct0 = &x[off * kx];
            auto int32_t_vy_off_kx_padded_ct1 =
                (int32_t*)vy + off * (kx_padded / 32 * 9);

            cgh.parallel_for(
                sycl::nd_range<3>(num_blocks * block_size, block_size),
                [=](sycl::nd_item<3> item_ct1) {
                    quantize_q8_1(x_off_kx_ct0, int32_t_vy_off_kx_padded_ct1,
                                  kx, kx_padded, item_ct1);
                });
        });
  }
}

}