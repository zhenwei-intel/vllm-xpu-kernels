# SPDX-License-Identifier: Apache-2.0

from vllm_xpu_kernels.quantization._quantize_convert import (
    dequantize_gguf,
    dequantize_gguf_q4_k,
    dequantize_gguf_q8_0,
)

__all__ = [
    "dequantize_gguf",
    "dequantize_gguf_q4_k",
    "dequantize_gguf_q8_0",
]
