# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

import torch

import vllm_xpu_kernels._C  # noqa: F401


class GGUFQuantType(IntEnum):
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14


_ALIASES = {
    "q4_0": GGUFQuantType.Q4_0,
    "q4_1": GGUFQuantType.Q4_1,
    "q5_0": GGUFQuantType.Q5_0,
    "q5_1": GGUFQuantType.Q5_1,
    "q8_0": GGUFQuantType.Q8_0,
    "q2_k": GGUFQuantType.Q2_K,
    "q3_k": GGUFQuantType.Q3_K,
    "q4_k": GGUFQuantType.Q4_K,
    "q5_k": GGUFQuantType.Q5_K,
    "q6_k": GGUFQuantType.Q6_K,
}

_TRAITS = {
    GGUFQuantType.Q4_0: (32, 18),
    GGUFQuantType.Q4_1: (32, 20),
    GGUFQuantType.Q5_0: (32, 22),
    GGUFQuantType.Q5_1: (32, 24),
    GGUFQuantType.Q8_0: (32, 34),
    GGUFQuantType.Q2_K: (256, 84),
    GGUFQuantType.Q3_K: (256, 110),
    GGUFQuantType.Q4_K: (256, 144),
    GGUFQuantType.Q5_K: (256, 176),
    GGUFQuantType.Q6_K: (256, 210),
}


def _normalize_ggml_type(ggml_type: int | str | GGUFQuantType) -> GGUFQuantType:
    if isinstance(ggml_type, GGUFQuantType):
        return ggml_type
    if isinstance(ggml_type, str):
        key = ggml_type.strip().lower()
        if key not in _ALIASES:
            raise ValueError(f"Unsupported GGUF quantization type: {ggml_type}")
        return _ALIASES[key]
    return GGUFQuantType(int(ggml_type))


def gguf_dequantize(
    packed: torch.Tensor,
    ggml_type: int | str | GGUFQuantType,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    qtype = _normalize_ggml_type(ggml_type)
    values_per_block, bytes_per_block = _TRAITS[qtype]

    if packed.dtype != torch.uint8:
        raise TypeError(f"packed must be torch.uint8, got {packed.dtype}")
    if packed.ndim not in (1, 2):
        raise ValueError(f"packed must be 1D or 2D, got ndim={packed.ndim}")
    if packed.shape[-1] % bytes_per_block != 0:
        raise ValueError(
            f"packed.shape[-1]={packed.shape[-1]} is not divisible by "
            f"bytes_per_block={bytes_per_block} for {qtype.name}")

    packed_2d = packed.contiguous()
    squeeze_out = False
    if packed_2d.ndim == 1:
        packed_2d = packed_2d.unsqueeze(0)
        squeeze_out = True

    num_blocks = packed_2d.shape[-1] // bytes_per_block
    out = torch.empty(
        (packed_2d.shape[0], num_blocks * values_per_block),
        device=packed_2d.device,
        dtype=out_dtype,
    )
    torch.ops._C.gguf_dequantize(out, packed_2d, int(qtype))
    return out.squeeze(0) if squeeze_out else out