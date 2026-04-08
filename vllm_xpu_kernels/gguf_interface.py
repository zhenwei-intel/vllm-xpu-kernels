# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import vllm_xpu_kernels._C  # noqa: F401  # Ensure GGUF ops are registered.
import torch

def ggml_dequantize(
    W: torch.Tensor,
    type: int,
    m: int,
    n: int,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    return torch.ops._C.ggml_dequantize(W, type, m, n, dtype)


def ggml_mul_mat_vec_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    type: int,
    row: int,
) -> torch.Tensor:
    return torch.ops._C.ggml_mul_mat_vec_a8(W, X, type, row)


def ggml_mul_mat_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    type: int,
    row: int,
) -> torch.Tensor:
    return torch.ops._C.ggml_mul_mat_a8(W, X, type, row)
