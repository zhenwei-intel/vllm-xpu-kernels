# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

import torch

_DEF_LIB = torch.library.Library("_xpu_C", "FRAGMENT")
_DEF_LIB.define(
    "ggml_dequantize(Tensor W, int type, int m, int n, ScalarType? dtype) "
    "-> Tensor")
_DEF_LIB.define("ggml_mul_mat_vec_a8(Tensor W, Tensor X, int type, int row) "
                "-> Tensor")
_DEF_LIB.define(
    "ggml_mul_mat_a8(Tensor W, Tensor X, int type, int row) -> Tensor")

_IMPL_LIB = torch.library.Library("_xpu_C", "IMPL",
                                  "CompositeExplicitAutograd")
_META_LIB = torch.library.Library("_xpu_C", "IMPL", "Meta")

_GGUF_IMPORT_ERROR: Optional[ImportError] = None

try:
    import gguf as _gguf
except ImportError as exc:
    _gguf = None
    _GGUF_IMPORT_ERROR = exc


def _require_gguf():
    if _gguf is None:
        raise ImportError(
            "gguf is required to use GGUF XPU operators. "
            "Please install the 'gguf' package."
        ) from _GGUF_IMPORT_ERROR
    return _gguf


def _to_dtype(dtype: Optional[torch.dtype]) -> torch.dtype:
    return dtype if dtype is not None else torch.float16


def _dequantize_to_tensor(
    W: torch.Tensor,
    quant_type: int,
    m: int,
    n: int,
    dtype: Optional[torch.dtype],
) -> torch.Tensor:
    gguf = _require_gguf()
    dequantized = gguf.dequantize(
        W.detach().cpu().numpy(),
        gguf.GGMLQuantizationType(int(quant_type)),
    )
    return torch.from_numpy(dequantized).reshape(int(m), int(n)).to(
        device=W.device,
        dtype=_to_dtype(dtype),
    )


def _check_mm_args(W: torch.Tensor, X: torch.Tensor, row: int) -> None:
    if W.device != X.device:
        raise RuntimeError("W and X must be on the same device.")
    if X.dim() != 2:
        raise RuntimeError("X must be a 2D tensor.")
    if int(row) <= 0:
        raise RuntimeError("row must be a positive integer.")


def _ggml_dequantize_impl(
    W: torch.Tensor,
    type: int,
    m: int,
    n: int,
    dtype: Optional[torch.dtype],
) -> torch.Tensor:
    return _dequantize_to_tensor(W, type, m, n, dtype)


def _ggml_mul_mat_impl(
    W: torch.Tensor,
    X: torch.Tensor,
    type: int,
    row: int,
) -> torch.Tensor:
    _check_mm_args(W, X, row)
    weight = _dequantize_to_tensor(W, type, int(row), int(X.shape[1]), X.dtype)
    return torch.matmul(X, weight.transpose(0, 1))


def _ggml_dequantize_meta(
    W: torch.Tensor,
    type: int,
    m: int,
    n: int,
    dtype: Optional[torch.dtype],
) -> torch.Tensor:
    del W, type
    return torch.empty((int(m), int(n)), device="meta", dtype=_to_dtype(dtype))


def _ggml_mul_mat_meta(
    W: torch.Tensor,
    X: torch.Tensor,
    type: int,
    row: int,
) -> torch.Tensor:
    del W, type
    return torch.empty((X.shape[0], int(row)), device="meta", dtype=X.dtype)


_IMPL_LIB.impl("ggml_dequantize", _ggml_dequantize_impl)
_IMPL_LIB.impl("ggml_mul_mat_vec_a8", _ggml_mul_mat_impl)
_IMPL_LIB.impl("ggml_mul_mat_a8", _ggml_mul_mat_impl)

_META_LIB.impl("ggml_dequantize", _ggml_dequantize_meta)
_META_LIB.impl("ggml_mul_mat_vec_a8", _ggml_mul_mat_meta)
_META_LIB.impl("ggml_mul_mat_a8", _ggml_mul_mat_meta)


def ggml_dequantize(
    W: torch.Tensor,
    type: int,
    m: int,
    n: int,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    return torch.ops._xpu_C.ggml_dequantize(W, type, m, n, dtype)


def ggml_mul_mat_vec_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    type: int,
    row: int,
) -> torch.Tensor:
    return torch.ops._xpu_C.ggml_mul_mat_vec_a8(W, X, type, row)


def ggml_mul_mat_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    type: int,
    row: int,
) -> torch.Tensor:
    return torch.ops._xpu_C.ggml_mul_mat_a8(W, X, type, row)
