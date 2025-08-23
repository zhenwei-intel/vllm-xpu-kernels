# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F

import tests.register_ops as ops
from tests.ops.custom_ops import CustomOp


class GGUF(CustomOp):
    """An activation function for SwiGLU.
    TODO
    """

    def __init__(self):
        super().__init__()
        self.op = ops.gguf_mul_mat_vec_a8

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = (x.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        self.op(out, x)
        return out
