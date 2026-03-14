# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

import vllm_xpu_kernels.gguf_interface as gguf_interface


class _FakeQuantizationType:

    def __init__(self, value: int):
        self.value = value


class _FakeGGUF:
    GGMLQuantizationType = _FakeQuantizationType

    @staticmethod
    def dequantize(data, quant_type):
        return data.astype(np.float32) + quant_type.value


def test_ggml_dequantize(monkeypatch):
    monkeypatch.setattr(gguf_interface, "_gguf", _FakeGGUF)
    monkeypatch.setattr(gguf_interface, "_GGUF_IMPORT_ERROR", None)

    qweight = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.uint8)
    output = torch.ops._xpu_C.ggml_dequantize(qweight, 2, 2, 3, torch.float32)

    expected = torch.tensor([[3, 4, 5], [6, 7, 8]], dtype=torch.float32)
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("op_name", ["ggml_mul_mat_vec_a8", "ggml_mul_mat_a8"])
def test_ggml_matmul_ops(monkeypatch, op_name: str):
    monkeypatch.setattr(gguf_interface, "_gguf", _FakeGGUF)
    monkeypatch.setattr(gguf_interface, "_GGUF_IMPORT_ERROR", None)

    qweight = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.uint8)
    x = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]], dtype=torch.float32)

    output = getattr(torch.ops._xpu_C, op_name)(qweight, x, 1, 2)

    dequantized = torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.float32)
    expected = torch.matmul(x, dequantized.transpose(0, 1))
    torch.testing.assert_close(output, expected)
