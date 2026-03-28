# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_xpu_kernels.quantization._quantize_convert import (
    GGUF_BLOCK_SIZE_Q4_K,
    GGUF_BLOCK_SIZE_Q8_0,
    GGUF_QK8_0,
    dequantize_gguf,
)


def _fp16_bytes(value: float) -> torch.Tensor:
    return torch.tensor([value], dtype=torch.float16).view(torch.uint8)


def _pack_q8_0_block(scale: float, quants: torch.Tensor) -> torch.Tensor:
    assert quants.shape == (GGUF_QK8_0, )
    return torch.cat((_fp16_bytes(scale), quants.to(torch.int8).view(
        torch.uint8)))


def _pack_q4_k_scales(scale_values: list[int],
                      min_values: list[int]) -> torch.Tensor:
    assert len(scale_values) == 8
    assert len(min_values) == 8
    packed = torch.zeros(12, dtype=torch.uint8)
    for index, (scale, minimum) in enumerate(zip(scale_values, min_values)):
        assert 0 <= scale < 64
        assert 0 <= minimum < 64
        if index < 4:
            packed[index] = scale
            packed[index + 4] = minimum
        else:
            packed[index + 4] = (scale & 0x0F) | ((minimum & 0x0F) << 4)
            packed[index - 4] |= (scale >> 4) << 6
            packed[index] |= (minimum >> 4) << 6
    return packed


def _pack_q4_k_block(scale_values: list[int], min_values: list[int],
                     low_values: list[int],
                     high_values: list[int]) -> torch.Tensor:
    assert len(low_values) == 4
    assert len(high_values) == 4
    packed = torch.empty(GGUF_BLOCK_SIZE_Q4_K, dtype=torch.uint8)
    packed[:2] = _fp16_bytes(1.0)
    packed[2:4] = _fp16_bytes(1.0)
    packed[4:16] = _pack_q4_k_scales(scale_values, min_values)

    quant_bytes = []
    for low, high in zip(low_values, high_values):
        assert 0 <= low < 16
        assert 0 <= high < 16
        quant_bytes.append(torch.full((32, ),
                                      low | (high << 4),
                                      dtype=torch.uint8))
    packed[16:] = torch.cat(quant_bytes)
    return packed


def test_dequantize_gguf_q8_0():
    quants = torch.arange(-16, 16, dtype=torch.int8)
    packed = _pack_q8_0_block(0.5, quants)

    output = dequantize_gguf(packed, "Q8_0")
    expected = quants.to(torch.float32) * 0.5

    torch.testing.assert_close(output, expected)


def test_dequantize_gguf_q4_k_m():
    scale_values = [1, 2, 3, 4, 5, 6, 7, 8]
    min_values = [0] * 8
    packed = _pack_q4_k_block(
        scale_values=scale_values,
        min_values=min_values,
        low_values=[1, 3, 5, 7],
        high_values=[2, 4, 6, 8],
    )

    output = dequantize_gguf(packed, "Q4_K_M")
    expected = torch.cat([
        torch.full((32, ), scale * quant, dtype=torch.float32)
        for scale, quant in zip(scale_values, [1, 2, 3, 4, 5, 6, 7, 8])
    ])

    torch.testing.assert_close(output, expected)


def test_dequantize_gguf_rejected_invalid_block_size():
    packed = torch.zeros(GGUF_BLOCK_SIZE_Q8_0 - 1, dtype=torch.uint8)
    with pytest.raises(ValueError):
        dequantize_gguf(packed, "Q8_0")


def test_dequantize_gguf_rejected_unsupported_type():
    packed = torch.zeros(GGUF_BLOCK_SIZE_Q8_0, dtype=torch.uint8)
    with pytest.raises(NotImplementedError):
        dequantize_gguf(packed, "Q5_K")
