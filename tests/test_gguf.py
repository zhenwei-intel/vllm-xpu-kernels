# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import struct

import pytest
import torch


GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8
QK = 32


def _pack_half(value: float) -> bytes:
    return struct.pack("<e", value)


def _pack_q8_0_row(values: list[float], scale: float) -> bytes:
    assert len(values) % QK == 0
    row = bytearray()
    for block_start in range(0, len(values), QK):
        block = values[block_start:block_start + QK]
        quants = [int(round(v / scale)) for v in block]
        row.extend(_pack_half(scale))
        row.extend(struct.pack("<32b", *quants))
    return bytes(row)


def _pack_q4_0_row(quants: list[int], scale: float) -> bytes:
    assert len(quants) % QK == 0
    row = bytearray()
    for block_start in range(0, len(quants), QK):
        block = quants[block_start:block_start + QK]
        row.extend(_pack_half(scale))
        for idx in range(QK // 2):
            lo = block[idx] + 8
            hi = block[idx + QK // 2] + 8
            row.append((hi << 4) | lo)
    return bytes(row)


def _make_qweight(rows: list[bytes]) -> torch.Tensor:
    width = len(rows[0])
    assert all(len(row) == width for row in rows)
    data = [list(row) for row in rows]
    return torch.tensor(data, dtype=torch.uint8)


def _reference_dequantize_q8_0(scales: list[float],
                               quants: list[list[int]]) -> torch.Tensor:
    rows = []
    for scale, row_quants in zip(scales, quants):
        rows.append([scale * q for q in row_quants])
    return torch.tensor(rows, dtype=torch.float32)


def _reference_dequantize_q4_0(scales: list[float],
                               quants: list[list[int]]) -> torch.Tensor:
    rows = []
    for scale, row_quants in zip(scales, quants):
        rows.append([scale * q for q in row_quants])
    return torch.tensor(rows, dtype=torch.float32)


def _get_ops():
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        pytest.skip("XPU is required")
    pytest.importorskip("vllm_xpu_kernels._C", reason="Extension is not built")
    from tests.register_ops import (ggml_dequantize, ggml_mul_mat_a8,
                                    ggml_mul_mat_vec_a8)
    return ggml_dequantize, ggml_mul_mat_a8, ggml_mul_mat_vec_a8


def test_q8_0_reference_pack_roundtrip():
    scales = [0.5, 0.25]
    quants = [list(range(-16, 16)), list(range(15, -17, -1))]
    qweight = _make_qweight([
        _pack_q8_0_row([scales[0] * q for q in quants[0]], scales[0]),
        _pack_q8_0_row([scales[1] * q for q in quants[1]], scales[1]),
    ])

    assert qweight.shape == (2, 34)
    ref = _reference_dequantize_q8_0(scales, quants)
    assert ref.shape == (2, 32)


@pytest.mark.parametrize("quant_type", [GGML_TYPE_Q4_0, GGML_TYPE_Q8_0])
def test_ggml_dequantize_qx_0(quant_type: int):
    ggml_dequantize, _, _ = _get_ops()
    if quant_type == GGML_TYPE_Q8_0:
        scales = [0.5, 0.25]
        quants = [list(range(-16, 16)), list(range(15, -17, -1))]
        qweight_cpu = _make_qweight([
            _pack_q8_0_row([scales[0] * q for q in quants[0]], scales[0]),
            _pack_q8_0_row([scales[1] * q for q in quants[1]], scales[1]),
        ])
        ref = _reference_dequantize_q8_0(scales, quants)
    else:
        scales = [0.5, 0.25]
        quants = [
            [((idx % 16) - 8) for idx in range(32)],
            [7 - (idx % 16) for idx in range(32)],
        ]
        qweight_cpu = _make_qweight([
            _pack_q4_0_row(quants[0], scales[0]),
            _pack_q4_0_row(quants[1], scales[1]),
        ])
        ref = _reference_dequantize_q4_0(scales, quants)

    qweight = qweight_cpu.to("xpu")
    output = ggml_dequantize(qweight, quant_type, 2, 32, torch.float32)
    torch.testing.assert_close(output.cpu(), ref)


@pytest.mark.parametrize("op_name", ["ggml_mul_mat_vec_a8", "ggml_mul_mat_a8"])
@pytest.mark.parametrize("quant_type", [GGML_TYPE_Q4_0, GGML_TYPE_Q8_0])
def test_ggml_matmul_qx_0(quant_type: int, op_name: str):
    _, ggml_mul_mat_a8, ggml_mul_mat_vec_a8 = _get_ops()
    if quant_type == GGML_TYPE_Q8_0:
        scales = [0.5, 0.25]
        quants = [list(range(-16, 16)), list(range(15, -17, -1))]
        qweight_cpu = _make_qweight([
            _pack_q8_0_row([scales[0] * q for q in quants[0]], scales[0]),
            _pack_q8_0_row([scales[1] * q for q in quants[1]], scales[1]),
        ])
        ref_weight = _reference_dequantize_q8_0(scales, quants)
    else:
        scales = [0.5, 0.25]
        quants = [
            [((idx % 16) - 8) for idx in range(32)],
            [7 - (idx % 16) for idx in range(32)],
        ]
        qweight_cpu = _make_qweight([
            _pack_q4_0_row(quants[0], scales[0]),
            _pack_q4_0_row(quants[1], scales[1]),
        ])
        ref_weight = _reference_dequantize_q4_0(scales, quants)

    x = torch.linspace(-1.0, 1.0, 64, dtype=torch.float32).reshape(2, 32)
    ref = torch.matmul(x, ref_weight.transpose(0, 1))

    qweight = qweight_cpu.to("xpu")
    x_xpu = x.to("xpu")
    op = ggml_mul_mat_vec_a8 if op_name == "ggml_mul_mat_vec_a8" else ggml_mul_mat_a8
    out = op(qweight, x_xpu, quant_type, 2)
    torch.testing.assert_close(out.cpu(), ref, atol=1e-4, rtol=1e-4)
