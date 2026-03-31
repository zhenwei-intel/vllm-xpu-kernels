# SPDX-License-Identifier: Apache-2.0

import random
import struct

import pytest
import torch

from vllm_xpu_kernels.gguf import GGUFQuantType, gguf_dequantize


QK_K = 256


def _pack_f16(value: float) -> bytes:
    return struct.pack("<e", value)


def _pack_u32(value: int) -> bytes:
    return struct.pack("<I", value)


def _pack_i8(values: list[int]) -> bytes:
    return struct.pack(f"<{len(values)}b", *values)


def _decode_q3_scale(scales: list[int], is_idx: int) -> int:
    if is_idx < 4:
        return (scales[is_idx] & 0x0F) | (((scales[is_idx + 8] >> 0) & 0x03) << 4)
    if is_idx < 8:
        return (scales[is_idx] & 0x0F) | (((scales[is_idx + 4] >> 2) & 0x03) << 4)
    if is_idx < 12:
        return (scales[is_idx - 8] >> 4) | (((scales[is_idx] >> 4) & 0x03) << 4)
    return (scales[is_idx - 8] >> 4) | (((scales[is_idx - 4] >> 6) & 0x03) << 4)


def _get_scale_min_k4(j: int, q: list[int]) -> tuple[int, int]:
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


def _make_q4_0_block(rng: random.Random):
    d = rng.uniform(0.05, 1.5)
    qs = [rng.randrange(256) for _ in range(16)]
    out = [0.0] * 32
    for i, q in enumerate(qs):
        out[i] = d * ((q & 0x0F) - 8)
        out[i + 16] = d * ((q >> 4) - 8)
    return _pack_f16(d) + bytes(qs), out


def _make_q4_1_block(rng: random.Random):
    d = rng.uniform(0.05, 1.5)
    m = rng.uniform(-1.0, 1.0)
    qs = [rng.randrange(256) for _ in range(16)]
    out = [0.0] * 32
    for i, q in enumerate(qs):
        out[i] = (q & 0x0F) * d + m
        out[i + 16] = (q >> 4) * d + m
    return _pack_f16(d) + _pack_f16(m) + bytes(qs), out


def _make_q5_0_block(rng: random.Random):
    d = rng.uniform(0.05, 1.5)
    qh = rng.getrandbits(32)
    qs = [rng.randrange(256) for _ in range(16)]
    out = [0.0] * 32
    for i, q in enumerate(qs):
        xh0 = ((qh >> i) << 4) & 0x10
        xh1 = (qh >> (i + 12)) & 0x10
        out[i] = d * (((q & 0x0F) | xh0) - 16)
        out[i + 16] = d * (((q >> 4) | xh1) - 16)
    return _pack_f16(d) + _pack_u32(qh) + bytes(qs), out


def _make_q5_1_block(rng: random.Random):
    d = rng.uniform(0.05, 1.5)
    m = rng.uniform(-1.0, 1.0)
    qh = rng.getrandbits(32)
    qs = [rng.randrange(256) for _ in range(16)]
    out = [0.0] * 32
    for i, q in enumerate(qs):
        xh0 = ((qh >> i) << 4) & 0x10
        xh1 = (qh >> (i + 12)) & 0x10
        out[i] = ((q & 0x0F) | xh0) * d + m
        out[i + 16] = ((q >> 4) | xh1) * d + m
    return _pack_f16(d) + _pack_f16(m) + _pack_u32(qh) + bytes(qs), out


def _make_q8_0_block(rng: random.Random):
    d = rng.uniform(0.05, 1.5)
    qs = [rng.randrange(-128, 128) for _ in range(32)]
    out = [d * q for q in qs]
    return _pack_f16(d) + _pack_i8(qs), out


def _make_q2_k_block(rng: random.Random):
    scales = [rng.randrange(256) for _ in range(16)]
    qs = [rng.randrange(256) for _ in range(64)]
    d = rng.uniform(0.01, 0.5)
    dmin = rng.uniform(0.01, 0.5)
    out = [0.0] * QK_K
    for n in range(2):
        for l in range(32):
            is_idx = 8 * n + l // 16
            q = qs[32 * n + l]
            base = 128 * n
            out[base + l] = d * (scales[is_idx] & 0x0F) * ((q >> 0) & 0x03) - dmin * (scales[is_idx] >> 4)
            out[base + l + 32] = d * (scales[is_idx + 2] & 0x0F) * ((q >> 2) & 0x03) - dmin * (scales[is_idx + 2] >> 4)
            out[base + l + 64] = d * (scales[is_idx + 4] & 0x0F) * ((q >> 4) & 0x03) - dmin * (scales[is_idx + 4] >> 4)
            out[base + l + 96] = d * (scales[is_idx + 6] & 0x0F) * ((q >> 6) & 0x03) - dmin * (scales[is_idx + 6] >> 4)
    return bytes(scales) + bytes(qs) + _pack_f16(d) + _pack_f16(dmin), out


def _make_q3_k_block(rng: random.Random):
    hmask = [rng.randrange(256) for _ in range(32)]
    qs = [rng.randrange(256) for _ in range(64)]
    scales = [rng.randrange(256) for _ in range(12)]
    d = rng.uniform(0.01, 0.25)
    out = [0.0] * QK_K
    for n in range(2):
        q = qs[32 * n:32 * (n + 1)]
        for j in range(4):
            m = 1 << (4 * n + j)
            shift = 2 * j
            chunk = 128 * n + 32 * j
            for is0 in range(2):
                is_idx = 8 * n + 2 * j + is0
                dl = d * (_decode_q3_scale(scales, is_idx) - 32)
                for l in range(16 * is0, 16 * is0 + 16):
                    out[chunk + l] = dl * (((q[l] >> shift) & 0x03) - (0 if (hmask[l] & m) else 4))
    return bytes(hmask) + bytes(qs) + bytes(scales) + _pack_f16(d), out


def _make_q4_k_block(rng: random.Random):
    d = rng.uniform(0.01, 0.25)
    dmin = rng.uniform(0.01, 0.25)
    scales = [rng.randrange(256) for _ in range(12)]
    qs = [rng.randrange(256) for _ in range(128)]
    out = [0.0] * QK_K
    for il in range(4):
        sc1, m1 = _get_scale_min_k4(2 * il, scales)
        sc2, m2 = _get_scale_min_k4(2 * il + 1, scales)
        d1 = d * sc1
        d2 = d * sc2
        min1 = dmin * m1
        min2 = dmin * m2
        for ir in range(8):
            q = qs[32 * il + 4 * ir:32 * il + 4 * ir + 4]
            base = 64 * il + 4 * ir
            for l, qv in enumerate(q):
                out[base + l] = d1 * (qv & 0x0F) - min1
                out[base + 32 + l] = d2 * (qv >> 4) - min2
    return _pack_f16(d) + _pack_f16(dmin) + bytes(scales) + bytes(qs), out


def _make_q5_k_block(rng: random.Random):
    d = rng.uniform(0.01, 0.25)
    dmin = rng.uniform(0.01, 0.25)
    scales = [rng.randrange(256) for _ in range(12)]
    qh = [rng.randrange(256) for _ in range(32)]
    qs = [rng.randrange(256) for _ in range(128)]
    out = [0.0] * QK_K
    for il in range(4):
        sc1, m1 = _get_scale_min_k4(2 * il, scales)
        sc2, m2 = _get_scale_min_k4(2 * il + 1, scales)
        d1 = d * sc1
        d2 = d * sc2
        min1 = dmin * m1
        min2 = dmin * m2
        for ir in range(16):
            q0 = qs[32 * il + 2 * ir]
            q1 = qs[32 * il + 2 * ir + 1]
            hm = 1 << (2 * il)
            base = 64 * il + 2 * ir
            out[base + 0] = d1 * ((q0 & 0x0F) + (16 if (qh[2 * ir] & hm) else 0)) - min1
            out[base + 1] = d1 * ((q1 & 0x0F) + (16 if (qh[2 * ir + 1] & hm) else 0)) - min1
            hm <<= 1
            out[base + 32] = d2 * ((q0 >> 4) + (16 if (qh[2 * ir] & hm) else 0)) - min2
            out[base + 33] = d2 * ((q1 >> 4) + (16 if (qh[2 * ir + 1] & hm) else 0)) - min2
    return _pack_f16(d) + _pack_f16(dmin) + bytes(scales) + bytes(qh) + bytes(qs), out


def _make_q6_k_block(rng: random.Random):
    ql = [rng.randrange(256) for _ in range(128)]
    qh = [rng.randrange(256) for _ in range(64)]
    scales = [rng.randrange(-32, 32) for _ in range(16)]
    d = rng.uniform(0.01, 0.25)
    out = [0.0] * QK_K
    for ip in range(2):
        for il in range(32):
            is_idx = 8 * ip + il // 16
            base = 128 * ip + il
            qh_byte = qh[32 * ip + il]
            sc = scales[is_idx:is_idx + 8]
            out[base + 0] = d * sc[0] * ((((ql[64 * ip + il] & 0x0F) | (((qh_byte >> 0) & 0x03) << 4)) - 32))
            out[base + 32] = d * sc[2] * ((((ql[64 * ip + il + 32] & 0x0F) | (((qh_byte >> 2) & 0x03) << 4)) - 32))
            out[base + 64] = d * sc[4] * ((((ql[64 * ip + il] >> 4) | (((qh_byte >> 4) & 0x03) << 4)) - 32))
            out[base + 96] = d * sc[6] * ((((ql[64 * ip + il + 32] >> 4) | (((qh_byte >> 6) & 0x03) << 4)) - 32))
    return bytes(ql) + bytes(qh) + _pack_i8(scales) + _pack_f16(d), out


BUILDERS = {
    GGUFQuantType.Q4_0: _make_q4_0_block,
    GGUFQuantType.Q4_1: _make_q4_1_block,
    GGUFQuantType.Q5_0: _make_q5_0_block,
    GGUFQuantType.Q5_1: _make_q5_1_block,
    GGUFQuantType.Q8_0: _make_q8_0_block,
    GGUFQuantType.Q2_K: _make_q2_k_block,
    GGUFQuantType.Q3_K: _make_q3_k_block,
    GGUFQuantType.Q4_K: _make_q4_k_block,
    GGUFQuantType.Q5_K: _make_q5_k_block,
    GGUFQuantType.Q6_K: _make_q6_k_block,
}


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required")
@pytest.mark.parametrize("qtype", list(BUILDERS))
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_gguf_dequantize_matches_reference(qtype: GGUFQuantType,
                                           out_dtype: torch.dtype) -> None:
    rng = random.Random(0)
    rows = 2
    blocks_per_row = 3
    packed_rows = []
    expected_rows = []

    for _ in range(rows):
        packed = bytearray()
        expected = []
        for _ in range(blocks_per_row):
            packed_block, expected_block = BUILDERS[qtype](rng)
            packed.extend(packed_block)
            expected.extend(expected_block)
        packed_rows.append(list(packed))
        expected_rows.append(expected)

    packed_tensor = torch.tensor(packed_rows, dtype=torch.uint8, device="xpu")
    expected_tensor = torch.tensor(expected_rows, dtype=torch.float32)

    actual = gguf_dequantize(packed_tensor, qtype, out_dtype=out_dtype)

    assert actual.dtype == out_dtype
    torch.testing.assert_close(actual.cpu().float(), expected_tensor, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required")
def test_gguf_dequantize_accepts_string_alias() -> None:
    rng = random.Random(1)
    packed_block, expected = _make_q4_0_block(rng)
    packed = torch.tensor(list(packed_block), dtype=torch.uint8, device="xpu")

    actual = gguf_dequantize(packed, "q4_0")

    torch.testing.assert_close(actual.cpu(), torch.tensor(expected, dtype=torch.float32), rtol=1e-4, atol=1e-4)