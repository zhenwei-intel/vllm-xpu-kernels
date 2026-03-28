# SPDX-License-Identifier: Apache-2.0
import torch


class GPTQUtils:

    def __init__(self, bits=4, blocksize=128):
        super(GPTQUtils, self).__init__()  # noqa: UP008
        self.bits = bits
        self.blocksize = blocksize

    def convert_idx(self, g_idx, k):
        ret_idx = torch.zeros(k, dtype=int).to(g_idx.device)
        groups = k // self.blocksize
        remainder = k % self.blocksize
        g_idx_2 = g_idx * self.blocksize
        if remainder > 0:
            g_idx_2[g_idx == groups] += torch.arange(remainder).to(
                g_idx.device)
        arrange_tensor = torch.arange(self.blocksize).to(g_idx.device)
        for i in range(groups):
            g_idx_2[g_idx == i] += arrange_tensor
        ret_idx[g_idx_2] = torch.arange(k).to(g_idx.device)
        return ret_idx.to(torch.int32)

    def unpack_weight(self, qweight_int32):
        s32_bits = 32

        assert self.bits == 4
        # Int32 can store 8 * 4bits data. This is the offset for each data.
        wf = (torch.tensor(list(range(0, s32_bits, self.bits)),
                           dtype=torch.int32).unsqueeze(0).to(
                               qweight_int32.device))
        weight = torch.bitwise_right_shift(
            torch.unsqueeze(qweight_int32, 1).expand(-1, 32 // self.bits, -1),
            wf.unsqueeze(-1),
        ).to(torch.int16 if self.bits == 8 else torch.int8)
        torch.bitwise_and(weight, (2**self.bits) - 1, out=weight)

        return weight

    def unpack_zp(self, qzeros_int32):
        s32_bits = 32

        assert self.bits == 4
        # Int32 can store 8 * 4bits data. This is the offset for each data.
        wf = (torch.tensor(list(range(0, s32_bits, self.bits)),
                           dtype=torch.int32).unsqueeze(0).to(
                               qzeros_int32.device))
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(qzeros_int32, 2).expand(-1, -1, 32 // self.bits),
            wf.unsqueeze(0),
        ).to(torch.int8)
        torch.bitwise_and(zeros, (2**self.bits) - 1, out=zeros)

        return zeros

    def pack(self, qweight_int8):
        i = 0
        row = 0
        qweight_int32_shape = (
            qweight_int8.shape[0] // 32 * self.bits,
            qweight_int8.shape[1],
        )
        qweight_int32 = torch.zeros(qweight_int32_shape,
                                    dtype=torch.int32,
                                    device=qweight_int8.device)

        while row < qweight_int32.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight_int32[row] |= qweight_int8[j].to(
                        torch.int32) << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        return qweight_int32

    def shuffle(self, qweight_int32, g_idx):
        k = qweight_int32.shape[0] * 8
        g_idx4kernel = self.convert_idx(g_idx, k).to(qweight_int32.device)
        qweight_int8 = self.unpack_weight(qweight_int32)
        qweight_int8 = qweight_int8.reshape(-1, qweight_int8.shape[-1])
        qweight_int8_shuffled = qweight_int8[g_idx4kernel, :]
        qweight_int32_shuffled = self.pack(qweight_int8_shuffled)
        return qweight_int32_shuffled, g_idx4kernel


class AWQUtils:
    AWQ_PACK_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
    REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

    @classmethod
    def pack(cls, imatrix: torch.Tensor, direction: str = "column"):
        """
        Packs a 4-bit integer matrix into a packed 32-bit integer matrix.
        Args:
            imatrix (torch.Tensor): matrix of integers
            direction (str): direction of packing, either "column" or "row"
        Returns:
            qmatrix (torch.Tensor): packed matrix of integers
        """
        shifts = torch.arange(0,
                              32,
                              4,
                              dtype=torch.int32,
                              device=imatrix.device)

        imatrix = imatrix.to(torch.int8) & 0x0F  # eventually correct overflow

        if direction == "column":
            imatrix = imatrix.view(-1, imatrix.shape[1] // (32 // 4),
                                   (32 // 4))
            qmatrix = torch.bitwise_left_shift(imatrix,
                                               shifts[None,
                                                      None, :]).sum(dim=-1)

        elif direction == "row":
            imatrix = imatrix.view(imatrix.shape[0] // (32 // 4), (32 // 4),
                                   -1)
            qmatrix = torch.bitwise_left_shift(imatrix,
                                               shifts[None, :,
                                                      None]).sum(dim=1)

        qmatrix = qmatrix.to(torch.int32)

        return qmatrix

    @classmethod
    def unpack(cls, qmatrix: torch.Tensor, direction: str = "column"):
        """
        Unpacks a 32-bit packed integer matrix into a 4-bit integer matrix.
        Args:
            qmatrix (torch.Tensor): matrix of packed integers
            direction (str): direction of unpacking, either "column" or "row"
        Returns:
            imatrix (torch.Tensor): matrix of integers
        """
        shifts = torch.arange(0, 32, 4, device=qmatrix.device)

        if direction == "column":
            imatrix = torch.bitwise_right_shift(qmatrix[:, :, None],
                                                shifts[None, None, :]).view(
                                                    qmatrix.shape[0], -1)

        elif direction == "row":
            imatrix = torch.bitwise_right_shift(qmatrix[:, None, :],
                                                shifts[None, :, None]).view(
                                                    -1, qmatrix.shape[-1])

        imatrix = imatrix.to(torch.int8) & 0x0F  # eventually correct overflow

        return imatrix

    @classmethod
    def apply_order(
        cls,
        imatrix: torch.Tensor,
        direction: str = "column",
        order: list[int] = None,
    ):
        """
        Applies the order to a 4-bit integer matrix.
        Args:
            imatrix (torch.Tensor): matrix of integers
            direction (str): direction of applying order, "column" or "row"
            order (List[int]): order to apply, default is AWQ_PACK_ORDER
        Returns:
            imatrix (torch.Tensor): matrix of integers
        """
        if direction == "column":
            imatrix = imatrix.view(-1, (32 // 4))[:, order].view(imatrix.shape)
        elif direction == "row":
            imatrix = imatrix.view((32 // 4), -1)[order, :].view(imatrix.shape)

        return imatrix

    @classmethod
    def repack(cls, qweight, qzeros):
        # awq uses column packing for both weights and zeros
        izeros = cls.unpack(qzeros, direction="column")
        iweights = cls.unpack(qweight, direction="column")

        # Reverse the order of the iweight and izeros tensors
        izeros = cls.apply_order(izeros,
                                 direction="column",
                                 order=cls.REVERSE_AWQ_PACK_ORDER)
        iweights = cls.apply_order(iweights,
                                   direction="column",
                                   order=cls.REVERSE_AWQ_PACK_ORDER)

        # exllama uses row packing for weights and column packing for zeros
        qzeros = cls.pack(izeros, direction="column")
        qweight = cls.pack(iweights, direction="row")

        return qweight, qzeros


def transpose_onednn_woq_format(layer: torch.nn.Module,
                                method: str,
                                is_sym: bool = True):
    # The oneDNN int4 GEMM has the following requirements:
    # - Weights need to be contiguous along the 'k' dimension,
    #   but the shape should remain (k, n/8).
    # - Scales remains unchanged.
    # - Zero-point is a scalar value of 8 in symmetric (symm) scenarios,
    #   allowing oneDNN to broadcast it.
    # - Zero-point remains unchanged in asymmetric (asymm) scenarios.
    reshaped_tensor = layer.qweight.transpose(0,
                                              1).contiguous().transpose(0, 1)
    layer.qweight.as_strided_(reshaped_tensor.shape, reshaped_tensor.stride())
    layer.qweight.copy_(reshaped_tensor)
    layer.scales.data = layer.scales.contiguous()
    if method == "gptq":
        if is_sym:
            qzeros = torch.Tensor([8]).to(torch.int8).to("xpu")
        else:
            qzeros = layer.qzeros + 0x11111111
        layer.qzeros.as_strided_(qzeros.shape, qzeros.stride())
        layer.qzeros.copy_(qzeros)


def dequantize(qweight, scales, qzeros, group_size, g_idx=None):
    # Dequantize the weight from int4 to scales data type
    gptq_utils = GPTQUtils(bits=4, blocksize=group_size)
    weight = gptq_utils.unpack_weight(qweight)
    if len(weight.shape) > 2:
        weight = weight.reshape(-1, weight.shape[-1])
    infeatures = weight.shape[0]
    if g_idx is None:
        g_idx = torch.tensor(
            [i // group_size for i in range(infeatures)],
            dtype=torch.int32,
        )
    if qzeros is None:
        return (weight - 8) * scales[g_idx]
    else:
        gptq_zeros = gptq_utils.unpack_zp(qzeros)
        gptq_zeros = gptq_zeros.reshape(scales.shape)
        return (weight - gptq_zeros[g_idx]) * scales[g_idx]


def dequantize_s8_to_float(quantized, scale, zero_point):
    repeat_dims = [1] * (quantized.dim() - 1) + [quantized.shape[-1]]
    return (quantized -
            zero_point.repeat(repeat_dims)) * scale.repeat(repeat_dims)


def dynamic_per_token_quant_ref(input, use_sym_quant, bits):
    original_sizes = input.size()
    input = input.view(
        -1, original_sizes[-1])  # Flatten except for the last dimension
    k = input.shape[-1]
    qmin = -(2**(bits - 1)) if use_sym_quant else 0
    qmax = 2**(bits - 1) - 1 if use_sym_quant else 2**bits - 1
    min_val = torch.min(input, dim=-1)[0].to(dtype=torch.float32).unsqueeze(-1)
    max_val = torch.max(input, dim=-1)[0].to(dtype=torch.float32).unsqueeze(-1)
    if use_sym_quant:
        scale = torch.maximum(torch.abs(min_val), torch.abs(max_val)) / qmax
        zero_point = torch.zeros_like(scale).to(dtype=torch.int32)
    else:
        scale = (max_val - min_val) / qmax
        zero_point = -1 * torch.round(min_val / scale).to(dtype=torch.int32)
    scale = scale.to(dtype=input.dtype)
    quantized = torch.clamp(
        torch.round(input / scale.repeat(1, k).to(dtype=torch.float32) +
                    zero_point.repeat(1, k)),
        qmin,
        qmax,
    ).to(dtype=torch.int8 if use_sym_quant else torch.uint8)
    return (
        quantized.view(original_sizes),
        scale.view(original_sizes[:-1] + (1, )),
        zero_point.view(original_sizes[:-1] + (1, )),
    )


def dynamic_per_tensor_quant_ref(input, use_sym_quant, bits):
    original_sizes = input.size()
    input = input.view(
        -1, original_sizes[-1])  # Flatten except for the last dimension
    qmin = -(2**(bits - 1)) if use_sym_quant else 0
    qmax = 2**(bits - 1) - 1 if use_sym_quant else 2**bits - 1
    min_val = torch.min(input)
    max_val = torch.max(input)
    if use_sym_quant:
        scale_val = torch.maximum(torch.abs(min_val),
                                  torch.abs(max_val)) / qmax
        scale = torch.tensor([scale_val]).to("xpu")
        zero_point = torch.tensor([0], dtype=torch.int32, device="xpu")
    else:
        scale = (max_val - min_val) / qmax
        zero_point = -1 * torch.round(min_val / scale).to(dtype=torch.int32)
    scale = scale.to(dtype=input.dtype)
    quantized = torch.clamp(
        torch.round(input / scale.to(dtype=torch.float32) + zero_point),
        qmin,
        qmax,
    ).to(dtype=torch.int8 if use_sym_quant else torch.uint8)
    return (
        quantized.view(original_sizes),
        scale,
        zero_point,
    )


GGUF_QK8_0 = 32
GGUF_QK_K = 256
GGUF_Q4_K_SCALE_SIZE = 12
GGUF_BLOCK_SIZE_Q8_0 = 2 + GGUF_QK8_0
GGUF_BLOCK_SIZE_Q4_K = 4 + GGUF_Q4_K_SCALE_SIZE + GGUF_QK_K // 2


def _gguf_as_blocks(packed: torch.Tensor, block_size: int) -> torch.Tensor:
    packed = packed.contiguous().view(torch.uint8).reshape(-1)
    if packed.numel() % block_size != 0:
        raise ValueError(
            "Expected packed GGUF tensor size to be divisible by "
            f"{block_size}, got {packed.numel()}.")
    return packed.reshape(-1, block_size)


def _gguf_bytes_to_fp16(raw: torch.Tensor) -> torch.Tensor:
    raw = raw.to(dtype=torch.uint8)
    words = (raw[:, 0].to(dtype=torch.int32) |
             (raw[:, 1].to(dtype=torch.int32) << 8))
    return words.to(dtype=torch.uint16).view(torch.float16)


def _gguf_get_scale_min_k4(scales: torch.Tensor,
                           index: int) -> tuple[torch.Tensor, torch.Tensor]:
    if index < 4:
        scale = scales[:, index] & 63
        minimum = scales[:, index + 4] & 63
    else:
        scale = (scales[:, index + 4] & 0x0F) | (
            (scales[:, index - 4] >> 6) << 4)
        minimum = (scales[:, index + 4] >> 4) | (
            (scales[:, index] >> 6) << 4)
    return scale.to(torch.float32), minimum.to(torch.float32)


def dequantize_gguf_q8_0(packed: torch.Tensor) -> torch.Tensor:
    blocks = _gguf_as_blocks(packed, GGUF_BLOCK_SIZE_Q8_0)
    scales = _gguf_bytes_to_fp16(blocks[:, :2]).to(torch.float32)
    quants = blocks[:, 2:].view(torch.int8).to(torch.float32)
    return (quants * scales.unsqueeze(-1)).reshape(-1)


def dequantize_gguf_q4_k(packed: torch.Tensor) -> torch.Tensor:
    blocks = _gguf_as_blocks(packed, GGUF_BLOCK_SIZE_Q4_K)
    d = _gguf_bytes_to_fp16(blocks[:, 0:2]).to(torch.float32)
    dmin = _gguf_bytes_to_fp16(blocks[:, 2:4]).to(torch.float32)
    scales = blocks[:, 4:4 + GGUF_Q4_K_SCALE_SIZE]
    quants = blocks[:, 4 + GGUF_Q4_K_SCALE_SIZE:]

    dequant = torch.empty((blocks.shape[0], GGUF_QK_K),
                          dtype=torch.float32,
                          device=packed.device)
    low = (quants & 0x0F).to(torch.float32)
    high = (quants >> 4).to(torch.float32)

    for block_index in range(GGUF_QK_K // 64):
        scale_0, min_0 = _gguf_get_scale_min_k4(scales, 2 * block_index)
        scale_1, min_1 = _gguf_get_scale_min_k4(scales, 2 * block_index + 1)
        q_slice = slice(block_index * 32, (block_index + 1) * 32)
        out_offset = block_index * 64

        d_0 = (d * scale_0).unsqueeze(-1)
        d_1 = (d * scale_1).unsqueeze(-1)
        m_0 = (dmin * min_0).unsqueeze(-1)
        m_1 = (dmin * min_1).unsqueeze(-1)

        dequant[:, out_offset:out_offset + 32] = d_0 * low[:, q_slice] - m_0
        dequant[:, out_offset + 32:out_offset + 64] = (
            d_1 * high[:, q_slice] - m_1)

    return dequant.reshape(-1)


def dequantize_gguf(packed: torch.Tensor, gguf_type: str) -> torch.Tensor:
    gguf_type = gguf_type.upper()
    if gguf_type == "Q8_0":
        return dequantize_gguf_q8_0(packed)
    if gguf_type in {"Q4_K", "Q4_K_M"}:
        return dequantize_gguf_q4_k(packed)
    raise NotImplementedError(
        f"Unsupported GGUF type '{gguf_type}'. Supported types: Q8_0, Q4_K, "
        "Q4_K_M.")
