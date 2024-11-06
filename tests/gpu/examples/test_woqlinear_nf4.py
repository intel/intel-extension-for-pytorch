import torch
import intel_extension_for_pytorch  # noqa
from torch.testing._internal.common_utils import TestCase


NF4_QUANT_TABLE = [
    -1.0 - 1e-2,  # 0b0000
    -0.8480964004993439,  # 0b0001
    -0.6106329262256622,  # 0b0010
    -0.4599952697753906,  # 0b0011
    -0.33967943489551544,  # 0b0100
    -0.23460740596055984,  # 0b0101
    -0.13791173323988914,  # 0b0110
    -0.045525018125772476,  # 0b0111
    0.03979014977812767,  # 0b1000
    0.1202552504837513,  # 0b1001
    0.2035212516784668,  # 0b1010
    0.2920137718319893,  # 0b1011
    0.3893125355243683,  # 0b1100
    0.5016634166240692,  # 0b1101
    0.6427869200706482,  # 0b1110
    0.8614784181118011,  # 0b1111
]


NF4_DEQUANT_TABLE = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]


checking_atol = 1e-2
checking_rtol = 1e-2


class Test4bitDequant(TestCase):

    def quantize_nf4(
        self,
        A,
        blocksize=64,
        quant_type="nf4",
    ):
        n = A.numel()
        input_shape = A.shape
        blocks = n // blocksize
        blocks += 1 if n % blocksize > 0 else 0

        absmax = torch.zeros((blocks,), device=A.device, dtype=A.dtype)
        out = torch.zeros(((n + 1) // 2), dtype=torch.uint8, device=A.device)

        rem = n % blocksize
        has_rem = rem > 0

        # Scale tensor to [-1, 1]
        A_reshaped = A.reshape(n)
        A_com = A_reshaped[: n - rem]
        A_com_reshaped = A_com.reshape(n // blocksize, blocksize)
        absmax[: blocks - has_rem] = torch.abs(A_com_reshaped).max(dim=-1)[0]
        scaled_A = torch.clamp(
            A_com_reshaped * (1 / absmax[: blocks - has_rem].view(-1, 1)), -1, 1
        )
        scaled_A = scaled_A.reshape(-1)

        if has_rem:
            absmax[-1] = torch.abs(A_reshaped[n - rem :]).max()
            scaled_A_rem = torch.clamp(A_reshaped[n - rem :] * (1 / absmax[-1]), -1, 1)
            scaled_A = torch.cat([scaled_A, scaled_A_rem], dim=0)

        # map [-1, 1] to nf4
        out_uint8 = torch.empty(scaled_A.shape, dtype=torch.uint8, device=A.device)

        for i in range(len(NF4_QUANT_TABLE)):
            out_uint8[scaled_A > NF4_QUANT_TABLE[i]] = i

        if out_uint8.size(-1) % 2:
            out_uint8 = torch.nn.functional.pad(out_uint8, (0, 1), value=0)
        out[:] = out_uint8[1::2].bitwise_left_shift(4).bitwise_or_(out_uint8[::2])

        state = {}
        state["absmax"] = absmax
        state["shape"] = input_shape
        state["dtype"] = A.dtype
        state["blocksize"] = blocksize
        state["quant_type"] = quant_type

        return out, state

    def dequantize_nf4(
        self,
        A,
        quant_state,
        quant_type="nf4",
    ):
        absmax = quant_state["absmax"]
        blocksize = quant_state["blocksize"]

        out = torch.empty(
            quant_state["shape"], dtype=quant_state["dtype"], device=A.device
        )

        n = out.numel()
        # Map nf4 to [-1, 1]
        out_uint8 = torch.empty(A.size(0) * 2, dtype=torch.uint8, device=A.device)
        out_uint8[::2] = A.bitwise_and(0xF)
        out_uint8[1::2] = A.bitwise_right_shift(4)
        out_dq = torch.empty(out_uint8.shape).to(quant_state["dtype"]).to(A.device)

        NF4_LUT = torch.tensor(NF4_DEQUANT_TABLE, device=A.device)
        for i in range(len(NF4_LUT)):
            NF4_LUT = NF4_LUT.to(quant_state["dtype"])
            out_dq[out_uint8 == i] = NF4_LUT[i]

        # Apply scales
        if out_dq.numel() != n:
            assert out_dq.numel() == n + 1
            out_dq = torch.narrow(out_dq, 0, 0, n)
        blocks = n // blocksize
        blocks += 1 if n % blocksize > 0 else 0
        rem = n % blocksize
        has_rem = rem > 0
        out_reshaped = out.reshape(-1)
        out_reshaped[: n - rem] = (
            out_dq[: n - rem].view(-1, blocksize)
            * absmax[: blocks - has_rem].view(-1, 1)
        ).reshape(-1)
        if has_rem:
            out_reshaped[n - rem :] = out_dq[n - rem :] * absmax[-1]

        return out

    def test_nf4_dequant_woqlinear(self):
        shapes = [(4096, 4096), (4096, 11008), (11008, 4096)]
        blocksize = 128
        for dtype in [torch.bfloat16, torch.float16]:
            for shape in shapes:
                weight = torch.randn(shape, dtype=dtype, device="xpu")
                input = torch.randn((8, shape[0]), dtype=dtype, device="xpu")

                # quantize
                quant_weight, state = self.quantize_nf4(
                    weight, blocksize=blocksize, quant_type="nf4"
                )

                # dequantize ref
                dequant_ref = self.dequantize_nf4(quant_weight, state, quant_type="nf4")
                linear_ref = torch.nn.functional.linear(input, dequant_ref.t())

                # nf4 dequantize
                dequant_output = torch.ops.torch_ipex.dequantize_4bit(
                    quant_weight,
                    "nf4",
                    state["shape"],
                    state["absmax"],
                    None,
                    state["blocksize"],
                )

                # nf4 woq linear
                linear_output = torch.ops.torch_ipex.woq_linear(
                    input,
                    quant_weight.reshape([weight.shape[0], weight.shape[1] // 2]),
                    "nf4",
                    weight.shape,
                    state["absmax"].view(weight.shape[0], weight.shape[1] // blocksize),
                    None,
                    None,
                    None,
                    blocksize,
                    0,
                    -1,
                    None,
                )

                self.assertEqual(
                    dequant_ref, dequant_output, atol=checking_atol, rtol=checking_rtol
                )
                self.assertEqual(
                    linear_ref, linear_output, atol=checking_atol, rtol=checking_rtol
                )
