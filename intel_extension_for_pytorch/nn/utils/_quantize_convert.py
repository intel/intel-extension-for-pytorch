import torch
import intel_extension_for_pytorch as ipex  # noqa
import torch.nn as nn
from torch import Tensor
from typing import Optional, List
import os
from intel_extension_for_pytorch.llm.quantization.utils import (
    XPUWoqActQuantMode,
    XPU_UNSUPPORTED_ACT_QUANT_MODES,
    VLLM_ACT_QUANT_MODE_TO_XPU,
)

DTYPE_BITS_MAPPING = {
    "nf4": 4,
    "fp4_e2m1_bnb": 4,
    "fp4_e2m1": 4,
    "int4_fullrange": 4,
    "int4_clip": 4,
    "int8": 8,
    "fp8_e5m2": 8,
    "fp8_e4m3fn": 8,
}

# currently disable W4A8 computation by default
USE_W4A8_GEMM = os.environ.get("USE_W4A8_COMPUTE", "OFF").upper() in [
    "1",
    "Y",
    "ON",
    "YES",
    "TRUE",
]


class GPTQShuffle(nn.Module):
    def __init__(self, bits=4, blocksize=128):
        super(GPTQShuffle, self).__init__()
        self.bits = bits
        self.blocksize = blocksize

    def convert_idx(self, g_idx, k):
        ret_idx = torch.zeros(k, dtype=int).to(g_idx.device)
        groups = k // self.blocksize
        remainder = k % self.blocksize
        g_idx_2 = g_idx * self.blocksize
        if remainder > 0:
            g_idx_2[g_idx == groups] += torch.arange(remainder).to(g_idx.device)
        arange_tensor = torch.arange(self.blocksize).to(g_idx.device)
        for i in range(groups):
            g_idx_2[g_idx == i] += arange_tensor
        ret_idx[g_idx_2] = torch.arange(k).to(g_idx.device)
        return ret_idx.to(torch.int32)

    def unpack(self, qweight_int32):
        s32_bits = 32

        assert self.bits == 4
        # Int32 can store 8 * 4bits data. This is the offset for each data.
        wf = (
            torch.tensor(list(range(0, s32_bits, self.bits)), dtype=torch.int32)
            .unsqueeze(0)
            .to(qweight_int32.device)
        )
        weight = torch.bitwise_right_shift(
            torch.unsqueeze(qweight_int32, 1).expand(-1, 32 // self.bits, -1),
            wf.unsqueeze(-1),
        ).to(torch.int16 if self.bits == 8 else torch.int8)
        torch.bitwise_and(weight, (2**self.bits) - 1, out=weight)

        return weight

    def pack(self, qweight_int8):
        i = 0
        row = 0
        qweight_int32_shape = (
            qweight_int8.shape[0] // 32 * self.bits,
            qweight_int8.shape[1],
        )
        qweight_int32 = torch.zeros(
            qweight_int32_shape, dtype=torch.int32, device=qweight_int8.device
        )

        while row < qweight_int32.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight_int32[row] |= qweight_int8[j].to(torch.int32) << (
                        self.bits * (j - i)
                    )
                i += 32 // self.bits
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        return qweight_int32

    def forward(self, qweight_int32, g_idx):
        k = qweight_int32.shape[0] * 8
        g_idx4kernel = self.convert_idx(g_idx, k).to(qweight_int32.device)
        qweight_int8 = self.unpack(qweight_int32)
        qweight_int8 = qweight_int8.reshape(-1, qweight_int8.shape[-1])
        qweight_int8_shuffled = qweight_int8[g_idx4kernel, :]
        qweight_int32_shuffled = self.pack(qweight_int8_shuffled)
        return qweight_int32_shuffled, g_idx4kernel


def convert_dtype_str2torch(str_dtype):
    if str_dtype == "int8":
        return torch.int8
    elif str_dtype == "fp32" or str_dtype == "auto":
        return torch.float
    elif str_dtype == "fp16":
        return torch.float16
    elif str_dtype == "bf16":
        return torch.bfloat16
    elif "fp8" in str_dtype:
        return torch.float32
    else:
        AssertionError(False), "Unsupport str dtype {} to torch dtype".format(str_dtype)


AWQ_PACK_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def pack(imatrix: torch.Tensor, direction: str = "column"):
    """
    Packs a 4-bit integer matrix into a packed 32-bit integer matrix.
    Args:
        imatrix (torch.Tensor): matrix of integers
        direction (str): direction of packing, either "column" or "row"
    Returns:
        qmatrix (torch.Tensor): packed matrix of integers
    """
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=imatrix.device)

    imatrix = imatrix.to(torch.int8) & 0x0F  # eventually correct overflow

    if direction == "column":
        imatrix = imatrix.view(-1, imatrix.shape[1] // (32 // 4), (32 // 4))
        qmatrix = torch.bitwise_left_shift(imatrix, shifts[None, None, :]).sum(dim=-1)

    elif direction == "row":
        imatrix = imatrix.view(imatrix.shape[0] // (32 // 4), (32 // 4), -1)
        qmatrix = torch.bitwise_left_shift(imatrix, shifts[None, :, None]).sum(dim=1)

    qmatrix = qmatrix.to(torch.int32)

    return qmatrix


def unpack(qmatrix: torch.Tensor, direction: str = "column"):
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
        imatrix = torch.bitwise_right_shift(
            qmatrix[:, :, None], shifts[None, None, :]
        ).view(qmatrix.shape[0], -1)

    elif direction == "row":
        imatrix = torch.bitwise_right_shift(
            qmatrix[:, None, :], shifts[None, :, None]
        ).view(-1, qmatrix.shape[-1])

    imatrix = imatrix.to(torch.int8) & 0x0F  # eventually correct overflow

    return imatrix


def apply_order(
    imatrix: torch.Tensor,
    direction: str = "column",
    order: List[int] = AWQ_PACK_ORDER,
):
    """
    Applies the order to a 4-bit integer matrix.
    Args:
        imatrix (torch.Tensor): matrix of integers
        direction (str): direction of applying order, either "column" or "row"
        order (List[int]): order to apply, default is AWQ_PACK_ORDER
    Returns:
        imatrix (torch.Tensor): matrix of integers
    """
    if direction == "column":
        imatrix = imatrix.view(-1, (32 // 4))[:, order].view(imatrix.shape)
    elif direction == "row":
        imatrix = imatrix.view((32 // 4), -1)[order, :].view(imatrix.shape)

    return imatrix


def fast_awq_to_gptq(qweight, qzeros):
    # awq uses column packing for both weights and zeros
    izeros = unpack(qzeros, direction="column")
    iweights = unpack(qweight, direction="column")

    # Reverse the order of the iweight and izeros tensors
    izeros = apply_order(izeros, direction="column", order=REVERSE_AWQ_PACK_ORDER)
    iweights = apply_order(iweights, direction="column", order=REVERSE_AWQ_PACK_ORDER)

    # exllama uses row packing for weights and column packing for zeros
    qzeros = pack(izeros, direction="column")
    qweight = pack(iweights, direction="row")

    return qweight, qzeros


def fast_gptq_to_awq(qweight, qzeros):
    # gptq uses row packing for both weights and zeros
    izeros = unpack(qzeros, direction="column")
    iweight = unpack(qweight, direction="row")

    izeros = apply_order(izeros, direction="column", order=AWQ_PACK_ORDER)
    iweight = apply_order(iweight, direction="row", order=AWQ_PACK_ORDER)

    izeros = izeros + 1

    qzeros = pack(izeros, direction="column")
    qweight = pack(iweight, direction="column")

    return qweight, qzeros


class ParamsLowBits(torch.nn.Parameter):
    def __new__(
        cls,
        data=None,
        requires_grad=True,
        quant_state=None,
        blocksize=32,
        compress_statistics=True,
        quant_dtype=None,
        scale_dtype="fp32",
        double_quant_scale_dtype=None,
        compression_dtype=None,
    ):
        if data is None:
            data = torch.empty((0))

        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.blocksize = blocksize
        self.compress_statistics = compress_statistics
        self.quant_dtype = quant_dtype
        self.scale_dtype = scale_dtype
        self.double_quant_scale_dtype = double_quant_scale_dtype
        self.quant_state = quant_state
        self.data = data
        self.compression_dtype = compression_dtype
        return self


# There are two implementations of int4 woq gemm.
# Use the onednn implementation on client machines (LNL and BMG) and pvc.
# On other platforms, use the xetla implementation if available
def xpu_gemm_use_xetla(force_xetla=False):
    has_xmx = torch.xpu.has_xmx()
    has_2d_load = torch.xpu.has_2d_block_array()
    compute_eng = torch.xpu.get_compute_eng()

    if has_xmx and has_2d_load:
        if force_xetla:
            return True
        return compute_eng == torch.xpu.XPUComputeEng.XETLA

    return (
        torch.xpu.has_xetla()
        and not torch.xpu.using_onednn_layout()
        and compute_eng
        in (torch.xpu.XPUComputeEng.RECOMMEND, torch.xpu.XPUComputeEng.XETLA)
    )


class WeightOnlyQuantizedLinear(nn.Module):
    __constants__ = ["in_features", "out_features", "blocksize"]
    in_features: int
    out_features: int
    blocksize: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        compute_dtype="fp32",
        compress_statistics=True,
        weight_dtype="int4_fullrange",
        scale_dtype="fp16",
        blocksize: int = 64,
        scheme="sym",
        double_quant_scale_dtype=None,
        compression_dtype=torch.int32,
        compression_dim=1,
        device=None,
        use_optimum_format=False,
        quant_method=0,  # QuantMethod(GPTQ_GEMM)
        act_quant_mode=XPUWoqActQuantMode.UNQUANT_A,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        assert compute_dtype in [
            "fp32",
            "fp16",
        ], "compute_dtype must be 'fp32', 'fp16'."
        assert scale_dtype in [
            "fp16",
            "fp32",
        ], "scale_dtype only support 'fp32', 'fp16'. now."
        self.scale_dtype = scale_dtype
        self.double_quant_scale_dtype = double_quant_scale_dtype
        self.compute_dtype = compute_dtype
        self.compress_statistics = compress_statistics
        self.blocksize = blocksize if blocksize != -1 else self.in_features
        self.scheme = scheme
        self.weight_dtype = weight_dtype
        self.device = device
        # `compression_dim` indicates in which dimension to be compressed in data.
        self.compression_dim = compression_dim
        self.weight_transposed = False
        assert compression_dtype in [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float8_e5m2,
            torch.float8_e4m3fn,
        ], "Only support torch.int8|16|32|64 as compressed dtype."
        dtype_bits_mapping = {
            torch.int8: 8,
            torch.int16: 16,
            torch.int32: 32,
            torch.int64: 64,
            torch.float8_e5m2: 8,
            torch.float8_e4m3fn: 8,
        }
        self.bits = DTYPE_BITS_MAPPING[weight_dtype]
        self.compress_bits = dtype_bits_mapping[compression_dtype]
        self.n_pack = self.compress_bits // self.bits
        self.compression_dtype = compression_dtype
        # K is input channel, N is output channel
        assert compression_dim in [0, 1], (
            "Only support 0 or 1 as compression dimension, "
            + "0 is output channel, 1 is input channel."
        )

        # `use_optimum_format` is for GPTQ model, if it is True, it's weight is k x n,
        # so it needn't to transpose in optimized operator.
        self.use_optimum_format = use_optimum_format

        self.register_parameter("qweight", None)
        self.register_parameter("bias", None)
        self.register_buffer("g_idx", None)
        self.force_xetla = False
        self.quant_method = quant_method
        if act_quant_mode not in XPU_UNSUPPORTED_ACT_QUANT_MODES:
            self.act_quant_mode = VLLM_ACT_QUANT_MODE_TO_XPU[act_quant_mode]
        else:
            # otherwise use per-token asym quantization for activation
            self.act_quant_mode = XPUWoqActQuantMode.QUANT_A_PER_M

    def transpose_xetla_woq_format(self):
        # The xetla int4 GEMM has the following requirements:
        # - Weights need to be contiguous along the 'k' dimension.
        # - Scales also need to be contiguous along the 'k' dimension.
        # - Zero-point is set to 'none' in symmetric (symm) scenarios.
        # - Zero-point remains unchanged in asymmetric (asymm) scenarios.
        self.qweight.data = self.qweight.t().contiguous()
        self.scales.data = self.scales.t().contiguous().to(torch.float16)
        if self.bias is not None:
            self.bias.data = self.bias.contiguous().to(torch.float16)
        if self.scheme == "sym" and self.quant_method == 0:
            self.qzeros = None
        elif self.quant_method == 0:
            self.qzeros += 0x11111111

    def transpose_onednn_woq_format(self):
        # The oneDNN int4 GEMM has the following requirements:
        # - Weights need to be contiguous along the 'k' dimension, but the shape should remain (k, n/8).
        # - Scales remains unchanged.
        # - Zero-point is a scalar value of 8 in symmetric (symm) scenarios, allowing oneDNN to broadcast it.
        # - Zero-point remains unchanged in asymmetric (asymm) scenarios.
        reshaped_tensor = self.qweight.transpose(0, 1).contiguous().transpose(0, 1)
        self.qweight.as_strided_(reshaped_tensor.shape, reshaped_tensor.stride())
        self.qweight.copy_(reshaped_tensor)
        self.scales.data = self.scales.contiguous().to(torch.float16)
        if self.bias is not None:
            self.bias.data = self.bias.contiguous().to(torch.float16)
        if (
            self.scheme == "sym"
            and self.quant_method == 0
            and self.weight_dtype == "int4_fullrange"
        ):
            self.qzeros = torch.Tensor([8]).to(torch.int8).to("xpu")
        elif self.quant_method == 0 and self.weight_dtype == "int4_fullrange":
            self.qzeros += 0x11111111

    @classmethod
    def from_weight(
        cls,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        in_feature: int,
        out_feature: int,
        qconfig=None,
        bias: Optional[torch.Tensor] = None,
        group_size: int = -1,
        g_idx: Optional[torch.Tensor] = None,
        quant_method=0,  # QuantMethod(GPTQ_GEMM)
        dtype=0,  # QuantDtype(INT4)
        **kwargs
    ):
        r"""Create a weight-only quantized module from weight

        Args:
            qweight (Tensor): tensor in int32 dtype and contains actually int4 data
            scales (Tensor): scales for qweight
            zero_points (Tensor): zero points for qweight
            in_feature (int): size of each input sample
            out_feature (int): size of each output sample
            qconfig (object): Defining the IPEX quantization recipe for Weight only quantization.
                Default value is ``None``.
            bias (Tensor or None): bias for linear
            group_size: Group size for weight quantization
            g_idx: Indices of groups for each input channel of weight. Generated by
                GPTQ with act-order.
            quant_method: Quantization method, such as GPTQ, AWQ, ...
            dtype (QuantDtype): quantization data type

        """
        from intel_extension_for_pytorch.llm.quantization.utils import (
            QuantDtype,
            QuantMethod,
        )

        assert (
            dtype == QuantDtype.INT4
            or dtype == QuantDtype.FP8_E5M2
            or dtype == QuantDtype.FP8_E4M3FN
        ), "IPEX only support INT4 FP8_E5M2 FP8_E4M3FN as quantization data type for now."
        assert quant_method in [
            QuantMethod.GPTQ_GEMM,
            QuantMethod.AWQ_GEMM,
        ], "IPEX only support GPTQ_GEMM and AWQ_GEMM as quantization method for now."
        compression_dim = None
        compression_dtype = None
        if quant_method == QuantMethod.GPTQ_GEMM:
            scheme = "sym"
            compression_dim = 1
            compression_dtype = torch.int32
        if quant_method == QuantMethod.AWQ_GEMM:
            scheme = "asym"
            compression_dim = 0
            compression_dtype = torch.int32
        if dtype == QuantDtype.FP8_E5M2:
            compression_dtype = torch.float8_e5m2
        elif dtype == QuantDtype.FP8_E4M3FN:
            compression_dtype = torch.float8_e4m3fn
        act_quant_mode = XPUWoqActQuantMode.UNQUANT_A
        if qconfig is not None and hasattr(qconfig, "global_qconfig"):
            if hasattr(qconfig.global_qconfig, "act_quant_mode"):
                act_quant_mode = qconfig.global_qconfig.act_quant_mode
        cls_inst = WeightOnlyQuantizedLinear(
            in_features=in_feature,
            out_features=out_feature,
            bias=True if bias is not None else False,
            compute_dtype="fp32",
            compress_statistics=True,
            weight_dtype=(
                "int4_fullrange"
                if dtype == QuantDtype.INT4
                else "fp8_e5m2" if dtype == QuantDtype.FP8_E5M2 else "fp8_e4m3fn"
            ),
            scale_dtype="fp16" if scales.dtype == torch.float16 else "fp32",
            blocksize=group_size,
            scheme=scheme,
            double_quant_scale_dtype=None,
            compression_dtype=compression_dtype,
            compression_dim=compression_dim,
            device="xpu",
            use_optimum_format=False,
            quant_method=quant_method,
            act_quant_mode=act_quant_mode,
        )

        if g_idx is not None and quant_method == QuantMethod.GPTQ_GEMM:
            shuffler = GPTQShuffle(bits=4, blocksize=group_size)
            qweight_new, g_idx_new = shuffler(qweight, g_idx)
            qweight.data.copy_(qweight_new)
            g_idx.data.copy_(g_idx_new)
            qweight_new = None
            g_idx_new = None
            del qweight_new, g_idx_new

        cls_inst.set_weights_bias(qweight, bias)
        cls_inst.set_scales_zps_gidx(scales, zero_points, g_idx, quant_method)

        if quant_method == QuantMethod.AWQ_GEMM:
            qweight_new, qzeros_new = fast_awq_to_gptq(
                cls_inst.qweight, cls_inst.qzeros
            )
            if qweight_new.shape != cls_inst.qweight.data.shape:
                cls_inst.qweight.data = cls_inst.qweight.data.view_as(qweight_new)
            if qzeros_new.shape != cls_inst.qzeros.data.shape:
                cls_inst.qzeros.data = cls_inst.qzeros.data.view_as(qzeros_new)
            cls_inst.qweight.data.copy_(qweight_new)
            cls_inst.qzeros.data.copy_(qzeros_new)

        if not cls_inst.weight_transposed and dtype == QuantDtype.INT4:
            if xpu_gemm_use_xetla():
                # Transpose the weight/scale/zp to xetla format
                cls_inst.transpose_xetla_woq_format()

            else:
                # Transpose the weight/scale/zp to oneDNN format
                cls_inst.transpose_onednn_woq_format()

            cls_inst.weight_transposed = True
            cls_inst.use_optimum_format = False

        if dtype == QuantDtype.FP8_E5M2 or dtype == QuantDtype.FP8_E4M3FN:
            if qweight.shape[0] != in_feature:
                cls_inst.weight_transposed = True

        return cls_inst

    def forward(self, input: Tensor, bias: Optional[torch.Tensor] = None) -> Tensor:
        if self.compute_dtype == "fp16":
            input = input.to(convert_dtype_str2torch(self.compute_dtype))
        if not self.weight_transposed and self.weight_dtype == "int4_fullrange":
            if xpu_gemm_use_xetla():
                # Transpose the weight/scale/zp to xetla format
                self.transpose_xetla_woq_format()

            else:
                # Transpose the weight/scale/zp to oneDNN format
                self.transpose_onednn_woq_format()

            self.weight_transposed = True
            self.use_optimum_format = False

        if self.weight_dtype == "int4_fullrange":
            if xpu_gemm_use_xetla():
                # TODO input.shape[1] > 1 seems not work on gidx scenario, need to fix this bug
                if input.dim() == 3:
                    m = input.size(1)
                else:
                    m = input.size(0)
                if m > 1:
                    return dequant_gemm_block(input, self)
                return torch.ops.torch_ipex.mm_low_bits(
                    input,
                    self.qweight,
                    self.scales,
                    self.qzeros,
                    self.bias,
                    self.bias is not None,
                    self.compute_dtype,
                    self.weight_dtype,
                    self.blocksize,
                    self.g_idx,
                )
            elif self.bias is not None:
                if USE_W4A8_GEMM:
                    return torch.ops.torch_ipex.mm_bias_w4a8(
                        input,
                        self.qweight,
                        self.bias,
                        self.scales,
                        self.qzeros,
                        self.act_quant_mode,
                        self.blocksize,
                        self.g_idx,
                    )
                else:
                    return torch.ops.torch_ipex.mm_bias_int4(
                        input,
                        self.qweight,
                        self.bias,
                        self.scales,
                        self.qzeros,
                        self.blocksize,
                        self.g_idx,
                    )
            else:
                if USE_W4A8_GEMM:
                    return torch.ops.torch_ipex.mm_w4a8(
                        input,
                        self.qweight,
                        self.scales,
                        self.qzeros,
                        self.act_quant_mode,
                        self.blocksize,
                        self.g_idx,
                    )
                else:
                    return torch.ops.torch_ipex.mm_int4(
                        input,
                        self.qweight,
                        self.scales,
                        self.qzeros,
                        self.blocksize,
                        self.g_idx,
                    )
        else:
            return torch.ops.torch_ipex.fp8_gemm_w8a16(
                input,
                self.qweight,
                self.weight_transposed,
                self.scales,
                bias,
            )

    def set_weights_bias(self, weight_data, bias=None, update_g_idx=True):
        qweight = ParamsLowBits(
            data=weight_data,
            requires_grad=False,
            quant_state={"scheme": self.scheme},
            blocksize=self.blocksize,
            compress_statistics=self.compress_statistics,
            quant_dtype=self.weight_dtype,
            scale_dtype=self.scale_dtype,
            double_quant_scale_dtype=self.double_quant_scale_dtype,
            compression_dtype=self.compression_dtype,
        )
        self.qweight = qweight
        if bias is not None:
            self.bias = torch.nn.Parameter(
                bias.contiguous().to(torch.float16), requires_grad=False
            )
        if hasattr(self, "g_idx") and self.g_idx is not None and update_g_idx:
            # The prerequisite for this to work is that set_scales_zps_gidx is called first.
            assert self.qweight.data.dtype == torch.int32
            shuf_weight = GPTQShuffle(self.bits, self.blocksize)
            self.qweight.data, self.g_idx = shuf_weight(self.qweight.data, self.g_idx)

    def set_scales_zps_gidx(
        self, scales, qzeros=None, g_idx=None, quant_method=0  # QuantMethod(GPTQ_GEMM)
    ):
        self.register_buffer("scales", scales)
        self.register_buffer("qzeros", qzeros)
        unuse_g_idx = torch.tensor(
            [i // self.blocksize for i in range(self.in_features)],
            dtype=torch.int32,
            device=scales.device,
        )
        if g_idx is not None and not torch.equal(g_idx, unuse_g_idx):
            self.g_idx = g_idx
        if quant_method == 1:  # QuantMethod(AWQ_GEMM)
            self.qzeros = qzeros
        else:
            # WA: for sym scheme with xetla impl., qzeros needs to be None
            if self.scheme == "sym" and xpu_gemm_use_xetla():
                self.qzeros = None
            elif xpu_gemm_use_xetla():
                self.qzeros += 0x11111111
            else:
                self.qzeros = torch.Tensor([8]).to(torch.int8).to("xpu")

    def extra_repr(self) -> str:
        tmp_str = (
            "in_features={}, out_features={}, bits={}, blocksize={}, bias={}".format(
                self.in_features,
                self.out_features,
                self.bits,
                self.blocksize,
                self.bias is not None,
            )
        )
        if self.use_optimum_format:
            tmp_str += ", use_optimum_format=True"
        return tmp_str


def convert_qmodel(model, dtype, group_size):
    def convert_qmodel_recursive(module):
        for name, child in module.named_children():
            if type(child) == torch.nn.Linear:
                qmodule = WeightOnlyQuantizedLinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    group_size=group_size,
                    bias=True if child.bias is not None else False,
                    dtype=dtype,
                )
                setattr(module, name, qmodule)
            else:
                convert_qmodel_recursive(child)

    convert_qmodel_recursive(model)
    return model


def dequant_gemm_block(input, quant_layer, output=None):
    if quant_layer.g_idx is not None:
        input = input[..., quant_layer.g_idx]
    if output is None:
        output = torch.ops.torch_ipex.mm_common(
            input,
            torch.ops.torch_ipex.int4x8_dequantize(
                quant_layer.qweight,
                quant_layer.scales,
                quant_layer.qzeros,
                quant_layer.blocksize,
            ),
        )
    else:
        torch.ops.torch_ipex.mm_common_out(
            input,
            torch.ops.torch_ipex.int4x8_dequantize(
                quant_layer.qweight,
                quant_layer.scales,
                quant_layer.qzeros,
                quant_layer.blocksize,
            ),
            output,
        )
    if quant_layer.bias is not None:
        output += quant_layer.bias
    return output


def dequant_gemm_block_with_params(
    input, qweight, scales, qzeros, blocksize, bias=None, g_idx=None, output=None
):
    if g_idx is not None:
        input = input[:, :, g_idx]
    if output is None:
        output = torch.ops.torch_ipex.mm_common(
            input,
            torch.ops.torch_ipex.int4x8_dequantize(
                qweight,
                scales,
                qzeros,
                blocksize,
            ),
        )
    else:
        torch.ops.torch_ipex.mm_common_out(
            input,
            torch.ops.torch_ipex.int4x8_dequantize(
                qweight,
                scales,
                qzeros,
                blocksize,
            ),
            output,
        )
    if bias is not None:
        output += bias
    return output


class FP8ScaledQuant:
    @classmethod
    def scaled_fp8_quant(
        cls,
        input: torch.Tensor,
        out_dtype: torch.dtype,
        scale: Optional[torch.Tensor] = None,
        num_token_padding: Optional[int] = None,
        scale_ub: Optional[torch.Tensor] = None,
        use_per_token_if_dynamic: bool = False,
    ):
        """
        Quantize input tensor to FP8 and return quantized tensor and scale.

        This function supports both static and dynamic quantization: If you
        provide the scale, it will use static scaling and if you omit it,
        the scale will be determined dynamically. The function also allows
        optional padding of the output tensors for downstream kernels that
        will benefit from padding.

        Args:
            input: The input tensor to be quantized to FP8
            scale: Optional scaling factor for the FP8 quantization
            scale_ub: Optional upper bound for scaling factor in dynamic
                per token case
            num_token_padding: If specified, pad the first dimension
                of the output to at least this value.
            use_per_token_if_dynamic: Whether to do per_tensor or per_token
                in the dynamic quantization case.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
                scaling factor.
        """
        # This code assumes batch_dim and num_tokens are flattened
        assert input.ndim == 2
        shape: Union[tuple[int, int], torch.Size] = input.shape
        if num_token_padding:
            shape = (max(num_token_padding, input.shape[0]), shape[1])
        output = torch.empty(shape, device=input.device, dtype=out_dtype)

        if scale is None:
            if use_per_token_if_dynamic:
                # TODO: add per-token support
                pass
            else:
                scale = torch.zeros(1, device=input.device, dtype=torch.float32)
                torch.ops.torch_ipex.dynamic_scaled_fp8_quant(output, input, scale)
        else:
            # num_token_padding not implemented for this case
            assert scale.numel() == 1 or num_token_padding is None
            torch.ops.torch_ipex.static_scaled_fp8_quant(output, input, scale)

        return output, scale
