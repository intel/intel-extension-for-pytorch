import torch
import intel_extension_for_pytorch as ipex  # noqa
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
from torch import Tensor


DTYPE_BITS_MAPPING = {
    "nf4": 4,
    "fp4_e2m1_bnb": 4,
    "fp4_e2m1": 4,
    "int4_fullrange": 4,
    "int4_clip": 4,
    "int8": 8,
}


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
        assert False, "Unsupport str dtype {} to torch dtype".format(str_dtype)


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


class WeightOnlyLinear(nn.Module):
    __constants__ = ["in_features", "out_features", "group_size"]
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
        scale_dtype="fp32",
        blocksize: int = 64,
        scheme="sym",
        double_quant_scale_dtype=None,
        compression_dtype=torch.int32,
        compression_dim=1,
        g_idx=False,
        device=None,
        use_optimum_format=False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        assert compute_dtype in [
            "fp32",
            "fp16",
        ], "compute_dtype must be 'fp32', 'fp16'."
        assert scale_dtype in [
            "fp32",
        ], "scale_dtype only support 'fp32' now."
        self.scale_dtype = scale_dtype
        self.double_quant_scale_dtype = double_quant_scale_dtype
        self.compute_dtype = compute_dtype
        self.compress_statistics = compress_statistics
        self.blocksize = blocksize
        self.scheme = scheme
        self.weight_dtype = weight_dtype
        self.device = device
        self.compression_dim = compression_dim
        assert compression_dtype in [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ], "Only support torch.int8|16|32|64 as compressed dtype."
        dtype_bits_mapping = {
            torch.int8: 8,
            torch.int16: 16,
            torch.int32: 32,
            torch.int64: 64,
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
        self.use_optimum_format = use_optimum_format
        if self.use_optimum_format:
            self.register_buffer(
                "scales",
                torch.empty(
                    (math.ceil(in_features / self.blocksize), out_features),
                    dtype=convert_dtype_str2torch(self.scale_dtype),
                    device=device,
                ),
            )
            self.scales = self.scales.T
            weight = torch.empty(
                (math.ceil(in_features / self.n_pack), out_features),
                dtype=self.compression_dtype,
                device=device,
            )
            self.register_buffer(
                "qweight",
                ParamsLowBits(
                    weight,
                    requires_grad=False,
                    quant_state={"scheme": self.scheme},
                    blocksize=self.blocksize,
                    compress_statistics=self.compress_statistics,
                    quant_dtype=self.weight_dtype,
                    scale_dtype=self.scale_dtype,
                    double_quant_scale_dtype=self.double_quant_scale_dtype,
                    compression_dtype=self.compression_dtype,
                ),
            )
            self.qweight = self.qweight.T
            self.register_buffer(
                "qzeros",
                torch.empty(
                    (
                        math.ceil(self.in_features / self.blocksize),
                        math.ceil(self.out_features / self.n_pack),
                    ),
                    dtype=self.compression_dtype,
                    device=device,
                ),
            )
            self.qzeros = self.qzeros.T
        else:
            self.register_buffer(
                "scales",
                torch.empty(
                    (out_features, math.ceil(in_features / self.blocksize)),
                    dtype=convert_dtype_str2torch(self.scale_dtype),
                    device=device,
                ),
            )
            if compression_dim == 1:
                weight = torch.empty(
                    (out_features, math.ceil(in_features / self.n_pack)),
                    dtype=self.compression_dtype,
                    device=device,
                )
                self.register_buffer(
                    "qweight",
                    ParamsLowBits(
                        weight,
                        requires_grad=False,
                        quant_state={"scheme": self.scheme},
                        blocksize=self.blocksize,
                        compress_statistics=self.compress_statistics,
                        quant_dtype=self.weight_dtype,
                        scale_dtype=self.scale_dtype,
                        double_quant_scale_dtype=self.double_quant_scale_dtype,
                        compression_dtype=self.compression_dtype,
                    ),
                ),
                self.register_buffer(
                    "qzeros",
                    torch.zeros(
                        (
                            self.out_features,
                            math.ceil(self.in_features / self.blocksize / self.n_pack),
                        ),
                        dtype=self.compression_dtype,
                        device=device,
                    ),
                )
            else:
                weight = torch.empty(
                    (math.ceil(out_features / self.n_pack), in_features),
                    dtype=self.compression_dtype,
                    device=device,
                )
                self.register_buffer(
                    "qweight",
                    ParamsLowBits(
                        weight,
                        requires_grad=False,
                        quant_state={"scheme": self.scheme},
                        blocksize=self.blocksize,
                        compress_statistics=self.compress_statistics,
                        quant_dtype=self.weight_dtype,
                        scale_dtype=self.scale_dtype,
                        double_quant_scale_dtype=self.double_quant_scale_dtype,
                        compression_dtype=self.compression_dtype,
                    ),
                )
                self.register_buffer(
                    "qzeros",
                    torch.zeros(
                        (
                            math.ceil(self.out_features / self.n_pack),
                            math.ceil(self.in_features / self.blocksize),
                        ),
                        dtype=self.compression_dtype,
                        device=device,
                    ),
                )
        if g_idx:
            self.register_buffer(
                "g_idx", torch.empty(in_features, dtype=torch.int32, device=device)
            )
        else:
            self.g_idx = None
        if bias:
            self.register_buffer(
                "bias",
                torch.empty(
                    self.out_features,
                    dtype=convert_dtype_str2torch(self.compute_dtype),
                    device=device,
                ),
            )
        else:
            self.bias = None

    def pack(self, int_weight, scale, zp, bias, g_idx=None):
        int_weight = int_weight.to(self.device)
        if self.use_optimum_format and zp is None:
            # to avoid overflow
            int_weight = int_weight.type(torch.int32)
            shift_bias = 2 ** (self.bits - 1)
            int_weight += shift_bias
            zp = torch.zeros_like(scale, dtype=torch.uint8) + shift_bias
        if bias is not None:
            assert hasattr(self, "bias"), "bias is not set when initializing."
            self.bias = bias.type(convert_dtype_str2torch(self.compute_dtype)).to(
                self.device
            )
        if g_idx is not None:
            assert hasattr(self, "g_idx"), "g_idx is not set when initializing."
            self.g_idx = g_idx.type(torch.int32).to(self.device)
            if self.use_optimum_format:
                invperm = torch.argsort(self.g_idx)
                self.g_idx = invperm // self.blocksize
                self.g_idx = self.g_idx.type(torch.int32).to(self.device)
        assert scale.shape == self.scales.shape, "Scale shape is mismatched."
        self.scales = scale.type(self.scale_type).to(self.device)
        if not self.use_optimum_format and self.compression_dim == 0:
            int_weight = int_weight.T
            self.qweight = self.qweight.T
        origin_shape = int_weight.shape
        target_shape = self.qweight.shape
        assert (
            origin_shape[0] == target_shape[0]
        ), "output channels mismatch, please check."
        mask = torch.tensor(2**self.bits - 1, dtype=self.compression_dtype).to(
            self.device
        )

        # pack weight
        for j in range(target_shape[1]):
            start = self.n_pack * j
            end = self.n_pack * (j + 1)
            tmp = int_weight[:, start:end].type(self.compression_dtype)
            for e in range(tmp.shape[1]):
                tmp[:, e] &= mask
                tmp[:, e] = tmp[:, e] << (self.bits * e)
                self.qweight[:, j] |= tmp[:, e]
        if not self.use_optimum_format and self.compression_dim == 0:
            self.qweight = self.qweight.T

        if zp is not None:
            zp = zp.to(self.device)
            if self.use_optimum_format:
                zp -= 1
            if self.use_optimum_format or self.compression_dim == 0:
                zp = zp.T
                self.qzeros = self.qzeros.T
            assert hasattr(self, "qzeros"), "zp is not set when initializing."
            target_shape = self.qzeros.shape
            for j in range(target_shape[1]):
                start = self.n_pack * j
                end = self.n_pack * (j + 1)
                tmp = zp[:, start:end].type(self.compression_dtype)
                for e in range(tmp.shape[1]):
                    tmp[:, e] &= mask
                    tmp[:, e] = tmp[:, e] << (self.bits * e)
                    self.qzeros[:, j] |= tmp[:, e]
            if self.use_optimum_format or self.compression_dim == 0:
                self.qzeros = self.qzeros.T
        if self.use_optimum_format:
            self.scales = self.scales.T
            self.qweight = self.qweight.T
            self.qzeros = self.qzeros.T

    def recover(self):
        scales = self.scales.T if self.use_optimum_format else self.scales
        qweight = self.qweight.T if self.use_optimum_format else self.qweight

        device = scales.device
        fp32_weight = torch.zeros(
            self.out_features,
            self.in_features,
            dtype=convert_dtype_str2torch(self.compute_dtype),
            device=device,
        )
        if self.g_idx is None:
            # used for recovering fp32_weight
            self.g_idx = torch.tensor(
                [i // self.blocksize for i in range(self.in_features)],
                dtype=torch.int32,
            )
        mask = torch.tensor(2**self.bits - 1, dtype=self.compression_dtype).to(device)
        if hasattr(self, "qzeros"):
            weight_dtype = torch.uint8
        else:
            weight_dtype = torch.int8
        # unpack weight
        weight = torch.zeros(
            self.out_features, self.in_features, dtype=weight_dtype
        ).to(device)
        if not self.use_optimum_format and self.compression_dim == 0:
            weight = weight.T
            qweight = qweight.T
        origin_shape = weight.shape
        target_shape = qweight.shape
        for j in range(target_shape[1]):
            for e in range(self.n_pack):
                index = j * self.n_pack + e
                if index >= origin_shape[1]:
                    continue
                tmp = qweight[:, j]
                tmp = tmp << (self.compress_bits - self.bits * (e + 1))
                tmp = tmp >> self.compress_bits - self.bits
                if weight_dtype == torch.uint8:
                    tmp &= mask  # remove sign bit
                weight[:, index] = tmp.type(weight_dtype)
        if not self.use_optimum_format and self.compression_dim == 0:
            weight = weight.T
        if "int" not in self.dtype:
            new_weight = torch.zeros(self.out_features, self.in_features).to(device)
            for k, v in self.int2float_mapping.items():
                new_weight += torch.where(weight == k, v, 0)
            weight = new_weight
        # unpack zero_point
        if hasattr(self, "qzeros"):
            zp_dtype = self.compression_dtype  # to avoid overflow when weight-zp
            zp = torch.zeros(scales.shape, dtype=zp_dtype).to(device)
            qzeros = self.qzeros.T if self.use_optimum_format else self.qzeros
            if self.use_optimum_format or self.compression_dim == 0:
                zp = zp.T
                qzeros = qzeros.T
            origin_shape = zp.shape
            target_shape = qzeros.shape
            for j in range(target_shape[1]):
                for e in range(self.n_pack):
                    index = j * self.n_pack + e
                    if index >= origin_shape[1]:
                        continue
                    tmp = qzeros[:, j]
                    tmp = tmp << (self.compress_bits - self.bits * (e + 1))
                    tmp = tmp >> self.compress_bits - self.bits
                    tmp &= mask
                    zp[:, index] = tmp.type(zp_dtype)
            if self.use_optimum_format or self.compression_dim == 0:
                zp = zp.T
            if self.use_optimum_format:
                # zp -= 1 may cause zp == -1, after recover it becomes 2**self.bits - 1
                zp += 1
                zp = torch.where(zp > (2**self.bits - 1), 0, zp)
            # recover fp32 weight with int_weight, scale, and zero_point
            for idx in range(self.in_features):
                fp32_weight[:, idx] = (
                    weight[:, idx] - zp[:, self.g_idx[idx]]
                ) * scales[:, self.g_idx[idx]]
        else:
            # recover fp32 weight with int_weight, scale
            for idx in range(self.in_features):
                fp32_weight[:, idx] = weight[:, idx] * scales[:, self.g_idx[idx]]
        return fp32_weight

    def forward(self, input: Tensor) -> Tensor:
        return torch.ops.torch_ipex.mm_low_bits(
            input,
            self.qweight.byte(),
            self.scales,
            self.qzeros.byte() if hasattr(self, "qzeros") else None,
            self.bias,
            self.bias is not None,
            self.compute_dtype,
            self.weight_dtype,
            self.blocksize,
        )

    def set_weights_bias(self, weight_data, bias=None):
        self.qweight = ParamsLowBits(
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
        if bias is not None:
            self.bias = torch.nn.Parameter(bias, requires_grad=False)

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
                qmodule = WeightOnlyLinear(
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
