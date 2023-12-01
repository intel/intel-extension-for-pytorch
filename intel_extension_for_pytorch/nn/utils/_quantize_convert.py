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
    "int8": 8
}


class ParamsLowBits(torch.nn.Parameter):
    def __new__(
            cls,
            data=None,
            requires_grad=True,
            quant_state=None,
            blocksize=32,
            compress_statistics=True,
            quant_dtype='int8'
    ):
        if data is None:
            data = torch.empty(0)

        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.blocksize = blocksize
        self.compress_statistics = compress_statistics
        self.quant_dtype = quant_dtype
        self.quant_state = quant_state
        self.data = data
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
        blocksize: int = 64,
        scheme="sym",
        zp=False,
        scale_dtype=torch.float32,
        compression_dtype=torch.int32,
        compression_dim=1,
        g_idx=False,
        device=None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        assert compute_dtype in ['fp32', 'fp16'], \
            "compute_dtype must be 'fp32', 'fp16' on intel CPU device."
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
        dtype_bits_mapping = {torch.int8: 8, torch.int16: 16, torch.int32: 32, torch.int64: 64}
        self.bits = DTYPE_BITS_MAPPING[weight_dtype]
        self.compress_bits = dtype_bits_mapping[compression_dtype]
        self.n_pack = self.compress_bits // self.bits
        self.compressed_dtype = compression_dtype
        self.float_type = scale_dtype
        # K is input channel, N is output channel
        assert compression_dim in [0, 1], (
            "Only support 0 or 1 as compression dimension, " + "0 is output channel, 1 is input channel."
        )
        if self.use_hf_format:
            self.register_buffer(
                "scales",
                torch.empty(
                    (math.ceil(in_features / self.groupsize), out_features),
                    dtype=self.float_type, device=device
                ),
            )
            self.scales = self.scales.T
            weight = torch.empty(
                (math.ceil(in_features / self.n_pack), out_features),
                dtype=self.compressed_dtype, device=device
            )
            self.register_parameter(
                "weight",
                ParamsLowBits(weight, requires_grad=False, quant_state={"scheme": self.scheme}, blocksize=self.blocksize,
                    compress_statistics=self.compress_statistics, quant_dtype=self.weight_dtype),
            )
            self.weight = self.weight.T
            self.register_buffer(
                "qzeros",
                torch.empty(
                    (math.ceil(self.in_features / self.groupsize), math.ceil(self.out_features / self.n_pack)),
                    dtype=self.compressed_dtype, device=device
                ),
            )
            self.qzeros = self.qzeros.T
        else:
            self.register_buffer(
                "scales",
                torch.empty(
                    (out_features, math.ceil(in_features / self.groupsize)),
                    dtype=self.float_type, device=device
                ),
            )
            if compression_dim == 1:
                weight = torch.empty(
                    (out_features, math.ceil(in_features / self.n_pack)),
                    dtype=self.compressed_dtype, device=device
                )
                self.register_parameter(
                    "weight",
                    ParamsLowBits(weight, requires_grad=False, quant_state={"scheme": self.scheme}, blocksize=self.blocksize,
                        compress_statistics=self.compress_statistics, quant_dtype=self.weight_dtype),
                ),
                if zp:
                    self.register_buffer(
                        "qzeros",
                        torch.empty(
                            (self.out_features, math.ceil(self.in_features / self.groupsize / self.n_pack)),
                            dtype=self.compressed_dtype, device=device
                        ),
                    )
            else:
                weight = torch.empty(
                    (math.ceil(out_features / self.n_pack), in_features),
                    dtype=self.compressed_dtype, device=device
                )
                self.register_buffer(
                    "weight",
                    ParamsLowBits(weight, requires_grad=False, quant_state={"scheme": self.scheme}, blocksize=self.blocksize,
                        compress_statistics=self.compress_statistics, quant_dtype=self.weight_dtype),
                )
                if zp:
                    self.register_buffer(
                        "qzeros",
                        torch.empty(
                            (math.ceil(self.out_features / self.n_pack), math.ceil(self.in_features / self.groupsize)),
                            dtype=self.compressed_dtype, device=device
                        ),
                    )
        if g_idx:
            self.register_buffer("g_idx", torch.empty(in_features, dtype=torch.int32, device=device)
        else:
            self.g_idx = None
        if bias:
            bias = torch.empty(self.out_features, dtype=self.float_type, device=device)
            self.register_parameter("bias", torch.nn.Parameter(bias, requires_grad=False))
        else:
            self.bias = None

    def forward(self, input: Tensor) -> Tensor:
        if self.bias is None:
            return torch.ops.torch_ipex.mm_low_bits(
                input, self.weight, self.scales, self.qzeros, self.compute_dtype, self.weight_dtype, self.blocksize
            )
        else:
            return torch.ops.torch_ipex.mm_low_bits(
                input,
                self.weight,
                self.bias,
                self.scales,
                self.qzeros,
                self.compute_dtype,
                self.weight_dtype,
                self.blocksize,
            )

    def init_weights_bias(self, weight_data, bias=None):
        weight = gbits.quantize(
            weight_data, True, self.blocksize, self.compute_dtype, self.weight_dtype
        )
        self.weight = ParamsGBits(
            data=weight, requires_grad=False, quant_state={"scheme": self.scheme}, blocksize=self.blocksize,
            compress_statistics=self.compress_statistics, quant_dtype=self.weight_dtype
        )
        if bias is not None:
            self.bias = torch.nn.Parameter(bias, requires_grad=False)

    def extra_repr(self) -> str:
        tmp_str = "in_features={}, out_features={}, bits={}, blocksize={}, bias={}".format(
            self.in_features,
            self.out_features,
            self.bits,
            self.blocksize,
            self.bias is not None,
        )
        if self.use_hf_format:
            tmp_str += ", use_hf_format=True"
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
