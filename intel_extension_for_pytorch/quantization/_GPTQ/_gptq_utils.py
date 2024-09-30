import logging
import torch
import math
from torch.nn import functional as F

format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=format_str)
logger = logging.getLogger("GPTQ")
logger.setLevel(logging.INFO)


def gptq_quantize(
    model,
    weight_config=None,
    dataloader=None,
    nsamples=128,
    use_max_length=True,
    pad_max_length=2048,
    device=torch.device("cpu"),
    layer_wise=False,
    model_path=None,
):
    """Run weight-only quantization with weight configs.

    Args:
        model (torch.nn.Module): fp32 model to quantize
        weight_config (dict, optional): contains all info required by GPTQ. Defaults to {}.
            For example:
            weight_config={
                'layer1':
                {
                    'bits': 4,
                    'group_size': 32,
                    'sym': False,
                    'percdamp': .01,
                    'act_order': False
                }
                ...
            }
        dataloader (torch.utils.data.DataLoader): an iterable containing calibration datasets.
        nsamples (int): calibration samples' size.
        use_max_length (bool): whether to align calibration data to a fixed length.
        pad_max_length (int): whether to align calibration data to a fixed length.
        device: set to torch.device("cpu").
        layer_wise (bool): whether to do LWQ.
        model_path (str): path to register LWQ weight hooks.
    """
    assert isinstance(model, torch.nn.Module), "only support torch module"
    if layer_wise:
        assert (
            model_path is not None
        ), "model_path should not be None when use layer_wise mode"

    from .gptq import GPTQuantizer

    if weight_config is None:
        weight_config = {}

    gptq_quantizer = GPTQuantizer(
        model,
        weight_config,
        dataloader,
        nsamples,
        use_max_length,
        pad_max_length,
        device,
        layer_wise=layer_wise,
    )
    fp32_modified_model, gptq_config = gptq_quantizer.execute_quantization(
        model_path=model_path
    )
    logger.info("GPTQ quantizing done.")
    return fp32_modified_model, gptq_config


def gptq_export(
    model,
    weight_config,
    gptq_config,
    compression_dtype=torch.int32,
    compression_dim=1,
    scale_dtype=torch.float16,
):
    for k, v in weight_config.items():
        if "lm_head" in k or "output_layer" in k or "embed_out" in k:
            continue
        logger.info(f"Exporting {k}")
        num_bits = v["wbits"]
        group_size = v["group_size"]
        sym = v["sym"]
        m = fetch_module(model, k)
        gptq_conf = gptq_config[k]
        if "perm" in gptq_conf:
            gptq_perm = torch.tensor(gptq_conf["perm"])
            fp32_weight = m.weight.data[:, gptq_perm]
        else:
            fp32_weight = m.weight.data
            gptq_perm = None
        gptq_scale = torch.tensor(gptq_conf["scale"], dtype=torch.float32)
        gptq_zp = None if sym else torch.tensor(gptq_conf["zero"], dtype=torch.int32)
        int_weight = quant_weight_w_scale(fp32_weight, gptq_scale, gptq_zp, group_size)
        int_weight = int_weight.type(torch.int32)
        if "perm" in gptq_conf:
            invperm = torch.argsort(gptq_perm)
            int_weight = int_weight[:, invperm]
        new_module = WeightOnlyLinear(
            m.in_features,
            m.out_features,
            num_bits,
            group_size,
            dtype="int",
            zp=gptq_zp is not None,
            bias=m.bias is not None,
            g_idx=gptq_perm is not None,
            compression_dtype=compression_dtype,
            compression_dim=compression_dim,
            scale_dtype=scale_dtype,
            device=torch.device("cpu"),
            use_optimum_format=True,
        )
        new_module.pack(int_weight, gptq_scale, gptq_zp, m.bias, gptq_perm)
        set_module(model, k, new_module)
    return model


def fetch_module(model, op_name):
    """Get module with a given op name.

    Args:
        model (object): the input model.
        op_name (str): name of op.

    Returns:
        module (object).
    """
    module = model
    name_list = op_name.split(".")
    for name in name_list:
        if hasattr(module, name):
            module = getattr(module, name)
        else:
            module = module
    return module


def set_module(model, op_name, new_module):
    """Set module with a given op name.

    Args:
        model (object): the input model.
        op_name (str): name of op.
        new_module (object): the input model.

    Returns:
        module (object).
    """
    module = model
    name_list = op_name.split(".")
    for name in name_list[:-1]:
        if hasattr(module, name):
            module = getattr(module, name)
        else:
            module = module
    setattr(module, name_list[-1], new_module)


def quant_weight_w_scale(weight, scale, zp, group_size=128):
    """Quant and dequant tensor with group size.

    Args:
        weight: input weight
        scale: scale
        zp: zero point
        group_size (int, optional): how many elements share one scale/zp.

    Returns:
        output: int weight.
    """
    device = weight.device
    scale = scale.to(device)
    if zp is not None:
        zp = zp.to(device)
    if group_size == -1:
        return (
            torch.round(weight / scale)
            if zp is None
            else torch.round(weight / scale + zp)
        )
    int_weight = torch.zeros(weight.shape).to(device)
    leng = weight.shape[1] // group_size
    tail_flag = False if weight.shape[1] % group_size == 0 else True
    for i in range(leng):
        int_weight_tmp = weight[:, i * group_size : (i + 1) * group_size] / scale[
            :, i
        ].unsqueeze(1)
        if zp is not None:
            int_weight_tmp += zp[:, i].unsqueeze(1)
        int_weight[:, i * group_size : (i + 1) * group_size] = torch.round(
            int_weight_tmp
        )
    if tail_flag:
        int_weight_tmp = weight[:, leng * group_size :] / scale[:, -1].unsqueeze(1)
        if zp is not None:
            int_weight_tmp += zp[:, -1].unsqueeze(1)
        int_weight[:, leng * group_size :] = torch.round(int_weight_tmp)
    return int_weight


class WeightOnlyLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bits,
        groupsize,
        dtype="int",
        zp=False,
        bias=False,
        scale_dtype=torch.float32,
        compression_dtype=torch.int32,
        compression_dim=1,
        g_idx=False,
        device="cpu",
        use_optimum_format=False,
    ):
        super().__init__()
        self.use_optimum_format = use_optimum_format
        self.dtype = dtype
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.groupsize = groupsize if groupsize != -1 else in_features
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
        self.compress_bits = dtype_bits_mapping[compression_dtype]
        self.n_pack = self.compress_bits // self.bits
        # K is input channel, N is output channel
        assert compression_dim in [0, 1], (
            "Only support 0 or 1 as compression dimension, "
            + "0 is output channel, 1 is input channel."
        )
        if self.use_optimum_format:
            self.float_type = torch.float16
            self.compressed_dtype = torch.int32
            self.register_buffer(
                "scales",
                torch.zeros(
                    (math.ceil(in_features / self.groupsize), out_features),
                    dtype=self.float_type,
                ).to(device),
            )
            self.scales = self.scales.T
            self.register_buffer(
                "qweight",
                torch.zeros(
                    (math.ceil(in_features / self.n_pack), out_features),
                    dtype=self.compressed_dtype,
                ).to(device),
            )
            self.qweight = self.qweight.T
            self.register_buffer(
                "qzeros",
                torch.zeros(
                    (
                        math.ceil(self.in_features / self.groupsize),
                        math.ceil(self.out_features / self.n_pack),
                    ),
                    dtype=self.compressed_dtype,
                ).to(device),
            )
            self.qzeros = self.qzeros.T
            self.register_buffer(
                "bias", torch.zeros(self.out_features, dtype=self.float_type).to(device)
            )
        else:
            self.compressed_dtype = compression_dtype
            self.float_type = scale_dtype
            self.register_buffer(
                "scales",
                torch.zeros(
                    (out_features, math.ceil(in_features / self.groupsize)),
                    dtype=self.float_type,
                ).to(device),
            )
            if compression_dim == 1:
                self.register_buffer(
                    "qweight",
                    torch.zeros(
                        (out_features, math.ceil(in_features / self.n_pack)),
                        dtype=self.compressed_dtype,
                    ).to(device),
                )
                if zp:
                    self.register_buffer(
                        "qzeros",
                        torch.zeros(
                            (
                                self.out_features,
                                math.ceil(
                                    self.in_features / self.groupsize / self.n_pack
                                ),
                            ),
                            dtype=self.compressed_dtype,
                        ).to(device),
                    )
            else:
                self.register_buffer(
                    "qweight",
                    torch.zeros(
                        (math.ceil(out_features / self.n_pack), in_features),
                        dtype=self.compressed_dtype,
                    ).to(device),
                )
                if zp:
                    self.register_buffer(
                        "qzeros",
                        torch.zeros(
                            (
                                math.ceil(self.out_features / self.n_pack),
                                math.ceil(self.in_features / self.groupsize),
                            ),
                            dtype=self.compressed_dtype,
                        ).to(device),
                    )
            if bias:
                self.register_buffer(
                    "bias",
                    torch.zeros(self.out_features, dtype=self.float_type).to(device),
                )
            else:
                self.bias = None
        if g_idx:
            self.register_buffer(
                "g_idx", torch.zeros(in_features, dtype=torch.int32).to(device)
            )
        else:
            self.g_idx = None

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
            self.bias = bias.type(self.float_type).to(self.device)
        if g_idx is not None:
            assert hasattr(self, "g_idx"), "g_idx is not set when initializing."
            self.g_idx = g_idx.type(torch.int32).to(self.device)
            if self.use_optimum_format:
                invperm = torch.argsort(self.g_idx)
                self.g_idx = invperm // self.groupsize
                self.g_idx = self.g_idx.type(torch.int32).to(self.device)
        assert scale.shape == self.scales.shape, "Scale shape is mismatched."
        self.scales = scale.type(self.float_type).to(self.device)
        if not self.use_optimum_format and self.compression_dim == 0:
            int_weight = int_weight.T
            self.qweight = self.qweight.T
        origin_shape = int_weight.shape
        target_shape = self.qweight.shape
        assert (
            origin_shape[0] == target_shape[0]
        ), "output channels mismatch, please check."
        mask = torch.tensor(2**self.bits - 1, dtype=self.compressed_dtype).to(
            self.device
        )

        # pack weight
        for j in range(target_shape[1]):
            start = self.n_pack * j
            end = self.n_pack * (j + 1)
            tmp = int_weight[:, start:end].type(self.compressed_dtype)
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
                tmp = zp[:, start:end].type(self.compressed_dtype)
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
        logger.debug(f"Recovering {self} weight")
        scales = self.scales.T if self.use_optimum_format else self.scales
        qweight = self.qweight.T if self.use_optimum_format else self.qweight

        device = scales.device
        fp32_weight = torch.zeros(
            self.out_features, self.in_features, dtype=self.float_type
        ).to(device)
        if self.g_idx is None:
            # used for recovering fp32_weight
            self.g_idx = torch.tensor(
                [i // self.groupsize for i in range(self.in_features)],
                dtype=torch.int32,
            )
        mask = torch.tensor(2**self.bits - 1, dtype=self.compressed_dtype).to(device)
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
            zp_dtype = self.compressed_dtype  # to avoid overflow when weight-zp
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

    def forward(self, input):
        weight = self.recover()
        device = self.scales.device
        if weight.dtype == torch.float16 and device.type == "cpu":
            weight = weight.float()
            self.bias = self.bias.float() if self.bias is not None else None
            input = input.type(weight.dtype)
            return F.linear(input, weight, self.bias)

    def extra_repr(self) -> str:
        tmp_str = (
            "in_features={}, out_features={}, bits={}, group_size={}, bias={}".format(
                self.in_features,
                self.out_features,
                self.bits,
                self.groupsize,
                self.bias is not None,
            )
        )
        if self.use_optimum_format:
            tmp_str += ", use_optimum_format=True"
        return tmp_str
