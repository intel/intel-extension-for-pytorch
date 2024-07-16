import torch
import torch.nn as nn
import os
import torch.distributed as dist
from intel_extension_for_pytorch.nn.utils._quantize_convert import (
    WeightOnlyQuantizedLinear,
)


class EnvParam:
    @classmethod
    def collect_all_env_param(cls):
        flag_list = {"LLM_ACC_TEST": "acc_test"}
        number_list = {"TP_SIZE": "tp_size"}
        for flag, flag_name in flag_list.items():
            flag = os.environ.get(flag, "OFF").upper() in [
                "1",
                "Y",
                "ON",
                "YES",
                "TRUE",
            ]
            setattr(EnvParam, flag_name, flag)
        for num, name in number_list.items():
            num = os.environ.get(num, 0)
            setattr(EnvParam, name, num)

    @classmethod
    def set_env(cls, name, value):
        setattr(EnvParam, name, value)


EnvParam.collect_all_env_param()


class IPEXOpForInference(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.acc_test = EnvParam.acc_test

    def port_data(self):
        raise NotImplementedError


class IpexFastLayerNorm(IPEXOpForInference):
    def __init__(self, module: nn.LayerNorm):
        super().__init__(module)
        self.weight = None
        self.bias = None
        self.eps = None
        self.normalized_shape = None
        self.port_data()

    def port_data(self):
        self.weight = self.module.weight
        self.bias = self.module.bias
        self.eps = self.module.eps
        self.normalized_shape = self.module.normalized_shape

    def forward(self, input_tensor):
        out = torch.ops.torch_ipex.fast_layer_norm(
            input_tensor, self.normalized_shape, self.weight, self.bias, self.eps
        )
        return out


class IpexFastLinear(IPEXOpForInference):
    def __init__(self, module):
        super().__init__(module)
        self.weight = None
        self.bias = None
        self.port_data()

    def port_data(self):
        self.weight = self.module.weight.transpose(0, 1).contiguous()
        self.module.weight.data = self.weight.data
        self.bias = self.module.bias

    def forward(self, input_tensor):
        if input_tensor.dim() > 3:
            input_tensor = input_tensor.reshape(
                [-1, input_tensor.shape[-2], input_tensor.shape[-1]]
            )
        shape = input_tensor.shape[:-1] + (self.weight.shape[1],)
        if self.bias is not None:
            return torch.addmm(
                self.bias, input_tensor.flatten(0, -2), self.weight
            ).view(shape)
        else:
            return torch.matmul(input_tensor.flatten(0, -2), self.weight).view(shape)


class IpexFastAllReduceLinear(IPEXOpForInference):
    def __init__(self, module):
        super().__init__(module)
        self.weight = None
        self.bias = None
        self.mp_group = None
        self.port_data()

    def port_data(self):
        self.weight = self.module.weight.transpose(0, 1).contiguous()
        self.module.weight.data = self.weight.data
        self.bias = self.module.bias
        self.mp_group = self.module.mp_group

    def forward(self, input):
        output = torch.matmul(input, self.weight)
        if self.mp_group is not None:
            dist.all_reduce(output, group=self.mp_group)
        if self.bias is not None:
            output += self.bias
        return output


class IPEXLmHeadLinearAllreduceWithPadding(IPEXOpForInference):
    def __init__(self, module):
        super().__init__(module)
        self.weight = None
        self.bias = None
        self.mp_group = None
        self.rank = None
        self.world_size = None
        self.port_data()

    def port_data(self):
        self.n_dim = self.module.weight.shape[0]
        self.bias = self.module.bias
        self.weight = self.module.weight.transpose(-1, -2).contiguous()
        if dist.is_initialized():
            self.mp_group = self.module.mp_group
            self.rank = self.module.rank
            self.world_size = self.module.world_size

    def forward(self, input):
        if input.dim() > 3:
            input = input.reshape([-1, input.shape[-2], input.shape[-1]])
        if not self.acc_test:
            shape = list(input.size())
            shape[1] = 1
            input = input[:, -1, :].view(shape)
        # input.dim() == 3 and weight.dim() == 2
        # The function `matmul` will check if input is contiguous or weight requires grad to determine whether to go into mm or bmm.
        # mm will fold the input while bmm will expand the weight. Expanding the weight will cause performance regression.
        # The weight should not require grad in inference. So, we make sure that the input is contiguous.
        # Refer https://github.com/pytorch/pytorch/blob/v2.1.0/aten/src/ATen/native/LinearAlgebra.cpp#L1882 for the code logic.
        input = input.contiguous()
        if dist.is_initialized():
            assert (
                input.shape[-1] % self.world_size == 0
            ), "Please ensure that self.world_size is divisible by input.shape[-1]"
            input_shard = input.shape[-1] // self.world_size
            output = torch.matmul(
                input[:, :, self.rank * input_shard : (self.rank + 1) * input_shard],
                self.weight,
            )
            if self.mp_group is not None:
                dist.all_reduce(output, group=self.mp_group)
            if self.bias is not None:
                output += self.bias
            output = output[:, :, : self.n_dim]
            return output
        else:
            output = torch.matmul(input, self.weight)
            if self.bias is not None:
                output += self.bias
            output = output[:, :, : self.n_dim]
            return output


class IPEXLmHeadLinearAllreduceWithPaddingInt4(IPEXOpForInference):
    def __init__(self, module: WeightOnlyQuantizedLinear):
        super().__init__(module)
        self.qweight = None
        self.bias = None
        self.scales = None
        self.qzeros = None
        self.blocksize = None
        self.port_data()

    def port_data(self):
        self.n_dim = self.module.scales.shape[1]  # 32000 or 128256
        self.bias = self.module.bias
        self.qweight = self.module.qweight
        self.scales = self.module.scales
        self.qzeros = self.module.qzeros
        self.blocksize = self.module.blocksize

    def forward(self, input):
        if input.dim() > 3:
            input = input.reshape([-1, input.shape[-2], input.shape[-1]])
        if not self.acc_test:
            shape = list(input.size())
            shape[1] = 1
            input = input[:, -1, :].view(shape)
        if self.bias is None:
            return torch.ops.torch_ipex.mm_int4(
                input, self.qweight, self.scales, self.qzeros, self.blocksize
            )[:, :, : self.n_dim]
        else:
            return torch.ops.torch_ipex.mm_bias_int4(
                input,
                self.qweight,
                self.bias,
                self.scales,
                self.qzeros,
                self.blocksize,
            )[:, :, : self.n_dim]


class IPEXLmHeadLinearAllreduceWithPaddingBaichuan(
    IPEXLmHeadLinearAllreduceWithPadding
):
    def __init__(self, module):
        super().__init__(module)
        self.first_flag = True

    def port_data(self):
        self.n_dim = self.module.weight.shape[0]
        self.bias = self.module.bias if hasattr(self.module, "bias") else None
        self.weight = self.module.weight.transpose(-1, -2).contiguous()
        if dist.is_initialized():
            self.mp_group = self.module.mp_group
            self.rank = self.module.rank
            self.world_size = self.module.world_size

    def forward(self, input):
        if self.first_flag:
            self.first_flag = False
            self.weight.data = torch.nn.functional.normalize(self.weight)
        return super().forward(input)
