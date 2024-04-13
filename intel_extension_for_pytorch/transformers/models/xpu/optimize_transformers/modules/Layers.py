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
        if dist.is_initialized():
            self.weight = self.module.weight.transpose(-1, -2).contiguous()
            self.mp_group = self.module.mp_group
            self.rank = self.module.rank
            self.world_size = self.module.world_size
        else:
            self.weight = self.module.weight

    def forward(self, input):
        if input.dim() > 3:
            input = input.reshape([-1, input.shape[-2], input.shape[-1]])
        if not self.acc_test:
            shape = list(input.size())
            shape[1] = 1
            input = input[:, -1, :].view(shape)
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
            return torch.nn.functional.linear(input, self.weight, bias=self.bias)[
                :, :, : self.n_dim
            ]


class IPEXLmHeadLinearAllreduceWithPaddingInt4(IPEXOpForInference):
    def __init__(self, module):
        super().__init__(module)
        self.qlinear = WeightOnlyQuantizedLinear(
            in_features=4096, out_features=128256
        )
        # print("self.module: ", self.module)
        # print("self.module.weight.size(): ", self.module.weight.size())
        self.n_dim = self.module.weight.shape[0]
        # print("self.n_dim: ", self.n_dim)
        ## WA since we don't have any lm_head int4 linear weight
        group_size = 32  # 128 -> 32
        group_num = int(4096 / group_size)
        scales = torch.rand([group_num, 128256], device="xpu", dtype=torch.float16)
        # zero_points = torch.rand([group_num, 512], device="xpu").byte()
        zero_points = torch.zeros(group_num, 64128, device="xpu").byte()
        weight = (torch.rand([4096, 64128], device="xpu") * 10 * 16).byte()
        bias = None
        self.qlinear.set_weights_bias(weight, bias)
        self.qlinear.set_scales_zps_gidx(scales, zero_points)
        self.qlinear.blocksize = 32
        self.qlinear.qweight.data = self.qlinear.qweight.transpose(
            0, 1
        ).contiguous()
        try_linear_out_reorder_input = torch.empty(
                1, 1, 4096, dtype=torch.float16, device="xpu"
        )
        try_linear_out_reorder = torch.ops.torch_ipex.mm_esimd_int4(
            try_linear_out_reorder_input,
            self.qlinear.qweight,
            self.qlinear.scales,
            self.qlinear.qzeros,
            self.qlinear.blocksize,
            True,
        )

    def port_data(self):
        # print("self.module.qweight.size(): ", self.module.qweight.size())
        self.n_dim = self.module.weight.shape[0]
        self.bias = self.module.bias
        if dist.is_initialized():
            self.weight = self.module.weight.transpose(-1, -2).contiguous()
            self.mp_group = self.module.mp_group
            self.rank = self.module.rank
            self.world_size = self.module.world_size
        else:
            self.weight = self.module.weight

    def forward(self, input):
        if input.dim() > 3:
            input = input.reshape([-1, input.shape[-2], input.shape[-1]])
        if not self.acc_test:
            shape = list(input.size())
            shape[1] = 1
            input = input[:, -1, :].view(shape)
        # if dist.is_initialized():
        #     assert (
        #         input.shape[-1] % self.world_size == 0
        #     ), "Please ensure that self.world_size is divisible by input.shape[-1]"
        #     input_shard = input.shape[-1] // self.world_size
        #     output = torch.matmul(
        #         input[:, :, self.rank * input_shard : (self.rank + 1) * input_shard],
        #         self.weight,
        #     )
        #     if self.mp_group is not None:
        #         dist.all_reduce(output, group=self.mp_group)
        #     if self.bias is not None:
        #         output += self.bias
        #     output = output[:, :, : self.n_dim]
        #     return output
        # else:
        #     return torch.nn.functional.linear(input, self.weight, bias=self.bias)[
        #         :, :, : self.n_dim
        #     ]
        return torch.ops.torch_ipex.mm_esimd_int4(
            input,
            self.qlinear.qweight,
            self.qlinear.scales,
            self.qlinear.qzeros,
            self.qlinear.blocksize,
            False,
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
        if dist.is_initialized():
            self.weight = self.module.weight.transpose(-1, -2).contiguous()
            self.mp_group = self.module.mp_group
            self.rank = self.module.rank
            self.world_size = self.module.world_size
        else:
            self.weight = self.module.weight

    def forward(self, input):
        if self.first_flag:
            self.first_flag = False
            self.weight.data = torch.nn.functional.normalize(self.weight)
        return super().forward(input)
