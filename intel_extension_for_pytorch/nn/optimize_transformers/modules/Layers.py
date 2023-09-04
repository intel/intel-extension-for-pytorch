import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.distributed as dist


class EnvParam:

    @classmethod
    def collect_all_env_param(cls):
        flag_list = {"COL_MAJOR": "col_major"}
        number_list = {"TP_SIZE": "tp_size"}
        for flag, flag_name in flag_list.items():
            flag = os.environ.get(flag, "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]
            setattr(EnvParam, flag_name, flag)
        for num, name in number_list.items():
            num = os.environ.get(num, 0)
            setattr(EnvParam, name, num)
    
    @classmethod
    def set_env(name, value):
        setattr(EnvParam, name, value)

EnvParam.collect_all_env_param()

class IPEXOpForInference(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.row_major = EnvParam.col_major

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
        out = torch.ops.torch_ipex.fast_layer_norm(input_tensor, self.normalized_shape, self.weight, self.bias, self.eps)
        return out

class IpexFastLinear(IPEXOpForInference):
    def __init__(self, module):
        super().__init__(module)
        self.weight = None
        self.bias = None
        self.port_data()
    
    def port_data(self):
        if self.row_major:
            self.weight = self.module.weight.transpose(0, 1).contiguous()
            self.module.weight.data = self.weight.data
        else:
            self.weight = self.module.weight
        self.bias = self.module.bias
    
    def forward(self, input_tensor):
        if self.row_major:
            shape = [input_tensor.shape[0], input_tensor.shape[1], self.weight.shape[1]]
            if self.bias is not None:
                return torch.addmm(self.bias, input_tensor.flatten(0, -2), self.weight).view(shape)
            else:
                return torch.matmul(input_tensor.flatten(0, -2), self.weight).view(shape)
        else:
            return F.linear(input_tensor, self.weight, self.bias)

class IpexFastAllReduceLinear(IPEXOpForInference):
    def __init__(self, module):
        super().__init__(module)
        self.weight = None
        self.bias = None
        self.mp_group = None
        self.port_data()
    
    def port_data(self):
        if self.row_major:
            self.weight = self.module.weight.transpose(0, 1).contiguous()
            self.module.weight.data = self.weight.data
        else:
            self.weight = self.module.weight
        self.bias = self.module.bias
        self.mp_group = self.module.mp_group
    
    def forward(self, input):
        if self.row_major:
            output = torch.matmul(input, self.weight)
        else:
            output = F.linear(input, self.weight)
        if self.mp_group is not None:
            dist.all_reduce(output, group=self.mp_group)
        if self.bias is not None:
            output += self.bias
        return output