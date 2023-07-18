import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class IPEXOpForInference(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        col_major = os.environ.get("COL_MAJOR", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]
        self.row_major = not col_major

    def port_data(self):
        raise NotImplementedError



class IpexFastLayerNorm(IPEXOpForInference):
    def __init__(self, module):
        super().__init__(module)
        self.weight = None
        self.bias = None
        self.eps = None
        self.normalize_shape = None

    def port_data(self):
        self.weight = self.module.weight
        self.bias = self.module.bias
        self.eps = self.module.eps
        self.normalize_shape = self.module.normalized_shape

    def forward(self, input_tensor):
        out, _, _ = torch.ops.torch_ipex.fast_layer_norm(input_tensor, self.normalize_shape, self.weight, self.bias, self.eps)
        return out

class IpexFastLinear(IPEXOpForInference):
    def __init__(self, module):
        super().__init__(module)
        self.weight = None
        self.bias = None
    
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
    
    def port_data(self):
        if self.row_major:
            self.weight = self.module.weight.transpose(0, 1).contiguous()
            self.module.weight.data = self.weight.data
        else:
            self.weight = self.module.weight
        self.bias = self.module.bias
        self.mp_group = self.module.mp_group
    
    def forward(self, input):
        from deepspeed import comm as dist
        if self.row_major:
            output = torch.matmul(input, self.weight)
        else:
            output = F.linear(input, self.weight)
        if self.mp_group is not None:
            dist.all_reduce(output, group=self.mp_group)
        if self.bias is not None:
            output += self.bias
        return output

class OpConverter:
    supported_ops = {
        nn.LayerNorm: IpexFastLayerNorm,
        nn.Linear: IpexFastLinear,
    }
    def __init__(self):
        pass

    @staticmethod
    def valid_op_for_convert(module):
        return type(module) in OpConverter.supported_ops.keys()
    
    @staticmethod
    def update_custom_policy(policy):
        OpConverter.supported_ops.update(policy)
        OpConverter.supported_ops.update(policy)

    @staticmethod
    def update_deepspeed_supported_ops():
        try:
            import deepspeed
        except ImportError:
            print("Can't find deepspeed in your environment, some related replace policy will automatically disable")
            return
        deepspeed_replace_dict = {
            deepspeed.module_inject.layers.LinearLayer: IpexFastLinear,
            deepspeed.module_inject.layers.LinearAllreduce: IpexFastAllReduceLinear
        }
        OpConverter.supported_ops.update(deepspeed_replace_dict)
        return

    @staticmethod
    def convert_op(module):
        converted_op = OpConverter.supported_ops[type(module)](module)
        converted_op.port_data()
        return converted_op

