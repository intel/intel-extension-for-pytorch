import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Tuple, Union

from .Activation import ACT2FN
from .._transformer_configuration import IPEXTransformerConfig
import os
import math
import dataclasses
from .Linear import IPEXTransformerLinear, IPEXTransformerQLinear, matmul_add_add

class IPEXTransformerBaseMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.tp_size = config.tp_size
        self.tp_group = config.tp_group
        self.module = None

    @staticmethod
    def release_resources():
        pass

    def load_parameter(self, module):
        self.module = module

    def transpose_parameter(self, module):
        raise NotImplementedError

    def all_reduce_if_necessary(self, target):
        if self.tp_group is not None:
            dist.all_reduce(target, group=self.tp_group)
        return target

    def inter_mm(self, hidden_states):
        raise NotImplementedError

    def out_mm(self, hidden_states, residual=None):
        raise NotImplementedError

    def forward(self, hidden_states, residual=None):
        inter_output = self.inter_mm(hidden_states)
        output = self.out_mm(inter_output, residual)
        return output


class IPEXTransformerMLP(IPEXTransformerBaseMLP):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.fc_in = IPEXTransformerLinear()
        self.fc_out = IPEXTransformerLinear()
        self.act = ACT2FN[config.activation_function]


    def load_parameter(self, fc_in, fc_out):
        self.origin_fc_in = fc_in
        self.origin_fc_out = fc_out

        self.fc_in.weight  = fc_in.weight
        self.fc_out.weight = fc_out.weight

        self.fc_in.bias  = fc_in.bias
        self.fc_out.bias = fc_out.bias


    def transpose_parameter(self):
        self.fc_in.weight.data  = self.fc_in.weight.transpose(0, 1).contiguous()
        self.fc_out.weight.data = self.fc_out.weight.transpose(0, 1).contiguous()

        self.origin_fc_in.weight.data  = self.fc_in.weight
        self.origin_fc_out.weight.data = self.fc_out.weight

    def inter_mm(self, hidden_states):
        hidden_states = self.fc_in(hidden_states)
        return self.act(hidden_states)

    def out_mm(self, hidden_states, residual=None):
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.all_reduce_if_necessary(hidden_states)
        if self.fc_out.bias:
            hidden_states += self.fc_out.bias
        if residual:
            hidden_states += residual
        return hidden_states

class IPEXTransformerMLPOptimizedFp16(IPEXTransformerMLP):
    def __init__(self, config) -> None:
        super().__init__(config)

    def inter_mm(self, hidden_states):
        hidden_states = torch.ops.torch_ipex.matmul_bias_out(hidden_states, self.fc_in.weight, self.fc_in.bias)
        return self.act(hidden_states)

    def out_mm(self, hidden_states, residual=None):
        hidden_states = matmul_add_add(hidden_states, self.fc_out.weight, self.tp_size, self.fc_out.bias, residual)
        hidden_states = self.all_reduce_if_necessary(hidden_states)
        return hidden_states

class IPEXTransformerMLPOptimizedFp16Gelu(IPEXTransformerMLPOptimizedFp16):
    def __init__(self, config) -> None:
        super().__init__(config)

    def inter_mm(self, hidden_states):
        hidden_states = torch.ops.torch_ipex.matmul_gelu(hidden_states, self.fc_in.weight, self.fc_in.bias, 1.0, self.act.approximate)
        return hidden_states

class IPEXTransformerMLPOptimizedFp16GeluGptj(IPEXTransformerMLPOptimizedFp16Gelu):
    def __init__(self, config) -> None:
        super().__init__(config)
    
    def forward(self, hidden_states, attn_output, residual):
        inter_out = self.inter_mm(hidden_states)
        out = self.out_mm(inter_out, attn_output, residual)
        return out
    
    def out_mm(self, inter_out, attn_out, residual):
        hidden_states = torch.ops.torch_ipex.mm_bias_resadd_resadd(inter_out, self.fc_out.weight, self.fc_out.bias, 1.0/self.tp_size, attn_out, 1.0/self.tp_size, residual, 1.0/self.tp_size)
        hidden_states = self.all_reduce_if_necessary(hidden_states)
        return hidden_states

class IPEXTransformerMLPOptimizedInt4(IPEXTransformerMLP):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.qfc_in = IPEXTransformerQLinear()
        self.qfc_out = IPEXTransformerQLinear()

    # TODO(ganyi): int4 load data
    def load_parameter(self, fc_in, fc_out, qfc_in, qfc_out):
        super().load_parameter(fc_in, fc_out)
        self.qfc_in.weight  = qfc_in.qweight
        self.qfc_out.weight = qfc_out.qweight
        
        self.qfc_in.scale = qfc_in.scales
        self.qfc_out.scale= qfc_out.scales
        
        self.qfc_in.zp = qfc_in.qzeros
        self.qfc_out.zp = qfc_out.qzeros

        self.qfc_in.gs = qfc_in.group_size.data.item()
        self.qfc_out.gs = qfc_out.group_size.data.item()

    def inter_mm(self, hidden_states):
        if hidden_states.shape[0] == 1:
            hidden_states = torch.ops.torch_ipex.mm_bias_int4(hidden_states, self.qfc_in.weight, self.qfc_in.scale, self.qfc_in.zp, self.qfc_in.bias)
        else:
            hidden_states = torch.ops.torch_ipex.matmul_bias_out(hidden_states, self.fc_in.weight, self.fc_in.bias)
        return self.act(hidden_states)

    def out_mm(self, hidden_states, attn_output=None, residual=None):
        if hidden_states.shape[0] == 1:
            hidden_states = torch.ops.torch_ipex.mm_bias_resadd_resadd_int4(hidden_states, self.qfc_out.weight, self.qfc_out.bias, attn_output, residual, self.qfc_out.scale, self.qfc_out.zp, self.qfc_out.gs)
        else:
            hidden_states = torch.ops.torch_ipex.mm_bias_resadd_resadd(hidden_states, self.fc_out.weight, self.fc_out.bias, attn_output, residual)

        hidden_states = self.all_reduce_if_necessary(hidden_states)
        return hidden_states


class IPEXTransformerMLPOptimizedInt4Gelu(IPEXTransformerMLPOptimizedInt4):
    def __init__(self, config) -> None:
        super().__init__(config)

    def inter_mm(self, hidden_states):
        if hidden_states.shape[0] == 1:
            hidden_states = torch.ops.torch_ipex.mm_bias_gelu_int4(hidden_states, self.qfc_in.weight, self.qfc_in.scale, self.qfc_in.zp, self.qfc_in.bias)
        else:
            hidden_states = torch.ops.torch_ipex.matmul_gelu(hidden_states, self.fc_in.weight, self.fc_in.bias)
        return hidden_states

class IPEXTransformerMLPOptimizedFp16Silu(IPEXTransformerMLPOptimizedFp16):
    def __init__(self, config):
        super().__init__(config)


    def inter_mm(self, hidden_states):
        if self.fc_in.bias is None:
            hidden_states = torch.ops.torch_ipex.mm_silu(hidden_states, self.fc_in.weight)
        else:
            hidden_states = self.fc_in(hidden_states)
        return hidden_states

class IPEXTransformerMLPOptimizedFp16SiluLlama(IPEXTransformerMLPOptimizedFp16Silu):
    def __init__(self, config):
        super().__init__(config)
        self.up_proj = IPEXTransformerLinear()

    def load_parameter(self, fc_in, fc_out, up_proj):
        super().load_parameter(fc_in, fc_out)
        self.origin_up_proj = up_proj
        self.up_proj.weight = up_proj.weight
        self.up_proj.bias   = up_proj.bias

    def transpose_parameter(self):
        super().transpose_parameter()
        self.up_proj.weight.data = self.up_proj.weight.transpose(0, 1).contiguous()
        self.origin_up_proj.data = self.up_proj.weight.data

    def inter_mm(self, hidden_states):
        hidden_states1 = torch.ops.torch_ipex.mm_silu(hidden_states, self.fc_in.weight)
        hidden_states = torch.ops.torch_ipex.mm_resmul(hidden_states, self.up_proj.weight, hidden_states1)
        return hidden_states
