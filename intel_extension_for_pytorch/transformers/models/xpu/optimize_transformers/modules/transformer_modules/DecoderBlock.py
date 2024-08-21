import torch
import torch.nn as nn

import sys

from .BaseAttention import IPEXTransformerAttn  # noqa
from .QuantizedAttention import *  # noqa
from .NaiveAttention import IPEXTransformerAttnNaive  # noqa
from .GroupedAttention import *  # noqa
from .Attention import *  # noqa
from .CrossedAttention import *  # noqa
from .Mlp import *  # noqa
from .QuantizedMlp import *  # noqa
from .model_utils import xpu_gemm_use_xetla


class IPEXTransformerBlock(nn.Module):
    def __init__(self, module, config, dtype, device, module_name) -> None:
        super().__init__()
        self.module = module
        self.config = config
        self.device = device
        self.module_name = module_name

    def build_attention_from_config(self, model_name=None, grouped=False):
        dtype = self.ipex_config.dtype
        impl = self.ipex_config.impl
        attn_type = IPEXTransformerAttn
        attn_type_str = "IPEXTransformerAttn"
        attn_list = [impl.name, dtype]
        if grouped:
            attn_list.append("Grouped")
        if model_name is not None:
            attn_list.append(model_name)
        if not xpu_gemm_use_xetla():
            attn_list.append("OneDNN")
        for elem in attn_list:
            attn_type_str = (
                attn_type_str + elem.capitalize()[0] + elem[1:]
            )  # For ChatGLM
            if hasattr(sys.modules[__name__], attn_type_str):
                attn_type = getattr(sys.modules[__name__], attn_type_str)
        return attn_type(self.ipex_config)

    def build_mlp_from_config(self, model_name=None):
        dtype = self.ipex_config.dtype
        impl = self.ipex_config.impl
        activation = self.ipex_config.ipex_act
        mlp_type = IPEXTransformerMLP
        mlp_type_str = "IPEXTransformerMLP"
        mlp_list = [impl.name, dtype, activation.name]
        if model_name is not None:
            mlp_list.append(model_name)
        if not xpu_gemm_use_xetla():
            mlp_list.append("OneDNN")
        for elem in mlp_list:
            mlp_type_str = mlp_type_str + elem.capitalize()[0] + elem[1:]
            if hasattr(sys.modules[__name__], mlp_type_str):
                mlp_type = getattr(sys.modules[__name__], mlp_type_str)
        return mlp_type(self.ipex_config)

    def contruct_module_from_config(self):
        pass

    def print_all_paramter_with_name(self):
        for name, param in self.ipex_optimized_module.named_parameters():
            name = self.module_name + "." + name
            print("module name: ", name)
            print("module param: ", param)

    def transpose_parameter(self):
        pass

    def port_attn_parameter(self):
        raise NotImplementedError

    def port_mlp_parameter(self):
        raise NotImplementedError

    def port_norm_parameter(self):
        raise NotImplementedError

    def port_module_specific_parameter(self):
        pass

    def port_all_parameters_to_new_module(self):
        self.port_attn_parameter()
        self.port_mlp_parameter()
        self.port_norm_parameter()
        self.port_module_specific_parameter()
        torch.xpu.empty_cache()
        # for debug
        # self.print_all_paramter_with_name

    def get_optimized_module(self):
        return self

    def release_resources(self):
        for module in self.ipex_optimized_module.children():
            if hasattr(module, "release_resources"):
                module.release_resources()
