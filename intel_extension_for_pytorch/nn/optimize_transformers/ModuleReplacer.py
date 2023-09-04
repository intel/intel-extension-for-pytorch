from .modules.Functions import ipex_convert_to_bloom_cache, ipex_prepare_model_inputs, ipex_beam_search, ipex_beam_search_without_optimize, gptj_forward_hook, llama_forward_hook, bloom_forward_hook, opt_forward_hook, ipex_build_bloom_alibi_tensor
from typing import List
import torch
import torch.nn as nn
from .modules.Layers import IpexFastLinear, IpexFastAllReduceLinear, IpexFastLayerNorm
from .modules.gptj import IPEXGPTJConverter
from .modules.llama import IPEXLlamaConverter
from .modules.opt import IPEXOptConverter
from .modules.bloom import IPEXBloomConverter


def default_replaced_module_dict():
    import transformers
    default_replace_modules = {
        transformers.models.gptj.modeling_gptj.GPTJBlock: IPEXGPTJConverter,
        transformers.models.llama.modeling_llama.LlamaDecoderLayer: IPEXLlamaConverter,
        transformers.models.opt.modeling_opt.OPTDecoderLayer: IPEXOptConverter,
        transformers.models.bloom.modeling_bloom.BloomBlock: IPEXBloomConverter
    }
    return default_replace_modules

def linear_op(module):
    all_reduce_flag = getattr(module, "all_reduce", False)
    if all_reduce_flag:
        return IpexFastAllReduceLinear(module)
    else:
        return IpexFastLinear(module)

def default_replaced_layer_dict():
    default_replace_layers = {
        # nn.LayerNorm: IpexFastLayerNorm,
        nn.Linear: linear_op,
    }
    return default_replace_layers

def default_override_function_list() -> List:
    default_fn_list = [ipex_convert_to_bloom_cache, ipex_prepare_model_inputs, ipex_beam_search, gptj_forward_hook, llama_forward_hook, bloom_forward_hook, opt_forward_hook, ipex_build_bloom_alibi_tensor]
    return default_fn_list

class ModuleReplacer:
    def __init__(self, module_dict = None, layer_dict = None, fn_list = None) -> None:
        self.module_dict = default_replaced_module_dict()
        self.layer_dict = default_replaced_layer_dict()
        self.fn_dict = default_override_function_list()
        if module_dict is not None:
            self.module_dict.update(module_dict)
        if layer_dict is not None:
            self.layer_dict.update(layer_dict)
        if fn_list is not None:
            self.fn_dict.extend(fn_list)
        self.optimized_model = None
    
    def replace_module(self, model, dtype, config=None, prefix=""):
        if config is None and hasattr(model, "config"):
            config = model.config
            config.dtype = dtype
            config.device = "xpu"
        module_name = "" if prefix == "" else prefix + "."
        for name, child in model.named_children():
            if type(child) in self.module_dict.keys():
                module_converter = self.module_dict[type(child)](child, config, dtype=dtype, device="xpu", name=module_name + name)
                new_module = module_converter.get_transformed_module()
                # IPEXLLMResourceContrainer.push(new_module)
                setattr(model, name, new_module)
            else:
                self.replace_module(child, dtype, config, module_name + name)

    def replace_op(self, model):
        for name, child in model.named_children():
            if type(child) in self.layer_dict.keys():
                new_layer = self.layer_dict[type(child)](child)
                setattr(model, name, new_layer)
            else:
                self.replace_op(child)

    def replace_func(self, model):
        for fn in self.fn_dict:
            fn(model)
        for name, child in model.named_children():
            for fn in self.fn_dict:
                fn(child)
            self.replace_func(child)

    def update_deepspeed_supported_op(self):
        import deepspeed
        self.layer_dict.update({
            deepspeed.module_inject.layers.LinearLayer: IpexFastLinear,
            deepspeed.module_inject.layers.LinearAllreduce: IpexFastAllReduceLinear
        })
