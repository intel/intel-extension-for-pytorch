from .modules.Functions import (
    ipex_convert_to_bloom_cache,
    ipex_prepare_model_inputs,
    ipex_beam_search,
    gptj_forward_hook,
    llama_forward_hook,
    bloom_forward_hook,
    opt_forward_hook,
    ipex_build_bloom_alibi_tensor,
)
from .modules.utils import is_int4
from typing import List
import torch
from .modules._transformer_configuration import ImplementMode
from .modules.Layers import (
    IpexFastLinear,
    IpexFastAllReduceLinear,
    IPEXLmHeadLinearAllreduceWithPadding,
)
from .modules.gptj import NewIPEXGPTJBlock
from .modules.bloom import NewIPEXBloomBlock
from .modules.llama import NewIPEXLLAMABlock
from .modules.opt import NewIPEXOPTBlock
import os


def default_replaced_module_dict():
    import transformers

    default_replace_modules = {
        transformers.models.gptj.modeling_gptj.GPTJBlock: NewIPEXGPTJBlock,
        transformers.models.llama.modeling_llama.LlamaDecoderLayer: NewIPEXLLAMABlock,
        transformers.models.opt.modeling_opt.OPTDecoderLayer: NewIPEXOPTBlock,
        transformers.models.bloom.modeling_bloom.BloomBlock: NewIPEXBloomBlock,
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
        # torch.nn.LayerNorm: IpexFastLayerNorm,
        torch.nn.Linear: linear_op,
    }
    return default_replace_layers


def default_override_function_list() -> List:
    default_fn_list = [
        ipex_convert_to_bloom_cache,
        ipex_prepare_model_inputs,
        ipex_beam_search,
        gptj_forward_hook,
        llama_forward_hook,
        bloom_forward_hook,
        opt_forward_hook,
        ipex_build_bloom_alibi_tensor,
    ]
    return default_fn_list


class ModuleReplacer:
    def __init__(
        self, module_dict=None, layer_dict=None, fn_list=None, tp_size=1, tp_group=None
    ) -> None:
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
        self.tp_size = tp_size
        self.tp_group = tp_group

    def replace_module(self, model, dtype, config=None, prefix=""):
        if config is None and hasattr(model, "config"):
            config = model.config
            config.dtype = dtype
            config.device = "xpu"
        module_name = "" if prefix == "" else prefix + "."
        is_replace_success = False
        enable_naive_path = os.environ.get("ENABLE_NAIVE_PATH", "OFF").upper() in [
            "1",
            "Y",
            "ON",
            "YES",
            "TRUE",
        ]
        impl_mode = (
            ImplementMode.naive if enable_naive_path else ImplementMode.optimized
        )
        for name, child in model.named_children():
            if type(child) in self.module_dict.keys():
                new_module = self.module_dict[type(child)](
                    child,
                    config,
                    dtype=dtype,
                    device="xpu",
                    module_name=module_name + name,
                    impl_mode=impl_mode,
                    tp_size=self.tp_size,
                    tp_group=self.tp_group,
                )
                if new_module is not None:
                    # IPEXLLMResourceContrainer.push(new_module)
                    setattr(model, name, new_module)
                    is_replace_success = True
            else:
                is_replace_success = is_replace_success or self.replace_module(
                    child, dtype, config, module_name + name
                )
        return is_replace_success

    def replace_op(self, model):
        for name, child in model.named_children():
            if type(child) in self.module_dict.keys():
                continue
            if name == "lm_head" and (not is_int4(model)):
                setattr(model, name, IPEXLmHeadLinearAllreduceWithPadding(child))
            elif type(child) in self.layer_dict.keys():
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

        self.layer_dict.update(
            {
                deepspeed.module_inject.layers.LinearLayer: IpexFastLinear,
                deepspeed.module_inject.layers.LinearAllreduce: IpexFastAllReduceLinear,
                deepspeed.module_inject.layers.LmHeadLinearAllreduce: IPEXLmHeadLinearAllreduceWithPadding,
            }
        )
