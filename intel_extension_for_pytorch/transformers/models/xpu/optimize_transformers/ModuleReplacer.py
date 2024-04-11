from .modules.Functions import (
    ipex_convert_to_bloom_cache,
    ipex_prepare_model_inputs,
    ipex_beam_search,
    ipex_beam_sample,
    gptj_forward_hook,
    llama_forward_hook,
    bloom_forward_hook,
    opt_forward_hook,
    falcon_forward_hook,
    baichuan_forward_hook,
    qwen_forward_hook,
    chatglm_forward_hook,
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
    IPEXLmHeadLinearAllreduceWithPaddingBaichuan,
)
from .modules.gptj import NewIPEXGPTJBlock
from .modules.bloom import NewIPEXBloomBlock
from .modules.llama import NewIPEXLLAMABlock
from .modules.opt import NewIPEXOPTBlock
from .modules.falcon import NewIPEXFalconBlock
from .modules.qwen import NewIPEXQWENBlock
from .modules.chatglm import (
    NewIPEXCHATGLMBlock,
    prepare_inputs_for_generation,
    NewIPEXRotaryEmbedding,
)
from .modules.DiffusersTransformer import NewIPEXBasicTransformerBlock
from .modules.bert import NewIPEXBertSelfAttention

import os


def default_replaced_module_dict():
    import transformers
    from diffusers.models.attention import BasicTransformerBlock

    default_replace_modules = {
        transformers.models.gptj.modeling_gptj.GPTJBlock: NewIPEXGPTJBlock,
        transformers.models.llama.modeling_llama.LlamaDecoderLayer: NewIPEXLLAMABlock,
        transformers.models.opt.modeling_opt.OPTDecoderLayer: NewIPEXOPTBlock,
        transformers.models.bloom.modeling_bloom.BloomBlock: NewIPEXBloomBlock,
        # only support transformers version model, not in-library model
        transformers.models.falcon.modeling_falcon.FalconDecoderLayer: NewIPEXFalconBlock,
        transformers.models.bert.modeling_bert.BertSelfAttention: NewIPEXBertSelfAttention,
        BasicTransformerBlock: NewIPEXBasicTransformerBlock,
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
        ipex_beam_sample,
        gptj_forward_hook,
        llama_forward_hook,
        bloom_forward_hook,
        opt_forward_hook,
        falcon_forward_hook,
        baichuan_forward_hook,
        qwen_forward_hook,
        chatglm_forward_hook,
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
            if child.__class__.__name__ == "Transformer2DModel":
                config = child.config
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
            # BaichuanLayer is a custom model in transformers
            elif child.__class__.__name__ == "BaichuanLayer":
                from .modules.baichuan import NewIPEXBaichuanBlock

                new_module = NewIPEXBaichuanBlock(
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
                    setattr(model, name, new_module)
                    is_replace_success = True
            # QWenBlock is a customized model in transformers
            elif child.__class__.__name__ == "QWenBlock":
                new_module = NewIPEXQWENBlock(
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
                    setattr(model, name, new_module)
                    is_replace_success = True
            # GLMBlock is a customized model in transformers
            elif child.__class__.__name__ == "GLMBlock":
                new_module = NewIPEXCHATGLMBlock(
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
                    setattr(model, name, new_module)
                    is_replace_success = True
            # Replace RotaryEmbedding in ChatGLMModel
            elif (
                model.__class__.__name__ == "ChatGLMModel"
                and child.__class__.__name__ == "RotaryEmbedding"
            ):
                new_module = NewIPEXRotaryEmbedding(
                    child,
                    config,
                    device="xpu",
                )
                if new_module is not None:
                    setattr(model, name, new_module)
            else:
                is_replace_success = (
                    self.replace_module(child, dtype, config, module_name + name)
                    or is_replace_success
                )
        return is_replace_success

    def replace_op(self, model):
        if model.__class__.__name__ == "ChatGLMForConditionalGeneration":
            model.prepare_inputs_for_generation = prepare_inputs_for_generation.__get__(
                model, model.__class__
            )
        for name, child in model.named_children():
            if (
                type(child) in self.module_dict.keys()
                or child.__class__.__name__ == "BaichuanLayer"
                or child.__class__.__name__ == "QWenBlock"
                or child.__class__.__name__ == "BertPreTrainingHeads"
                or child.__class__.__name__ == "BertModel"
                or child.__class__.__name__ == "GLMBlock"
            ):
                continue
            if name == "lm_head" and (not is_int4(model)):
                if model.__class__.__name__ == "BaichuanForCausalLM":
                    setattr(
                        model, name, IPEXLmHeadLinearAllreduceWithPaddingBaichuan(child)
                    )
                else:
                    setattr(model, name, IPEXLmHeadLinearAllreduceWithPadding(child))
            elif name == "ChatGLMModel" and (not is_int4(model)):
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
