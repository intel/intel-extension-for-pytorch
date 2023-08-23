import torch
import os
from ._transformers import IPEXEmptyLinearWithPadding, IPEXEmptyINT4LinearWithPadding, IPEXTransformerConverter

from functools import partial
from ._utils import ipex_beam_search, _ipex_prepare_model_inputs, ipex_beam_search_without_optimize, IPEXLLMResourceContrainer
from .gptj import IPEXGPTJForCausalLMForward
from .llama import IPEXLlamaForCausalLMForward
from .bloom import IPEXBloomForCausalLMForward
from ._inference_ops import OpConverter

def int4_gemm_padding(qdata):
    k, n = qdata.shape
    if n % 8 != 0:
        padded_n = (n + 8 - 1) // 8 * 8
        padded_qdata = torch.empty(k, padded_n, dtype=qdata.dtype, device=qdata.device)
        padded_qdata[:, :n] = qdata
        return padded_qdata
    else:
        return qdata

def int4_gemm_bias_padding(qdata):
    n = qdata.shape[0]
    if n % 16 != 0:
        padded_n = (n + 16 - 1) // 16 * 16
        padded_qdata = torch.empty(padded_n, dtype=qdata.dtype, device=qdata.device)
        padded_qdata[:n] = qdata
        return padded_qdata
    else:
        return qdata

def int4_gemm_scale_padding(scale):
    k, n = scale.shape
    if n % 4 != 0:
        padded_n = (n + 4 - 1) // 4 * 4
        padded_scale = torch.empty(k, padded_n, dtype=scale.dtype, device=scale.device)
        padded_scale[:, :n] = scale
        return padded_scale
    else:
        return scale

def gemm_padding(weight, bias=None):
    n, k = weight.shape
    if n % 4 != 0:
        padded_n = (n + 4 - 1) // 4 * 4
        padded_weight = torch.zeros(padded_n, k, dtype=weight.dtype, device=weight.device)
        padded_weight[:n, :] = weight
        if bias is not None:
            padded_bias = torch.zeros(padded_n, dtype=bias.dtype, device=bias.device)
            padded_bias[:n] = bias
        else:
            padded_bias = None
        return padded_weight, padded_bias
    else:
        return weight, bias

def pad_for_gptj_lm_head(model, is_int4=False):
    if is_int4:
        n = model.lm_head.qweight.shape[1] * 2 - 1 #specific for 50401(25201) int4 weight

        lm_head_new = IPEXEmptyINT4LinearWithPadding(n)
        lm_head_new.qweight = model.lm_head.qweight
        lm_head_new.weight = model.lm_head.weight
        lm_head_new.bias = model.lm_head.bias if model.lm_head.bias is not None else None
        lm_head_new.scales = model.lm_head.scales
        lm_head_new.qzeros = model.lm_head.qzeros
        lm_head_new.group_size = model.lm_head.group_size.data.item()
        model.lm_head = lm_head_new

        model.lm_head.qweight.data = int4_gemm_padding(model.lm_head.qweight)
        model.lm_head.scales.data = int4_gemm_scale_padding(model.lm_head.scales)
        model.lm_head.qzeros.data = int4_gemm_padding(model.lm_head.qzeros)

        if model.lm_head.bias is not None:
            model.lm_head.bias.data = int4_gemm_bias_padding(model.lm_head.bias)

    else:
        n = model.lm_head.weight.shape[0] #[n, k]

        lm_head_new = IPEXEmptyLinearWithPadding(n)
        lm_head_new.weight = model.lm_head.weight
        lm_head_new.bias = model.lm_head.bias
        model.lm_head = lm_head_new

        if model.lm_head.bias is not None:
            model.lm_head.weight.data, model.lm_head.bias.data = gemm_padding(model.lm_head.weight, model.lm_head.bias)
        else:
            model.lm_head.weight.data, _ = gemm_padding(model.lm_head.weight)

def transformer_frontend_replace(model, config = None, dtype = torch.float, is_int4=False):
    import transformers
    enable_ds = False
    try:
        import deepspeed
    except ImportError as e:
        print("Warning: we didn't find Deepspeed in your env, multi-tile optimization will be closed")
    else:
        enable_ds = True
        OpConverter.update_deepspeed_supported_ops()
        if isinstance(model, deepspeed.InferenceEngine):
            IPEXTransformerConverter.update_tp_data(model._config.tensor_parallel.tp_size, model._config.tensor_parallel.tp_group)

    from .gptj import IPEXGPTJConverter
    from .llama import IPEXLlamaConverter
    from .opt import IPEXOptConverter
    from .bloom import IPEXBloomConverter, _convert_to_bloom_cache_ipex
    transformers_converter = {
        transformers.models.gptj.modeling_gptj.GPTJBlock: IPEXGPTJConverter,
        transformers.models.llama.modeling_llama.LlamaDecoderLayer: IPEXLlamaConverter,
        transformers.models.opt.modeling_opt.OPTDecoderLayer: IPEXOptConverter,
        transformers.models.bloom.modeling_bloom.BloomBlock: IPEXBloomConverter
    }

    transformers_int4 = {
        transformers.models.gptj.modeling_gptj.GPTJBlock,
    }    

    def recursive_module_replace(module, config, dtype, enable_deepspeed=False, is_int4=is_int4):
        not_deepspeed_engine = not enable_deepspeed or not isinstance(module, deepspeed.InferenceEngine)
        if config is None and hasattr(module, "config") and not_deepspeed_engine:
            config = module.config
            config.dtype = dtype
            config.device = module.device

        if hasattr(module, "_convert_to_bloom_cache"):
            setattr(module, "_convert_to_bloom_cache", _convert_to_bloom_cache_ipex)

        if hasattr(module, "_prepare_model_inputs"):
            setattr(module, "_prepare_model_inputs", partial(_ipex_prepare_model_inputs, module))

        if type(module) == transformers.models.gptj.modeling_gptj.GPTJForCausalLM:
            pad_for_gptj_lm_head(module, is_int4)
            if hasattr(module, "forward"):
                setattr(module, "forward", partial(IPEXGPTJForCausalLMForward, module))
        elif type(module) == transformers.models.llama.modeling_llama.LlamaForCausalLM:
            pad_for_gptj_lm_head(module)
            if hasattr(module, "forward"):
                setattr(module, "forward", partial(IPEXLlamaForCausalLMForward, module))
        elif type(module) == transformers.models.bloom.modeling_bloom.BloomForCausalLM:
            pad_for_gptj_lm_head(module)
            if hasattr(module, "forward"):
                setattr(module, "forward", partial(IPEXBloomForCausalLMForward, module))


        if os.environ.get("DISABLE_KV_CACHE", "OFF") not in ["1", "Y", "YES", "TRUE", "ON"]:
            if hasattr(module, "beam_search"):
                setattr(module, "beam_search", partial(ipex_beam_search, module))

        for name, named_module in module.named_children():
            if type(named_module) in transformers_converter.keys():
                if type(named_module) in transformers_int4:
                    module_converter = transformers_converter[type(named_module)](named_module, config, dtype=dtype, device=config.device, is_int4=is_int4)
                else:
                    module_converter = transformers_converter[type(named_module)](named_module, config, dtype=dtype, device=config.device)
                module_transformed = module_converter.get_transformed_module()
                setattr(module, name, module_transformed)
                IPEXLLMResourceContrainer.push(module_transformed)
            # elif OpConverter.valid_op_for_convert(named_module):
            #     op_transformed = OpConverter.convert_op(named_module)
            #     setattr(module, name, op_transformed)
            else:
                recursive_module_replace(named_module, config, dtype=dtype, is_int4=is_int4)
        return module

    replaced_model = recursive_module_replace(model, None, dtype=dtype, enable_deepspeed=enable_ds, is_int4=is_int4)

    return replaced_model
