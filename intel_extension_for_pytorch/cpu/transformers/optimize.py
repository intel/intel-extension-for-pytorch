import torch
import copy
import re
import warnings
import pkg_resources
from intel_extension_for_pytorch.cpu._auto_kernel_selection import _enable_tpp
import intel_extension_for_pytorch
from intel_extension_for_pytorch.frontend import optimize


def convert_class(m, target_m, new_class, config):
    for name, sub_m in m.named_children():
        if isinstance(sub_m, target_m):
            new_m = new_class(sub_m, config)
            setattr(m, name, new_m)
        convert_class(sub_m, target_m, new_class, config)


def convert_forward(m, target_m, new_forward):
    for _, sub_m in m.named_children():
        if isinstance(sub_m, target_m):
            bound_method = new_forward.__get__(sub_m, sub_m.__class__)
            sub_m.forward = bound_method
        convert_forward(sub_m, target_m, new_forward)


def convert_functions(m, target_m, new_function_name, new_function):
    for _, sub_m in m.named_children():
        if isinstance(sub_m, target_m):
            bound_method = new_function.__get__(sub_m, sub_m.__class__)
            setattr(sub_m, new_function_name, bound_method)
        convert_functions(sub_m, target_m, new_function_name, new_function)


def convert_function(m, func_name, new_function):
    bound_method = new_function.__get__(m, m.__class__)
    setattr(m, func_name, bound_method)


distributed = False


def is_distributed(m):
    for _, sub_m in m.named_children():
        if isinstance(
            sub_m,
            intel_extension_for_pytorch.nn.utils._weight_prepack._IPEXLinearAllreduce,
        ):
            global distributed
            distributed = True
            return
        is_distributed(sub_m)


def _set_optimized_model_for_generation(
    model,
    optimized_model,
    first_token_optimized_model=None,
):
    if first_token_optimized_model is not None:
        model.trace_graph_first = first_token_optimized_model

    model.trace_graph = optimized_model


def _optimize_transformers(
    model,
    dtype=torch.float,
    inplace=False,
    device="cpu",
):
    r"""
    Apply optimizations at Python frontend to the given transformers model (nn.Module) for inference only.
    This API focus on transformers models, especially for generation tasks inference.
    Well supported model list: Llama, GPT-J, GPT-Neox, OPT, Falcon, Bloom, ChatGLM2, CodeGen.

    Args:
        model (torch.nn.Module): User model to apply optimizations on.
        dtype (torch.dtype): Only works for ``torch.bfloat16`` and ``torch.int8`` and ``torch.float``.
        inplace (bool): Whether to perform inplace optimization. Default value is ``False``.
        device (str): Perform optimization on which device. Curentlty only support cpu. Default value is ``cpu``.

    Returns:
        optimized Model

    .. warning::
        Please invoke ``_optimize_transformers`` function AFTER invoking DeepSpeed in Tensor Parallel
        inference scenario.

    Examples:

        >>> # bfloat16 inference case.
        >>> model = ...
        >>> model.load_state_dict(torch.load(PATH))
        >>> model.eval()
        >>> optimized_model = ipex._optimize_transformers(model, dtype=torch.bfloat16)
        >>> # running evaluation step.


    """
    if isinstance(model, torch.jit.ScriptModule):
        return model
    if model.training:
        return model

    if device == "cpu":
        try:
            # tpp rope optimization has transformers version requirements
            installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
            min_version = "4.28.0"
            if "transformers" not in installed_pkg:
                raise RuntimeError(
                    "optimize_transformers optimization requires transformers package and its version at least {} , \
                        fallback due to not meet".format(
                        min_version
                    )
                )

            import transformers
            from packaging import version

            trans_version = transformers.__version__
            if version.parse(trans_version) < version.parse(min_version):
                raise RuntimeError(
                    "optimize_transformers optimization requires the transformers with version: \
                        at least {} while now transformers== {}, fallback due to not meet".format(
                        min_version, trans_version
                    )
                )

            from .generation import (
                _beam_search,
                _greedy_search,
                _extract_past_from_model_output,
            )
            from .attentions import (
                _prepare_decoder_attention_mask,
                _prepare_attn_mask_falcon,
                _LlamaAttention,
                _LlamaAttention_GQA,
                _LlamaRMSNorm_forward_v1,
                _LlamaRMSNorm_forward_v2,
                _GPTJAttention,
                _GPTNeoXAttention,
                _OPTAttention,
                _FalconAttention,
                _BloomAttention,
                _GLM2Attention,
                _CodeGenAttention,
                _reorder_cache,
                GLM2_get_masks,
            )
            from intel_extension_for_pytorch.cpu.tpp.fused_llm import (
                GPTJBlock_forward,
                GPTJMLP_forward,
                GPTJMLP_forward_distributed,
                LlamaMLP_forward,
                LlamaMLP_forward_distributed,
                LlamaDecoderLayer_forward,
                OPTDecoderLayer_forward,
                OPTDecoderLayer_forward_distributed,
                FalconMLP_forward,
                FalconDecoderLayer_forward,
                FalconMLP_forward_distributed,
                FalconDecoderLayer_forward_distributed,
                BloomMLP_forward,
                BloomMLP_forward_distributed,
                GLMMLP_forward,
                GLMBlock_forward,
            )
            from intel_extension_for_pytorch.cpu.woq.fused_llm import (
                GPTJMLP_woq_forward,
                GPTJBlock_woq_forward,
            )
            from .models import (
                GPTJModel_forward,
                LlamaModel_forward,
                GPTNeoXModel_forward,
                OPTDecoder_forward,
                FalconModel_forward,
                BloomModel_forward,
                CodeGenModel_forward,
                GPTJForCausalLM_forward,
                LlamaForCausalLM_forward,
                GPTNeoXForCausalLM_forward,
                OPTForCausalLM_forward,
                CodeGenForCausalLM_forward,
                prepare_inputs_for_generation,
            )

            well_supported_model = (
                re.search("GPTJ", model.config.architectures[0], re.IGNORECASE)
                or re.search("llama", model.config.architectures[0], re.IGNORECASE)
                or re.search("gptneox", model.config.architectures[0], re.IGNORECASE)
                or re.search("OPT", model.config.architectures[0], re.IGNORECASE)
                or re.search("falcon", model.config.architectures[0], re.IGNORECASE)
                or re.search("rw", model.config.architectures[0], re.IGNORECASE)
                or re.search("bloom", model.config.architectures[0], re.IGNORECASE)
                or re.search("chatglm", model.config.architectures[0], re.IGNORECASE)
                or re.search("codegen", model.config.architectures[0], re.IGNORECASE)
            )
            if not well_supported_model:
                warnings.warn(
                    "optimize_transformers currently well supports Llama, GPT-J, GPT-Neox, OPT, falcon, bloom, ChatGLM2, CodeGen"
                )

            if not inplace:
                _model = copy.deepcopy(model)
            else:
                _model = model

            if dtype == torch.bfloat16 or dtype == torch.float or dtype == torch.int8:
                # generation-wise optimizations

                convert_function(_model, "beam_search", _beam_search)
                convert_function(_model, "greedy_search", _greedy_search)
                convert_function(
                    _model,
                    "_extract_past_from_model_output",
                    _extract_past_from_model_output,
                )
                if well_supported_model:
                    convert_function(_model, "_reorder_cache", _reorder_cache)
                # model-wise optimizations
                if re.search("GPTJ", model.config.architectures[0], re.IGNORECASE):
                    convert_function(
                        _model,
                        "forward",
                        GPTJForCausalLM_forward,
                    )
                elif re.search("llama", model.config.architectures[0], re.IGNORECASE):
                    convert_function(
                        _model,
                        "forward",
                        LlamaForCausalLM_forward,
                    )
                    if dtype != torch.int8:
                        convert_forward(
                            _model,
                            transformers.models.llama.modeling_llama.LlamaRMSNorm,
                            _LlamaRMSNorm_forward_v1,
                        )
                elif re.search("gptneox", model.config.architectures[0], re.IGNORECASE):
                    convert_function(
                        _model,
                        "forward",
                        GPTNeoXForCausalLM_forward,
                    )
                elif re.search("OPT", model.config.architectures[0], re.IGNORECASE):
                    convert_function(
                        _model,
                        "forward",
                        OPTForCausalLM_forward,
                    )
                elif re.search(
                    "falcon", model.config.architectures[0], re.IGNORECASE
                ) or re.search("rw", model.config.architectures[0], re.IGNORECASE):
                    convert_class(
                        _model,
                        type(_model.transformer.h[0].self_attention),
                        _FalconAttention,
                        _model.config,
                    )
                    convert_forward(
                        _model,
                        type(_model.transformer),
                        FalconModel_forward,
                    )
                    convert_function(
                        _model,
                        "prepare_inputs_for_generation",
                        prepare_inputs_for_generation,
                    )
                    convert_function(
                        _model.transformer,
                        "_prepare_attn_mask",
                        _prepare_attn_mask_falcon,
                    )
                elif re.search("chatglm", model.config.architectures[0], re.IGNORECASE):
                    convert_class(
                        _model,
                        type(_model.transformer.encoder.layers[0].self_attention),
                        _GLM2Attention,
                        _model.config,
                    )
                    convert_function(_model.transformer, "get_masks", GLM2_get_masks)
                elif re.search("codegen", model.config.architectures[0], re.IGNORECASE):
                    convert_function(
                        _model,
                        "forward",
                        CodeGenForCausalLM_forward,
                    )

                convert_forward(
                    _model,
                    transformers.models.gptj.modeling_gptj.GPTJModel,
                    GPTJModel_forward,
                )
                convert_forward(
                    _model,
                    transformers.models.llama.modeling_llama.LlamaModel,
                    LlamaModel_forward,
                )
                convert_forward(
                    _model,
                    transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXModel,
                    GPTNeoXModel_forward,
                )
                convert_forward(
                    _model,
                    transformers.models.opt.modeling_opt.OPTDecoder,
                    OPTDecoder_forward,
                )
                convert_forward(
                    _model,
                    transformers.models.bloom.modeling_bloom.BloomModel,
                    BloomModel_forward,
                )
                convert_forward(
                    _model,
                    transformers.models.codegen.modeling_codegen.CodeGenModel,
                    CodeGenModel_forward,
                )
                convert_function(
                    transformers.models.bloom.modeling_bloom.BloomModel,
                    "_prepare_attn_mask",
                    _prepare_attn_mask_falcon,
                )
                convert_class(
                    _model,
                    transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXAttention,
                    _GPTNeoXAttention,
                    _model.config,
                )
                convert_class(
                    _model,
                    transformers.models.bloom.modeling_bloom.BloomAttention,
                    _BloomAttention,
                    _model.config,
                )
                if hasattr(_model.config, "num_key_value_heads"):
                    convert_class(
                        _model,
                        transformers.models.llama.modeling_llama.LlamaAttention,
                        _LlamaAttention_GQA,
                        _model.config,
                    )
                else:
                    convert_class(
                        _model,
                        transformers.models.llama.modeling_llama.LlamaAttention,
                        _LlamaAttention,
                        _model.config,
                    )
                convert_class(
                    _model,
                    transformers.models.gptj.modeling_gptj.GPTJAttention,
                    _GPTJAttention,
                    _model.config,
                )
                convert_class(
                    _model,
                    transformers.models.opt.modeling_opt.OPTAttention,
                    _OPTAttention,
                    _model.config,
                )
                convert_class(
                    _model,
                    transformers.models.codegen.modeling_codegen.CodeGenAttention,
                    _CodeGenAttention,
                    _model.config,
                )

                if dtype == torch.int8:
                    convert_functions(
                        _model,
                        transformers.models.llama.modeling_llama.LlamaModel,
                        "_prepare_decoder_attention_mask",
                        _prepare_decoder_attention_mask,
                    )
                    convert_forward(
                        _model,
                        transformers.models.llama.modeling_llama.LlamaRMSNorm,
                        _LlamaRMSNorm_forward_v2,
                    )
                    if getattr(_model.config, "weight_only_quantization", False):
                        convert_forward(
                            _model,
                            transformers.models.gptj.modeling_gptj.GPTJBlock,
                            GPTJBlock_woq_forward,
                        )
                        convert_forward(
                            _model,
                            transformers.models.gptj.modeling_gptj.GPTJMLP,
                            GPTJMLP_woq_forward,
                        )
                        convert_forward(
                            _model,
                            transformers.models.codegen.modeling_codegen.CodeGenBlock,
                            GPTJBlock_woq_forward,
                        )
                        convert_forward(
                            _model,
                            transformers.models.codegen.modeling_codegen.CodeGenMLP,
                            GPTJMLP_woq_forward,
                        )
                else:
                    # linear-wise optimizations
                    _enable_tpp()
                    _model = optimize(_model.eval(), dtype=dtype, inplace=True)

                    # linear-postop-wise optimizations
                    is_distributed(_model)
                    if not distributed:
                        convert_forward(
                            _model,
                            transformers.models.gptj.modeling_gptj.GPTJBlock,
                            GPTJBlock_forward,
                        )
                        convert_forward(
                            _model,
                            transformers.models.gptj.modeling_gptj.GPTJMLP,
                            GPTJMLP_forward,
                        )
                        convert_forward(
                            _model,
                            transformers.models.llama.modeling_llama.LlamaDecoderLayer,
                            LlamaDecoderLayer_forward,
                        )
                        convert_forward(
                            _model,
                            transformers.models.llama.modeling_llama.LlamaMLP,
                            LlamaMLP_forward,
                        )
                        convert_forward(
                            _model,
                            transformers.models.opt.modeling_opt.OPTDecoderLayer,
                            OPTDecoderLayer_forward,
                        )
                        convert_forward(
                            _model,
                            transformers.models.codegen.modeling_codegen.CodeGenBlock,
                            GPTJBlock_forward,
                        )
                        convert_forward(
                            _model,
                            transformers.models.codegen.modeling_codegen.CodeGenMLP,
                            GPTJMLP_forward,
                        )
                        if re.search(
                            "falcon", model.config.architectures[0], re.IGNORECASE
                        ) or re.search(
                            "rw", model.config.architectures[0], re.IGNORECASE
                        ):
                            convert_forward(
                                _model,
                                type(_model.transformer.h[0].mlp),
                                FalconMLP_forward,
                            )
                            convert_forward(
                                _model,
                                type(_model.transformer.h[0]),
                                FalconDecoderLayer_forward,
                            )
                        convert_forward(
                            _model,
                            transformers.models.bloom.modeling_bloom.BloomMLP,
                            BloomMLP_forward,
                        )
                        if re.search(
                            "chatglm", model.config.architectures[0], re.IGNORECASE
                        ):
                            convert_forward(
                                _model,
                                type(_model.transformer.encoder.layers[0].mlp),
                                GLMMLP_forward,
                            )
                            convert_forward(
                                _model,
                                type(_model.transformer.encoder.layers[0]),
                                GLMBlock_forward,
                            )
                    else:
                        convert_forward(
                            _model,
                            transformers.models.llama.modeling_llama.LlamaMLP,
                            LlamaMLP_forward_distributed,
                        )
                        convert_forward(
                            _model,
                            transformers.models.gptj.modeling_gptj.GPTJMLP,
                            GPTJMLP_forward_distributed,
                        )
                        convert_forward(
                            _model,
                            transformers.models.codegen.modeling_codegen.CodeGenMLP,
                            GPTJMLP_forward_distributed,
                        )
                        convert_forward(
                            _model,
                            transformers.models.opt.modeling_opt.OPTDecoderLayer,
                            OPTDecoderLayer_forward_distributed,
                        )
                        if re.search(
                            "falcon", model.config.architectures[0], re.IGNORECASE
                        ) or re.search(
                            "rw", model.config.architectures[0], re.IGNORECASE
                        ):
                            convert_forward(
                                _model,
                                type(_model.transformer.h[0].mlp),
                                FalconMLP_forward_distributed,
                            )
                            convert_forward(
                                _model,
                                type(_model.transformer.h[0]),
                                FalconDecoderLayer_forward_distributed,
                            )
                        convert_forward(
                            _model,
                            transformers.models.bloom.modeling_bloom.BloomMLP,
                            BloomMLP_forward_distributed,
                        )
            else:
                raise RuntimeError(
                    "optimize_transformers optimization currently supports dtype: \
                        torch.float, torch.bfloat16, torch.int8, will cover more soon."
                )

            return _model

        except RuntimeError:
            return model

    return model
