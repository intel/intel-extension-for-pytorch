import torch
import torch.nn as nn
import copy
import re
import warnings
import pkg_resources
from intel_extension_for_pytorch.cpu._auto_kernel_selection import (
    _enable_tpp,
    _disable_tpp,
)
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
            setattr(sub_m, "forward", bound_method)
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


def _optimize_transformers(
    model,
    dtype=torch.float,
    inplace=False,
    device="cpu",
):
    r"""
    Apply optimizations at Python frontend to the given transformers model (nn.Module) for inference only.
    This API focus on transformers models, especially for generation tasks inference.
    Well supported model list: Llama, GPT-J, GPT-Neox.

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
            max_version = "4.30.0"
            if "transformers" not in installed_pkg:
                raise RuntimeError(
                    "optimize_transformers optimization requires transformers package and its version between {} and {}, fallback due to not meet".format(
                        min_version, max_version
                    )
                )

            import transformers
            from packaging import version

            trans_version = transformers.__version__
            if version.parse(trans_version) < version.parse(
                min_version
            ) or version.parse(trans_version) > version.parse(max_version):
                raise RuntimeError(
                    "optimize_transformers optimization requires the transformers with version: between {} and {} while now transformers== {}, fallback due to not meet".format(
                        min_version, max_version, trans_version
                    )
                )

            from .generation import (
                _beam_search,
                _greedy_search,
                _extract_past_from_model_output,
            )
            from .attentions import (
                _prepare_decoder_attention_mask,
                _LlamaAttention,
                _LlamaAttention_GQA,
                _GPTJAttention,
                _GPTNeoXAttention,
                _reorder_cache,
            )
            from intel_extension_for_pytorch.cpu.tpp.fused_llm import (
                GPTJBlock_forward,
                GPTJMLP_forward,
                GPTJMLP_forward_distributed,
                GPTNeoXMLP_forward,
                GPTNeoXLayer_forward,
                LlamaMLP_forward,
                LlamaDecoderLayer_forward,
            )
            from .models import (
                GPTJModel_forward,
                LlamaModel_forward,
                GPTNeoXModel_forward,
                GPTJForCausalLM_forward,
                LlamaForCausalLM_forward,
                GPTNeoXForCausalLM_forward,
            )

            well_supported_model = (
                re.search("GPTJ", model.config.architectures[0], re.IGNORECASE)
                or re.search("llama", model.config.architectures[0], re.IGNORECASE)
                or re.search("gptneox", model.config.architectures[0], re.IGNORECASE)
            )
            if not well_supported_model:
                warnings.warn(
                    "optimize_transformers currently well supports Llama, GPT-J, GPT-Neox"
                )

            if not inplace:
                _model = copy.deepcopy(model)
            else:
                _model = model

            if dtype == torch.bfloat16 or dtype == torch.float or dtype == torch.int8:
                # generation-wise optimizations
                convert_function(_model, "_reorder_cache", _reorder_cache)
                convert_function(_model, "beam_search", _beam_search)
                convert_function(_model, "greedy_search", _greedy_search)
                convert_function(
                    _model,
                    "_extract_past_from_model_output",
                    _extract_past_from_model_output,
                )

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
                elif re.search("gptneox", model.config.architectures[0], re.IGNORECASE):
                    convert_function(
                        _model,
                        "forward",
                        GPTNeoXForCausalLM_forward,
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
                convert_class(
                    _model,
                    transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXAttention,
                    _GPTNeoXAttention,
                    _model.config,
                )
                if hasattr(_model.config, "num_attention_kv_heads"):
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

                if dtype == torch.int8:
                    convert_functions(
                        _model,
                        transformers.models.llama.modeling_llama.LlamaModel,
                        "_prepare_decoder_attention_mask",
                        _prepare_decoder_attention_mask,
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
                    else:
                        convert_forward(
                            _model,
                            transformers.models.gptj.modeling_gptj.GPTJMLP,
                            GPTJMLP_forward_distributed,
                        )
            else:
                raise RuntimeError(
                    "optimize_transformers optimization currently supports dtype: torch.float, torch.bfloat16, torch.int8, will cover more soon."
                )

            return _model

        except RuntimeError:
            return model

    return model
