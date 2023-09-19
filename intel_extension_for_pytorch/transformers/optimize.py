import torch
import copy
import re
import warnings
import pkg_resources
from intel_extension_for_pytorch.cpu._auto_kernel_selection import (
    _enable_tpp,
    _disable_tpp,
    _using_tpp,
)
import intel_extension_for_pytorch as ipex
from torch.ao.quantization import PlaceholderObserver


def convert_functions(m, target_m, new_function_name, new_function):
    for _, sub_m in m.named_children():
        if isinstance(sub_m, target_m):
            bound_method = new_function.__get__(sub_m, sub_m.__class__)
            setattr(sub_m, new_function_name, bound_method)
        convert_functions(sub_m, target_m, new_function_name, new_function)


def convert_function(m, func_name, new_function):
    bound_method = new_function.__get__(m, m.__class__)
    setattr(m, func_name, bound_method)


def convert_class(m, target_m, new_class, config, distributed=False):
    for name, sub_m in m.named_children():
        if isinstance(sub_m, target_m):
            new_m = new_class(sub_m, config, distributed)
            setattr(m, name, new_m)
        convert_class(sub_m, target_m, new_class, config, distributed)


def lowering_class_cpu(m, target_m, new_class, config, tpp=False, woq=False):
    for name, sub_m in m.named_children():
        if isinstance(sub_m, target_m):
            new_m = new_class(sub_m, config, tpp, woq)
            setattr(m, name, new_m)
        lowering_class_cpu(sub_m, target_m, new_class, config, tpp, woq)


distributed = False


def is_distributed(m, ds_layers):
    for _, sub_m in m.named_children():
        if isinstance(
            sub_m,
            ds_layers[0],
        ) or isinstance(
            sub_m,
            ds_layers[1],
        ):
            global distributed
            distributed = True
            return
        is_distributed(sub_m, ds_layers)


def _is_woq_qconfig(qconfig):
    return isinstance(qconfig.global_qconfig.activation(), PlaceholderObserver) and \
        not qconfig.global_qconfig.activation().is_dynamic


def _set_optimized_model_for_generation(
    model,
    optimized_model,
    first_token_optimized_model=None,
):
    if first_token_optimized_model is not None:
        setattr(model, "trace_graph_first", first_token_optimized_model)  # noqa: B010

    setattr(model, "trace_graph", optimized_model)  # noqa: B010
    print(
        "ipex.optimize_transformers has set the optimized or quantization model for model.generate()"
    )
    return model


def model_convert_reference(_model):
    import transformers

    # generation wise optimization
    from .generation.utils import (
        _extract_past_from_model_output,
    )
    from .generation import (
        _beam_search,
        _greedy_search,
    )

    # model wise optimization for MHA module
    from .models.reference.modules.attentions import (
        _IPEXAttentionRef,
        _reorder_cache,
        _convert_cache_to_standard_format,
        _convert_to_rw_cache,
        _prepare_decoder_attention_mask,
        _prepare_attn_mask_falcon,
    )

    # model wise optimization for Feedforward and Decoder layer modules
    from .models.reference.modules.decoder import _IPEXDecoderLayerRef

    # generation length or model forward order
    from .models.reference.models import (
        GPTJForCausalLM_forward,
        LlamaForCausalLM_forward,
        GPTNeoXForCausalLM_forward,
        OPTForCausalLM_forward,
        prepare_inputs_for_generation,
    )

    if not hasattr(_model.config, "architectures"):
        AssertionError(
            False,
            "Cannot detect the model transformers family by model.config.architectures",
        )
    # generation-wise optimizations
    convert_function(_model, "_reorder_cache", _reorder_cache)
    convert_function(_model, "beam_search", _beam_search)
    convert_function(_model, "greedy_search", _greedy_search)
    convert_function(
        _model,
        "_extract_past_from_model_output",
        _extract_past_from_model_output,
    )
    convert_functions(
        _model,
        transformers.models.llama.modeling_llama.LlamaModel,
        "_prepare_decoder_attention_mask",
        _prepare_decoder_attention_mask,
    )

    # model-wise changes for adoption
    # forward order
    if (
        hasattr(_model, "__class__")
        and _model.__class__ == transformers.models.gptj.modeling_gptj.GPTJForCausalLM
    ):
        convert_function(
            _model,
            "forward",
            GPTJForCausalLM_forward,
        )
    elif (
        hasattr(_model, "__class__")
        and _model.__class__
        == transformers.models.llama.modeling_llama.LlamaForCausalLM
    ):
        convert_function(
            _model,
            "forward",
            LlamaForCausalLM_forward,
        )
    elif (
        hasattr(_model, "__class__")
        and _model.__class__
        == transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXForCausalLM
    ):
        convert_function(
            _model,
            "forward",
            GPTNeoXForCausalLM_forward,
        )
    elif (
        hasattr(_model, "__class__")
        and _model.__class__ == transformers.models.opt.modeling_opt.OPTForCausalLM
    ):
        convert_function(
            _model,
            "forward",
            OPTForCausalLM_forward,
        )

    # checking if model has been wrapped by deepspeed (distributed or not)
    try:
        from deepspeed.module_inject.layers import LinearAllreduce, LinearLayer

        ds_layers = [LinearAllreduce, LinearLayer]
        is_distributed(_model, ds_layers)
    except ImportError:
        # distributed uses default False
        pass

    # model-wise optimizations - MHA module
    for supported_mha_class in [
        transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXAttention,
        transformers.models.llama.modeling_llama.LlamaAttention,
        transformers.models.gptj.modeling_gptj.GPTJAttention,
        transformers.models.opt.modeling_opt.OPTAttention,
    ]:
        convert_class(
            _model,
            supported_mha_class,
            _IPEXAttentionRef,
            _model.config,
            distributed=distributed,
        )
    # model-wise optimizations - Feedforward/Decoder layer modules
    for supported_decoder_class in [
        transformers.models.llama.modeling_llama.LlamaDecoderLayer,
        transformers.models.gptj.modeling_gptj.GPTJBlock,
        transformers.models.opt.modeling_opt.OPTDecoderLayer,
    ]:
        convert_class(
            _model,
            supported_decoder_class,
            _IPEXDecoderLayerRef,
            _model.config,
            distributed=distributed,
        )

    # special list that has not official transformers design
    if re.search("falcon", _model.config.architectures[0], re.IGNORECASE) or re.search(
        "rw", _model.config.architectures[0], re.IGNORECASE
    ):
        with torch.no_grad():
            ipex.nn.utils._model_convert.replace_customized_linear_with_linear(
                _model.eval()
            )
        convert_class(
            _model,
            type(_model.transformer.h[0].self_attention),
            _IPEXAttentionRef,
            _model.config,
            distributed=distributed,
        )
        convert_class(
            _model,
            type(_model.transformer.h[0]),
            _IPEXDecoderLayerRef,
            _model.config,
            distributed=distributed,
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
        convert_function(
            _model.transformer,
            "_convert_cache_to_standard_format",
            _convert_cache_to_standard_format,
        )
        convert_function(
            _model.transformer,
            "_convert_to_rw_cache",
            _convert_to_rw_cache,
        )
        if hasattr(_model.transformer, "use_alibi"):
            _model.transformer.alibi = _model.transformer.use_alibi
        elif hasattr(_model.transformer, "alibi"):
            _model.transformer.use_alibi = _model.transformer.alibi

    return _model


def get_dummy_input(_model, return_dict=False):
    sample_inputs = None

    if hasattr(_model.config, "n_layer"):
        model_num_layers = _model.config.n_layer
    elif hasattr(_model.config, "num_hidden_layers"):
        model_num_layers = _model.config.num_hidden_layers
    elif hasattr(_model.config, "num_layers"):
        model_num_layers = _model.config.num_layers
    else:
        AssertionError(
            False,
            "Cannot support the dummy sample_inputs for your model, please use your sample_inputs as the inputs and run again",
        )
    past_key_values = tuple(
        [
            (
                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                torch.zeros([1, 1, 1, 1]).contiguous(),
                torch.zeros([1, 1, 1, 1]).contiguous(),
                torch.zeros(1, 4, dtype=torch.long),
            )
            for i in range(model_num_layers)
        ]
    )

    input_ids = torch.ones(32).to(torch.long)
    attention_mask = torch.ones(len(input_ids))
    position_ids = torch.arange(len(input_ids))
    if re.search("opt", _model.config.architectures[0], re.IGNORECASE):
        sample_inputs = (
            {
                "input_ids": input_ids.unsqueeze(0),
                "attention_mask": attention_mask.unsqueeze(0),
                "past_key_values": past_key_values,
            }
            if return_dict
            else (input_ids.unsqueeze(0), attention_mask.unsqueeze(0), past_key_values)
        )
    elif re.search(
        "falcon", _model.config.architectures[0], re.IGNORECASE
    ) or re.search("rw", _model.config.architectures[0], re.IGNORECASE):
        sample_inputs = (
            {
                "input_ids": input_ids.unsqueeze(0),
                "past_key_values": past_key_values,
                "attention_mask": attention_mask.unsqueeze(0),
            }
            if return_dict
            else (input_ids.unsqueeze(0), past_key_values, attention_mask.unsqueeze(0))
        )
    else:
        sample_inputs = (
            {
                "input_ids": input_ids.unsqueeze(0),
                "attention_mask": attention_mask.unsqueeze(0),
                "position_ids": position_ids.unsqueeze(0),
                "past_key_values": past_key_values,
            }
            if return_dict
            else (
                input_ids.unsqueeze(0),
                attention_mask.unsqueeze(0),
                position_ids.unsqueeze(0),
                past_key_values,
            )
        )
    return sample_inputs


def load_int4_weight_and_convert_woq(model, qconfig, state_dict, inplace=True):
    import copy
    from ipex.nn.modules import IpexWoqLinear

    def _convert(mod, attr_name):
        if isinstance(mod, torch.nn.Linear):
            mod.qconfig = qconfig.global_qconfig
            # deal with concat linear
            if attr_name.endswith("concat_qkv"):
                attr_base = ".".join(attr_name.split(".")[:-1])
                w_q = state_dict[attr_base + ".q_proj.qweight"]
                s_q = state_dict[attr_base + ".q_proj.scales"]
                z_q = state_dict[attr_base + ".q_proj.qzeros"]
                w_k = state_dict[attr_base + ".k_proj.qweight"]
                s_k = state_dict[attr_base + ".k_proj.scales"]
                z_k = state_dict[attr_base + ".k_proj.qzeros"]
                w_v = state_dict[attr_base + ".v_proj.qweight"]
                s_v = state_dict[attr_base + ".v_proj.scales"]
                z_v = state_dict[attr_base + ".v_proj.qzeros"]
                w = torch.cat([w_q, w_k, w_v], dim=0)
                s = torch.cat([s_q, s_k, s_v], dim=-1)
                z = torch.cat([z_q, z_k, z_v], dim=-1)
                mod_new = IpexWoqLinear.from_float_and_int4_weight(mod, w, s, z)
            else:
                qweight = state_dict[attr_name + ".qweight"]
                scales = state_dict[attr_name + ".scales"]
                qzeros = state_dict[attr_name + ".qzeros"]
                mod_new = IpexWoqLinear.from_float_and_int4_weight(
                    mod, qweight.float(), scales, qzeros
                )
            return mod_new
        mod_new = mod

        for name, child in mod.named_children():
            attr = attr_name + "." + name if attr_name != "" else name
            setattr(mod_new, name, _convert(child, attr))
        return mod_new

    if not inplace:
        model_new = copy.deepcopy(model)
    else:
        model_new = model
    return _convert(model_new, "")


def ipex_quantization_flow(
    _model, dtype, sample_inputs, qconfig, static_qconfig_file=None
):
    from intel_extension_for_pytorch.quantization import prepare, convert

    if (
        not _is_woq_qconfig(qconfig) and sample_inputs is None
    ):
        sample_inputs = get_dummy_input(_model)

    prepared_model = prepare(
        _model.eval(), qconfig, example_inputs=sample_inputs, inplace=True
    )
    if static_qconfig_file is not None:
        prepared_model.load_qconf_summary(qconf_summary=static_qconfig_file)
        print("ipex.optimize_transformers is doing the static quantization")
    else:
        print("ipex.optimize_transformers is doing the weight only quantization")
    with torch.no_grad(), torch.cpu.amp.autocast(
        enabled=True if dtype is torch.bfloat16 else False
    ):
        convert_model = convert(prepared_model.eval(), inplace=True).eval()
        if _is_woq_qconfig(qconfig) and dtype is torch.bfloat16:
            convert_model = convert_model.to(dtype)
    return convert_model


def model_convert_lowering(
    _model,
    device,
    dtype,
    sample_inputs,
    deployment_mode,
    is_quantization=False,
    woq=False,
):
    from .models.reference.modules.attentions import _IPEXAttentionRef
    from .models.reference.modules.decoder import _IPEXDecoderLayerRef

    if device == "cpu":
        from .models.cpu.modules.attentions import _IPEXAttentionCPU
        from .models.cpu.fusions.mha_fusion import _IPEXRMSNorm
        from .models.cpu.modules.decoder import _IPEXDecoderLayerCPU

        _disable_tpp()
        if not is_quantization:
            if dtype is torch.bfloat16:
                _enable_tpp()
            _model = ipex.optimize(_model.eval(), dtype=dtype, inplace=True)

        if not is_quantization or woq:
            import transformers

            for supported_class in [
                transformers.models.llama.modeling_llama.LlamaRMSNorm
            ]:
                lowering_class_cpu(
                    _model,
                    supported_class,
                    _IPEXRMSNorm,
                    _model.config,
                    tpp=False,
                    woq=False,
                )

        for supported_mlp_class in [_IPEXDecoderLayerRef]:
            lowering_class_cpu(
                _model,
                supported_mlp_class,
                _IPEXDecoderLayerCPU,
                _model.config,
                tpp=True if _using_tpp() else False,
                woq=woq,
            )

        for supported_mha_class in [_IPEXAttentionRef]:
            lowering_class_cpu(
                _model,
                supported_mha_class,
                _IPEXAttentionCPU,
                _model.config,
                tpp=True if _using_tpp() else False,
                woq=woq,
            )

        if deployment_mode:
            sample_inputs = (
                get_dummy_input(_model, return_dict=True)
                if sample_inputs is None
                else sample_inputs
            )
            with torch.no_grad(), torch.cpu.amp.autocast(
                enabled=True if dtype is torch.bfloat16 else False
            ):
                trace_model = torch.jit.trace(
                    _model,
                    example_kwarg_inputs=sample_inputs,
                    strict=False,
                    check_trace=False,
                )
                trace_model = torch.jit.freeze(trace_model)
                _model = _set_optimized_model_for_generation(
                    _model, optimized_model=trace_model
                )

    return _model


def optimize_transformers(
    model,
    optimizer=None,
    dtype=torch.float,
    inplace=False,
    device="cpu",
    quantization_config=None,
    qconfig_summary_file=None,
    low_precision_checkpoint=None,
    sample_inputs=None,
    deployment_mode=True,
):
    r"""
    Apply optimizations at Python frontend to the given transformers model (nn.Module).
    This API focus on transformers models, especially for generation tasks inference.
    Well supported model family: Llama, GPT-J, GPT-Neox, OPT, Falcon.

    Args:
        model (torch.nn.Module): User model to apply optimizations.
        optimizer (torch.optim.Optimizer): User optimizer to apply optimizations
            on, such as SGD. The default value is ``None``, meaning inference case.
        dtype (torch.dtype): Now it works for ``torch.bfloat16`` and ``torch.float``.
            The default value is ``torch.float``. When working with quantization, it means the mixed dtype with quantization.
        inplace (bool): Whether to perform inplace optimization. Default value is ``False``.
        device (str): Perform optimization on which device. Curentlty only support cpu. Default value is ``cpu``.
        quantization_config (object): Defining the IPEX quantization recipe (Weight only quant or static quant).
            Default value is ``None``. Once used, meaning using IPEX quantizatization model for model.generate().
        qconfig_summary_file (str): Path to the IPEX static quantization config json file.
            Default value is ``None``. Work with quantization_config under static quantization use case.
            Need to do IPEX static quantization calibration and generate this file.
        low_precision_checkpoint (str): Path to the IPEX weight only quantization converted (via GPTQ) weight
            checkpoint file. Default value is ``None``.
            Work with quantization_config under weight only quantization INT4 use case.
        sample_inputs (Tuple tensors): sample inputs used for model quantization or torchscript.
            Default value is ``None``, and for well supported model, we provide this sample inputs automaticlly.
        deployment_mode (bool): Whether to apply the optimized model for deployment of model generation.
            It means there is no need to further apply optimization like torchscirpt. Default value is ``True``.

    Returns:
        optimized model object for model.generate(), also workable with model.forward

    .. warning::
        Please invoke ``optimize_transformers`` function AFTER invoking DeepSpeed in Tensor Parallel
        inference scenario.

    Examples:

        >>> # bfloat16 generation inference case.
        >>> model = ...
        >>> model.load_state_dict(torch.load(PATH))
        >>> model.eval()
        >>> optimized_model = ipex.optimize_transformers(model, dtype=torch.bfloat16)
        >>> optimized_model.generate()

    """
    if isinstance(model, torch.jit.ScriptModule):
        return model
    if model.training or optimizer is not None:
        warnings.warn(
            "fail to apply optimize_transformers, this API supports inference for now, fallback to default path"
        )
        return model, optimizer

    try:
        installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
        min_version = "4.28.1"
        validated_version = "4.31.0"
        if "transformers" not in installed_pkg:
            raise RuntimeError(
                "optimize_transformers requires transformers package with version at least {} , fallback".format(
                    min_version
                )
            )

        import transformers
        from packaging import version

        trans_version = transformers.__version__
        if version.parse(trans_version) < version.parse(min_version):
            raise RuntimeError(
                "optimize_transformers requires transformers: at least {} while but your transformers== {}, fallback".format(
                    min_version, trans_version
                )
            )
        if version.parse(trans_version) > version.parse(validated_version):
            warnings.warn(
                f"The transformers version is {trans_version}, bigger than validated {validated_version}, may have risks"
            )
        if not hasattr(model, "config"):
            warnings.warn(
                "Can not check transformers model config to detect its model family, fallback to origin model"
            )
            return model

        well_supported_model = (
            re.search("GPTJ", model.config.architectures[0], re.IGNORECASE)
            or re.search("llama", model.config.architectures[0], re.IGNORECASE)
            or re.search("gptneox", model.config.architectures[0], re.IGNORECASE)
            or re.search("OPT", model.config.architectures[0], re.IGNORECASE)
            or re.search("falcon", model.config.architectures[0], re.IGNORECASE)
            or re.search("rw", model.config.architectures[0], re.IGNORECASE)
        )
        if not well_supported_model:
            warnings.warn(
                "optimize_transformers supports Llama, GPT-J, GPT-Neox, Falcon, and OPT, fallback to origin model"
            )
            return model

        if not inplace:
            _model = copy.deepcopy(model)
        else:
            _model = model

        # model reference conversion
        _model = model_convert_reference(_model)

        # model quantization if needed
        is_quantization = False
        is_woq = False
        if quantization_config is not None:
            is_quantization = True
            if not _is_woq_qconfig(quantization_config):  # static quantization
                deployment_mode = False

                if qconfig_summary_file is not None:
                    # static quantization is cpu only, thus doing model lowering firstly
                    _model = model_convert_lowering(
                        _model,
                        device,
                        dtype,
                        sample_inputs,
                        deployment_mode,
                        is_quantization,
                        woq=is_woq,
                    )
                    _model = ipex_quantization_flow(
                        _model,
                        dtype,
                        sample_inputs,
                        quantization_config,
                        qconfig_summary_file,
                    )
                    sample_inputs = (
                        get_dummy_input(_model, return_dict=True)
                        if sample_inputs is None
                        else sample_inputs
                    )
                    with torch.no_grad(), torch.cpu.amp.autocast(
                        enabled=True if dtype is torch.bfloat16 else False
                    ):
                        trace_model = torch.jit.trace(
                            _model,
                            example_kwarg_inputs=sample_inputs,
                            strict=False,
                            check_trace=False,
                        )
                        trace_model = torch.jit.freeze(trace_model)
                        _model = _set_optimized_model_for_generation(
                            _model, optimized_model=trace_model
                        )
                    return _model
                else:
                    print(
                        "ipex.optimize_transformers is prepared for the calibration of the static quantization"
                    )

            elif _is_woq_qconfig(quantization_config):  # weight only quantization
                is_woq = True
                if low_precision_checkpoint is not None:
                    state_dict = torch.load(low_precision_checkpoint)
                    _model = load_int4_weight_and_convert_woq(
                        _model, quantization_config, state_dict
                    )
                else:
                    _model = ipex_quantization_flow(
                        _model,
                        dtype,
                        sample_inputs,
                        quantization_config,
                        None,
                    )

        # model lowering conversion
        _model = model_convert_lowering(
            _model,
            device,
            dtype,
            sample_inputs,
            deployment_mode,
            is_quantization,
            is_woq,
        )

        return _model

    except RuntimeError as e:
        warnings.warn(
            f"fail to apply optimize_transformers due to: {e}, fallback to the origin model"
        )
        return model

    return model
