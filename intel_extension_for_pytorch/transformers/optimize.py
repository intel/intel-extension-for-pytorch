import torch
import copy
import warnings
import pkg_resources
from intel_extension_for_pytorch.cpu._auto_kernel_selection import (
    _enable_tpp,
    _disable_tpp,
    _using_tpp,
)
import intel_extension_for_pytorch as ipex
from ..utils.weight_only_quantization import (
    _is_woq_qconfig,
    _convert_woq_with_low_precision_checkpoint,
)


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


def _set_optimized_model_for_generation(
    model,
    optimized_model,
    first_token_optimized_model=None,
):
    from .models.reference.models import IPEX_LLM_Model_Return

    if first_token_optimized_model is not None:
        model.trace_graph_first = IPEX_LLM_Model_Return(
            model, first_token_optimized_model
        )

    model.trace_graph = IPEX_LLM_Model_Return(model, optimized_model)
    print(
        "ipex.llm.optimize has set the optimized or quantization model for model.generate()"
    )
    return model


def model_convert_reference(_model):
    import transformers
    from packaging import version

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
        _gen_baichuan_alibi_mask,
        GLM2_get_masks,
        _relative_position_bucket,
        _to_4d,
    )

    # model wise optimization for Feedforward and Decoder layer modules
    from .models.reference.modules.decoder import _IPEXDecoderLayerRef

    # generation length or model forward order
    from .models.reference.models import (
        GPTJForCausalLM_forward,
        LlamaForCausalLM_forward,
        GPTNeoXForCausalLM_forward,
        OPTForCausalLM_forward,
        BloomForCausalLM_forward,
        FalconForCausalLM_forward,
        CodeGenForCausalLM_forward,
        BaichuanForCausalLM_forward,
        BaichuanModel_forward,
        ChatGLMModel_forward,
        GLMTransformer_forward,
        ChatGLMForConditionalGeneration_forward,
        GPTBigCodeForCausalLM_forward,
        GPTBigCodeModel_forward,
        T5ForConditionalGeneration_forward,
        T5DenseGatedActDense_forward,
        T5DenseActDense_forward,
        MistralForCausalLM_forward,
        MptForCausalLM_forward,
        prepare_inputs_for_generation,
        prepare_inputs_for_generation_gptbigcode,
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

    if version.parse(transformers.__version__) > version.parse("4.34.1"):
        from transformers.modeling_attn_mask_utils import AttentionMaskConverter

        AttentionMaskConverter.to_4d = _to_4d

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
    elif (
        hasattr(_model, "__class__")
        and _model.__class__
        == transformers.models.bloom.modeling_bloom.BloomForCausalLM
    ):
        convert_function(
            _model,
            "forward",
            BloomForCausalLM_forward,
        )
    elif (
        hasattr(_model, "__class__")
        and _model.__class__
        == transformers.models.codegen.modeling_codegen.CodeGenForCausalLM
    ):
        convert_function(
            _model,
            "forward",
            CodeGenForCausalLM_forward,
        )
    elif (
        hasattr(_model, "__class__")
        and _model.__class__
        == transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBigCodeForCausalLM
    ):
        convert_function(
            _model,
            "forward",
            GPTBigCodeForCausalLM_forward,
        )
        convert_function(_model.transformer, "forward", GPTBigCodeModel_forward)
        convert_function(
            _model,
            "prepare_inputs_for_generation",
            prepare_inputs_for_generation_gptbigcode,
        )
    elif (
        hasattr(_model, "__class__")
        and _model.__class__
        == transformers.models.t5.modeling_t5.T5ForConditionalGeneration
    ):
        convert_function(_model, "forward", T5ForConditionalGeneration_forward)
        convert_function(
            transformers.models.t5.modeling_t5.T5Attention,
            "_relative_position_bucket",
            _relative_position_bucket,
        )
        convert_functions(
            _model,
            transformers.models.t5.modeling_t5.T5DenseActDense,
            "forward",
            T5DenseActDense_forward,
        )
        convert_functions(
            _model,
            transformers.models.t5.modeling_t5.T5DenseGatedActDense,
            "forward",
            T5DenseGatedActDense_forward,
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
        transformers.models.bloom.modeling_bloom.BloomAttention,
        transformers.models.codegen.modeling_codegen.CodeGenAttention,
        transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBigCodeAttention,
        transformers.models.t5.modeling_t5.T5Attention,
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
        transformers.models.codegen.modeling_codegen.CodeGenBlock,
        transformers.models.opt.modeling_opt.OPTDecoderLayer,
        transformers.models.bloom.modeling_bloom.BloomBlock,
        transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBigCodeBlock,
        transformers.models.t5.modeling_t5.T5Block,
    ]:
        convert_class(
            _model,
            supported_decoder_class,
            _IPEXDecoderLayerRef,
            _model.config,
            distributed=distributed,
        )

    # special list that has not official transformers design
    if _model.config.architectures[0] == "BloomForCausalLM":
        convert_function(
            _model.transformer,
            "_prepare_attn_mask",
            _prepare_attn_mask_falcon,
        )
        convert_function(
            _model,
            "prepare_inputs_for_generation",
            prepare_inputs_for_generation,
        )
    if (
        _model.config.architectures[0] == "FalconForCausalLM"
        or _model.config.architectures[0] == "RWForCausalLM"
    ):
        with torch.no_grad():
            ipex.nn.utils._model_convert.replace_customized_linear_with_linear(
                _model.eval()
            )
        convert_function(_model, "forward", FalconForCausalLM_forward)
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
    elif _model.config.architectures[0] == "BaichuanForCausalLM":
        convert_function(_model, "forward", BaichuanForCausalLM_forward)
        convert_class(
            _model,
            type(_model.model.layers[0].self_attn),
            _IPEXAttentionRef,
            _model.config,
            distributed=distributed,
        )
        convert_class(
            _model,
            type(_model.model.layers[0]),
            _IPEXDecoderLayerRef,
            _model.config,
            distributed=distributed,
        )
        if not distributed:
            convert_function(_model.model, "forward", BaichuanModel_forward)
            _model.model.future_mask = _gen_baichuan_alibi_mask(
                _model.model.layers[0].self_attn.num_heads,
                _model.model.layers[0].self_attn.max_position_embeddings,
            )
    elif _model.config.architectures[0] == "ChatGLMModel":
        convert_function(_model, "forward", ChatGLMForConditionalGeneration_forward)
        convert_function(_model.transformer, "forward", ChatGLMModel_forward)
        convert_function(_model.transformer.encoder, "forward", GLMTransformer_forward)
        convert_class(
            _model,
            type(_model.transformer.encoder.layers[0].self_attention),
            _IPEXAttentionRef,
            _model.config,
            distributed=distributed,
        )
        convert_class(
            _model,
            type(_model.transformer.encoder.layers[0]),
            _IPEXDecoderLayerRef,
            _model.config,
            distributed=distributed,
        )
        convert_function(_model.transformer, "get_masks", GLM2_get_masks)
    elif _model.config.architectures[0] == "MistralForCausalLM":
        convert_function(_model, "forward", MistralForCausalLM_forward)
        convert_class(
            _model,
            transformers.models.mistral.modeling_mistral.MistralAttention,
            _IPEXAttentionRef,
            _model.config,
            distributed=distributed,
        )
        convert_class(
            _model,
            transformers.models.mistral.modeling_mistral.MistralDecoderLayer,
            _IPEXDecoderLayerRef,
            _model.config,
            distributed=distributed,
        )
    elif _model.config.architectures[0] == "MptForCausalLM":
        convert_function(_model, "forward", MptForCausalLM_forward)
        convert_class(
            _model,
            transformers.models.mpt.modeling_mpt.MptAttention,
            _IPEXAttentionRef,
            _model.config,
            distributed=distributed,
        )
        convert_class(
            _model,
            transformers.models.mpt.modeling_mpt.MptBlock,
            _IPEXDecoderLayerRef,
            _model.config,
            distributed=distributed,
        )
    return _model


def get_dummy_input(_model, return_dict=False):
    sample_inputs = None

    if hasattr(_model.config, "n_layer"):
        model_num_layers = _model.config.n_layer
    elif hasattr(_model.config, "num_hidden_layers"):
        model_num_layers = _model.config.num_hidden_layers
    elif hasattr(_model.config, "num_layers"):
        model_num_layers = _model.config.num_layers
    elif hasattr(_model.config, "n_layers"):
        model_num_layers = _model.config.n_layers
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
            if _model.config.architectures[0] != "T5ForConditionalGeneration"
            else (
                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                torch.zeros([1, 1, 1, 1]).contiguous(),
                torch.zeros([1, 1, 1, 1]).contiguous(),
                torch.zeros(1, 4, dtype=torch.long),
                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                torch.zeros(
                    [
                        32,
                        1,
                        _model.decoder.block[i].layer[1].EncDecAttention.n_heads,
                        _model.decoder.block[i]
                        .layer[1]
                        .EncDecAttention.key_value_proj_dim,
                    ]
                ).contiguous(),
                torch.zeros(
                    [
                        32,
                        1,
                        _model.decoder.block[i].layer[1].EncDecAttention.n_heads,
                        _model.decoder.block[i]
                        .layer[1]
                        .EncDecAttention.key_value_proj_dim,
                    ]
                ).contiguous(),
                torch.zeros(1, 4, dtype=torch.long),
            )
            for i in range(model_num_layers)
        ]
    )

    input_ids = torch.ones(32).to(torch.long)
    model_inputs = _model.prepare_inputs_for_generation(input_ids.unsqueeze(0))
    has_position_ids = "position_ids" in model_inputs
    attention_mask = torch.ones(len(input_ids))
    position_ids = torch.arange(len(input_ids))
    if has_position_ids:
        sample_inputs = (
            {
                "input_ids": input_ids.unsqueeze(0),
                "attention_mask": attention_mask.unsqueeze(0),
                "past_key_values": past_key_values,
                "position_ids": position_ids.unsqueeze(0),
            }
            if return_dict
            else (
                input_ids.unsqueeze(0),
                attention_mask.unsqueeze(0),
                past_key_values,
                position_ids.unsqueeze(0),
            )
        )
    elif _model.config.architectures[0] == "T5ForConditionalGeneration":
        last_hidden_state = torch.rand([1, 32, 2048])
        sample_inputs = (
            (
                {
                    "decoder_input_ids": torch.ones(1).to(torch.long).unsqueeze(0),
                    "attention_mask": attention_mask.unsqueeze(0),
                    "past_key_values": past_key_values,
                    "encoder_outputs": (last_hidden_state,),
                }
            )
            if return_dict
            else (
                torch.ones(1).to(torch.long).unsqueeze(0),
                attention_mask.unsqueeze(0),
                past_key_values,
                (last_hidden_state,),
            )
        )
    else:
        sample_inputs = (
            {
                "input_ids": input_ids.unsqueeze(0),
                "attention_mask": attention_mask.unsqueeze(0),
                "past_key_values": past_key_values,
            }
            if return_dict
            else (input_ids.unsqueeze(0), attention_mask.unsqueeze(0), past_key_values)
        )

    if "return_last_logit" in model_inputs:
        if return_dict:
            sample_inputs["return_last_logit"] = torch.tensor(True)
        else:
            sample_inputs = sample_inputs + (torch.tensor(True),)
    return sample_inputs


def ipex_quantization_flow(
    _model, dtype, sample_inputs, qconfig, static_qconfig_file=None
):
    from intel_extension_for_pytorch.quantization import prepare, convert

    is_woq = _is_woq_qconfig(qconfig)
    if not is_woq and sample_inputs is None:
        sample_inputs = get_dummy_input(_model)

    prepared_model = prepare(
        _model.eval(), qconfig, example_inputs=sample_inputs, inplace=True
    )

    if static_qconfig_file is not None:
        prepared_model.load_qconf_summary(qconf_summary=static_qconfig_file)
        print("ipex.llm.optimize is doing the static quantization")
    else:
        print("ipex.llm.optimize is doing the weight only quantization")

    with torch.no_grad(), torch.cpu.amp.autocast(
        enabled=True if dtype is torch.bfloat16 else False
    ):
        convert_model = convert(prepared_model.eval(), inplace=True).eval()
        if is_woq and dtype is torch.bfloat16:
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

            supported_classes = [
                transformers.models.llama.modeling_llama.LlamaRMSNorm,
            ]
            if _model.config.architectures[0] == "BaichuanForCausalLM":
                supported_classes.append(type(_model.model.layers[0].input_layernorm))
            if (
                _model.config.architectures[0] == "ChatGLMModel"
                and _model.config.rmsnorm
            ):
                supported_classes.append(
                    type(_model.transformer.encoder.layers[0].input_layernorm)
                )
            if hasattr(transformers.models, "mistral"):
                supported_classes.append(
                    transformers.models.mistral.modeling_mistral.MistralRMSNorm
                )

            for supported_class in supported_classes:
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


# TODO: refine this check in other specific path
def validate_device_avaliable(device: str):
    def error_message(device):
        raise RuntimeError(
            f"Device [{device}] is not avaliable in your IPEX package, need to re-install IPEX with [{device}] support, exiting..."
        )

    if device == "xpu":
        if not ipex._C._has_xpu():
            error_message(device)
    elif device == "cpu":
        if not ipex._C._has_cpu():
            error_message(device)
    else:
        raise RuntimeError(
            f"Device [{device}] is not supported in the IPEX package. Options in [xpu, cpu], exiting..."
        )


def optimize(
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
    Well supported model family:
    Llama, GPT-J, GPT-Neox, OPT, Falcon, Bloom, CodeGen, Baichuan, ChatGLM, GPTBigCode, T5, Mistral, MPT.

    Args:
        model (torch.nn.Module): User model to apply optimizations.
        optimizer (torch.optim.Optimizer): User optimizer to apply optimizations
            on, such as SGD. The default value is ``None``, meaning inference case.
        dtype (torch.dtype): Now it works for ``torch.bfloat16`` and ``torch.float``.
            The default value is ``torch.float``. When working with quantization, it means the mixed dtype with quantization.
        inplace (bool): Whether to perform inplace optimization. Default value is ``False``.
        device (str): Specifying the device on which the optimization will be performed.
            Can be either 'cpu' or 'xpu' ('xpu' is not applicable for cpu only packages). The default value is 'cpu'.
        quantization_config (object): Defining the IPEX quantization recipe (Weight only quant or static quant).
            Default value is ``None``. Once used, meaning using IPEX quantizatization model for model.generate().
        qconfig_summary_file (str): Path to the IPEX static quantization config json file.
            Default value is ``None``. Work with quantization_config under static quantization use case.
            Need to do IPEX static quantization calibration and generate this file.
        low_precision_checkpoint (dict or tuple of dict): For weight only quantization with INT4 weights.
            If it's a dict, it should be the state_dict of checkpoint (`.pt`) generated by GPTQ, etc.
            If a tuple is provided, it should be `(checkpoint, checkpoint config)`,
            where `checkpoint` is the state_dict and `checkpoint config` is dict specifying
            keys of weight/scale/zero point/bias in the state_dict.
            The default config is {'weight_key': 'packed_weight', 'scale_key': 'scale',
            'zero_point_key': 'packed_zp', bias_key: 'bias'}. Change the values of the dict to make a custom config.
            Weights shape should be N by K and they are quantized to UINT4 and compressed along K, then stored as
            `torch.int32`. Zero points are also UINT4 and stored as INT32. Scales and bias are floating point values.
            Bias is optional. If bias is not in state dict, bias of the original model is used.
            Default value is ``None``.
        sample_inputs (Tuple tensors): sample inputs used for model quantization or torchscript.
            Default value is ``None``, and for well supported model, we provide this sample inputs automaticlly.
        deployment_mode (bool): Whether to apply the optimized model for deployment of model generation.
            It means there is no need to further apply optimization like torchscirpt. Default value is ``True``.

    Returns:
        Optimized model object for model.generate(), also workable with model.forward

    .. warning::
        Please invoke ``ipex.llm.optimize`` function AFTER invoking DeepSpeed in Tensor Parallel
        inference scenario.

    Examples:

        >>> # bfloat16 generation inference case.
        >>> model = ...
        >>> model.load_state_dict(torch.load(PATH))
        >>> model.eval()
        >>> optimized_model = ipex.llm.optimize(model, dtype=torch.bfloat16)
        >>> optimized_model.generate()

    """
    if isinstance(model, torch.jit.ScriptModule):
        return model
    if model.training or optimizer is not None:
        warnings.warn(
            "fail to apply ipex.llm.optimize, this API supports inference for now, fallback to default path"
        )
        return model, optimizer

    validate_device_avaliable(device)

    try:
        installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
        min_version = "4.28.1"
        validated_version = "4.35.2"
        if "transformers" not in installed_pkg:
            raise RuntimeError(
                "ipex.llm.optimize requires transformers package with version at least {} , fallback".format(
                    min_version
                )
            )

        import transformers
        from packaging import version

        trans_version = transformers.__version__
        if version.parse(trans_version) < version.parse(min_version):
            raise RuntimeError(
                "ipex.llm.optimize requires transformers: at least {} while but your transformers== {}, fallback".format(
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

        well_supported_model = model.config.architectures[0] in [
            "GPTJForCausalLM",
            "LlamaForCausalLM",
            "GPTNeoXForCausalLM",
            "OPTForCausalLM",
            "FalconForCausalLM",
            "RWForCausalLM",
            "BloomForCausalLM",
            "CodeGenForCausalLM",
            "BaichuanForCausalLM",
            "ChatGLMModel",
            "GPTBigCodeForCausalLM",
            "T5ForConditionalGeneration",
            "MistralForCausalLM",
            "MptForCausalLM",
        ]
        if not well_supported_model:
            warnings.warn(
                "ipex.llm.optimize supports Llama, GPT-J, GPT-Neox, Falcon, OPT, Bloom, CodeGen, Baichuan, ChatGLM, \
                    GPTBigCode, T5, Mistral, and MPT, fallback to origin model"
            )
            return model

        if not inplace:
            _model = copy.deepcopy(model)
        else:
            _model = model

        # profiling mode is disabled in ChatGLM (https://huggingface.co/THUDM/chatglm3-6b/blob/main/modeling_chatglm.py#L33-L34)
        # Enable profiling mode to apply jit optimizations
        if model.config.architectures[0] == "ChatGLMModel":
            torch._C._jit_set_profiling_mode(True)
            torch._C._jit_set_profiling_executor(True)
            torch._C._jit_override_can_fuse_on_cpu(False)
            torch._C._jit_override_can_fuse_on_gpu(False)

        is_quantization = False
        is_woq = False
        if quantization_config is not None:
            is_quantization = True
            if _is_woq_qconfig(quantization_config):
                is_woq = True

        # Load low precision checkpoint (generated by GPTQ, etc.) for WOQ before any conversion
        if device == "cpu" and is_woq and low_precision_checkpoint is not None:
            state_dict, config = None, None
            if isinstance(low_precision_checkpoint, tuple):
                assert (
                    len(low_precision_checkpoint) == 2
                    and isinstance(low_precision_checkpoint[0], dict)
                    and isinstance(low_precision_checkpoint[1], dict)
                ), "Invalid low_precision_checkpoint"
                state_dict, config = low_precision_checkpoint
            else:
                assert isinstance(
                    low_precision_checkpoint, dict
                ), "Invalid low_precision_checkpoint argument"
                state_dict = low_precision_checkpoint
            _model = _convert_woq_with_low_precision_checkpoint(
                _model, quantization_config, state_dict, config
            )

        # model reference conversion
        _model = model_convert_reference(_model)

        # model quantization if needed
        if is_quantization:
            if not is_woq:  # static quantization
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
                        "ipex.llm.optimize is prepared for the calibration of the static quantization"
                    )

            else:  # weight only quantization
                # Note that GPTQ is already handled at the beginning.
                # Here we only deal with linear layers that have not beed quantized so far.
                # It's a choice of the user or algorithm.
                # Those layers will also be converted here.
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
        # do not register output hook when doing calibration
        if not (is_quantization and qconfig_summary_file is None):
            from .models.reference.models import output_hook

            _model.register_forward_hook(output_hook, with_kwargs=True)
        return _model

    except RuntimeError as e:
        warnings.warn(
            f"fail to apply ipex.llm.optimize due to: {e}, fallback to the origin model"
        )
        return model

    return model


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
    warnings.warn(
        "ipex.optimize_transformers API is going to be deprecated, please use ipex.llm.optimize instead."
    )
    return optimize(
        model=model,
        optimizer=optimizer,
        dtype=dtype,
        inplace=inplace,
        device=device,
        quantization_config=quantization_config,
        qconfig_summary_file=qconfig_summary_file,
        low_precision_checkpoint=low_precision_checkpoint,
        sample_inputs=sample_inputs,
        deployment_mode=deployment_mode,
    )
