import copy
import torch
from torch.ao.quantization import PlaceholderObserver, QConfigMapping
from intel_extension_for_pytorch.utils.utils import has_cpu
from intel_extension_for_pytorch.nn.utils._model_convert import (
    _convert_awq_scales_qzeros,
    _convert_optimum_format_to_desired,
    _convert_gptq_scales_qzeros,
)

if has_cpu():
    from intel_extension_for_pytorch.quantization import (
        QConfigWoq,
        WoqLowpMode,
        WoqWeightDtype,
    )

# The config describes how to load low precision checkpoint for weight only quantization.
# Weight shape is N by K if transposed is False otherwise K by N.
# Bias is optional. If bias is not provided in the checkpoint, we read the original model.
GPTQ_LOWP_CHECKPOINT_CONFIG = {
    "name": "gptq",
    "weight_key": "qweight",
    "scale_key": "scales",
    "zero_point_key": "qzeros",
    "bias_key": "bias",
    "g_idx_key": "g_idx",
    "desc_act": None,
}

AWQ_LOWP_CHECKPOINT_CONFIG = {
    "name": "awq",
    "weight_key": "qweight",
    "scale_key": "scales",
    "zero_point_key": "qzeros",
    "bias_key": "bias",
}

RTN_LOWP_CHECKPOINT_CONFIG = {
    "name": "rtn",
    "weight_key": "qweight",
    "scale_key": "scales",
    "zero_point_key": "qzeros",
    "bias_key": "bias",
}

# For now, it is for DeepSeek-V3/R1 only
FP8_LOWP_CHECKPOINT_CONFIG = {
    "name": "fp8",
    "weight_key": "weight",
    "scale_key": "weight_scale_inv",
    "zero_point_key": "qzeros",
    "bias_key": "bias",
}

# For now, it is for meituan/DeepSeek-R1-Channel-INT8 only
INT8_LOWP_CHECKPOINT_CONFIG = {
    "name": "int8",
    "weight_key": "weight",
    "scale_key": "weight_scale",
    "zero_point_key": "qzeros",
    "bias_key": "bias",
}


def _is_woq_qconfig(qconfig_mapping):
    qconfig = (
        qconfig_mapping.global_qconfig
        if isinstance(qconfig_mapping, QConfigMapping)
        else qconfig_mapping
    )
    return (
        isinstance(qconfig.activation(), PlaceholderObserver)
        and not qconfig.activation().is_dynamic
    )


def _woq_enable_weight_cache_for_large_batch(qconfig_mapping):
    qconfig = (
        qconfig_mapping.global_qconfig
        if isinstance(qconfig_mapping, QConfigMapping)
        else qconfig_mapping
    )
    assert qconfig.lowp_mode in [
        WoqLowpMode.BF16,
        WoqLowpMode.INT8,
    ], "Weight cache is only supported for lowp-mode=BF16 and INT8"
    qconfig_dict = qconfig._asdict()
    qconfig_dict["cache_weight_for_large_batch"] = True
    if isinstance(qconfig_mapping, QConfigMapping):
        qconfig_mapping.set_global(QConfigWoq(**qconfig_dict))
        return qconfig_mapping
    return QConfigWoq(**qconfig_dict)


def _gptq_lowp_checkpoint_config():
    return GPTQ_LOWP_CHECKPOINT_CONFIG


def _awq_lowp_checkpoint_config():
    return AWQ_LOWP_CHECKPOINT_CONFIG


def _fp8_lowp_checkpoint_config():
    return FP8_LOWP_CHECKPOINT_CONFIG


def _get_keys_from_config(checkpoint_config):
    weight_key = checkpoint_config.get("weight_key", "qweight")
    scales_key = checkpoint_config.get("scale_key", "scales")
    zeros_key = checkpoint_config.get("zero_point_key", "qzeros")
    bias_key = checkpoint_config.get("bias_key", "bias")
    g_idx_key = checkpoint_config.get("g_idx_key", "g_idx")
    return weight_key, scales_key, zeros_key, bias_key, g_idx_key


def _get_linear_parameters(attr_name, state_dict, checkpoint_config, quant_config):
    weight_key, scales_key, zeros_key, bias_key, g_idx_key = _get_keys_from_config(
        checkpoint_config
    )
    w_key = attr_name + "." + weight_key
    fw_key = attr_name + ".weight"
    s_key = attr_name + "." + scales_key
    z_key = attr_name + "." + zeros_key
    b_key = attr_name + "." + bias_key
    g_key = attr_name + "." + g_idx_key
    # all are tensors
    qweight = state_dict.get(w_key, None)
    weight = state_dict.get(fw_key, None)
    scales = state_dict.get(s_key, None)
    qzeros = state_dict.get(z_key, None)
    bias = state_dict.get(b_key, None)
    g_idx = state_dict.get(g_key, None)
    group_size = -1
    from intel_extension_for_pytorch.nn.modules import WoqWeightFormat  # noqa F401

    weight_format = WoqWeightFormat.PLAIN_FORMAT
    quant_method = quant_config["quant_method"]
    weight_block_size = quant_config.get("weight_block_size", None)

    if qweight is None:
        return weight, scales, qzeros, bias, group_size, g_idx, weight_format

    if quant_method == "gptq":
        # weight shape = [K // 8, N]
        # scales shape = [K // G, N]
        # qzeros shape = [K // G, N // 8]
        desc_act = checkpoint_config.get("desc_act", None)
        K = qweight.size(0) * 8
        if scales is not None:
            assert scales.dim() == 2, "Unexpected scales tensor dimension"
            if scales.size(-1) != 1:
                group_size = K // scales.size(0)
                # Ensure group_size is a power of two
                assert group_size > 0
                group_size = 2 ** (group_size - 1).bit_length()

        if desc_act is False:
            g_idx = None
        if g_idx is not None and group_size > 0:
            # ignore dummy g_idx
            # qweight is compressed along the last dim int4 * 8 -> int32
            dummy = torch.tensor([i // group_size for i in range(K)], dtype=torch.int32)
            if torch.equal(g_idx, dummy):
                g_idx = None
            elif g_idx.nonzero().numel() == 0:
                g_idx = None

        # if g_idx is None, pack weight with GPTQ format directly
        # Otherwise, convert GPTQ format to plain format then pack
        # This is because we need to do channel shuffling if g_idx presents
        if g_idx is not None:
            qweight, scales, qzeros = _convert_optimum_format_to_desired(
                qweight, scales, qzeros
            )
        else:
            scales, qzeros = _convert_gptq_scales_qzeros(scales, qzeros)
            weight_format = WoqWeightFormat.GPTQ_FORMAT

    elif quant_method == "awq":
        if scales is not None:
            assert (
                qweight.size(0) % scales.size(0) == 0
            ), "Uneven group sizes are not supported"
            group_size = qweight.size(0) // scales.size(0)
            scales, qzeros = _convert_awq_scales_qzeros(scales, qzeros)
            weight_format = WoqWeightFormat.AWQ_FORMAT
            g_idx = None
    elif quant_method == "fp8":
        if scales is not None:
            block_n = weight_block_size[0]
            block_k = weight_block_size[1]
            assert (
                qweight.size(1) % block_k == 0
            ), f"Weight size is not divisible by {block_n}, got {qweight.shape}"
            if scales.size(0) != qweight.size(0):
                assert (qweight.size(0) + block_n - 1) // block_n == scales.size(0), (
                    "WOQ FP8 got misaligned shapes of weight and scales: "
                    f"{qweight.shape}, {scales.shape}, blocks = {weight_block_size}"
                )
                scales = torch.repeat_interleave(scales, block_n, 0)
                scales = scales[: qweight.size(0), :].contiguous()
            group_size = block_k
    elif quant_method == "rtn":
        # weight shape = [K // 4, N] in int32
        # scales shape = [K // G, N] or [1, N] in float
        # qzeros shape = [K // G, N // 4] or [1, N // 4] in int32
        assert scales.shape[0] == qzeros.shape[0]
        if scales.shape[0] == 1:
            scales = scales.squeeze(0)
            qzeros = qzeros.squeeze(0).view(torch.uint8)
        else:
            scales = scales.t().contiguous()
            qzeros = qzeros.view(torch.uint8).t().contiguous()
        qzeros += 1
        qzeros = qzeros.view(torch.int8)
        g_idx = None
        qweight = qweight.t().contiguous().view(torch.int8)
        weight_format = WoqWeightFormat.GPTQ_FORMAT

    return qweight, scales, qzeros, bias, group_size, g_idx, weight_format


def _convert_woq_with_low_precision_checkpoint(
    model,
    qconfig_mapping,
    low_precision_checkpoint,
    quant_config,
    inplace=True,
):
    r"""
    Method to convert fp32 model to WOQ model with checkpoint generated by GPTQ, AWQ and intel/autoround.
    Official FP8 checkpoints of DeepSeek-V3/R1 are also supported.
    Args:
        model: original model
        qconfig_mapping: QConfigMapping object containing observer info, lowp mode, etc.
        low_precision_checkpoint (dict): checkpoint generated by GPTQ/AWQ, etc.
        quant_config (dict): containing info like quantization method ("gptq" or "awq") and group size.
        inplace: do conversion in-place or make a copy of original model
    Return:
        Converted model
    """

    assert isinstance(
        low_precision_checkpoint, dict
    ), "low_precision_checkpoint should be a state_dict"
    quantization_method = quant_config["quant_method"]
    quant_group_size = quant_config.get("group_size", None)
    target_weight_dtype = WoqWeightDtype.INT4

    if quantization_method == "gptq":
        checkpoint_config = _gptq_lowp_checkpoint_config()
        if "desc_act" in quant_config:
            checkpoint_config["desc_act"] = quant_config.get("desc_act", None)
    elif quantization_method == "awq":
        checkpoint_config = _awq_lowp_checkpoint_config()
    elif quantization_method == "fp8":
        checkpoint_config = _fp8_lowp_checkpoint_config()
        target_weight_dtype = WoqWeightDtype.FP8
    elif quantization_method == "rtn":
        checkpoint_config = RTN_LOWP_CHECKPOINT_CONFIG
        bits = quant_config.get("bits", 8)
        target_weight_dtype = WoqWeightDtype.INT8 if bits == 8 else WoqWeightDtype.INT4
    elif quantization_method == "int8":
        checkpoint_config = INT8_LOWP_CHECKPOINT_CONFIG
        bits = quant_config.get("bits", 8)
        target_weight_dtype = WoqWeightDtype.INT8
    else:
        raise AssertionError(
            f"{quantization_method} is not supported, quantization_method choice in [`gptq`, `awq`, `fp8`]."
        )

    state_dict = low_precision_checkpoint
    # Check that keys can be found in the state dict. Bias and g_idx are optional.
    weight_key, scales_key, *_ = _get_keys_from_config(checkpoint_config)
    keys_found = [False] * 2
    for k, _ in state_dict.items():
        if k.endswith("." + weight_key):
            keys_found[0] = True
        if k.endswith("." + scales_key):
            keys_found[1] = True
        if all(keys_found):
            break
    assert all(keys_found), "Error: Format of checkpoint and config do not match"
    from intel_extension_for_pytorch.nn.modules import (  # noqa F401
        WeightOnlyQuantizedLinear,
        IpexWoqLinearAllreduce,
        WoqWeightFormat,
    )

    q_op_map = {
        torch.nn.Linear: WeightOnlyQuantizedLinear,
    }

    from intel_extension_for_pytorch.nn.utils._weight_prepack import (
        may_import_deepspeed_modules,
    )

    deepspeed_modules = may_import_deepspeed_modules()
    if deepspeed_modules is not None:
        LinearAllreduce, LinearLayer, LmHeadLinearAllreduce = deepspeed_modules[:]
        q_op_map.update(
            {
                LinearAllreduce: IpexWoqLinearAllreduce,
                LinearLayer: WeightOnlyQuantizedLinear,
            }
        )

    linear_modules = tuple(q_op_map.keys())

    def _convert(mod, attr_name):
        # lm_head is not quantized in int4 checkpoint, LmHeadLinearAllReduce is not handled here
        if isinstance(mod, linear_modules):
            mod.qconfig = qconfig_mapping.global_qconfig
            weight, scales, qzeros, bias, group_size, g_idx, w_format = (
                _get_linear_parameters(
                    attr_name, state_dict, checkpoint_config, quant_config
                )
            )
            if quant_group_size is not None:
                group_size = quant_group_size
            if scales is None:
                # lm_head
                if weight is not None and weight.dtype in [
                    torch.float,
                    torch.bfloat16,
                    torch.half,
                ]:
                    mod.weight = torch.nn.Parameter(weight)
                    if hasattr(mod, "bias") and isinstance(
                        mod.bias, torch.nn.Parameter
                    ):
                        mod.bias = torch.nn.Parameter(bias)
                return mod
            mod_new = q_op_map[type(mod)].from_float_and_qweight(
                mod,
                weight,
                target_weight_dtype,
                scales,
                qzeros,
                bias,
                group_size=group_size,
                g_idx=g_idx,
                weight_format=w_format,
            )
            return mod_new
        elif hasattr(mod, "weight") and isinstance(mod.weight, torch.nn.Parameter):
            new_w = state_dict.get(attr_name + ".weight", mod.weight.data)
            mod.weight = torch.nn.Parameter(new_w)
            if hasattr(mod, "bias") and isinstance(mod.bias, torch.nn.Parameter):
                new_b = state_dict.get(attr_name + ".bias", mod.bias.data)
                mod.bias = torch.nn.Parameter(new_b)
            if hasattr(mod, "e_score_correction_bias"):
                new_b = state_dict.get(
                    attr_name + ".e_score_correction_bias",
                    mod.e_score_correction_bias.data,
                )
                mod.e_score_correction_bias = torch.nn.Parameter(new_b)
            return mod

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
