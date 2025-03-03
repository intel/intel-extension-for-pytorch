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
    from intel_extension_for_pytorch.quantization import QConfigWoq, WoqLowpMode

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


def _get_keys_from_config(checkpoint_config):
    weight_key = checkpoint_config.get("weight_key", "qweight")
    scales_key = checkpoint_config.get("scale_key", "scales")
    zeros_key = checkpoint_config.get("zero_point_key", "qzeros")
    bias_key = checkpoint_config.get("bias_key", "bias")
    g_idx_key = checkpoint_config.get("g_idx_key", "g_idx")
    return weight_key, scales_key, zeros_key, bias_key, g_idx_key


def _get_linear_parameters(attr_name, state_dict, checkpoint_config):
    weight_key, scales_key, zeros_key, bias_key, g_idx_key = _get_keys_from_config(
        checkpoint_config
    )
    w_key = attr_name + "." + weight_key
    s_key = attr_name + "." + scales_key
    z_key = attr_name + "." + zeros_key
    b_key = attr_name + "." + bias_key
    g_key = attr_name + "." + g_idx_key
    # all are tensors
    qweight = state_dict.get(w_key, None)
    scales = state_dict.get(s_key, None)
    qzeros = state_dict.get(z_key, None)
    bias = state_dict.get(b_key, None)
    g_idx = state_dict.get(g_key, None)
    group_size = -1
    from intel_extension_for_pytorch.nn.modules import Int4WeightFormat

    weight_format = Int4WeightFormat.PLAIN_FORMAT

    if qweight is None:
        return qweight, scales, qzeros, bias, group_size, g_idx, weight_format

    if checkpoint_config["name"] == "gptq":
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

        # if g_idx is None, pack weight with GPTQ format directly
        # Otherwise, convert GPTQ format to plain format then pack
        # This is because we need to do channel shuffling if g_idx presents
        if g_idx is not None:
            qweight, scales, qzeros = _convert_optimum_format_to_desired(
                qweight, scales, qzeros
            )
        else:
            scales, qzeros = _convert_gptq_scales_qzeros(scales, qzeros)
            weight_format = Int4WeightFormat.GPTQ_FORMAT

    elif checkpoint_config["name"] == "awq":
        if scales is not None:
            assert (
                qweight.size(0) % scales.size(0) == 0
            ), "Uneven group sizes are not supported"
            group_size = qweight.size(0) // scales.size(0)
            scales, qzeros = _convert_awq_scales_qzeros(scales, qzeros)
            weight_format = Int4WeightFormat.AWQ_FORMAT
            g_idx = None
    return qweight, scales, qzeros, bias, group_size, g_idx, weight_format


def _convert_woq_with_low_precision_checkpoint(
    model,
    qconfig_mapping,
    low_precision_checkpoint,
    quant_config,
    inplace=True,
):
    r"""
    Method to convert fp32 model to WOQ model with checkpoint generated by GPTQ or AWQ
    Args:
        model: original model
        qconfig_mapping: QConfigMapping object containing observer info, lowp mode, etc.
        low_precision_checkpoint (dict): checkpoint generated by GPTQ/AWQ, etc.
        quant_config (dict): containing info like quantization method ("gptq" or "awq") and group size.
        inplace: do conversion in-place or make a copy of original model
    Return:
        Converted model

    By default, we use the checkpoint format generated by Intel(R) Neural Compressor (INC) GPTQ.
    The default format is described by `weight_only_low_precision_checkpoint_config.json`
    Users may use custom config to override the default.
    Default format:
    - Weights and zero points in UINT4 and compressed as INT32, scales in FP16.
    - Keys are 'packed_weight', 'scale', 'packed_zp'
    """

    assert isinstance(
        low_precision_checkpoint, dict
    ), "low_precision_checkpoint should be a state_dict"
    quantization_method = quant_config["quant_method"]
    quant_group_size = quant_config.get("group_size", None)

    if quantization_method == "gptq":
        checkpoint_config = _gptq_lowp_checkpoint_config()
        if "desc_act" in quant_config:
            checkpoint_config["desc_act"] = quant_config.get("desc_act", None)
    elif quantization_method == "awq":
        checkpoint_config = _awq_lowp_checkpoint_config()
    else:
        raise AssertionError(
            f"{quantization_method} is not supported, quantization_method choice in [`gptq`, `awq`]."
        )

    state_dict = low_precision_checkpoint
    # Check that keys can be found in the state dict. Bias and g_idx are optional.
    weight_key, scales_key, zeros_key, _, _ = _get_keys_from_config(checkpoint_config)
    keys_found = [False] * 3
    for k, _ in state_dict.items():
        if k.endswith("." + weight_key):
            keys_found[0] = True
        if k.endswith("." + scales_key):
            keys_found[1] = True
        if k.endswith("." + zeros_key):
            keys_found[2] = True
        if all(keys_found):
            break
    assert all(keys_found), "Error: Format of checkpoint and config do not match"
    from intel_extension_for_pytorch.nn.modules import (
        WeightOnlyQuantizedLinear,
        IpexWoqLinearAllreduce,
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
            qweight, scales, qzeros, bias, group_size, g_idx, w_format = (
                _get_linear_parameters(attr_name, state_dict, checkpoint_config)
            )
            if quant_group_size is not None:
                group_size = quant_group_size
            if any(i is None for i in [qweight, scales, qzeros]):
                return mod
            mod_new = q_op_map[type(mod)].from_float_and_int4_weight(
                mod,
                qweight,
                scales,
                qzeros,
                bias,
                group_size=group_size,
                g_idx=g_idx,
                weight_format=w_format,
            )
            return mod_new
        elif hasattr(mod, "weight") and isinstance(mod.weight, torch.nn.Parameter):
            mod.weight.data = state_dict.get(attr_name + ".weight", mod.weight.data)
            if hasattr(mod, "bias") and isinstance(mod.bias, torch.nn.Parameter):
                mod.bias.data = state_dict.get(attr_name + ".bias", mod.bias.data)
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
