import copy
import torch
from intel_extension_for_pytorch.nn.modules import WeightOnlyQuantizedLinear
from torch.ao.quantization import PlaceholderObserver, QConfigMapping

# The config describes how to load low precision checkpoint for weight only quantization.
# Weight shape is N by K if transposed is False otherwise K by N.
# Bias is optional. If bias is not provided in the checkpoint, we read the original model.
DEFAULT_LOWP_CHECKPOINT_CONFIG = {
    "name": "optimum",
    "use_optimum_format": True,
    "weight_key": "qweight",
    "scale_key": "scales",
    "zero_point_key": "qzeros",
    "bias_key": "bias",
    "g_idx_key": "g_idx",
}

LEGACY_LOWP_CHECKPOINT_CONFIG = {
    "name": "legacy",
    "use_optimum_format": False,
    "weight_key": "packed_weight",
    "scale_key": "scale",
    "zero_point_key": "packed_zp",
    "bias_key": "bias",
    "g_idx_key": "g_idx",
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


def _default_lowp_checkpoint_config():
    return DEFAULT_LOWP_CHECKPOINT_CONFIG


def _legacy_lowp_checkpoint_config():
    return LEGACY_LOWP_CHECKPOINT_CONFIG


def _get_keys_from_config(checkpoint_config):
    weight_key = checkpoint_config.get("weight_key", "qweight")
    scales_key = checkpoint_config.get("scale_key", "scales")
    zeros_key = checkpoint_config.get("zero_point_key", "qzeros")
    bias_key = checkpoint_config.get("bias_key", "bias")
    g_idx_key = checkpoint_config.get("g_idx_key", "bias")
    return weight_key, scales_key, zeros_key, bias_key, g_idx_key


def _convert_optimum_format_to_desired(qweight, scales, qzeros):
    """
    Optimum format:
        qweight: (math.ceil(IC / comp_ratio), OC)
        scales: (n_groups, OC)
        qzeros: (n_groups, math.ceil(OC / comp_ratio))
        qzeros are substracted by 1 before packing

    Desired format:
        compression_dim = 1
        qweight: (OC, math.ceil(IC / comp_ratio))
        scales: (OC, n_groups)
        qzeros: (OC, math.ceil(n_groups / comp_ratio))

    Note:
        IC = input channels or input features
        OC = output channels or output features
        n_groups = math.ceil(IC / group_size)
        comp_ratio = compression data type bits // weight or zeros data type bits
        E.g., compression dtype = int32, weight dtype = int4, comp_ratio = 32 / 4 = 8

    """
    if qweight is None:
        return qweight, scales, qzeros
    oc = qweight.shape[1]
    assert oc == scales.shape[1]
    n_groups = scales.shape[0]
    qweight = qweight.t_().contiguous()
    scales = scales.t_().contiguous()
    if qzeros is None:
        return qweight, scales, qzeros
    zp_dtype = torch.int32
    zp = torch.empty((n_groups, oc), dtype=zp_dtype)
    # Steps to convert qzeros:
    # (1) unpack qzeros to (n_groups, OC)
    # (2) take transpose
    # (3) plus one and handle overflow
    zp_bits = 4  # int4
    comp_dtype_bits = 32  # int32
    comp_ratio = comp_dtype_bits // zp_bits
    mask = torch.tensor(2**zp_bits - 1, dtype=zp_dtype)
    for j in range(qzeros.shape[1]):
        packed_data = qzeros[:, j]
        for e in range(comp_ratio):
            index = j * comp_ratio + e
            if index >= zp.shape[1]:
                continue
            data = (packed_data >> (zp_bits * e)) & mask
            zp[:, index] = data.type(zp_dtype)
    zp = zp.t_().contiguous()
    zp += 1
    # it may overflow after adding one
    zp = torch.where(zp > (2**zp_bits - 1), 0, zp)

    return qweight, scales, zp


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

    use_optimum_format = checkpoint_config.get("use_optimum_format", True)
    if use_optimum_format:
        qweight, scales, qzeros = _convert_optimum_format_to_desired(
            qweight, scales, qzeros
        )

    group_size = -1
    if qweight is not None and scales is not None:
        assert scales.dim() == 2, "Unexpected scales tensor dimension"
        if scales.size(-1) != 1:
            # qweight is compressed along the last dim int4 * 8 -> int32
            group_size = qweight.size(-1) * 8 // scales.size(-1)
    return qweight, scales, qzeros, bias, group_size, g_idx


def _convert_woq_with_low_precision_checkpoint(
    model,
    qconfig_mapping,
    low_precision_checkpoint,
    checkpoint_config=None,
    inplace=True,
):
    r"""
    Method to convert fp32 model to WOQ model with checkpoint generated by GPTQ
    Args:
        model: original model
        qconfig_mapping: QConfigMapping object containing observer info, lowp mode, etc.
        low_precision_checkpoint (dict): checkpoint generated by GPTQ, etc.
        checkpoint_config (dict): custom config to load the checkpoint. Use default if None
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
    assert checkpoint_config is None or isinstance(
        checkpoint_config, dict
    ), "checkpoint_config should be a dict"
    if checkpoint_config is None:
        checkpoint_config = _default_lowp_checkpoint_config()

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

    def _convert(mod, attr_name):
        if isinstance(mod, torch.nn.Linear):
            mod.qconfig = qconfig_mapping.global_qconfig
            qweight, scales, qzeros, bias, group_size, g_idx = _get_linear_parameters(
                attr_name, state_dict, checkpoint_config
            )
            if any(i is None for i in [qweight, scales, qzeros]):
                return mod
            mod_new = WeightOnlyQuantizedLinear.from_float_and_int4_weight(
                mod, qweight, scales, qzeros, bias, group_size=group_size, g_idx=g_idx
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
