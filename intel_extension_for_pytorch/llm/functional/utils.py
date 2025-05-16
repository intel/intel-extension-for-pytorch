import sys

from intel_extension_for_pytorch.transformers.models.xpu.fusions.activation_fusion import (  # noqa F401
    gelu_quick_xpu,
    silu_mul_xpu,
    silu_and_mul_xpu,
    gelu_mul_xpu,
    gelu_and_mul_xpu,
    add_rms_norm_xpu,
    add_layer_norm_xpu,
    rotary_embedding_batched_xpu,
    bgmv_shrink_xpu,
    bgmv_expand_xpu,
    bgmv_expand_slice_xpu,
    sgmv_shrink_xpu,
    sgmv_expand_xpu,
    sgmv_expand_slice_xpu,
)
from intel_extension_for_pytorch.transformers.models.cpu.fusions.mha_fusion import (  # noqa F401
    silu_mul_cpu,
    gelu_mul_cpu,
    add_rms_norm_cpu,
    add_layer_norm_cpu,
    bgmv_shrink_cpu,
    bgmv_expand_cpu,
    bgmv_expand_slice_cpu,
)


def _get_function_from_device(device_type: str, f):
    assert device_type in [
        "cpu",
        "xpu",
    ], "The device is not in the supported device list."
    target_f_name = f.__name__ + "_" + device_type
    assert hasattr(
        sys.modules[__name__], target_f_name
    ), f"Target function {f.__name__} on {device_type} haven't implemented yet."
    target_f = getattr(sys.modules[__name__], target_f_name)
    return target_f


def ipex_update_causal_mask(model):
    import torch
    import transformers

    target_list = [transformers.models.llama.modeling_llama.LlamaModel]
    origin_func = None

    def ipex_func(*args, **kwargs):
        causal_mask = origin_func(*args, **kwargs)
        if causal_mask is not None and causal_mask.device.type == "xpu":
            from transformers.modeling_attn_mask_utils import AttentionMaskConverter

            min_dtype = torch.finfo(causal_mask.dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )
        return causal_mask

    def convert_transformers_update_mask(model):
        for _, sub_m in model.named_children():
            if type(sub_m) in target_list:
                nonlocal origin_func
                origin_func = sub_m._update_causal_mask
                sub_m._update_causal_mask = ipex_func
                break
            convert_transformers_update_mask(sub_m)

    convert_transformers_update_mask(model)
