import torch
import sys
from intel_extension_for_pytorch.optim import _optimizer_utils
import types
from ._parameter_wrapper import get_shared_parameter_status, patch_state_dict


def weight_dtype_convert_with_ipex(
    model, optimizer, params_attr, master_weight_split, dtype=torch.bfloat16
):
    assert dtype in [
        torch.bfloat16,
        torch.float16,
    ], "weight convert only support bf16 and fp16"

    if len(params_attr) == 0:
        get_shared_parameter_status(model, params_attr)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for name, para in self.named_parameters():
            if not hasattr(self, name):
                continue
            para_name = prefix + name
            with torch.no_grad():
                if para_name in state_dict:
                    fp32_param = state_dict[para_name]
                    param_wrapper = getattr(self, name + "_wrapper")
                    param_wrapper.load(self, fp32_param)

    def convert(module):
        if not hasattr(module, "master_weight_split"):
            setattr(module, "master_weight_split", master_weight_split)  # noqa: B010
            # replace weight/bias
            for name, param in module._parameters.items():
                if param is None:
                    continue
                param_wrapper = params_attr[param]
                if param_wrapper.can_cast_training(dtype):
                    param_wrapper.cast_for_training(dtype, master_weight_split)
                    if not master_weight_split:
                        with torch.no_grad():
                            setattr(
                                module,
                                name,
                                param_wrapper.parameter,
                            )
                    setattr(module, name + "_wrapper", param_wrapper)
                    module._load_from_state_dict = types.MethodType(
                        _load_from_state_dict, module
                    )

    def isCLIPTextEmbeddings(module):
        mod = "transformers.models.clip.modeling_clip"
        return (
            mod in sys.modules
            and hasattr(sys.modules[mod], "CLIPTextEmbeddings")
            and isinstance(module, sys.modules[mod].CLIPTextEmbeddings)
        )

    def convert_rec(module):
        for sub_module in module.children():
            convert_rec(sub_module)
        if not isCLIPTextEmbeddings(module):
            convert(module)

    convert_rec(model)

    patch_state_dict(model, params_attr, "training")

    if optimizer is not None:
        _optimizer_utils.patch_load_state_dict(optimizer)
        if not hasattr(optimizer, "params_attr"):
            setattr(optimizer, "params_attr", params_attr)  # noqa: B010
        if not master_weight_split:
            _optimizer_utils.patch_step_for_master_weight_training(optimizer)
            _optimizer_utils.patch_zero_grad_for_master_weight_training(optimizer)

    return model, optimizer, params_attr
