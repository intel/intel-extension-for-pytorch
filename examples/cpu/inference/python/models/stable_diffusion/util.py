import os
import torch
import json
import torch.ao.quantization.fx._decomposed
from torch.nn import functional as F
from collections import namedtuple
import transformers

import torchao  # noqa: F401
import torchao.quantization.pt2e.quantizer.x86_inductor_quantizer  # noqa: F401
from torch._inductor.constant_folding import (
    add_dont_constant_fold,
    clear_dont_constant_fold,
)

clear_dont_constant_fold()
add_dont_constant_fold(torch.ops.torchao.dequantize_affine.default)
add_dont_constant_fold(torch.ops.torchao.quantize_affine.default)


class FP8QDQLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, dtype):
        super().__init__()
        self.qtype = torch.float8_e4m3fn
        self.weight = torch.randn((out_features, in_features)).to(self.qtype)
        self.weight_scale = 2.0
        self.scale = 2.0
        self.bias = None
        self.dtype = dtype

    def forward(self, input):
        weight = torch.ops.torchao.dequantize_affine_float8(
            tensor=self.weight.data,
            scale=torch.tensor(self.weight_scale),
            output_dtype=torch.float,
        )
        weight = weight.to(self.dtype)
        q_input = torch.ops.torchao.quantize_affine_float8(
            tensor=input,
            scale=torch.tensor(self.scale),
            float8_dtype=self.qtype,
        )
        dq_input = torch.ops.torchao.dequantize_affine_float8(
            tensor=q_input,
            scale=torch.tensor(self.scale),
            output_dtype=torch.float,
        )
        dq_input = dq_input.to(self.dtype)
        out = torch.nn.functional.linear(dq_input, weight, self.bias)
        return out


class FP8QDQConv2d(torch.nn.Module):
    def __init__(self, weight_shape, stride, padding, dilation, groups):
        super().__init__()
        self.weight = torch.empty(weight_shape)
        self.bias = None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight_scale = None
        self.scale = None

    def forward(self, input):
        dtype = input.dtype
        weight = self.weight.to(dtype) * self.weight_scale
        q_input = torch.clamp(
            (input / self.scale),
            torch.finfo(torch.float8_e4m3fn).min,
            torch.finfo(torch.float8_e4m3fn).max,
        ).to(torch.float8_e4m3fn)
        dq_input = q_input.to(dtype) * self.scale

        return F.conv2d(
            dq_input,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def generate_model_info(model):
    mod_inst_info = namedtuple("ModInstInfo", ["name", "parent"])
    parent_child_mod_dict = {}

    def create_mod_info_recursion(parent):
        for name, mod in parent.named_children():
            parent_child_mod_dict[mod] = mod_inst_info(name=name, parent=parent)
            create_mod_info_recursion(mod)

    create_mod_info_recursion(model)
    return parent_child_mod_dict


def convert(model, dtype):
    with open("data.json", "r") as fp:
        data = json.load(fp)
    parent_child_mod_dict = generate_model_info(model)
    with torch.no_grad():
        for name, mod in model.named_modules():
            mod_type_str = mod.__class__.__name__
            if mod_type_str not in ["Linear", "Conv2d"]:
                continue
            if name not in data:
                continue
            print(mod_type_str, name)

            # per-tensor
            param = mod.weight
            xmax = torch.max(torch.abs(param))
            weight_scale = xmax / torch.finfo(torch.float8_e4m3fn).max
            q_param = torch.clamp(
                (param / weight_scale),
                torch.finfo(torch.float8_e4m3fn).min,
                torch.finfo(torch.float8_e4m3fn).max,
            ).to(torch.float8_e4m3fn)

            scale = [i / torch.finfo(torch.float8_e4m3fn).max for i in data[name]]
            assert len(scale) == 1
            if mod_type_str == "Linear":
                patched_mod = FP8QDQLinear(mod.in_features, mod.out_features, dtype)
                patched_mod.weight.data = q_param
                patched_mod.weight_scale = weight_scale.item()
                patched_mod.scale = scale[0]
                patched_mod.bias = mod.bias
            else:
                patched_mod = FP8QDQConv2d(
                    list(mod.weight.shape),
                    mod.stride,
                    mod.padding,
                    mod.dilation,
                    mod.groups,
                )
                patched_mod.weight.data = q_param
                patched_mod.bias = mod.bias
                patched_mod.weight_scale = weight_scale.item()
                patched_mod.scale = scale[0]
            parent = parent_child_mod_dict[mod].parent
            name = parent_child_mod_dict[mod].name
            setattr(parent, name, patched_mod)
    model.eval()
    return model


input_maxes = {}


def _save_input_pt_hook(name):
    """A forward hook to save input max of a module
    :param name: the module name
    :return: A hook function."""

    def save_input_hook(module, inputs):
        for idx, input in enumerate(inputs):
            if not isinstance(input, torch.Tensor):
                continue
            if name not in input_maxes:
                input_maxes[name] = [torch.tensor(0) for i in range(len(inputs))]
            input_maxes[name][idx] = torch.maximum(
                torch.max(torch.abs(input)).detach(), input_maxes[name][idx]
            )

    return save_input_hook


hook_handles = []


def prepare(model):
    hook_modules = {}
    for n, module in model.named_modules():
        if n == "":
            continue
        hook_modules[n] = module

    for key in hook_modules.keys():
        hook_func = _save_input_pt_hook(key)
        hook_handle = hook_modules[key].register_forward_pre_hook(hook_func)
        hook_handles.append(hook_handle)

    return model


def finalize_calib():
    for k, v in input_maxes.items():
        input_maxes[k] = [i.item() for i in input_maxes[k]]
    import json

    if os.path.exists("data.json"):
        os.remove("data.json")
    with open("data.json", "w") as fp:
        json.dump(input_maxes, fp)
    for hook in hook_handles:
        hook.remove()


def load(model_name_or_path, **kwargs):
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model = convert(model)
    return model
