import os
import torch
import math
import json
import torch.ao.quantization.fx._decomposed
from collections import namedtuple
import transformers

quantize_affine_float8_non_decomposed = (
    torch.ops.torchao.quantize_affine_float8_non_decomposed.default
)
dequantize_affine_float8_non_decomposed = (
    torch.ops.torchao.dequantize_affine_float8_non_decomposed.default
)


def qdq(input, scale):
    dtype = input.dtype
    q_input = quantize_affine_float8_non_decomposed(
        input,
        torch.tensor([scale]),
        torch.float8_e4m3fn,
    )
    dq_input = dequantize_affine_float8_non_decomposed(
        q_input,
        torch.tensor([scale]),
        dtype,
    )
    return dq_input


class FP8QDQLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.qtype = torch.float8_e4m3fn
        self.weight = torch.empty(
            (out_features, in_features),
        )
        self.weight_scale = None
        self.scale = None
        self.bias = None

    def forward(self, input):
        dtype = input.dtype
        weight = dequantize_affine_float8_non_decomposed(
            self.weight.data,
            torch.tensor([self.weight_scale]),
            torch.float,
        )
        weight = weight.to(dtype)

        q_input = quantize_affine_float8_non_decomposed(
            input,
            torch.tensor([self.scale]),
            self.qtype,
        )
        dq_input = dequantize_affine_float8_non_decomposed(
            q_input,
            torch.tensor([self.scale]),
            torch.float,
        )
        dq_input = dq_input.to(dtype)

        out = torch.nn.functional.linear(dq_input, weight, self.bias)
        return out


class FP8QDQSDPA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.weight_scale = None
        self.q_out_scale = None
        self.k_out_scale = None
        self.attn_weights_scale = None
        self.v_out_scale = None
        self.attn_out_scale = None
        self.qk_out_scale = None

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))
        query = self.transpose_for_scores(self.query(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        query_qdq = qdq(query, self.q_out_scale)
        key_qdq = qdq(key.transpose(-1, -2), self.k_out_scale)
        attn_weights = torch.matmul(query_qdq, key_qdq) / math.sqrt(
            self.attention_head_size
        )

        # Normalize the attention scores to probabilities.
        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query.dtype)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        dropout = 0.0 if not self.training else self.dropout_prob
        attn_weights = torch.nn.functional.dropout(
            attn_weights, p=dropout, training=self.training
        )

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        value_qdq = qdq(value, self.v_out_scale)
        attn_weights_qdq = qdq(attn_weights, self.attn_weights_scale)
        attn_output = torch.matmul(attn_weights_qdq, value_qdq)
        attn_output = attn_output.transpose(1, 2).contiguous()

        new_context_layer_shape = attn_output.size()[:-2] + (self.all_head_size,)
        attn_output = attn_output.reshape(new_context_layer_shape)

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)

        return outputs


class MeasureSDPA(torch.nn.Module):
    def __init__(self, mod):
        super().__init__()
        self.__dict__.update(mod.__dict__)
        self.transpose_for_scores = mod.transpose_for_scores
        self.name = None

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))
        query = self.transpose_for_scores(self.query(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        if self.name + ".qk_matmul" not in input_maxes:
            input_maxes[self.name + ".qk_matmul"] = [
                torch.tensor(0) for i in range(2)
            ]  # [[], []]
        for idx, input in enumerate([query, key.transpose(-1, -2)]):
            input_maxes[self.name + ".qk_matmul"][idx] = torch.maximum(
                torch.max(torch.abs(input)).detach(),
                input_maxes[self.name + ".qk_matmul"][idx],
            )

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.name + ".qk_matmul_out" not in input_maxes:
            input_maxes[self.name + ".qk_matmul_out"] = [torch.tensor(0)]  # [[], []]
        input_maxes[self.name + ".qk_matmul_out"][0] = torch.maximum(
            torch.max(torch.abs(attn_weights)).detach(),
            input_maxes[self.name + ".qk_matmul_out"][0],
        )

        attn_weights = attn_weights / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query.dtype)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        dropout = 0.0 if not self.training else self.dropout_prob
        attn_weights = torch.nn.functional.dropout(
            attn_weights, p=dropout, training=self.training
        )

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        if self.name + ".v_matmul" not in input_maxes:
            input_maxes[self.name + ".v_matmul"] = [
                torch.tensor(0) for i in range(2)
            ]  # [[], []]
        for idx, input in enumerate([attn_weights, value]):
            input_maxes[self.name + ".v_matmul"][idx] = torch.maximum(
                torch.max(torch.abs(input)).detach(),
                input_maxes[self.name + ".v_matmul"][idx],
            )

        attn_output = torch.matmul(attn_weights, value)

        if self.name + ".v_matmul_out" not in input_maxes:
            input_maxes[self.name + ".v_matmul_out"] = [torch.tensor(0)]  # [[], []]
        input_maxes[self.name + ".v_matmul_out"][0] = torch.maximum(
            torch.max(torch.abs(attn_weights)).detach(),
            input_maxes[self.name + ".v_matmul_out"][0],
        )

        attn_output = attn_output.transpose(1, 2).contiguous()

        new_context_layer_shape = attn_output.size()[:-2] + (self.all_head_size,)
        attn_output = attn_output.reshape(new_context_layer_shape)

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)

        return outputs


def generate_model_info(model):
    mod_inst_info = namedtuple("ModInstInfo", ["name", "parent"])
    parent_child_mod_dict = {}

    def create_mod_info_recursion(parent):
        for name, mod in parent.named_children():
            parent_child_mod_dict[mod] = mod_inst_info(name=name, parent=parent)
            create_mod_info_recursion(mod)

    create_mod_info_recursion(model)
    return parent_child_mod_dict


def convert_fp8(model, config: str):
    with open(config, "r") as fp:
        data = json.load(fp)
    parent_child_mod_dict = generate_model_info(model)
    with torch.no_grad():
        for name, mod in model.named_modules():
            mod_type_str = mod.__class__.__name__
            if mod_type_str not in ["Linear", "ViTSelfAttention"]:
                continue
            if mod_type_str == "Linear":
                param = mod.weight
                xmax = torch.max(param)
                weight_scale = xmax / torch.finfo(torch.float8_e4m3fn).max
                q_param = torch.clamp(
                    (param / weight_scale),
                    torch.finfo(torch.float8_e4m3fn).min,
                    torch.finfo(torch.float8_e4m3fn).max,
                ).to(torch.float8_e4m3fn)

                scale = [i / torch.finfo(torch.float8_e4m3fn).max for i in data[name]]
                assert len(scale) == 1
                patched_mod = FP8QDQLinear(mod.in_features, mod.out_features)
                patched_mod.weight.data = q_param
                patched_mod.weight_scale = weight_scale.item()
                patched_mod.scale = scale[0]
                patched_mod.bias = mod.bias
            else:
                patched_mod = FP8QDQSDPA()
                patched_mod.__dict__.update(mod.__dict__)
                patched_mod.transpose_for_scores = mod.transpose_for_scores

                patched_mod.q_out_scale = (
                    data[name + ".qk_matmul"][0] / torch.finfo(torch.float8_e4m3fn).max
                )
                patched_mod.k_out_scale = (
                    data[name + ".qk_matmul"][1] / torch.finfo(torch.float8_e4m3fn).max
                )
                patched_mod.attn_weights_scale = (
                    data[name + ".v_matmul"][0] / torch.finfo(torch.float8_e4m3fn).max
                )
                patched_mod.v_out_scale = (
                    data[name + ".v_matmul"][1] / torch.finfo(torch.float8_e4m3fn).max
                )
                patched_mod.qk_out_scale = (
                    data[name + ".qk_matmul_out"][0]
                    / torch.finfo(torch.float8_e4m3fn).max
                )
                patched_mod.attn_out_scale = (
                    data[name + ".v_matmul_out"][0]
                    / torch.finfo(torch.float8_e4m3fn).max
                )

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
                input_maxes[name] = [
                    torch.tensor(0) for i in range(len(inputs))
                ]  # [[], []]
            input_maxes[name][idx] = torch.maximum(
                torch.max(torch.abs(input)).detach(), input_maxes[name][idx]
            )

    return save_input_hook


hook_handles = []


def prepare(model):
    parent_child_mod_dict = generate_model_info(model)
    with torch.no_grad():
        for name, mod in model.named_modules():
            mod_type_str = mod.__class__.__name__
            if mod_type_str not in ["ViTSelfAttention"]:
                continue
            new_module = MeasureSDPA(mod)
            new_module.name = name
            parent = parent_child_mod_dict[mod].parent
            setattr(parent, parent_child_mod_dict[mod].name, new_module)

    hook_modules = {}
    for n, module in model.named_modules():
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
    model = convert_fp8(model)
    return model
