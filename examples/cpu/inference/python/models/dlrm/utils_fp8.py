import torch
import torchao  # noqa: F401
import os

quantize_affine_float8_non_decomposed = (
    torch.ops.torchao.quantize_affine_float8_non_decomposed.default
)
dequantize_affine_float8_non_decomposed = (
    torch.ops.torchao.dequantize_affine_float8_non_decomposed.default
)


def inc_convert(model, dtype):
    def dequantize_per_tensor(
        tensor: torch.Tensor, scale: float, output_dtype: torch.dtype
    ) -> torch.Tensor:
        res = dequantize_affine_float8_non_decomposed(
            tensor=tensor, scale=torch.tensor([scale]), output_dtype=torch.float
        )
        return res

    def quantize_per_tensor(
        tensor: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        return quantize_affine_float8_non_decomposed(
            tensor=tensor,
            scale=torch.tensor([scale]),
            float8_dtype=torch.float8_e4m3fn,
        )

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

    class FP8QDQEmbeddingBag(torch.nn.Module):
        def __init__(
            self,
            weight_shape,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            mode,
            sparse,
            include_last_offset,
            padding_idx,
        ):
            super().__init__()
            # self.mod = mod
            self.max_norm = max_norm
            self.norm_type = norm_type
            self.scale_grad_by_freq = scale_grad_by_freq
            self.mode = mode
            self.sparse = sparse
            self.include_last_offset = include_last_offset
            self.padding_idx = padding_idx
            self.weight = torch.empty(weight_shape)
            self.weight_scale = None

        def forward(
            self,
            input,
            offsets=None,
            per_sample_weights=None,
        ):
            return torch.ops.torchao._scaled_embedding_bag(
                self.weight.data,
                input,
                offsets,
                torch.tensor([self.weight_scale]),
                1.0,
                0,
                True,
                torch.float,
            )

    import json
    from collections import namedtuple

    def generate_model_info(model):
        mod_inst_info = namedtuple("ModInstInfo", ["name", "parent"])
        parent_child_mod_dict = {}

        def create_mod_info_recursion(parent):
            for name, mod in parent.named_children():
                parent_child_mod_dict[mod] = mod_inst_info(name=name, parent=parent)
                create_mod_info_recursion(mod)

        create_mod_info_recursion(model)
        return parent_child_mod_dict

    parent_child_mod_dict = generate_model_info(model)

    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "fp8_data.json"), "r"
    ) as fp:
        data = json.load(fp)
    with torch.no_grad():
        for name, mod in model.model.named_modules():
            mod_type_str = mod.__class__.__name__
            if mod_type_str not in [
                "Linear",
                "EmbeddingBag",
            ]:
                continue
            if name not in data:
                continue
            print(mod_type_str, name)
            param = mod.weight
            if mod_type_str == "Linear":
                xmax = torch.max(param)
            else:
                xmax = torch.max(torch.abs(param))
            weight_scale = xmax / torch.finfo(torch.float8_e4m3fn).max
            mod.weight_scale = weight_scale
            q_param = torch.clamp(
                (param / weight_scale),
                torch.finfo(torch.float8_e4m3fn).min,
                torch.finfo(torch.float8_e4m3fn).max,
            ).to(torch.float8_e4m3fn)
            mod.weight.data = q_param
            if mod_type_str in ["Linear"]:
                scale = [i / torch.finfo(torch.float8_e4m3fn).max for i in data[name]]
                assert len(scale) == 1
                patched_mod = FP8QDQLinear(mod.in_features, mod.out_features)
                patched_mod.bias = mod.bias
                patched_mod.scale = scale[0]
                patched_mod.weight_scale = weight_scale.item()
                patched_mod.weight.data = q_param
            else:
                patched_mod = FP8QDQEmbeddingBag(
                    weight_shape=list(mod.weight.shape),
                    max_norm=mod.max_norm,
                    norm_type=mod.norm_type,
                    scale_grad_by_freq=mod.scale_grad_by_freq,
                    mode=mod.mode,
                    sparse=mod.sparse,
                    include_last_offset=mod.include_last_offset,
                    padding_idx=mod.padding_idx,
                )
                patched_mod.weight_scale = weight_scale.item()
                patched_mod.weight.data = q_param

            parent = parent_child_mod_dict[mod].parent
            name = parent_child_mod_dict[mod].name
            setattr(parent, name, patched_mod)
