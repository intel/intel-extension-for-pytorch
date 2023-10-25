import torch
import torch.nn as nn
import re


def default_tp_fn(model):
    a_model = model

    def supported():
        unsupported = [
            "deberta",
            "flaubert",
            "fsmt",
            "gpt2",
            "led",
            "longformer",
            "xlm",
            "xlnet",
        ]
        model = str(a_model)
        key = re.search(r": (.*?)Model", model)
        if key is None:
            key = re.search(r": (.*?)Stack", model)
        if key is None:
            key = re.match(r"(.*?)Model", model)
        assert (
            key is not None
        ), "Not able to determine model policy automatically. Please provide policy."
        if key.group(1).lower() in unsupported:
            return False
        return True

    def block_extract(module):
        module_list = []
        for child in module.children():
            if isinstance(child, nn.ModuleList):
                for child_module in child.children():
                    if not module_list:
                        module_list = [child_module]
                    elif type(child_module) not in [type(elem) for elem in module_list]:
                        module_list = module_list + [child_module]
            else:
                module_list = module_list + block_extract(child)
        return module_list

    def get_layer(module, outer_name=""):
        layer_list = []
        for key, submodule in module._modules.items():
            if isinstance(submodule, nn.Linear):
                current_layer = [outer_name + "." + key]
                layer_list = layer_list + current_layer
            elif (
                isinstance(submodule, nn.LayerNorm)
                or key == "LayerNorm"
                or key == "layer_norm"
            ):
                layer_list = layer_list + ["ln"]
            else:
                layer_list = layer_list + get_layer(submodule, key)
        return layer_list

    def update_target_list(target_list, module, slicing_list):
        # new_slice_list = set()
        if len(target_list):
            for i, target in enumerate(target_list):
                if target[0] == type(module):
                    slicing_list = set(slicing_list + target[1])
                    target_list[i] = tuple([type(module), slicing_list])  # noqa
                    return target_list
        target_list.append(([type(module), slicing_list]))
        return target_list

    if not supported():
        print(
            "We are not support this model for tensor parallel, original model will"
            "be returned for next phase of optimization. If you want this model runing"
            "on tensor parallel, please provide your own rule"
        )
        return model
    block_list = block_extract(model)
    target_list = []
    layer_list = []
    slicing_list = []
    for module in block_list:
        layer_list = layer_list + get_layer(module)
    for i, layer in enumerate(layer_list):
        if layer == "ln" and layer_list[i - 1] != "ln":
            slicing_list = slicing_list + [layer_list[i - 1]]
        elif "out_proj" in layer:
            slicing_list = slicing_list + [layer]
        elif "o_proj" in layer:
            slicing_list = slicing_list + [layer]
        elif "down_proj" in layer:
            slicing_list = slicing_list + [layer]
        elif "attention.dense" in layer and "GPTNeoX" in str(model):
            slicing_list = slicing_list + [layer]
        elif "self_attention.dense" in layer and "falcon" in str(type(module)):
            slicing_list = slicing_list + [layer]

    layer_list = []
    if slicing_list != []:
        slicing_list = list(set(slicing_list))
        update_target_list(target_list, module, slicing_list)
        slicing_list = []
    assert len(target_list), "Can not adopt TensorParallel on This model"
    return target_list


class TensorSlicer:
    def __init__(self, tp_size, tp_group, slicing_rule=None, dtype=torch.float) -> None:
        self.picking_rule = default_tp_fn if slicing_rule is None else slicing_rule
        self.tp_size = tp_size
        self.tp_group = tp_group
        self.dtype = dtype
        self.slicing_target_list = None
        self.slicing_rule = {
            nn.Linear: self.linear_slicing_rules,
        }

    def linear_slicing_rules(self, module: nn.Linear, name):
        if getattr(module, "replaced", False) is True:
            return
        w_shape = module.weight.shape
        if name in self.slicing_target_list[0][1]:
            slicing_weight = torch.empty(
                (w_shape[0], w_shape[1] // self.tp_size),
                dtype=self.dtype,
                device="meta",
            )
            setattr(module, "replaced", True)  # noqa
            setattr(module, "all_reduce", True)  # noqa
            setattr(module, "tp_weight", slicing_weight)  # noqa
        else:
            slicing_weight = torch.empty(
                (w_shape[0] // self.tp_size, w_shape[1]),
                dtype=self.dtype,
                device="meta",
            )
            slicing_bias = torch.empty(
                (w_shape[0] // self.tp_size), dtype=self.dtype, device="meta"
            )
            setattr(module, "replaced", True)  # noqa
            setattr(module, "all_reduce", False)  # noqa
            setattr(module, "tp_weight", slicing_weight)  # noqa
            setattr(module, "tp_bias", slicing_bias)  # noqa
        return

    def embedding_slicing_rule(self, module: nn.Embedding, name):
        if getattr(module, "replaced", False) is True:
            return
        w_shape = module.weight.shape
        slicing_weight = torch.empty(
            (w_shape[0], w_shape[1] // self.tp_size), dtype=self.dtype, device="meta"
        )
        setattr(module, "replaced", True)  # noqa
        setattr(module, "tp_weight", slicing_weight)  # noqa
        return

    def slicing_model(self, model):
        if self.tp_size >= 2:
            self.slicing_target_list = self.picking_rule(model)
            self.slicing_model_recursive(model, "")
            setattr(model, "slicing", True)  # noqa
        else:
            print("tp size less than 2, tensor parallel will be disabled")
        return model

    def update_parallel_params(self, child):
        if getattr(child, "replaced", False) is True:
            return
        for param in [
            "n_heads",
            "inner_dim",
            "num_heads",
            "num_kv",
            "num_attention_heads",
            "num_attn_heads",
            "all_head_size",
            "embed_dim",
            "hidden_size",
            "num_key_value_heads",
        ]:
            if hasattr(child, param):
                val = getattr(child, param)
                assert (
                    val % self.tp_size == 0
                ), f"{param} ({val}) must be divisible by mp_size ({self.tp_size})"
                para_name = "tp_" + param
                setattr(child, para_name, val // self.tp_size)
        setattr(child, "replaced", True)  # noqa

    def slicing_model_recursive(self, module, prev_name=""):
        for name, child in module.named_children():
            # fullname = prev_name + "." + name if prev_name is not "" else name
            fullname = prev_name + "." + name
            if type(child) in self.slicing_rule.keys():
                self.slicing_rule[type(child)](child, fullname)
            else:
                # self.update_parallel_params(child)
                self.slicing_model_recursive(child, name)
        return
