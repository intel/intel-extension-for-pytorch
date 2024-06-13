import logging
import torch
import torch.nn as nn
import transformers
from collections import UserDict

format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=format_str)
logger = logging.getLogger("GPTQ")
logger.setLevel(logging.INFO)


def move_input_to_device(input, device=torch.device("cpu")):
    if isinstance(input, dict) or isinstance(input, UserDict):
        for inp in input.keys():
            input[inp] = (
                input[inp].to(device)
                if isinstance(input[inp], torch.Tensor)
                else input[inp]
            )
    elif isinstance(input, list) or isinstance(input, tuple):
        input_res, prev_size = [], None
        for inp in input:
            if prev_size:
                if isinstance(inp, torch.Tensor):
                    if inp.size() == prev_size:
                        input_res.append(inp.to(device))
                else:
                    if torch.tensor(inp).size == prev_size:
                        input_res.append(inp)
            else:
                input_res.append(
                    inp.to(device) if isinstance(inp, torch.Tensor) else inp
                )
            prev_size = torch.tensor(inp).size()
        input = input_res
    else:
        input = input.to(device)
    return input


def is_leaf(module):
    """Determine whether a module has no children modules.

    Args:
        module: torch.nn.Module

    Returns:
        a bool: whether a module has no children modules.
    """
    children_cnt = 0
    for n in module.children():
        children_cnt += 1
    return True if children_cnt == 0 else False


def trace_gptq_target_blocks(
    module, module_types=(torch.nn.ModuleList, torch.nn.Sequential)
):
    """Search transformer stacked structures, which is critical in LLMs and GPTQ execution.

    Args:
        module: torch.nn.Module
        module_types: List of torch.nn.Module.

    Returns:
        gptq_related_blocks = {
            "embeddings": {}, # Dict embedding layers before transformer stack module
            "transformers_pre": {}, # TODO
            "transformers_name": "", # Str, LLMs' transformer stack module name
            "transformers": # torch.nn.ModuleList. LLMs' transformer stack module
            "transformers": {}, # TODO
        }
    """
    if type(module).__name__ == "MixFormerSequentialForCausalLM":
        gptq_related_blocks = {
            "embeddings": {},
            "transformers_pre": {},
            "transformers_name": "",
            "transformers": [],
            "transformers_post": {},
        }
        for n, m in module.named_modules():
            if type(m) in module_types:
                gptq_related_blocks["transformers_name"] = n
                gptq_related_blocks["transformers"] = m
                break
            else:
                continue
        for n, m in gptq_related_blocks["transformers"][0].named_modules():
            if is_leaf(m):
                gptq_related_blocks["embeddings"][n] = m
        gptq_related_blocks["transformers"] = gptq_related_blocks["transformers"][1:-1]
    else:
        gptq_related_blocks = {
            "embeddings": {},
            "transformers_pre": {},
            "transformers_name": "",
            "transformers": [],
            "transformers_post": {},
        }
        for n, m in module.named_modules():
            if type(m) in module_types:
                gptq_related_blocks["transformers_name"] = n
                gptq_related_blocks["transformers"] = m
                return gptq_related_blocks
            else:
                if is_leaf(m):
                    gptq_related_blocks["embeddings"][n] = m
    return gptq_related_blocks


def find_layers(
    module, layers=(nn.Conv2d, nn.Conv1d, nn.Linear, transformers.Conv1D), name=""
):
    """Get all layers with target types."""
    for layer in layers:
        if isinstance(module, layer):
            return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def find_layers_name(
    module, layers=(nn.Conv2d, nn.Conv1d, nn.Linear, transformers.Conv1D), name=""
):
    """Get all layers with target types."""
    for layer in layers:
        if isinstance(module, layer):
            return [name]
    res = []
    for name1, child in module.named_children():
        res += find_layers_name(
            child, layers=layers, name=name + "." + name1 if name != "" else name1
        )
    return res


def log_quantizable_layers_per_transformer(
    transformer_blocks, layers=(nn.Conv2d, nn.Conv1d, nn.Linear, transformers.Conv1D)
):
    """Print all layers which will be quantized in GPTQ algorithm."""
    logger.info("* * Layer to be quantized * *")

    for block_id in range(len(transformer_blocks["transformers"])):
        transformer_block = transformer_blocks["transformers"][block_id]
        layers_for_this_tblock = find_layers_name(transformer_block)
        layer_names = [
            (
                transformer_blocks["transformers_name"]
                + "."
                + str(block_id)
                + "."
                + layer_name
            )
            for layer_name in layers_for_this_tblock
        ]
        for name in layer_names:
            logger.info(name)


def quantize(x, scale, zero, maxq):
    """Do quantization."""
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)
