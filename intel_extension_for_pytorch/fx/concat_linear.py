import torch
import torch.nn as nn
import torch.fx as fx
import torch.fx.experimental.optimization as optimization
import _operator
import copy
from ..utils._logger import logger, WarningType


def concat_linear(model: fx.GraphModule, inplace=False) -> fx.GraphModule:
    def concat(compatible_layers, modules):
        if len(compatible_layers) < 2:
            return
        base_linear = modules[compatible_layers[0].target]
        input_channel = base_linear.weight.shape[1]
        dtype = base_linear.weight.dtype
        device = base_linear.weight.device
        with_bias = base_linear.bias is not None
        weights = [modules[layer.target].weight for layer in compatible_layers]
        output_channels = [w.shape[0] for w in weights]
        output_channel = sum(output_channels)
        concated_weights = torch.concat(weights, dim=0)
        if base_linear.bias is not None:
            bias = [modules[layer.target].bias for layer in compatible_layers]
            concated_bias = torch.concat(bias, dim=0)
        concat_linear_ = nn.Linear(
            input_channel, output_channel, with_bias, device, dtype
        )
        concat_linear_.weight = torch.nn.Parameter(
            concated_weights, weights[0].requires_grad
        )
        if with_bias:
            concat_linear_.bias = torch.nn.Parameter(
                concated_bias, bias[0].requires_grad
            )
        return concat_linear_, output_channels

    def collectLinearNodes(graph: fx.graph.Graph, modules: list):
        grouped_linear_nodes = {}
        linear_inputs = []
        for node in graph.nodes:
            if node.target not in modules:
                continue
            if type(modules[node.target]) != nn.Linear:
                continue
            linear_input = node.args[0]
            if linear_input not in grouped_linear_nodes:
                grouped_linear_nodes[linear_input] = [node]
                linear_inputs.append(linear_input)
            else:
                grouped_linear_nodes[linear_input].append(node)
        return grouped_linear_nodes, linear_inputs

    def canConcatLinear(base_linear, other_linear):
        def check_compatible(base_tensor, other_tensor):
            if base_tensor is None:
                return other_tensor is None
            if base_tensor.device != other_tensor.device:
                return False
            if base_tensor.dtype != other_tensor.dtype:
                return False
            if base_tensor.requires_grad != other_tensor.requires_grad:
                return False
            if base_tensor.dim() != other_tensor.dim():
                return False
            if base_tensor.dim() > 1:
                if base_tensor.shape[1:] != other_tensor.shape[1:]:
                    return False
            return True

        return check_compatible(
            base_linear.weight, other_linear.weight
        ) and check_compatible(base_linear.bias, other_linear.bias)

    def concatLinearNodes(
        grouped_linear_nodes: dict,
        linear_inputs: list,
        modules: list,
        graph: fx.graph.Graph,
    ):
        if len(linear_inputs) == 0:
            return
        for linear_input in linear_inputs:
            linear_nodes = grouped_linear_nodes[linear_input]
            if len(linear_nodes) < 2:
                continue
            base_node = linear_nodes[0]
            compatible_layers = [base_node]
            for other_node in linear_nodes[1:]:
                if canConcatLinear(
                    modules[base_node.target], modules[other_node.target]
                ):
                    compatible_layers.append(other_node)

            concated_linear_, output_channels = concat(compatible_layers, modules)
            with graph.inserting_after(base_node):
                split = graph.call_function(
                    torch.split, (base_node, output_channels), {"dim": -1}
                )
                with graph.inserting_after(split):
                    getitem_fn = _operator.getitem
                    getitems = [
                        graph.call_function(getitem_fn, (split, i))
                        for i in range(len(output_channels))
                    ]
                    for node, getitem_node in zip(compatible_layers, getitems):
                        node.replace_all_uses_with(getitem_node)
                        if node is not base_node:
                            graph.erase_node(node)
            split.update_arg(0, base_node)
            optimization.replace_node_module(base_node, modules, concated_linear_)

    _model: fx.GraphModule = model
    if not inplace:
        _model = copy.deepcopy(model)
    modules = dict(_model.named_modules())
    _graph: fx.graph.Graph = _model.graph
    if not inplace:
        _graph = copy.deepcopy(_graph)
    grouped_linear_nodes, linear_inputs = collectLinearNodes(_graph, modules)
    concatLinearNodes(grouped_linear_nodes, linear_inputs, modules, _graph)
    del grouped_linear_nodes
    del linear_inputs
    return fx.GraphModule(_model, _graph)


def _concat_linear(model: torch.nn.Module, inplace=False) -> fx.GraphModule:
    # if native symbolic trace failed, try transformer symbolic trace
    import sys

    if "diffusers" in sys.modules:
        diffusers = sys.modules["diffusers"]
        import torch._dynamo as dynamo

        def apply_concat_linear_on_unet(unet):
            def prepare_input_for_attn(BasicTransformerBlock):
                in1 = BasicTransformerBlock.attn1.to_q.in_features
                in2 = BasicTransformerBlock.attn2.to_v.in_features
                # The first dimension of hd/ehd (2) is related to user given batch size
                # The second dimension of hd (4096, 1024, 256) is related to user ginve h, w
                # The second dimension of ehd (77) is max-seq-lenght from text-encoder
                # All dimensions above cannot be got from unet model
                # We can hardcode this because the guards of dynamo export do not require
                # Concrete shapes on these dimensions with hd and ehd
                ehd = torch.rand(2, 77, in2)
                hd_dict = {
                    320: (2, 4096, 320),
                    640: (2, 1024, 640),
                    1280: (2, 256, 1280),
                }
                hd_shape = hd_dict[in1]
                hd = torch.rand(hd_shape)
                return hd, ehd

            def apply_concat_linear_on_BasicTransformerBlock(BasicTransformerBlock):
                if isinstance(
                    BasicTransformerBlock,
                    diffusers.models.attention.BasicTransformerBlock,
                ):
                    hd, ehd = prepare_input_for_attn(BasicTransformerBlock)
                    inputs1 = {
                        "hidden_states": hd,
                        "encoder_hidden_states": None,
                        "attention_mask": None,
                    }
                    inputs2 = {
                        "hidden_states": hd,
                        "encoder_hidden_states": ehd,
                        "attention_mask": None,
                    }
                    gm = dynamo.export(BasicTransformerBlock.attn1, **inputs1)[0]
                    concat_gm1 = concat_linear(gm)
                    BasicTransformerBlock.attn1 = concat_gm1
                    gm = dynamo.export(BasicTransformerBlock.attn2, **inputs2)[0]
                    concat_gm2 = concat_linear(gm)
                    BasicTransformerBlock.attn2 = concat_gm2
                return

            for child in unet.children():
                if isinstance(child, diffusers.models.attention.BasicTransformerBlock):
                    apply_concat_linear_on_BasicTransformerBlock(child)
                apply_concat_linear_on_unet(child)

        try:
            unet = diffusers.models.unet_2d_condition.UNet2DConditionModel
            if isinstance(model, unet):
                apply_concat_linear_on_unet(model)
                return model
        except BaseException:
            logger.warning(
                "failed to apply concat_linear on unet, please report bugs",
                _type=WarningType.NotSupported,
            )

    if "transformers" in sys.modules:

        def is_transfomer_model(model):
            name = model.__class__.__module__
            return name.startswith("transformers.models.")

        if is_transfomer_model(model):
            try:
                from transformers.utils.fx import symbolic_trace as hf_symbolic_trace
            except ImportError:
                # fx are not exposed in transformers.utils
                logger.warning(
                    "failed to import transformers symbolic_trace, cannnot apply concat linear",
                    _type=WarningType.NotSupported,
                )
            try:
                model: fx.GraphModule = hf_symbolic_trace(
                    model, input_names=["input_ids", "attention_mask", "token_type_ids"]
                )
                return concat_linear(model, inplace)
            except BaseException:
                logger.warning(
                    "failed to symbolic trace model with transformers symbolic_trace, cannnot apply concat linear",
                    _type=WarningType.NotSupported,
                )
    else:
        try:
            model: fx.GraphModule = fx.symbolic_trace(model)
            return concat_linear(model, inplace)
        except BaseException:
            logger.warning(
                "pytorch native symbolic trace failed, may cannnot apply concat linear",
                _type=WarningType.NotSupported,
            )
    return model
