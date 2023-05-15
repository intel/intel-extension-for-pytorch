import torch.nn as nn
import torch.fx as fx
import torch.fx.experimental.optimization as optimization
from torch.nn.utils.fusion import fuse_linear_bn_eval
import copy


def linear_bn_fuse(model: nn.Module, inplace=False) -> nn.Module:
    # implementation follows https://github.com/pytorch/pytorch/blob/master/torch/fx/experimental/optimization.py#L50
    patterns = [
        (nn.Linear, nn.BatchNorm1d),
        (nn.Linear, nn.BatchNorm2d),
        (nn.Linear, nn.BatchNorm3d),
    ]

    if not inplace:
        model = copy.deepcopy(model)
    fx_model = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())
    new_graph = copy.deepcopy(fx_model.graph)

    for pattern in patterns:
        for node in new_graph.nodes:
            if optimization.matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:
                    continue
                linear = modules[node.args[0].target]
                bn = modules[node.target]
                if not bn.track_running_stats:
                    continue
                fused_linear = fuse_linear_bn_eval(linear, bn)
                optimization.replace_node_module(node.args[0], modules, fused_linear)
                node.replace_all_uses_with(node.args[0])
                new_graph.erase_node(node)

    return fx.GraphModule(fx_model, new_graph)
