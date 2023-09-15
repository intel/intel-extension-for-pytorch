import torch
from torch._inductor.fx_passes.post_grad import register_lowering_pattern
from torch._inductor.pattern_matcher import Arg, CallFunction, PatternMatcherPass, Match
from torch._inductor.lowering import lowerings as L

patterns = PatternMatcherPass()


@register_lowering_pattern(
    CallFunction(
        torch.ops.aten.add, CallFunction(torch.ops.aten.bmm, Arg(), Arg()), Arg()
    ),
)
def bmm_add(match: Match, mat1, mat2, mat3):
    return L[torch.ops.torch_ipex.bmm_add](mat3, mat1, mat2, 1.0)


def _ipex_fusion_passes(gm: torch.fx.GraphModule):
    patterns.apply(gm.graph)
    gm.graph.lint()
    gm.recompile()
