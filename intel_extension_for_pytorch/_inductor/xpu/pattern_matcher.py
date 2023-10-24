import torch
import torch._inductor.pattern_matcher as pattern_matcher
from torch._inductor import config
from torch._inductor.pattern_matcher import PatternMatcherPass


class _PrecompiledPatternMatcherPass(PatternMatcherPass):
    def __init__(self):
        super().__init__()

    def __call__(self, g: torch.fx.graph.Graph):
        self.apply(g)


_precompiled_pattern_matcher_post_grad_pre_pass = _PrecompiledPatternMatcherPass()
config.post_grad_custom_pre_pass = _precompiled_pattern_matcher_post_grad_pre_pass


_precompiled_pattern_matcher_post_grad_post_pass = _PrecompiledPatternMatcherPass()
config.post_grad_custom_post_pass = _precompiled_pattern_matcher_post_grad_post_pass


"""
import torch
from torch._inductor.lowering import lowerings as L
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.pattern_matcher import _register_lowering_pattern_post_grad_pre_pass

@_register_lowering_pattern_post_grad_pre_pass(..)
def your_replacement(match, *args, **kwargs):
    computation_args = pack_your_args(args)
    return L[your_replacement_symbol](*computation_args)
"""


def _no_extra_check(m):
    return True


def _register_lowering_pattern_post_grad_pre_pass(pattern, extra_check=_no_extra_check):
    return pattern_matcher.register_lowering_pattern(
        pattern, extra_check, pass_dict=config.post_grad_custom_pre_pass
    )


def _register_lowering_pattern_post_grad_post_pass(
    pattern, extra_check=_no_extra_check
):
    return pattern_matcher.register_lowering_pattern(
        pattern, extra_check, pass_dict=config.post_grad_custom_post_pass
    )
