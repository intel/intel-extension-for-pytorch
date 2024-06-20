import builtins
import contextlib
from typing import List, Optional, Union, Dict
from unittest.mock import patch
import torch
from .decomposition import get_decompositions
from .lowering import patch_lowering
from torch._inductor.compile_fx import compile_fx_inner
from .ipex_fusion import _ipex_fusion_passes


def ipex_compile_fx_inner(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    cudagraphs=None,
    static_input_idxs=0,
    is_backward=False,
    graph_id=None,
    cpp_wrapper=False,
    aot_mode=False,
    is_inference=False,
    boxed_forward_device_index=None,
    user_visible_outputs=frozenset(),
    layout_opt=None,
):
    _ipex_fusion_passes(gm)
    return compile_fx_inner(
        gm,
        example_inputs,
        cudagraphs=cudagraphs,
        static_input_idxs=static_input_idxs,
        is_backward=is_backward,
        graph_id=graph_id,
        cpp_wrapper=cpp_wrapper,
        aot_mode=aot_mode,
        is_inference=is_inference,
        boxed_forward_device_index=boxed_forward_device_index,
        user_visible_outputs=user_visible_outputs,
        layout_opt=layout_opt,
    )


@contextlib.contextmanager
def patch_codegen():
    from torch._inductor.scheduler import Scheduler
    from .codegen.cpp import IpexCppScheduling

    def get_backend(scheduler, device):
        # TODO(jgong5): support xpu
        if device.type == "cpu":
            if device not in scheduler.backends or not isinstance(
                scheduler.backends[device], IpexCppScheduling
            ):
                scheduler.backends[device] = IpexCppScheduling(scheduler)
        else:
            if device not in scheduler.backends:
                scheduler.backends[device] = scheduler.create_backend(device)
        return scheduler.backends[device]

    with patch.object(Scheduler, "get_backend", get_backend):
        yield


@contextlib.contextmanager
def patch_functions():
    """
    On-the-fly patch:
    1. lowering registration
    2. codegen backends
    """
    with patch_lowering(), patch_codegen():
        yield


def compile_fx(
    model: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    mode: Union[str, None] = None,
    options: Optional[Dict[str, Union[str, builtins.int, builtins.bool]]] = None,
):
    from torch._inductor.compile_fx import compile_fx as inductor_compile

    with patch_functions():
        return inductor_compile(
            model,
            example_inputs,
            inner_compile=ipex_compile_fx_inner,
            decompositions=get_decompositions(),
        )
