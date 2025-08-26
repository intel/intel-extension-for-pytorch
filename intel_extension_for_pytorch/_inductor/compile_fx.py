import builtins
import contextlib
from typing import List, Optional, Union, Dict, Sequence
from typing_extensions import Unpack
from unittest.mock import patch
import torch
from .decomposition import get_decompositions
from .lowering import patch_lowering
from torch._inductor.compile_fx import compile_fx_inner, _CompileFxKwargs
from torch._inductor.utils import InputType
from .ipex_fusion import _ipex_fusion_passes


def ipex_compile_fx_inner(
    gm: torch.fx.GraphModule,
    example_inputs: Sequence[InputType],
    **kwargs: Unpack[_CompileFxKwargs],
):
    _ipex_fusion_passes(gm)
    return compile_fx_inner(
        gm,
        example_inputs,
        **kwargs,
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
