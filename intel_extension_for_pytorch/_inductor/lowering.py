# Custom lowerings overriding those from PyTorch

import contextlib
import functools
from torch._inductor.lowering import ELEMENTWISE_TYPE_PROMOTION_KIND

lowering_overrides = {}


def _register_lowering(
    aten_fn,
    decomp_fn,
    broadcast=False,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    convert_input_to_bool=False,
):
    if not isinstance(aten_fn, (list, tuple)):
        aten_fn = [aten_fn]
    else:
        aten_fn = list(aten_fn)
    for fn in aten_fn:
        lowering_overrides.update({fn: (decomp_fn, broadcast, type_promotion_kind, convert_input_to_bool)})


def register_lowering(
    aten_fn,
    broadcast=False,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    convert_input_to_bool=False,
):
    return functools.partial(
        _register_lowering,
        aten_fn,
        broadcast=broadcast,
        type_promotion_kind=type_promotion_kind,
        convert_input_to_bool=convert_input_to_bool,
    )


@contextlib.contextmanager
def patch_lowering():
    import copy
    from torch._inductor.lowering import lowerings
    from torch._inductor.lowering import register_lowering

    old_lowerings = lowerings
    lowerings = copy.copy(lowerings)
    for fn, (decomp_fn, broadcast, type_promotion_kind, convert_input_to_bool) in lowering_overrides.items():
        register_lowering(fn, broadcast=broadcast, type_promotion_kind=type_promotion_kind, convert_input_to_bool=convert_input_to_bool)(decomp_fn)
    yield
    lowerings = old_lowerings