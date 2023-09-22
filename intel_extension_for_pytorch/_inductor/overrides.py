import sys

import torch  # noqa


def override_decode_device():
    from .lowering import decode_device
    if not "torch._inductor.utils" in sys.modules:
        import torch._inductor.utils
        torch._inductor.utils.decode_device = decode_device
    else:
        import torch._inductor.lowering
        torch._inductor.lowering.decode_device = devoce_device
    return decode_device
