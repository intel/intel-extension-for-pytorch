import os
import sys

import torch


def override_size_asserts():
    """
    We have to mute assert_size_stride here because conv1D in IPEX has the
    support feature of the channel last. But conv1D in PyTorch has no support
    for channel last, resulting in a stride size assertion. So we have to turn
    off `TORCHINDUCTOR_SIZE_ASSERTS`.
    """
    if "torch._inductor.config" not in sys.modules:
        os.environ["TORCHINDUCTOR_SIZE_ASSERTS"] = "0"
        return

    from torch._inductor import config

    config.size_asserts = False

    if "torch._inductor.ir" not in sys.modules:
        torch._inductor.config = config
    else:
        torch._inductor.ir.config = config
