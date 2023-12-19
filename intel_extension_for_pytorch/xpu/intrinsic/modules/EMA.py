from copy import deepcopy
from torch import nn

import torch  # noqa
import intel_extension_for_pytorch  # noqa
import math


def is_parallel(model):
    return type(model) in (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include[...] and to exclude[...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


class EMA:
    """Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    To disable EMA set the `enabled` attribute to `False`.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, updates: float = 0):
        if not 0.0 <= decay:
            raise ValueError("Invalid decay value: {}".format(decay))

        self.ema = deepcopy(
            model.module if is_parallel(model) else model
        ).eval()  # FP32 EMA
        self.decay = lambda x: decay * (
            1 - math.exp(-x / 2000)
        )  # decay exponential ramp (to help early epochs) #decay
        self.updates = updates
        self.enabled = True
        self.updated_ema = list(self.ema.state_dict().values())

    def update(self, model):
        model_inputs = list(model.state_dict().values())
        with torch.no_grad():
            self.updates += 1
            decy = self.decay(self.updates)
            d = torch.tensor([decy]).to("xpu")

        torch.ops.torch_ipex.ema_fused_step(model_inputs, self.updated_ema, d)
        return self.ema.state_dict()

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        """Updates attributes and saves stripped model with optimizer removed."""
        if self.enabled:
            copy_attr(self.ema, model, include, exclude)
