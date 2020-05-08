import torch
from torch import nn
from torch.autograd import Function
import _torch_ipex as core

def interaction(*args):
    return InteractionFunc.apply(*args)

class InteractionFunc(Function):
    @staticmethod
    def forward(ctx, *args):
        ctx.save_for_backward(*args)
        output = core.interaction_forward(args)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        args = ctx.saved_tensors
        grad_in = core.interaction_backward(grad_out.contiguous(), args)
        return tuple(grad_in)
