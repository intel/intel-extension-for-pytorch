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
        grad_in = core.interaction_backward(grad_out, args)
        # FIXME: Change to dynamic unpack the return values
        return (grad_in[0], grad_in[1], grad_in[2], grad_in[3], grad_in[4], grad_in[5], grad_in[6], grad_in[7],
        grad_in[8], grad_in[9], grad_in[10], grad_in[11], grad_in[12], grad_in[13], grad_in[14], grad_in[15],
        grad_in[16], grad_in[17], grad_in[18], grad_in[19], grad_in[20], grad_in[21], grad_in[22], grad_in[23],
        grad_in[24], grad_in[25], grad_in[26])
