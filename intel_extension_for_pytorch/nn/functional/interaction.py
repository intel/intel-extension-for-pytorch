import torch
from torch.autograd import Function

def interaction(*args):
    # Current pytorch dose not support vector<Tensor> input for c++ custom function
    # So we preserve python custom function while need backward
    # Since python custom function will meet GIL when run multi-thread in one process
    # We will drop python custom function after c++ are supported
    if torch.is_grad_enabled():
        return InteractionFunc.apply(*args)
    return torch.ops.torch_ipex.interaction_forward(args)

class InteractionFunc(Function):
    @staticmethod
    def forward(ctx, *args):
        ctx.save_for_backward(*args)
        output = torch.ops.torch_ipex.interaction_forward(args)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        args = ctx.saved_tensors
        grad_in = torch.ops.torch_ipex.interaction_backward(grad_out.contiguous(), args)
        return tuple(grad_in)

