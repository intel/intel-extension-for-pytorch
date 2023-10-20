import torch
from torch.autograd import Function


def interaction(*args):
    r"""
    Get the interaction feature beyond different kinds of features (like gender
    or hobbies), used in DLRM model.

    For now, we only optimized "dot" interaction at `DLRM Github repo
    <https://github.com/facebookresearch/dlrm/blob/main/dlrm_s_pytorch.py#L475-L495>`_.
    Through this, we use the dot product to represent the interaction feature
    between two features.

    For example, if feature 1 is "Man" which is represented by [0.1, 0.2, 0.3],
    and feature 2 is "Like play football" which is represented by [-0.1, 0.3, 0.2].

    The dot interaction feature is
    ([0.1, 0.2, 0.3] * [-0.1, 0.3, 0.2]^T) =  -0.1 + 0.6 + 0.6 = 1.1

    Args:
        *args: Multiple tensors which represent different features.
            Input shape: ``N * (B, D)``, where N is the number of different kinds of features,
            B is the batch size, D is feature size.
            Output shape: ``(B, D + N * ( N - 1 ) / 2)``.
    """

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
