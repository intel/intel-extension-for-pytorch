import torch
import intel_extension_for_pytorch  # noqa F401
from torch.autograd import Function


class InteractionFuncion(Function):
    @staticmethod
    def forward(ctx, input_mlp, input_emb):
        return torch.ops.torch_ipex.interaction(input_mlp, input_emb)


Interaction = InteractionFuncion.apply
