import intel_extension_for_pytorch
from torch.autograd import Function


# TODO: XPU: no support bwd
class InteractionFuncion(Function):
    @staticmethod
    def forward(ctx, input_mlp, input_emb):
        return intel_extension_for_pytorch._C.interaction(input_mlp, input_emb)


Interaction = InteractionFuncion.apply
