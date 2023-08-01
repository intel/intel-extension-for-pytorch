import torch
import torch.nn as nn
from ._transformer_configuration import IPEXTransformerConfig

class LlamaRMSNorm(nn.Module):
    def __init__(self,
                 config: IPEXTransformerConfig):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.embed_dim))
        self.variance_epsilon = config.norm_eps

    def forward(self, hidden_states: torch.Tensor):
        hsz = hidden_states.shape[-1]
        hidden_states = torch.ops.torch_ipex.fast_rms_norm(hidden_states, [hsz], self.weight, None, self.variance_epsilon)
        #output = torch.ops.torch_ipex.rms_norm(hidden_states, [hsz], self.weight)
        #return output[0]
        return hidden_states
        '''
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states
        '''

