import torch
import torch.nn as nn
from ._transformer_configuration import IPEXTransformerConfig

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.hidden_size = hidden_size
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor):
        #hidden_states = torch.ops.torch_ipex.fast_rms_norm(hidden_states, [self.hidden_size], self.weight, None, self.variance_epsilon)
        #return hidden_states
        # llama-7b beam=4 has accuracy issue with fast_rms_norm
        # here we use the original rms_norm instead
        output = torch.ops.torch_ipex.rms_norm(hidden_states, [self.hidden_size], self.weight, self.variance_epsilon)
        return output[0]
        
        '''
        # Reference path in huggingface
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
        '''

