import torch
import torch.nn as nn


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
        output = torch.ops.torch_ipex.rms_norm(
            hidden_states, [self.hidden_size], self.weight, self.variance_epsilon
        )

        """
        # Reference path in huggingface
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
        """

        return output[0]


# QWen trying to import rms_norm from flash_attention
# from flash_attn.ops.rms_norm import rms_norm as __rms_norm
rms_norm = None


class QWenRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.hidden_size = dim
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # if rms_norm is not None and x.is_cuda:
        #     return rms_norm(x, self.weight, self.eps)
        # else:
        #     output = self._norm(x.float()).type_as(x)
        #     return output * self.weight
        output = torch.ops.torch_ipex.rms_norm(
            x, [self.hidden_size], self.weight, self.eps
        )
        return output[0]
