import torch
import torch.nn as nn
import intel_extension_for_pytorch  # noqa
from torch.testing._internal.common_utils import TestCase


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class TestNNMethod(TestCase):
    def test_fast_layer_norm_float(self, dtype=torch.float):
        batch, sentence_length, embedding_dim = 16, 128, 1024
        embedding = torch.randn(batch, sentence_length, embedding_dim)
        embedding_xpu = embedding.to("xpu")
        layer_norm = nn.LayerNorm(embedding_dim)
        ref = layer_norm(embedding)
        fast_ln = torch.ops.torch_ipex.fast_layer_norm(
            embedding_xpu,
            layer_norm.normalized_shape,
            layer_norm.weight.to("xpu"),
            layer_norm.bias.to("xpu"),
            layer_norm.eps,
        )
        self.assertEqual(ref, fast_ln.cpu())

    def test_fast_layer_norm_half(self, dtype=torch.half):
        batch, sentence_length, embedding_dim = 16, 128, 1024
        embedding = torch.randn(batch, sentence_length, embedding_dim)
        embedding_xpu = embedding.to(dtype).to("xpu")
        layer_norm = nn.LayerNorm(embedding_dim)
        ref = layer_norm(embedding)
        fast_ln = torch.ops.torch_ipex.fast_layer_norm(
            embedding_xpu,
            layer_norm.normalized_shape,
            layer_norm.weight.to(dtype).to("xpu"),
            layer_norm.bias.to(dtype).to("xpu"),
            layer_norm.eps,
        )
        self.assertEqual(ref, fast_ln.float().cpu(), atol=1e-3, rtol=1e-3)

    def test_rms_norm_float(self, dtype=torch.float):
        batch, sentence_length, embedding_dim = 16, 128, 1024
        hidden_states = torch.randn(batch, sentence_length, embedding_dim)
        hidden_states_xpu = hidden_states.to("xpu")
        RMS = LlamaRMSNorm(embedding_dim)
        ref = RMS(hidden_states)
        hsz = hidden_states.shape[-1]
        fast_rms = torch.ops.torch_ipex.fast_rms_norm(
            hidden_states_xpu, [hsz], RMS.weight.to("xpu"), None, RMS.variance_epsilon
        )
        self.assertEqual(ref, fast_rms.cpu())

    def test_rms_norm_half(self, dtype=torch.half):
        batch, sentence_length, embedding_dim = 16, 128, 1024
        hidden_states = torch.randn(batch, sentence_length, embedding_dim).to(dtype)
        hidden_states_xpu = hidden_states.to("xpu")
        RMS = LlamaRMSNorm(embedding_dim).to(dtype)
        ref = RMS(hidden_states)
        hsz = hidden_states.shape[-1]
        fast_rms = torch.ops.torch_ipex.fast_rms_norm(
            hidden_states_xpu, [hsz], RMS.weight.to("xpu"), None, RMS.variance_epsilon
        )
        self.assertEqual(ref, fast_rms.cpu())
