import unittest
import torch
from torch import nn
from common_utils import TestCase


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


class RMSNormTester(TestCase):
    def test_RMSNorm(self):
        for dim in [2, 3, 4, 5]:
            with torch.cpu.amp.autocast(), torch.no_grad():
                input_size = [
                    3,
                ]
                for _ in range(dim - 1):
                    input_size.append(10)
                x = torch.randn(input_size)
                # RMSNorm input is fp32
                model = RMSNorm(input_size).eval()
                trace_model = torch.jit.trace(model, x)
                y1_fp32 = model(x)
                y2_fp32 = trace_model(x)
                rmsnorm_graph = trace_model.graph_for(x)
                self.assertEqual(y1_fp32.dtype, torch.float32)
                self.assertEqual(y2_fp32.dtype, torch.float32)
                self.assertEqual(y1_fp32, y2_fp32)
                self.assertTrue(
                    any(n.kind() == "ipex::RMSNorm" for n in rmsnorm_graph.nodes())
                )


if __name__ == "__main__":
    test = unittest.main()
