import torch

# from torch.profiler import profile, ProfilerActivity
import intel_extension_for_pytorch  # noqa

from torch.testing._internal.common_utils import TestCase
import pytest

b = 16
n = 16
n_heads = 512
attn_head_size = 64


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(not torch.xpu.has_xetla(), reason="fallback is required")
    def test_sdp_backward_dropout_no_dropout(self, dtype=torch.bfloat16):
        query_states = torch.randn((b, n, n_heads, attn_head_size))
        key_states = torch.randn((b, n, n_heads, attn_head_size))
        value_states = torch.randn((b, n, n_heads, attn_head_size))
        bias = torch.randn((b, n, n_heads, n_heads))
        grad = torch.randn((b, n, n_heads, attn_head_size))

        query_states_xpu = query_states.bfloat16().xpu()
        key_states_xpu = key_states.bfloat16().xpu()
        value_states_xpu = value_states.bfloat16().xpu()
        bias_xpu = bias.bfloat16().xpu()
        grad_xpu = grad.bfloat16().xpu()

        query_states.requires_grad_(True)
        key_states.requires_grad_(True)
        value_states.requires_grad_(True)
        bias.requires_grad_(True)

        query_states_xpu.requires_grad_(True)
        key_states_xpu.requires_grad_(True)
        value_states_xpu.requires_grad_(True)
        bias_xpu.requires_grad_(True)

        r_cpu = torch.nn.functional.scaled_dot_product_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            attn_mask=bias,
        )
        r_cpu.backward(grad)

        r_xpu = torch.xpu.IpexSDP_dropout(
            query=query_states_xpu,
            key=key_states_xpu,
            value=value_states_xpu,
            attn_mask=bias_xpu,
        )
        r_xpu.backward(grad_xpu)

        self.assertEqual(
            query_states.grad, query_states_xpu.grad.cpu().float(), atol=1e-2, rtol=1e-2
        )
        self.assertEqual(
            key_states.grad, key_states_xpu.grad.cpu().float(), atol=1e-2, rtol=1e-2
        )
        self.assertEqual(
            value_states.grad, value_states_xpu.grad.cpu().float(), atol=1e-2, rtol=1e-2
        )

        self.assertEqual(bias.grad, bias_xpu.grad.cpu().float(), atol=1e-2, rtol=1e-1)

    @pytest.mark.skipif(not torch.xpu.has_xetla(), reason="fallback is required")
    def test_sdp_backward_dropout(self, dtype=torch.bfloat16):
        query_states = torch.randn((b, n, n_heads, attn_head_size))
        key_states = torch.randn((b, n, n_heads, attn_head_size))
        value_states = torch.randn((b, n, n_heads, attn_head_size))
        bias = torch.randn((b, n, n_heads, n_heads))
        grad = torch.randn((b, n, n_heads, attn_head_size))

        query_states_xpu = query_states.bfloat16().xpu()
        key_states_xpu = key_states.bfloat16().xpu()
        value_states_xpu = value_states.bfloat16().xpu()
        bias_xpu = bias.bfloat16().xpu()
        grad_xpu = grad.bfloat16().xpu()

        query_states.requires_grad_(True)
        key_states.requires_grad_(True)
        value_states.requires_grad_(True)
        bias.requires_grad_(True)

        query_states_xpu.requires_grad_(True)
        key_states_xpu.requires_grad_(True)
        value_states_xpu.requires_grad_(True)
        bias_xpu.requires_grad_(True)

        r_cpu = torch.nn.functional.scaled_dot_product_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            attn_mask=bias,
            dropout_p=0.1,
        )
        r_cpu.backward(grad)

        r_xpu = torch.xpu.IpexSDP_dropout(
            query=query_states_xpu,
            key=key_states_xpu,
            value=value_states_xpu,
            attn_mask=bias_xpu,
            dropout_p=0.1,
        )
        r_xpu.backward(grad_xpu)
