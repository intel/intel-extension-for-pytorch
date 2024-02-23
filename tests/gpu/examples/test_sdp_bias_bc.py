import torch
import intel_extension_for_pytorch  # noqa

from torch.testing._internal.common_utils import TestCase
import pytest

# torch.set_printoptions(profile="full")


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(not torch.xpu.has_xetla(), reason="fallback is required")
    def test_sdp_bias_bc(self, dtype=torch.bfloat16):
        b = 2
        n = 2
        attn_head_size = 64
        for seq_lenth in [77, 80]:
            q_len = seq_lenth
            kv_len = seq_lenth
            for bias_size in (
                [b, 1, q_len, kv_len],
                [b, n, 1, kv_len],
                [1, n, q_len, kv_len],
                [1, 1, q_len, kv_len],
                [b, 1, 1, kv_len],
            ):
                print("seq_lenth is ", seq_lenth)
                print("bias_size", bias_size)
                query_states = torch.rand((b, n, q_len, attn_head_size))
                key_states = torch.rand((b, n, kv_len, attn_head_size))
                value_states = torch.rand((b, n, kv_len, attn_head_size))
                grad = torch.rand((b, n, q_len, attn_head_size))
                bias = torch.rand(bias_size)

                query_states_xpu = query_states.to(dtype).xpu()
                key_states_xpu = key_states.to(dtype).xpu()
                value_states_xpu = value_states.to(dtype).xpu()
                grad_xpu = grad.to(dtype).xpu()
                bias_xpu = bias.to(dtype).xpu()

                query_states.requires_grad_(True)
                key_states.requires_grad_(True)
                value_states.requires_grad_(True)
                bias.requires_grad_(True)

                query_states_xpu.requires_grad_(True)
                key_states_xpu.requires_grad_(True)
                value_states_xpu.requires_grad_(True)
                bias_xpu.requires_grad_(True)

                r_cpu = torch.nn.functional.scaled_dot_product_attention(
                    query_states, key_states, value_states, bias
                )
                r_xpu = torch.nn.functional.scaled_dot_product_attention(
                    query_states_xpu, key_states_xpu, value_states_xpu, bias_xpu
                )
                r_cpu.backward(grad)
                r_xpu.backward(grad_xpu)

                self.assertEqual(r_cpu, r_xpu.cpu().float(), atol=1e-2, rtol=1e-1)
                self.assertEqual(
                    query_states.grad,
                    query_states_xpu.grad.cpu().float(),
                    atol=1e-2,
                    rtol=1e-1,
                )
                self.assertEqual(
                    key_states.grad,
                    key_states_xpu.grad.cpu().float(),
                    atol=1e-2,
                    rtol=1e-1,
                )
                self.assertEqual(
                    value_states.grad,
                    value_states_xpu.grad.cpu().float(),
                    atol=1e-2,
                    rtol=1e-1,
                )
                self.assertEqual(
                    bias.grad, bias_xpu.grad.cpu().float(), atol=1e-2, rtol=1e-1
                )
