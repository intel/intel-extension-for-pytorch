import torch
from torch.testing._internal.common_nn import NNTestCase
import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestTransformers(NNTestCase):
    def test_scaled_dot_product_attention(self, dtype=torch.float):
        for input_dim, attn_mask_dim, is_causal in [
            (3, None, False),
            (3, 2, False),
            (3, 3, False),
            (4, None, False),
            (3, None, True),
            (4, None, True),
        ]:
            N, N_prime, L, S, E = 5, 2, 4, 3, 6
            if input_dim == 3:
                query_cpu = torch.randn(N, L, E, dtype=dtype)
                key_cpu = torch.randn(N, S, E, dtype=dtype)
                value_cpu = torch.randn(N, S, E, dtype=dtype)
            elif input_dim == 4:
                query_cpu = torch.randn(N, N_prime, L, E, dtype=dtype)
                key_cpu = torch.randn(N, N_prime, S, E, dtype=dtype)
                value_cpu = torch.randn(N, N_prime, S, E, dtype=dtype)
            else:
                raise ValueError("Invalid input_dim: {}".format(input_dim))

            query_xpu = query_cpu.to(xpu_device)
            key_xpu = key_cpu.to(xpu_device)
            value_xpu = value_cpu.to(xpu_device)

            attn_mask_cpu = None
            if attn_mask_dim is not None:
                assert attn_mask_dim in [2, input_dim]
                mask_size = (
                    (L, S)
                    if attn_mask_dim == 2
                    else ((N, L, S) if input_dim == 3 else (N, N_prime, L, S))
                )
                attn_mask_cpu = (
                    torch.ones(mask_size, dtype=torch.bool).tril()
                    if is_causal
                    else torch.randint(0, 2, size=mask_size, dtype=torch.bool)
                )

            attn_mask_xpu = (
                None if attn_mask_dim is None else attn_mask_cpu.to(xpu_device)
            )

            dropout_p = 0
            need_attn_weights = True

            output_cpu, attn_cpu = torch.ops.aten._scaled_dot_product_attention(
                query_cpu,
                key_cpu,
                value_cpu,
                attn_mask_cpu,
                dropout_p,
                need_attn_weights,
                is_causal,
            )

            output_xpu, attn_xpu = torch.ops.aten._scaled_dot_product_attention(
                query_xpu,
                key_xpu,
                value_xpu,
                attn_mask_xpu,
                dropout_p,
                need_attn_weights,
                is_causal,
            )

            self.assertEqual(output_cpu, output_xpu.to(cpu_device))
            self.assertEqual(attn_cpu, attn_xpu.to(cpu_device))
