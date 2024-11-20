import torch
import intel_extension_for_pytorch as ipex  # noqa
import math

from torch.testing._internal.common_utils import TestCase


class TestTorchMethod(TestCase):

    def naive_sdp(self, query, key, value, attn_mask):
        head_dim = query.size(-1)
        scale_factor = 1.0 / math.sqrt(head_dim)
        # special case for phi3-small
        if query.size(1) == 32 and key.size(1) == 8:
            scale_factor = 1.0 / head_dim
        key = key.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
        value = value.repeat_interleave(query.shape[1] // value.shape[1], dim=1)
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights *= scale_factor

        if attn_mask is not None:
            attn_weights[attn_mask == 0] = float("-inf")

        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float
        )
        attn_output = torch.matmul(attn_weights, value)
        return attn_output

    def _test_esmid_sdp_cpu_ref_impl(self, b, n, n_kv, h, f, t, dtype=torch.float16):
        print(
            f"batch_size: {b}, num_q_head: {n}, num_kv_head: {n_kv}, head_dim: {h}, q_len: {f}, kv_len: {t}",
            flush=True,
        )
        assert n % n_kv == 0, "n should be a multiple of n_kv"
        q = torch.randn([b, n, f, h], dtype=dtype, device=torch.device("xpu"))
        k = torch.randn([b, n_kv, t, h], dtype=dtype, device=torch.device("xpu"))
        v = torch.randn([b, n_kv, t, h], dtype=dtype, device=torch.device("xpu"))
        attn_mask = torch.bernoulli(
            0.5 * torch.ones(b, n, f, t, device=torch.device("xpu"))
        ).to(torch.uint8)

        expected = self.naive_sdp(
            q.to(torch.float).to("cpu"),
            k.to(torch.float).to("cpu"),
            v.to(torch.float).to("cpu"),
            attn_mask.to(torch.float).to("cpu"),
        )

        actual = (
            torch.ops.torch_ipex.fmha_esimd(
                q,
                k,
                v,
                attn_mask,
                True,
            )
            .to(torch.float)
            .to("cpu")
        )

        print("esimd sdp vs naive: ", torch.max(torch.abs(expected - actual)).item())

        self.assertEqual(
            expected,
            actual,
            atol=1e-3,
            rtol=1e-3,
        )

    def test(self):
        # (bs, num_q_head, num_kv_head, head_dim, q_len, kv_len)
        sdp_config_list = [
            (1, 32, 32, 96, 4096, 4096),
            (1, 32, 32, 96, 1, 4097),
            (1, 32, 32, 96, 1, 5120),
            (1, 32, 8, 128, 32, 32),
            (1, 32, 8, 128, 1, 33),
            (1, 32, 8, 128, 1, 49),
            (1, 32, 8, 128, 4096, 4096),
            (1, 32, 8, 128, 1, 4097),
            (1, 32, 8, 128, 1, 5120),
        ]
        for config in sdp_config_list:
            b, n, n_kv, h, q_len, kv_len = config
            self._test_esmid_sdp_cpu_ref_impl(b, n, n_kv, h, q_len, kv_len)
