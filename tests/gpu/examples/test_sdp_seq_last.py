import torch
import intel_extension_for_pytorch  # noqa
import math

from torch.testing._internal.common_utils import TestCase
import pytest


def naive_sdp(query, key, value, attention_mask, head_mask, alibi, alpha):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    attn_weights *= alpha

    if attention_mask is not None:
        attn_weights += attention_mask
        # the attn_weights should anyway bigger than dtype.min, I wonder if this is necessary
        attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float).to(query.dtype)
    if head_mask is not None:
        attn_weights = attn_weights * head_mask
    attn_output = torch.matmul(attn_weights, value)
    return attn_output, attn_weights


beam_width = 1
num_heads = 14  # (/rank=8, 14)
head_dim = 128
q_len = 1023
kv_len = 1023  # 1152
alpha = 1.0 / math.sqrt(head_dim)
beta = 1.0
max_len = 2048

query_layer = torch.randn(q_len, beam_width, num_heads, head_dim).xpu().half()
key_layer = torch.randn(kv_len, beam_width, num_heads, head_dim).xpu().half()
value_layer = torch.randn(kv_len, beam_width, num_heads, head_dim).xpu().half()

# attention_mask = torch.zeros(beam_width, 1, q_len, kv_len).half()
# attention_mask[0][0][0] = -65504.
attention_mask = torch.zeros(beam_width, 1, q_len, kv_len).xpu().half()
attention_mask[0, 0, 0:q_len, 0] = -65504
attention_mask[0, 0, 0:q_len, kv_len - 1:kv_len] = -float('inf')
attention_mask[0, 0, 0, kv_len - 3:kv_len] = -float('inf')
# print(attention_mask)


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(not torch.xpu.has_xetla(), reason="ipex build without xetla")
    def test_fsdp_index_select(self):
        ref_out, _ = naive_sdp(
            query_layer.permute(1, 2, 0, 3),
            key_layer.permute(1, 2, 0, 3),
            value_layer.permute(1, 2, 0, 3),
            attention_mask,
            None,
            None,
            alpha)

        ref_out_float, _ = naive_sdp(
            query_layer.float().permute(1, 2, 0, 3),
            key_layer.float().permute(1, 2, 0, 3),
            value_layer.float().permute(1, 2, 0, 3),
            attention_mask.float(),
            None,
            None,
            alpha)

        ref_out_cpu, _ = naive_sdp(
            query_layer.cpu().float().permute(1, 2, 0, 3),
            key_layer.cpu().float().permute(1, 2, 0, 3),
            value_layer.cpu().float().permute(1, 2, 0, 3),
            attention_mask.cpu().float(),
            None,
            None,
            alpha)
        attention_mask_padded = torch.zeros(beam_width, 1, q_len, max_len).half()
        attention_mask_padded[:, :, :, 0:kv_len] = attention_mask

        res_out = torch.xpu.IpexSDP(
            query_layer.to('xpu').permute(
                1, 2, 0, 3), key_layer.to('xpu').permute(
                1, 2, 0, 3), value_layer.to('xpu').permute(
                1, 2, 0, 3), None, attention_mask_padded.to('xpu'), None, alpha, beta, 1.0, False, True)

        # print(ref_out)
        # print(res_out.cpu())
        print("sdp half vs naive xpu half: ", torch.max(torch.abs(ref_out.cpu() - res_out.cpu())).item())
        # print("sdp half vs sdp half non padded: ", torch.max(torch.abs(res_non_pad_out.cpu() - res_out.cpu())).item())
        print("sdp half vs naive cpu float: ", torch.max(torch.abs(res_out.cpu() - ref_out_cpu)).item())
        print("sdp half vs naive xpu float: ", torch.max(torch.abs(res_out.cpu() - ref_out_float.cpu())).item())
        print("naive xpu half vs naive cpu float: ", torch.max(torch.abs(ref_out.cpu() - ref_out_cpu)).item())
        print("naive xpu half vs naive xpu float: ", torch.max(torch.abs(ref_out.cpu() - ref_out_float.cpu())).item())

        self.assertEqual(ref_out.cpu(), res_out.cpu(), atol=1e-3, rtol=1e-4)
        self.assertEqual(res_out.cpu().float(), ref_out_cpu, atol=1e-3, rtol=1e-4)
        self.assertEqual(res_out.cpu().float(), ref_out_float, atol=1e-3, rtol=1e-4)
        self.assertEqual(ref_out.cpu().float(), ref_out_cpu, atol=1e-3, rtol=1e-4)
        self.assertEqual(ref_out.cpu().float(), ref_out_float, atol=1e-3, rtol=1e-4)
