import torch
import intel_extension_for_pytorch
import torch.nn.functional as F
import math

beam_width = 1
num_heads = 112 # (/rank=8, 14)
head_dim = 128
q_len = 1
kv_len = 1152

beta = 1.0
inv_norm_factor = 1.0 / math.sqrt(head_dim)

print("CPU sdp ...")
alibi = torch.randn(beam_width * num_heads, q_len, kv_len).xpu().half().fill_(0)
query_layer = torch.randn(beam_width * num_heads, q_len, head_dim).xpu().half()
key_layer = torch.randn(beam_width * num_heads, kv_len, head_dim).xpu().half()
value_layer = torch.randn(beam_width * num_heads, kv_len, head_dim).xpu().half()

gemm0_res = alibi.baddbmm(
    batch1=query_layer,
    batch2=key_layer.permute(0, 2, 1),
    beta=beta,
    alpha=inv_norm_factor,
)

attn_scores = gemm0_res.view(beam_width, num_heads, q_len, kv_len)
# attn_mask = torch.zeros(beam_width, kv_len, dtype=torch.bool, device=torch.device('xpu'))
# attn_weights = torch.masked_fill(attn_scores, attn_mask, torch.finfo(attn_scores.dtype).min)
attn_weights = attn_scores
attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
attn_probs_reshaped = attn_probs.view(beam_width * num_heads, q_len, kv_len)
context_layer = torch.bmm(attn_probs_reshaped, value_layer).view(beam_width, num_heads, q_len, head_dim)

print("XPU sdp ...")
alibi_sdp = alibi.view(beam_width, num_heads, q_len, kv_len)
query_layer_sdp = query_layer.view(beam_width, num_heads, q_len, head_dim).permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)
key_layer_sdp = key_layer.view(beam_width, num_heads, kv_len, head_dim).permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)
value_layer_sdp = value_layer.view(beam_width, num_heads, kv_len, head_dim).permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)

alibi_sdp = alibi_sdp.to('xpu')
query_layer_sdp = query_layer_sdp.to('xpu')
key_layer_sdp = key_layer_sdp.to('xpu')
value_layer_sdp = key_layer_sdp.to('xpu')
attn_mask = None
head_mask = None
alpha = inv_norm_factor
beta = 1.0
dropout = 0.0
is_causal = False

print(context_layer.cpu())
context_layer_sdp_ref = F.scaled_dot_product_attention(query_layer_sdp, key_layer_sdp, value_layer_sdp, is_causal=False)
print(context_layer_sdp_ref.cpu())
context_layer_cpu = F.scaled_dot_product_attention(query_layer_sdp.cpu().float(), key_layer_sdp.cpu().float(), value_layer_sdp.cpu().float(), is_causal=False)
print(context_layer_cpu)
context_layer_sdp = torch.xpu.IpexSDP(query_layer_sdp, key_layer_sdp, value_layer_sdp, alibi_sdp, attn_mask, head_mask, alpha, beta, dropout, is_causal)
print(context_layer_sdp.cpu())
# print(context_layer.cpu() - context_layer_sdp.cpu())
