import torch

from .._transformer_configuration import IPEXTransformerConfig
from .Attention import (  # noqa F401
    IPEXTransformerAttnOptimizedFp16,
    IPEXTransformerAttn,
)
from .Linear import IPEXTransformerLinear


class IPEXTransformerAttnOptimizedFp16Grouped(IPEXTransformerAttnOptimizedFp16):
    def __init__(self, config: IPEXTransformerConfig) -> None:
        super().__init__(config)
        self.num_kv_head = config.num_key_value_head // self.tp_size
        self.num_kv_group = self.num_attn_head // self.num_kv_head
        del self.q_proj
        del self.k_proj
        del self.v_proj

    def load_parameter(
        self, q_proj=None, k_proj=None, v_proj=None, out_proj=None, qkv_proj=None
    ):
        embed_dim = self.embed_dim
        num_head = self.num_attn_head
        num_kv_head = self.num_kv_head
        head_dim = self.head_dim
        group = num_head // num_kv_head + 2
        if qkv_proj is None:

            def helper(proj, g):
                w = proj.weight.view(num_kv_head, g, head_dim, embed_dim)
                del proj.weight
                if proj.bias is not None:
                    b = proj.bias.view(num_kv_head, g, head_dim)
                    del proj.bias
                else:
                    b = None
                return w, b

            wq, bq = helper(q_proj, group - 2)
            wk, bk = helper(k_proj, 1)
            wv, bv = helper(v_proj, 1)

            # [num_kv_head * (num_head//num_kv_head + 2) * head_dim, embed_dim]
            wqkv = torch.concat([wq, wk, wv], dim=1).contiguous()
            if bq is None:
                bqkv = None
            else:
                bqkv = torch.concat([bq, bk, bv], dim=1).contiguous()
            qkv_proj = IPEXTransformerLinear(wqkv, bqkv)

        self.qkv_proj.weight = qkv_proj.weight.view(
            num_kv_head, group, head_dim, embed_dim
        )
        if qkv_proj.bias is not None:
            self.qkv_proj.bias = qkv_proj.bias.view(num_kv_head, group, head_dim)
        else:
            self.qkv_proj.bias = None

        self.out_proj.weight = out_proj.weight
        self.out_proj.bias = out_proj.bias
        self.position_embed = self.config.rotary_embedding_class(
            self.config, qkv_proj.weight.dtype
        )

    def transpose_parameter(self):
        self.qkv_proj.weight.data = self.qkv_proj.weight.permute(
            3, 0, 1, 2
        ).contiguous()
        self.out_proj.weight.data = self.out_proj.weight.transpose(0, 1).contiguous()
        # Note: synchronize to ensure the completion of contiguous
        torch.xpu.synchronize()

    def cat_qkv(self):
        # GQA will directly use concatenated weight for qkv
        pass

    def prepare_kv_prompt(self, hidden_states, kv_head):
        return super().prepare_kv_prompt(hidden_states, self.num_kv_head)

    def prepare_kv_cache(self, hidden_states, kv_head):
        return super().prepare_kv_cache(hidden_states, self.num_kv_head)

    def prepare_qkv_input_1st_token_beam_search(self, hidden_states, **kwargs):
        bs_beam, seq_len, embed_dim = self.get_runtime_shape(hidden_states)
        kv_shape = [bs_beam, seq_len, self.num_kv_head * self.head_dim]

        query_shape = [
            bs_beam,
            seq_len,
            self.num_attn_head * self.head_dim,
        ]
        query = torch.empty(
            query_shape, device=hidden_states.device, dtype=hidden_states.dtype
        )
        return (
            query,
            self.runtime_cache.key_prompt.view(kv_shape),
            self.runtime_cache.value_prompt.view(kv_shape),
        )

    def prepare_qkv_input_2nd2last(self, hidden_states, **kwargs):
        bs_beam, seq_len, embed_dim = self.get_runtime_shape(hidden_states)
        query_shape = [
            seq_len,
            bs_beam,
            self.num_attn_head * self.head_dim,
        ]
        kv_shape = [seq_len, bs_beam, self.num_kv_head * self.head_dim]
        query = torch.empty(
            query_shape, device=hidden_states.device, dtype=hidden_states.dtype
        )

        key = self.runtime_cache.key_cache[
            self.prev_seq_len : self.seq_len, :, :, :
        ].view(kv_shape)
        value = self.runtime_cache.value_cache[
            self.prev_seq_len : self.seq_len, :, :, :
        ].view(kv_shape)
        return query, key, value

    def compute_qkv_gemm(self, hidden_states, query, key, value):
        torch.ops.torch_ipex.mm_qkv_group_out(
            hidden_states, self.qkv_proj.weight, self.qkv_proj.bias, query, key, value
        )
        return query, key, value

    def process_qkv_output(self, hidden_states, query, key, value):
        query_shape = hidden_states.size()[:-1] + (self.num_attn_head, self.head_dim)
        kv_shape = hidden_states.size()[:-1] + (self.num_kv_head, self.head_dim)
        if self.is_1st_token_beam_search():
            self.runtime_cache.key_prompt = self.runtime_cache.key_prompt.view(kv_shape)
            self.runtime_cache.value_prompt = self.runtime_cache.value_prompt.view(
                kv_shape
            )
        if self.is_beam_search():
            self.prev_seq_len = self.seq_len
        query = query.view(query_shape)
        key = key.view(kv_shape)
        value = value.view(kv_shape)
        return query, key, value

    def sdp_kv_preprocess_1st_token_beam_search(self, key, value):
        # first token will use IpexSDP which supports GQA and not need to repeat_kv
        return key, value, key, value

    def sdp_kv_preprocess_2nd2last(self, key, value):
        # next token for greedy will use IpexSDP which supports GQA and not need to repeat_kv
        if not self.is_beam_search():
            return (
                key,
                value,
                self.runtime_cache.key_prompt,
                self.runtime_cache.value_prompt,
            )
        key = self.repeat_kv(key, self.num_kv_group)
        value = self.repeat_kv(value, self.num_kv_group)
        key_prompt = self.repeat_kv(self.runtime_cache.key_prompt, self.num_kv_group)
        value_prompt = self.repeat_kv(
            self.runtime_cache.value_prompt, self.num_kv_group
        )
        return key, value, key_prompt, value_prompt


class IPEXTransformerAttnOptimizedFp16GroupedChatGLM(
    IPEXTransformerAttnOptimizedFp16Grouped
):
    def __init__(self, config: IPEXTransformerConfig) -> None:
        super().__init__(config)

    def load_parameter(self, qkv_proj=None, out_proj=None):
        embed_dim = self.embed_dim
        num_head = self.num_attn_head
        num_kv_head = self.num_kv_head
        head_dim = self.head_dim
        group = num_head // num_kv_head + 2
        query_weight_size = head_dim * num_head
        key_weight_size = value_weight_size = head_dim * num_kv_head

        def helper(proj, begin, end, g):
            weight = proj.weight[begin:end, ...].view(
                num_kv_head, g, head_dim, embed_dim
            )
            if proj.bias is not None:
                bias = proj.bias[begin:end, ...].view(num_kv_head, g, head_dim)
            else:
                bias = None
            return weight, bias

        wq, bq = helper(qkv_proj, 0, query_weight_size, group - 2)
        wk, bk = helper(
            qkv_proj, query_weight_size, query_weight_size + key_weight_size, 1
        )
        wv, bv = helper(
            qkv_proj, query_weight_size + key_weight_size, qkv_proj.weight.shape[0], 1
        )

        del qkv_proj.weight
        del qkv_proj.bias

        self.qkv_proj.weight = torch.concat([wq, wk, wv], dim=1).contiguous()
        if bq is not None:
            self.qkv_proj.bias = torch.concat([bq, bk, bv], dim=1).contiguous()
        else:
            self.qkv_proj.bias = None

        self.out_proj.weight = out_proj.weight
        self.out_proj.bias = out_proj.bias

        self.position_embed = self.config.rotary_embedding_class(
            self.config, self.qkv_proj.weight.dtype
        )

    def post_qkv(self, query, key, value, rotary_pos_emb, layer_past, **kwargs):
        bs_beam, seq, _ = self.get_runtime_shape(query)
        seq = seq if layer_past is None else layer_past[0].size(2) + 1
        query, key = self.position_embed(query, key, rotary_pos_emb)
        query, key, value = self.combine_kv_cache_interface(query, key, value)
        return query, key, value

    def prepare_sdp_input(self, query, key, value, attention_mask, alibi):
        (
            dropout,
            alpha,
            beta,
            is_casual,
            blocked_attn_mask,
            blocked_alibi,
        ) = super().prepare_sdp_input(query, key, value, attention_mask, alibi)
        is_causal = True if self.is_1st_token() else False
        return dropout, alpha, beta, is_causal, blocked_attn_mask, blocked_alibi
