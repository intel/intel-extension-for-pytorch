import torch

from .._transformer_configuration import IPEXTransformerConfig
from .Attention import IPEXTransformerAttnOptimizedFp16
from .QuantizedAttention import (
    IPEXTransformerAttnOptimizedInt4,
    IPEXTransformerAttnOptimizedInt4OneDNN,
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


class IPEXTransformerAttnOptimizedInt4Grouped(IPEXTransformerAttnOptimizedInt4):
    def __init__(self, config: IPEXTransformerConfig) -> None:
        super().__init__(config)
        self.num_kv_head = config.num_key_value_head // self.tp_size
        self.num_kv_group = self.num_attn_head // self.num_kv_head

    def prepare_kv_prompt(self, hidden_states, kv_head):
        return super().prepare_kv_prompt(hidden_states, self.num_kv_head)

    def prepare_kv_cache(self, hidden_states, kv_head):
        return super().prepare_kv_cache(hidden_states, self.num_kv_head)

    def prepare_qkv_input_1st_token_beam_search(self, hidden_states, **kwargs):
        if self.num_kv_group <= 1:
            return super().prepare_qkv_input_1st_token_beam_search(hidden_states)
        bs_beam, seq_len, embed_dim = self.get_runtime_shape(hidden_states)
        kv_shape = [bs_beam, seq_len, self.num_kv_head * self.head_dim]

        query_shape = [bs_beam, seq_len, self.num_attn_head * self.head_dim]
        query = torch.empty(
            query_shape, device=hidden_states.device, dtype=hidden_states.dtype
        )
        return (
            query,
            self.runtime_cache.key_prompt.view(kv_shape),
            self.runtime_cache.value_prompt.view(kv_shape),
        )

    def prepare_qkv_input_2nd2last(self, hidden_states, **kwargs):
        if self.num_kv_group <= 1:
            return super().prepare_qkv_input_2nd2last(hidden_states)
        bs_beam, seq_len, embed_dim = self.get_runtime_shape(hidden_states)
        query_shape = [seq_len, bs_beam, self.num_attn_head * self.head_dim]
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
        return super().compute_qkv_gemm(hidden_states, query, key, value)

    def process_qkv_output(self, hidden_states, query, key, value):
        if self.num_kv_group <= 1:
            return super().process_qkv_output(hidden_states, query, key, value)
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
        if self.num_kv_group <= 1:
            return super().sdp_kv_preprocess_1st_token_beam_search(key, value)
        # greedy search will use IpexSDP which supports GQA and not need to repeat_kv
        # future GQA will accept kv directly without repeating
        if not self.is_beam_search():
            return key, value, key, value
        else:
            key = self.repeat_kv(key, self.num_kv_group)
            value = self.repeat_kv(value, self.num_kv_group)
            key_prompt, value_prompt = key, value
            return key, value, key_prompt, value_prompt

    def sdp_kv_preprocess_2nd2last(self, key, value):
        if self.num_kv_group <= 1:
            return super().sdp_kv_preprocess_2nd2last(key, value)
        # next token for greedy will use IpexSDP which supports GQA and not need to repeat_kv
        if not self.is_beam_search():
            return (
                key,
                value,
                self.runtime_cache.key_prompt,
                self.runtime_cache.value_prompt,
            )
        else:
            key = self.repeat_kv(key, self.num_kv_group)
            value = self.repeat_kv(value, self.num_kv_group)
            key_prompt = self.repeat_kv(
                self.runtime_cache.key_prompt, self.num_kv_group
            )
            value_prompt = self.repeat_kv(
                self.runtime_cache.value_prompt, self.num_kv_group
            )
            return key, value, key_prompt, value_prompt


class IPEXTransformerAttnOptimizedInt4GroupedOneDNN(
    IPEXTransformerAttnOptimizedInt4OneDNN, IPEXTransformerAttnOptimizedInt4Grouped
):
    def __init__(self, config: IPEXTransformerConfig) -> None:
        super().__init__(config)
        self.num_kv_head = config.num_key_value_head // self.tp_size
        self.num_kv_group = self.num_attn_head // self.num_kv_head

    def cat_qkv(self):
        pass

    def compute_qkv_gemm(self, hidden_states, query, key, value):
        if self.num_kv_group <= 1:
            return super().compute_qkv_gemm(hidden_states, query, key, value)
        if (
            self.q_proj_quant.qweight is None
            and self.qkv_proj_quant.qweight is not None
        ):
            qkv_out = torch.ops.torch_ipex.mm_bias_int4(
                hidden_states,
                self.qkv_proj_quant.qweight,
                self.qkv_proj_quant.bias,
                self.qkv_proj_quant.scales,
                self.qkv_proj_quant.qzeros,
                self.qkv_proj_quant.blocksize,
                self.qkv_proj_quant.g_idx,
            )
            m_query = query.shape[-1]
            m_key = key.shape[-1]
            query = qkv_out[:, :, :m_query].contiguous()
            key.copy_(qkv_out[:, :, m_query : m_query + m_key])
            value.copy_(qkv_out[:, :, m_query + m_key :])
        else:
            query = torch.ops.torch_ipex.mm_bias_int4(
                hidden_states,
                self.q_proj_quant.qweight,
                self.q_proj_quant.bias,
                self.q_proj_quant.scales,
                self.q_proj_quant.qzeros,
                self.q_proj_quant.blocksize,
                self.q_proj_quant.g_idx,
            )
            key.copy_(
                torch.ops.torch_ipex.mm_bias_int4(
                    hidden_states,
                    self.k_proj_quant.qweight,
                    self.k_proj_quant.bias,
                    self.k_proj_quant.scales,
                    self.k_proj_quant.qzeros,
                    self.k_proj_quant.blocksize,
                    self.k_proj_quant.g_idx,
                )
            )
            value.copy_(
                torch.ops.torch_ipex.mm_bias_int4(
                    hidden_states,
                    self.v_proj_quant.qweight,
                    self.v_proj_quant.bias,
                    self.v_proj_quant.scales,
                    self.v_proj_quant.qzeros,
                    self.v_proj_quant.blocksize,
                    self.v_proj_quant.g_idx,
                )
            )
        return query, key, value
