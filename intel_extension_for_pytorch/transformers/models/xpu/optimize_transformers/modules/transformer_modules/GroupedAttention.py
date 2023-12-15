import torch

from .._transformer_configuration import IPEXTransformerConfig
from .Attention import (  # noqa F401
    IPEXTransformerAttnOptimizedFp16,
    IPEXTransformerAttn,
)


class IPEXTransformerAttnOptimizedFp16Grouped(IPEXTransformerAttnOptimizedFp16):
    def __init__(self, config: IPEXTransformerConfig) -> None:
        super().__init__(config)
        self.num_kv_head = config.num_key_value_head // self.tp_size
        self.num_kv_group = self.num_attn_head // self.num_kv_head
        del self.q_proj
        del self.k_proj
        del self.v_proj

    def load_parameter(self, qkv_proj, out_proj):
        self.qkv_proj.weight = qkv_proj.weight
        self.qkv_proj.bias = qkv_proj.bias
        self.out_proj.weight = out_proj.weight
        self.out_proj.bias = out_proj.bias
        self.position_embed = self.config.rotary_embedding_class(
            self.config, qkv_proj.weight.dtype
        )

    def transpose_parameter(self):
        self.qkv_proj.weight.data = self.qkv_proj.weight.transpose(1, 2).contiguous()
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
        if self.num_kv_group <= 1:
            return super().prepare_qkv_input_1st_token_beam_search(hidden_states)
        bs_beam, seq_len, embed_dim = self.get_runtime_shape(hidden_states)
        kv_shape = [bs_beam, seq_len, self.num_kv_head * self.head_dim]

        query_shape = [
            self.num_kv_group,
            bs_beam,
            seq_len,
            self.num_kv_head * self.head_dim,
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
        if self.num_kv_group <= 1:
            return super().prepare_qkv_input_2nd2last(hidden_states)
        bs_beam, seq_len, embed_dim = self.get_runtime_shape(hidden_states)
        query_shape = [
            self.num_kv_group,
            seq_len,
            bs_beam,
            self.num_kv_head * self.head_dim,
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
        if self.num_kv_group <= 1:
            return super().compute_qkv_gemm(hidden_states, query, key, value)
        torch.ops.torch_ipex.mm_qkv_group_out(
            hidden_states, self.qkv_proj.weight, self.qkv_proj.bias, query, key, value
        )
        return query, key, value

    def process_qkv_output(self, hidden_states, query, key, value):
        if self.num_kv_group <= 1:
            return super().process_qkv_output(hidden_states, query, key, value)
        query_shape = (
            (self.num_kv_group,)
            + hidden_states.size()[:-1]
            + (self.num_kv_head, self.head_dim)
        )
        kv_shape = hidden_states.size()[:-1] + (self.num_kv_head, self.head_dim)
        if self.is_1st_token_beam_search():
            self.runtime_cache.key_prompt = self.runtime_cache.key_prompt.view(kv_shape)
            self.runtime_cache.value_prompt = self.runtime_cache.value_prompt.view(
                kv_shape
            )
        if self.is_beam_search():
            self.prev_seq_len = self.seq_len
        query = (
            query.view(query_shape).permute(1, 2, 3, 0, 4).flatten(2, 3).contiguous()
        )
        key = key.view(kv_shape)
        value = value.view(kv_shape)
        return query, key, value

    def sdp_kv_preprocess_1st_token_beam_search(self, key, value):
        if self.num_kv_group <= 1:
            return super().sdp_kv_preprocess_1st_token_beam_search(key, value)
        key = self.repeat_kv(key, self.num_kv_group)
        value = self.repeat_kv(value, self.num_kv_group)
        key_prompt, value_prompt = key, value
        return key, value, key_prompt, value_prompt

    def sdp_kv_preprocess_2nd2last(self, key, value):
        if self.num_kv_group <= 1:
            return super().sdp_kv_preprocess_2nd2last(key, value)
        key = self.repeat_kv(key, self.num_kv_group)
        value = self.repeat_kv(value, self.num_kv_group)
        key_prompt = self.repeat_kv(self.runtime_cache.key_prompt, self.num_kv_group)
        value_prompt = self.repeat_kv(
            self.runtime_cache.value_prompt, self.num_kv_group
        )
        return key, value, key_prompt, value_prompt
