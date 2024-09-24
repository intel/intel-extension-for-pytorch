import torch
from .._transformer_configuration import IPEXTransformerConfig
from .NaiveAttention import IPEXTransformerAttnNaive
from .CacheUtils import IPEXStaticCache, CacheFormat
from typing import Optional
from typing import List
import torch.distributed as dist
import math
from .Linear import (
    IPEXQKVFusedGemm,
    IPEXLowbitGemmAdd,
    GemmDtype,
)


class IPEXAttention(IPEXTransformerAttnNaive):
    cache_type = None

    def __init__(
        self, config: IPEXTransformerConfig, layer_idx: Optional[int] = None
    ) -> None:
        super().__init__(config)
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.embedding_dim // self.tp_size
        self.use_causal_mask = config.use_causal_mask
        self.num_heads = config.num_attention_head // self.tp_size
        self.num_kv_heads = config.num_key_value_head // self.tp_size
        self.head_dim = config.embedding_dim // config.num_attention_head
        # TODO: add different Gemm type support here
        self.qkv_proj = IPEXQKVFusedGemm(
            self.num_heads,
            self.num_kv_heads,
            GemmDtype.FP16,
            None,
            None,
        )
        self.out_proj = IPEXLowbitGemmAdd(self.tp_size, GemmDtype.FP16, None, None)
        # for backward compatibility, transformers 4.38.1 will using o_proj as attr name in generation
        self.o_proj = self.out_proj

    def load_parameter(self, q_proj, k_proj, v_proj, o_proj):
        self.qkv_proj.load_parameter(q_proj, k_proj, v_proj)
        self.out_proj.load_parameter(o_proj)
        self.position_embed = self.config.rotary_embedding_class(
            self.config, q_proj.weight.dtype
        )

    def transpose_parameter(self):
        pass

    def cat_qkv(self):
        pass

    def repeat_kv(self, kv, num_group):
        # kv shape: [b, f, n, h]
        bs, _, seqlen, _ = kv.shape
        kv = kv.permute(2, 0, 1, 3).contiguous()
        kv = kv[:, :, :, None, :].expand(
            seqlen, bs, self.num_kv_heads, num_group, self.head_dim
        )
        kv = kv.reshape(seqlen, bs, self.num_heads, self.head_dim).contiguous()
        kv = kv.permute(1, 2, 0, 3)
        return kv

    def update_transformer_kv_cache(self, cache, key, value, kwargs):
        base_cache_module = type(cache).__base__.__module__
        base_cache_name = type(cache).__base__.__name__
        cache_relpath = base_cache_module + "." + base_cache_name

        # Using IPEXStaticCache for optimized path, otherwise would be functionality path with optimized
        # flow (more extra memory reorder for current kernel support, which may be removed after related kernel support is ready)
        if isinstance(cache, IPEXStaticCache):
            if (
                cache.cache_format == CacheFormat.FBNH
                and not self.beam_search_first_iter(key.size(2))
            ):
                key_ret, value_ret = cache.get_kv_slice_for_decoding(
                    self.layer_idx, key
                )
                cache.update_or_get_seq_cnt(self.layer_idx, key)
                return key_ret, value_ret
            if self.num_heads != self.num_kv_heads:
                # expand kv prompt for beam search
                num_group = self.num_heads // self.num_kv_heads
                key = self.repeat_kv(key, num_group)
                value = self.repeat_kv(value, num_group)
            return cache.update(
                key, value, layer_idx=self.layer_idx, cache_kwargs=kwargs
            )
        elif cache_relpath == "transformers.cache_utils.Cache":
            # huggingface's Cache have memory layout of [bs, num_heads, seqlen, head_dim] memory layout
            # we need to transpose it to [bs, seqlen, num_heads, head_dim] memory layout
            key, value = cache.update(key, value, self.layer_idx, kwargs)
            key = key.transpose(1, 2).contiguous().transpose(1, 2)
            value = value.transpose(1, 2).contiguous().transpose(1, 2)
            seqlen = cache.get_seq_length()
            return (key[:, :, :seqlen, :], value[:, :, :seqlen, :])
        elif isinstance(cache, List):
            key_cache = torch.cat([cache[0], key], dim=2)
            value_cache = torch.cat([cache[1], value], dim=2)
            return (key_cache, value_cache)
        elif cache is None:
            return key, value
        else:
            raise ValueError(f"Unrecognized cache type {cache_relpath}")

    def rotary_embedding(
        self, query, key, value, past_key_value, position_ids, layer_id, seqlen
    ):
        if (
            isinstance(past_key_value, IPEXStaticCache)
            and past_key_value.cache_format == CacheFormat.FBNH
            and not self.beam_search_first_iter(seqlen)
        ):
            query, key = self.position_embed(
                query,
                key,
                position_ids,
                self.layer_id,
                self.beam_idx,
                seqlen,
                past_key_value.cache_format,
            )

            # from [bs, seqlen, num_heads, head_dim] to [bs, num_heads, seqlen, head_dim]
            query = query.permute(1, 2, 0, 3)
            key = key.permute(1, 2, 0, 3)
            value = value.permute(1, 2, 0, 3)
        else:
            query, key = self.position_embed(
                query, key, position_ids, self.layer_id, self.beam_idx, seqlen
            )
            query = query.permute(0, 2, 1, 3)
            key = key.permute(0, 2, 1, 3)
            value = value.permute(0, 2, 1, 3)
        return query, key, value

    def all_reduce_if_necessary(self, reduce_target):
        if self.tp_group is not None:
            dist.all_reduce(reduce_target, group=self.tp_group)
        return

    def get_blocked_alibi(self, alibi, seq_len):
        if self.layer_id == 0:
            cache_len = (
                self.max_position
                if self.max_position > seq_len
                else seq_len + self.runtime_cache_size
            )
            shape = [
                alibi.shape[0],
                alibi.shape[1],
                cache_len,
            ]  # [beam*num_head, q_len, kv_len]
            IPEXAttention.blocked_alibi = torch.empty(
                shape, device=alibi.device, dtype=alibi.dtype
            )
            kv_len = alibi.shape[2]
            IPEXAttention.blocked_alibi[:, :, 0:kv_len] = alibi
        return IPEXAttention.blocked_alibi

    def get_blocked_attn_mask(self, attn_mask_):
        alignment = 64
        if self.layer_id == 0:
            attn_mask = attn_mask_.contiguous()
            seq_len = attn_mask_.size(-1)
            align_cache_len = (seq_len + alignment - 1) // alignment * alignment
            IPEXAttention.blocked_attn_mask = torch.empty(
                (
                    attn_mask.shape[0],
                    attn_mask.shape[1],
                    attn_mask.shape[2],
                    align_cache_len,
                ),
                device=attn_mask.device,
                dtype=attn_mask.dtype,
            )
            IPEXAttention.blocked_attn_mask.fill_(-65504.0)
            IPEXAttention.blocked_attn_mask[:, :, :, 0 : attn_mask.shape[3]] = attn_mask
        return IPEXAttention.blocked_attn_mask

    def sdp(self, query, key, value, past_key_value, attention_mask, head_mask, alibi):

        scale = 1.0 / math.sqrt(self.head_dim)
        use_casual = False
        if query.size(2) == key.size(2):
            use_casual = True

        # we are not plan to support attention mask here, for it should be in None
        # at all of our test case.

        # if attention_mask is not None:
        #     attention_mask = self.get_blocked_attn_mask(attention_mask)
        if alibi is not None:
            alibi = self.get_blocked_alibi(alibi, key.size(1))
        if (
            self.beam_idx is not None
            and query.size(-2) == 1
            and isinstance(past_key_value, IPEXStaticCache)
        ):
            use_casual = False
            key_prompt, value_prompt = past_key_value.get_prompt_for_beam_search(
                self.layer_idx
            )
            prompt_length = key_prompt.size(2)
            seqlen = key.size(2)
            # TODO: remove this after ifmha support combined kv cache with both prompt
            # and decode in [bs, seqlen, num_head, head_dim] layout
            key = key[:, :, prompt_length:, :]
            value = value[:, :, prompt_length:, :]
            # TODO: remove this after ifmha support [bs, seqlen, num_head, head_dim] layout
            if (
                isinstance(past_key_value, IPEXStaticCache)
                and past_key_value.cache_format == CacheFormat.BFNH
            ):  # for BFNH format
                key = key.permute(2, 0, 1, 3).contiguous().permute(1, 2, 0, 3)
                value = (
                    value.permute(2, 0, 1, 3)
                    .contiguous()
                    .permute(
                        1,
                        2,
                        0,
                    )
                )

            attention_output = torch.xpu.IpexSDP_Index(
                query,
                key_prompt,
                value_prompt,
                key,
                value,
                self.beam_idx,
                alibi,
                None,
                head_mask,
                seqlen,
                scale,
                1.0,
                0.0,
                use_casual,
            )
        else:
            # TODO: remove this after fmha support strided fmha on F dim
            if (
                isinstance(past_key_value, IPEXStaticCache)
                and past_key_value.cache_format == CacheFormat.FBNH
            ):  # for BFNH format
                attention_output = torch.xpu.IpexSDP(
                    query,
                    key,
                    value,
                    alibi,
                    None,
                    head_mask,
                    scale,
                    1.0,
                    0.0,
                    use_casual,
                    self.beam_idx is None,
                )
            else:  # for BFNH format
                # TODO: remove this after fmha support strided fmha on F dim
                if query.size(0) > 1:
                    key = key.transpose(1, 2).contiguous().transpose(1, 2)
                    value = value.transpose(1, 2).contiguous().transpose(1, 2)

                attention_output = torch.xpu.IpexSDP(
                    query,
                    key,
                    value,
                    alibi,
                    None,
                    head_mask,
                    scale,
                    1.0,
                    0.0,
                    use_casual,
                    False,
                )

        return attention_output, None

    def beam_search_first_iter(self, seqlen):
        return self.beam_idx is not None and seqlen != 1

    def beam_search_next_token(self, seqlen):
        return self.beam_idx is not None and seqlen == 1

    def compute_qkv_gemm(self, hidden_states, query, key, value):
        query, key, value = self.qkv_proj(hidden_states, query, key, value)
        return query, key, value

    def out_proj_compute(self, attn_output, residual):
        return self.out_proj(attn_output, residual)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        alibi: torch.Tensor = None,
        **kwargs,
    ):
        if IPEXAttention.cache_type is None:
            IPEXAttention.cache_type = (
                "static" if isinstance(past_key_value, IPEXStaticCache) else "dynamic"
            )

        bs, seqlen, _ = hidden_states.size()
        # TODO this wa will be removed after qkv gemm support strided write on F dim, we need this now
        # for performance reason
        if (
            isinstance(past_key_value, IPEXStaticCache)
            and past_key_value.cache_format == CacheFormat.FBNH
            and not self.beam_search_first_iter(seqlen)
        ):
            query = torch.empty(
                [seqlen, bs, self.num_heads * self.head_dim],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            key, value = past_key_value.get_kv_slice_for_qkv(
                self.layer_idx, cache_position=cache_position
            )

        else:
            query = torch.empty(
                [bs, seqlen, self.num_heads * self.head_dim],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            key = torch.empty(
                [bs, seqlen, self.num_kv_heads * self.head_dim],
                dtype=query.dtype,
                device=query.device,
            )
            value = torch.empty(
                [bs, seqlen, self.num_kv_heads * self.head_dim],
                dtype=query.dtype,
                device=query.device,
            )
        # qkv gemm
        query, key, value = self.compute_qkv_gemm(hidden_states, query, key, value)

        query = query.view(
            [query.shape[0], query.shape[1], self.num_heads, self.head_dim]
        )
        key = key.view([key.shape[0], key.shape[1], self.num_kv_heads, self.head_dim])
        value = value.view(
            [value.shape[0], value.shape[1], self.num_kv_heads, self.head_dim]
        )

        # apply rope to qk
        query, key, value = self.rotary_embedding(
            query, key, value, past_key_value, position_ids, self.layer_id, seqlen
        )

        kwargs_for_cache_update = {
            "cache_position": cache_position,
            "beam_search": self.beam_idx is not None,
            "num_heads": self.num_heads,
        }
        # for backward compatibility to older transformers, we should support both Cache and List[torch.Tensor] as kv_cache

        # key value should in shape of [B, N, F, H]
        present = self.update_transformer_kv_cache(
            past_key_value, key, value, kwargs_for_cache_update
        )
        key, value = present
        past_key_value = (
            [key, value]
            if isinstance(past_key_value, List) or past_key_value is None
            else past_key_value
        )
        # need to repeat kv for beam search next token
        if self.num_heads != self.num_kv_heads and self.beam_search_next_token(
            query.size(2)
        ):
            num_group = self.num_heads // self.num_kv_heads
            key = self.repeat_kv(key, num_group)
            value = self.repeat_kv(value, num_group)

        attn_output, attn_weight = self.sdp(
            query, key, value, past_key_value, attention_mask, head_mask, alibi
        )

        # transpose back to [bs, seqlen, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).reshape(
            [bs, seqlen, self.hidden_size]
        )
        # o proj + residual
        residual = kwargs.get("residual", None)
        attn_output = self.out_proj_compute(attn_output, residual)

        self.all_reduce_if_necessary(attn_output)
        attn_output = attn_output.view([bs, seqlen, self.hidden_size * self.tp_size])

        outputs = (attn_output, past_key_value)
        if output_attentions:
            outputs += (attn_weight,)
        else:
            outputs += (None,)
        return outputs
