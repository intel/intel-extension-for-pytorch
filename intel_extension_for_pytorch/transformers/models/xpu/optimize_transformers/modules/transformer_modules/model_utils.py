import torch
import intel_extension_for_pytorch as ipex
from .RoPE import apply_rotary_pos_emb


def qwen_post_qkv(self, query, key, value, position_ids, layer_past, **kwargs):
    bs_beam, seq, _ = self.get_runtime_shape(query)
    seq = seq if layer_past is None else layer_past[0].size(2) + 1
    rotary_pos_emb_list = kwargs.pop("rotary_pos_emb_list", None)
    if rotary_pos_emb_list is not None:
        if self.is_first_token_beam_search():
            query, key = apply_rotary_pos_emb(query, key, rotary_pos_emb_list)
            self.runtime_cache.key_prompt = key
        else:
            # FIX: need to optimize
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            query, key = apply_rotary_pos_emb(query, key, rotary_pos_emb_list)
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            if self.is_beam_search():
                self.runtime_cache.key_cache[
                    self.seq_len - 1 : self.seq_len, :, :, :
                ] = key
            else:
                self.runtime_cache.key_cache[
                    self.prev_seq_len : self.seq_len, :, :, :
                ] = key
    else:
        query, key = self.position_embed(
            query, key, position_ids, self.layer_id, self.beam_size, seq
        )
    query, key, value = self.combine_kv_cache_interface(query, key, value)
    return query, key, value


def qwen_sdp(self, query, key, value, attention_mask, head_mask, alibi):
    # QWen needs to initilize attention_mask inside attn module
    causal_mask = None
    key_size = key.shape[2]
    if query.shape[2] == key_size:
        causal_mask = torch.tril(
            torch.ones((key_size, key_size), dtype=torch.bool, device=query.device)
        ).view(1, 1, key_size, key_size)
    if attention_mask is not None:
        attention_mask = attention_mask.expand(-1, -1, query.size(2), -1)
        if causal_mask is not None:
            attention_mask = attention_mask.masked_fill(
                ~causal_mask, torch.finfo(query.dtype).min
            )
    else:
        if causal_mask is not None:
            attention_mask = causal_mask
            new_attention_mask = torch.zeros_like(
                attention_mask, dtype=query.dtype, device=query.device
            )
            attention_mask = new_attention_mask.masked_fill_(
                attention_mask.logical_not(), torch.finfo(query.dtype).min
            )
    if not ipex._C._has_2d_block_array(0):
        return self.naive_sdp(query, key, value, attention_mask, head_mask, alibi)
    key, value, key_prompt, value_prompt = self.sdp_kv_preprocess(key, value)
    (
        dropout,
        alpha,
        beta,
        is_casual,
        blocked_attn_mask,
        blocked_alibi,
    ) = self.prepare_sdp_input(query, key, value, attention_mask, alibi)
    attention_output, attn_weight = self.compute_sdp(
        query,
        key,
        value,
        key_prompt,
        value_prompt,
        blocked_attn_mask,
        blocked_alibi,
        head_mask,
        alpha,
        beta,
        dropout,
        is_casual,
    )
    attention_output = self.process_sdp_output(attention_output)
    attention_output = attention_output.reshape(
        attention_output.size()[:-2] + (self.head_dim * self.num_attn_head,)
    )
    return attention_output, attn_weight
