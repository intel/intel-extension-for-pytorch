import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Tuple


from .transformer_modules.Activation import ACT2FN

from ._transformer_configuration import IPEXTransformerConfig
import os
import math

MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "0"))
MAX_OUT_SEQ_LEN = max(128, int(os.environ.get("MAX_OUT_SEQ_LEN", "0")))
acc_test = os.environ.get("LLM_ACC_TEST", "OFF").upper() in [
    "1",
    "ON",
    "Y",
    "YES",
    "TRUE",
]


def activation_replace(module):
    from transformers.activations import NewGELUActivation

    replace_dict = {NewGELUActivation: torch.nn.GELU(approximate="tanh")}
    for m in replace_dict.keys():
        if isinstance(module, m):
            return replace_dict[m]
    return module


class IPEXEmptyLinear(nn.Module):
    def __init__(self):
        super(IPEXEmptyLinear, self).__init__()
        # we set the weight and bias to None to avoid any possible memory presure
        self.weight = None
        self.bias = None

    def forward(self, input):
        return torch.nn.functional.linear(input, self.weight, bias=self.bias)


class IPEXEmptyINT4Linear(nn.Module):
    def __init__(self):
        super(IPEXEmptyINT4Linear, self).__init__()
        self.qweight = None
        self.bias = None
        self.scales = None
        self.qzeros = None
        self.group_size = 1

    def forward(self, input):
        if self.bias is None:
            return torch.ops.torch_ipex.mm_int4(
                input, self.qweight, self.scales, self.qzeros, self.group_size
            )
        else:
            return torch.ops.torch_ipex.mm_bias_int4(
                input,
                self.qweight,
                self.bias,
                self.scales,
                self.qzeros,
                self.group_size,
            )


class IPEXEmptyINT4LinearWithPadding(nn.Module):
    def __init__(self, n_dim):
        super(IPEXEmptyINT4LinearWithPadding, self).__init__()
        self.qweight = None
        self.scales = None
        self.qzeros = None
        self.group_size = 1
        self.bias = None
        self.n_dim = n_dim

    def forward(self, input):
        if self.bias is None:
            return torch.ops.torch_ipex.mm_int4(
                input, self.qweight, self.scales, self.qzeros, self.group_size
            )[:, :, : self.n_dim]
        else:
            return torch.ops.torch_ipex.mm_bias_int4(
                input,
                self.qweight,
                self.bias,
                self.scales,
                self.qzeros,
                self.group_size,
            )[:, :, : self.n_dim]


class IPEXTransformerAtten(nn.Module):
    layer_id_static = 0
    casual_attention_mask = None
    blocked_alibi = None
    blocked_attn_mask = None
    beam_index = None
    batch_size = 1
    beam_size = 0
    runtime_bs = 0

    def __init__(self, config, is_int4=False) -> None:
        super(IPEXTransformerAtten, self).__init__()
        self.config: IPEXTransformerConfig = config
        self.seq_first = self.config.seq_first
        self.kv_cache_optimize = self.config.kv_cache_optimize
        self.layer_id = IPEXTransformerAtten.layer_id_static
        self.max_positions = self.config.max_positions
        self.max_out_positions = self.config.max_out_positions
        self.use_casual_mask = self.config.use_casual_mask
        self.layer_id = IPEXTransformerAtten.layer_id_static
        self.embed_dim = self.config.embed_dim
        self.num_attn_head = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.tp_size = self.config.tp_size
        self.tp_group = self.config.tp_group
        self.num_attn_head = self.num_attn_head // self.tp_size
        self.num_key_value_heads = self.num_key_value_heads // self.tp_size
        self.num_key_value_groups = self.num_attn_head // self.num_key_value_heads
        IPEXTransformerAtten.layer_id_static += 1
        self.head_dim = self.config.embed_dim // self.config.num_attention_heads
        self.is_int4 = is_int4
        self.position_emb = self.config.rotary_embedding_class(config=self.config)
        self.device = self.config.device
        if self.config.scale_attention:
            self.scale_attn = torch.sqrt(
                torch.tensor(self.head_dim, device=self.config.device)
            )
        else:
            self.scale_attn = None
        self.k_proj = IPEXEmptyLinear() if not is_int4 else IPEXEmptyINT4Linear()
        self.v_proj = IPEXEmptyLinear() if not is_int4 else IPEXEmptyINT4Linear()
        self.q_proj = IPEXEmptyLinear() if not is_int4 else IPEXEmptyINT4Linear()
        self.out_proj = IPEXEmptyLinear() if not is_int4 else IPEXEmptyINT4Linear()

        self.qkv_fused = True
        self.qkv_bias = None
        self.out_bias = None

        if is_int4:
            self.q_qwei = None
            self.k_qwei = None
            self.v_qwei = None
            self.qkv_qwei = None
            self.out_qwei = None
            self.q_scl = None
            self.q_zp = None
            self.k_scl = None
            self.k_zp = None
            self.v_scl = None
            self.v_zp = None
            self.out_scl = None
            self.out_zp = None
            self.qkv_scl = None
            self.qkv_zp = None
            self.qkv_gs = 1
            self.q_gs = 1
            self.k_gs = 1
            self.v_gs = 1
            self.out_gs = 1
        else:
            self.q_wei = None
            self.k_wei = None
            self.v_wei = None
            self.out_wei = None
            self.qkv_wei = None

        col_major = os.environ.get("COL_MAJOR", "OFF").upper() in [
            "1",
            "Y",
            "ON",
            "YES",
            "TRUE",
        ]
        self.row_major = not col_major
        self.key_cached = None
        self.value_cached = None
        self.kv_cache_invalid = True

        seq_first = os.environ.get("SEQ_FIRST", "OFF").upper() in [
            "1",
            "Y",
            "ON",
            "YES",
            "TRUE",
        ]
        disable_kv_cache = os.environ.get("DISABLE_KV_CACHE", "OFF").upper() in [
            "1",
            "Y",
            "ON",
            "YES",
            "TRUE",
        ]
        self.kv_cache = not disable_kv_cache

        self.is_decoder = self.config.is_decoder
        self.residual_drop = (
            nn.Dropout(self.config.residual_dropout)
            if self.config.residual_dropout is not None
            else nn.Identity()
        )
        self.attn_drop = (
            nn.Dropout(self.config.attn_dropout)
            if self.config.attn_dropout is not None
            else nn.Identity()
        )
        if self.use_casual_mask:
            mask = torch.ones(
                (self.max_positions, self.max_positions), dtype=torch.float
            )
            mask = (
                1 - torch.tril(mask).view(1, 1, self.max_positions, self.max_positions)
            ) * (-66504.0)
            IPEXTransformerAtten.attention_mask = mask.to(self.config.device)

        # the cached key/value for the input prompt
        self.key_prompt = None
        self.value_prompt = None
        self.prev_len = 0
        self.cur_len = 0

    @staticmethod
    def update_beam_index(beam_index):
        IPEXTransformerAtten.beam_index = beam_index

    @staticmethod
    def release_all_static_cached_resources():
        IPEXTransformerAtten.casual_attention_mask = None
        IPEXTransformerAtten.blocked_alibi = None
        IPEXTransformerAtten.blocked_attn_mask = None
        IPEXTransformerAtten.beam_index = None

    def release_resources(self):
        self.key_cached = None
        self.value_cached = None
        self.key_prompt = None
        self.value_prompt = None

    def checking_cache(self, layer_past):
        acc_test = os.environ.get("LLM_ACC_TEST", "OFF").upper() in [
            "1",
            "ON",
            "Y",
            "YES",
            "TRUE",
        ]
        if not acc_test:
            return True
        if layer_past is None:
            return True
        prev_key, prev_value = layer_past[0], layer_past[1]
        prev_key_len = prev_key.size(2)
        if not torch.equal(self.key_cached[:prev_key_len, :, :, :], prev_key):
            return False
        if not torch.equal(self.value_cached[:prev_key_len, :, :, :], prev_value):
            return False
        return True

    def qkv_cache_optimized_greedy(
        self, hidden_states, first_token=False, layer_past=None
    ):
        # greedy search path
        # hidden_states has already been converted to [seq, beam, hidden_size]
        seq_len, bs_beam, _ = hidden_states.shape
        if first_token:
            # the first timestep
            kv_shape = [
                self.max_positions,
                bs_beam,
                self.num_key_value_heads,
                self.head_dim,
            ]
            self.key_cached = torch.empty(
                kv_shape, device=hidden_states.device, dtype=hidden_states.dtype
            )
            self.value_cached = torch.empty(
                kv_shape, device=hidden_states.device, dtype=hidden_states.dtype
            )
            if layer_past is not None:
                # for acc test, user can give the layer_past value
                # and already have the corresponding response token
                # layer_past in shape of [bs*beam, head, seq, head_dim]
                self.prev_len = layer_past[0].size(2)
                layer_past_key = layer_past[0].permute(2, 0, 1, 3)
                layer_past_value = layer_past[1].permute(2, 0, 1, 3)
                self.key_cached[:, self.prev_len, :, :, :] = layer_past_key
                self.value_cached[:, self.prev_len, :, :, :] = layer_past_value
            else:
                # for text generation, layer_past should be None for the first token
                # has no response token
                self.prev_len = 0
        else:
            self.prev_len = layer_past[0].size(2)

        self.cur_len = self.prev_len + seq_len
        shape = [seq_len, bs_beam, self.num_attn_head * self.head_dim]
        query = torch.empty(
            shape, device=hidden_states.device, dtype=hidden_states.dtype
        )
        kv_shape = [seq_len, bs_beam, self.num_key_value_heads * self.head_dim]
        key = self.key_cached[self.prev_len : self.cur_len, :, :, :]
        value = self.value_cached[self.prev_len : self.cur_len, :, :, :]
        key = key.view(kv_shape)
        value = value.view(kv_shape)

        if self.num_key_value_groups > 1:
            hidden_states_flat = hidden_states.flatten(0, -2)
            if self.q_proj.bias is None:
                torch.matmul(hidden_states, self.q_wei, out=query)
            else:
                torch.addmm(
                    self.q_proj.bias,
                    hidden_states_flat,
                    self.q_wei,
                    out=query.flatten(0, -2),
                )

            if self.k_proj.bias is None:
                torch.matmul(hidden_states, self.k_wei, out=key)
            else:
                torch.addmm(
                    self.k_proj.bias,
                    hidden_states_flat,
                    self.k_wei,
                    out=key.flatten(0, -2),
                )

            if self.v_proj.bias is None:
                torch.matmul(hidden_states, self.v_wei, out=value)
            else:
                torch.addmm(
                    self.v_proj.bias,
                    hidden_states_flat,
                    self.v_wei,
                    out=value.flatten(0, -2),
                )
        else:
            if self.is_int4 and hidden_states.shape[0] == 1:
                torch.ops.torch_ipex.mm_qkv_out_int4(
                    hidden_states,
                    self.qkv_qwei,
                    self.qkv_scl,
                    self.qkv_zp,
                    self.qkv_bias,
                    query,
                    key,
                    value,
                    self.qkv_gs,
                )
            else:
                torch.ops.torch_ipex.mm_qkv_out(
                    hidden_states, self.qkv_wei, self.qkv_bias, query, key, value
                )

        return query, key, value

    def qkv_cache_optimized_beam(
        self, hidden_states, first_token=False, layer_past=None
    ):
        # beam search path
        # hidden_states keep the original shape [bs*beam, seq, hidden_size]
        device, dtype = hidden_states.device, hidden_states.dtype
        if first_token:
            # the first token, shape will be [bs, seq, hidden_size]
            bs_beam, seq, hidden_size = hidden_states.shape
            q_proj_size = self.num_attn_head * self.head_dim
            kv_proj_size = self.num_key_value_heads * self.head_dim
            shape, kv_shape = [bs_beam, seq, q_proj_size], [bs_beam, seq, kv_proj_size]
            if self.num_key_value_groups > 1:
                query = torch.empty(shape, device=device, dtype=dtype)
                self.key_prompt = torch.empty(kv_shape, device=device, dtype=dtype)
                self.value_prompt = torch.empty(kv_shape, device=device, dtype=dtype)
                hidden_states_flat = hidden_states.flatten(0, -2)
                if self.q_proj.bias is None:
                    torch.matmul(hidden_states, self.q_wei, out=query)
                else:
                    torch.addmm(
                        self.q_proj.bias,
                        hidden_states_flat,
                        self.q_wei,
                        out=query.flatten(0, -2),
                    )

                if self.k_proj.bias is None:
                    torch.matmul(hidden_states, self.k_wei, out=self.key_prompt)
                else:
                    torch.addmm(
                        self.k_proj.bias,
                        hidden_states_flat,
                        self.k_wei,
                        out=self.key_prompt.flatten(0, -2),
                    )

                if self.v_proj.bias is None:
                    torch.matmul(hidden_states, self.v_wei, out=self.value_prompt)
                else:
                    torch.addmm(
                        self.v_proj.bias,
                        hidden_states_flat,
                        self.v_wei,
                        out=self.value_prompt.flatten(0, -2),
                    )
            else:
                query = torch.empty(shape, device=device, dtype=dtype)
                self.key_prompt = torch.empty(shape, device=device, dtype=dtype)
                self.value_prompt = torch.empty(shape, device=device, dtype=dtype)
                if self.is_int4 and hidden_states.shape[0] == 1:
                    torch.ops.torch_ipex.mm_qkv_out_int4(
                        hidden_states,
                        self.qkv_qwei,
                        self.qkv_scl,
                        self.qkv_zp,
                        self.qkv_bias,
                        query,
                        self.key_prompt,
                        self.value_prompt,
                        self.qkv_gs,
                    )
                else:
                    torch.ops.torch_ipex.mm_qkv_out(
                        hidden_states,
                        self.qkv_wei,
                        self.qkv_bias,
                        query,
                        self.key_prompt,
                        self.value_prompt,
                    )
            if layer_past is None:
                # for text generation, layer_past should be None for the first_token
                key = self.key_prompt
                value = self.value_prompt
            else:
                # for accuracy check, layer_past may not be None for the first_token
                # user set the layer_past value
                # already have the response tokens
                # layer_past in format [bs*beam, head, seq, head_dim]
                bs_beam, head, seq, head_dim = layer_past[0].shape
                layer_past_key = (
                    layer_past[0]
                    .permute(2, 0, 1, 3)
                    .reshape([bs_beam, seq, head * head_dim])
                )
                layer_past_value = (
                    layer_past[1]
                    .permute(2, 0, 1, 3)
                    .reshape([bs_beam, seq, head * head_dim])
                )
                self.key_prompt = torch.cat(self.key_prompt, layer_past_key, dim=1)
                self.value_prompt = torch.cat(self.value_prompt, layer_past_key, dim=1)
                key = self.key_prompt
                value = self.value_prompt

            self.prev_len = 0
            self.cur_len = 0
            if bs_beam != IPEXTransformerAtten.runtime_bs:
                self.kv_cache_invalid = True
        else:
            # the 2nd to the last timestep
            # hidden_states has already been converted to [seq, bs*beam, hidden_size]
            seq, bs_beam, hidden_size = hidden_states.shape
            q_proj_size = self.num_attn_head * self.head_dim
            kv_proj_size = self.num_key_value_heads * self.head_dim
            if (
                self.key_cached is None
                or self.value_cached is None
                or self.kv_cache_invalid
            ):
                # the 2nd generated token, create the key_cached and value_cached buffers
                # kv_cahce shape/layout [max_seq, bs*beam, hidden_size]
                shape = [
                    self.max_out_positions,
                    bs_beam,
                    self.num_key_value_heads,
                    self.head_dim,
                ]
                self.key_cached = torch.empty(shape, device=device, dtype=dtype)
                self.value_cached = torch.empty(shape, device=device, dtype=dtype)
                self.prev_len = 0
                self.kv_cache_invalid = False
                if self.layer_id == IPEXTransformerAtten.layer_id_static - 1:
                    IPEXTransformerAtten.runtime_bs = bs_beam

            shape, kv_shape = [seq, bs_beam, q_proj_size], [seq, bs_beam, kv_proj_size]
            self.cur_len = self.prev_len + seq
            query = torch.empty(shape, device=device, dtype=dtype)
            key = self.key_cached[self.prev_len : self.cur_len, :, :, :].view(kv_shape)
            value = self.value_cached[self.prev_len : self.cur_len, :, :, :].view(
                kv_shape
            )

            if self.num_key_value_groups > 1:
                hidden_states_flat = hidden_states.flatten(0, -2)
                if self.q_proj.bias is None:
                    torch.matmul(hidden_states, self.q_wei, out=query)
                else:
                    torch.addmm(
                        self.q_proj.bias,
                        hidden_states_flat,
                        self.q_wei,
                        out=query.flatten(0, -2),
                    )

                if self.k_proj.bias is None:
                    torch.matmul(hidden_states, self.k_wei, out=key)
                else:
                    torch.addmm(
                        self.k_proj.bias,
                        hidden_states_flat,
                        self.k_wei,
                        out=key.flatten(0, -2),
                    )

                if self.v_proj.bias is None:
                    torch.matmul(hidden_states, self.v_wei, out=value)
                else:
                    torch.addmm(
                        self.v_proj.bias,
                        hidden_states_flat,
                        self.v_wei,
                        out=value.flatten(0, -2),
                    )
            else:
                if self.is_int4 and hidden_states.shape[0] == 1:
                    torch.ops.torch_ipex.mm_qkv_out_int4(
                        hidden_states,
                        self.qkv_qwei,
                        self.qkv_scl,
                        self.qkv_zp,
                        self.qkv_bias,
                        query,
                        key,
                        value,
                        self.qkv_gs,
                    )
                else:
                    torch.ops.torch_ipex.mm_qkv_out(
                        hidden_states, self.qkv_wei, self.qkv_bias, query, key, value
                    )
        self.prev_len = self.cur_len
        return query, key, value

    def qkv_normal(self, hidden_states, layer_past=None):
        if self.row_major:
            if self.is_int4 and hidden_states.shape[0] == 1:
                query = torch.ops.torch_ipex.mm_int4(
                    hidden_states, self.q_qwei, self.q_scl, self.q_zp, self.q_gs
                )
                key = torch.ops.torch_ipex.mm_int4(
                    hidden_states, self.k_qwei, self.k_scl, self.k_zp, self.k_gs
                )
                value = torch.ops.torch_ipex.mm_int4(
                    hidden_states, self.v_qwei, self.v_scl, self.v_zp, self.v_gs
                )
            else:
                query = torch.ops.torch_ipex.matmul_bias_out(
                    hidden_states, self.q_wei, self.q_proj.bias
                )
                key = torch.ops.torch_ipex.matmul_bias_out(
                    hidden_states, self.k_wei, self.k_proj.bias
                )
                value = torch.ops.torch_ipex.matmul_bias_out(
                    hidden_states, self.v_wei, self.v_proj.bias
                )
        else:
            query = self.q_proj(hidden_states)
            key = self.k_proj(hidden_states)
            value = self.v_proj(hidden_states)
        return query, key, value

    def compute_qkv(
        self,
        hidden_states: torch.Tensor,
        key_value_state: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        first_token=False,
    ):
        new_shape = hidden_states.size()[:-1] + (self.num_attn_head, self.head_dim)
        new_kv_shape = hidden_states.size()[:-1] + (
            self.num_key_value_heads,
            self.head_dim,
        )
        if self.kv_cache_optimize and self.kv_cache:
            if IPEXTransformerAtten.beam_size == 1:
                query, key, value = self.qkv_cache_optimized_greedy(
                    hidden_states=hidden_states,
                    first_token=first_token,
                    layer_past=layer_past,
                )
            else:
                # TODO: support greedy
                # update prompts status
                new_prompts = True if layer_past is None else False
                if new_prompts:
                    self.key_prompt = None
                    self.value_prompt = None
                query, key, value = self.qkv_cache_optimized_beam(
                    hidden_states=hidden_states,
                    first_token=first_token,
                    layer_past=layer_past,
                )
                if first_token:
                    if self.key_prompt is not None:
                        self.key_prompt = self.key_prompt.view(new_kv_shape)
                    if self.value_prompt is not None:
                        self.value_prompt = self.value_prompt.view(new_kv_shape)
        else:
            query, key, value = self.qkv_normal(
                hidden_states=hidden_states, layer_past=layer_past
            )

        # greedy_search: reshape the qkv size
        # all the tokens: from (seq_len, bs*beam, num_head*head_dim) to (seq_len, bs*beam, num_head, head_dim)
        # beam_search: reshape the qkv size
        # 1st token: from (bs*beam, seq_len, num_head*head_dim) to (bs*beam, seq_len, num_head, head_dim)
        # 2nd to last token: from (seq_len, bs*beam, num_head*head_dim) to (seq_len, bs*beam, num_head, head_dim)
        query = query.view(new_shape)
        key = key.view(new_kv_shape)
        value = value.view(new_kv_shape)
        return query, key, value

    def optimized_combine(self, query, key, value, first_token=False, layer_past=None):
        if first_token and IPEXTransformerAtten.beam_size > 1:
            # 1st token
            # input: shape and layout [bs*beam, seq, num_head, head_dim]
            # output: shape [bs*beam, num_head, seq, head_dim], layout [bs*beam, seq, num_head, head_dim]
            self.key_prompt = self.key_prompt.permute(0, 2, 1, 3)
            self.value_prompt = self.value_prompt.permute(0, 2, 1, 3)
            key = self.key_prompt
            value = self.value_prompt
            query = query.permute(0, 2, 1, 3)
        else:
            # greedy search or beam search 2nd to last token
            # input: shape and layout [seq, bs*beam, num_head, head_dim]
            # output: shape [bs*beam, num_head, seq, head_dim], layout [seq, bs*beam, num_head, head_dim]
            key = self.key_cached[: self.cur_len, :, :, :]
            value = self.value_cached[: self.cur_len, :, :, :]
            key = key.permute(1, 2, 0, 3)
            value = value.permute(1, 2, 0, 3)
            query = query.permute(1, 2, 0, 3)
        return query, key, value

    def normal_combine(self, query, key, value, first_token=False, layer_past=None):
        if self.row_major:
            if first_token and IPEXTransformerAtten.beam_size > 1:
                # 1st token
                # input: shape and layout [bs*beam, seq, num_head, head_dim]
                # output: shape [bs*beam, num_head, seq, head_dim], layout [bs*beam, seq, num_head, head_dim]
                query = query.permute(0, 2, 1, 3)
                key = key.permute(0, 2, 1, 3)
                value = value.permute(0, 2, 1, 3)
            else:
                # 2nd to last token
                # input shape/layout [seq, bs*beam, num_head, head_dim]
                # output: shape [bs*beam, num_head, seq, head_dim], layout [seq, bs*beam, num_head, head_dim]
                query = query.permute(1, 2, 0, 3)
                key = key.permute(1, 2, 0, 3)
                value = value.permute(1, 2, 0, 3)
        else:
            # from [bs*beam, seq, num_head, head_dim] to [bs*beam, num_head, seq, head_dim]
            query = query.permute(0, 2, 1, 3)
            key = key.permute(0, 2, 1, 3)
            value = value.permute(0, 2, 1, 3)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        self.cur_len = key.shape[-2]
        return query, key, value

    def combine_with_cache(self, query, key, value, first_token=False, layer_past=None):
        if self.kv_cache_optimize and self.kv_cache:
            query, key, value = self.optimized_combine(
                query, key, value, first_token, layer_past
            )
        else:
            query, key, value = self.normal_combine(
                query, key, value, first_token, layer_past
            )
        return query, key, value

    def apply_rotary_embedding(self, key, query, position_ids, kv_seq_len):
        return self.position_emb(
            key,
            query,
            position_ids,
            self.layer_id,
            IPEXTransformerAtten.beam_size,
            kv_seq_len,
        )

    def all_reduce_if_necessary(self, reduce_target):
        if self.tp_group is not None:
            dist.all_reduce(reduce_target, group=self.tp_group)
        return reduce_target

    def repeat_kv(
        self, hidden_states: torch.Tensor, n_rep: int, first_token
    ) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        if n_rep == 1:
            return hidden_states

        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if first_token and IPEXTransformerAtten.beam_size > 1:
            # beam search
            # 1st token
            # the shape  of key, value, [bs*beam, kv_head, seq, dim]
            # the layout of key, value, [bs*beam, seq, kv_head, dim]
            hidden_states = hidden_states.permute(0, 2, 1, 3)
            hidden_states = hidden_states[:, :, :, None, :].expand(
                batch, slen, num_key_value_heads, n_rep, head_dim
            )
            hidden_states = hidden_states.reshape(
                batch, slen, num_key_value_heads * n_rep, head_dim
            )
            hidden_states = hidden_states.permute(0, 2, 1, 3)
        else:
            # greedy search or beam search 2nd to last
            # the shape of query, key, value, [bs*beam, kv_head, seq, dim]
            # the layout of query, key, value, [seq, bs*beam, kv_head, dim]
            hidden_states = hidden_states.permute(2, 0, 1, 3)
            hidden_states = hidden_states[:, :, :, None, :].expand(
                slen, batch, num_key_value_heads, n_rep, head_dim
            )
            hidden_states = hidden_states.reshape(
                slen, batch, num_key_value_heads * n_rep, head_dim
            )
            hidden_states = hidden_states.permute(1, 2, 0, 3)
        return hidden_states

    def get_blocked_alibi(self, alibi):
        if self.layer_id == 0:
            shape = [
                alibi.shape[0],
                alibi.shape[1],
                self.max_positions,
            ]  # [beam*num_head, q_len, kv_len]
            IPEXTransformerAtten.blocked_alibi = torch.empty(
                shape, device=alibi.device, dtype=alibi.dtype
            )
            kv_len = alibi.shape[2]
            IPEXTransformerAtten.blocked_alibi[:, :, 0:kv_len] = alibi
        return IPEXTransformerAtten.blocked_alibi

    def get_blocked_attn_mask(self, attn_mask):
        if self.layer_id == 0:
            IPEXTransformerAtten.blocked_attn_mask = torch.empty(
                (
                    attn_mask.shape[0],
                    attn_mask.shape[1],
                    attn_mask.shape[2],
                    self.max_positions,
                ),
                device=attn_mask.device,
                dtype=attn_mask.dtype,
            )
            IPEXTransformerAtten.blocked_attn_mask.fill_(-65504.0)
            IPEXTransformerAtten.blocked_attn_mask[
                :, :, :, 0 : attn_mask.shape[3]
            ] = attn_mask
        return IPEXTransformerAtten.blocked_attn_mask

    def naive_self_attention(
        self,
        query,
        key,
        value,
        attention_mask=None,
        head_mask=None,
        alibi: torch.Tensor = None,
        first_token=False,
    ):
        if alibi is not None:
            bs_beam, num_heads, q_length, dim = query.shape
            _, _, kv_length, _ = key.shape
            # query, key result [bs*beam, num_head, q_len, kv_len]
            # alibi: [bs_beam*num_head, q_len, kv_len]
            if first_token and IPEXTransformerAtten.beam_size > 1:
                shape = [
                    IPEXTransformerAtten.batch_size,
                    IPEXTransformerAtten.beam_size,
                    num_heads,
                    -1,
                    kv_length,
                ]
                alibi = alibi.view(shape)[:, 0, :, :, :].reshape(
                    [IPEXTransformerAtten.batch_size * num_heads, -1, kv_length]
                )
            batch1 = query.view(-1, q_length, dim)
            batch2 = key.view(-1, kv_length, dim).transpose(1, 2)
            matmul_result = alibi.baddbmm(
                batch1=batch1,
                batch2=batch2,
                beta=self.beta,
                alpha=self.inv_norm_factor,
            )

            # change view to [bs_beam, num_heads, q_length, kv_length]
            attention_scores = matmul_result.view(
                bs_beam, num_heads, q_length, kv_length
            )
            attn_weights = torch.masked_fill(
                attention_scores,
                attention_mask,
                torch.finfo(attention_scores.dtype).min,
            )
            attention_probs = nn.functional.softmax(attn_weights, dim=-1)

            # [bs_beam, num_heads, q_length, kv_length]
            attention_probs = self.attn_drop(attention_probs)

            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            # matmul: [bs_beam * num_heads, q_length, head_dim]
            attn_output = torch.matmul(attention_probs, value)
        else:
            attn_weights = torch.matmul(query, key.transpose(-1, -2))

            if self.use_casual_mask:
                # convert the casual mask dtype to target dtype, this should only happen once
                IPEXTransformerAtten.attention_mask.to(attn_weights.dtype)
                query_length, key_length = query.size(-2), key.size(-2)
                casual_mask = IPEXTransformerAtten.attention_mask[
                    :, :, key_length - query_length : key_length, :key_length
                ]
                # # TODO: Maybe we can move this line to the initializer
                # casual_mask *= -66504.0
                # replace torch.where as torch.add might helps with the host overhead
                attn_weights += casual_mask
            if self.scale_attn:
                attn_weights /= self.scale_attn
            if attention_mask is not None:
                attn_weights += attention_mask
                # the attn_weights should anyway bigger than dtype.min, I wonder if this is necessary
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float
            ).to(query.dtype)
            attn_weights = self.attn_drop(attn_weights)
            if head_mask is not None:
                attn_weights = attn_weights * head_mask
            attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def self_attention(
        self,
        key_prompt,
        value_prompt,
        query,
        key,
        value,
        attention_mask=None,
        head_mask=None,
        alibi=None,
        first_token=False,
    ):
        # create the beam_idx for greedy search
        if self.layer_id == 0 and IPEXTransformerAtten.beam_size == 1:
            IPEXTransformerAtten.beam_index = torch.zeros(
                [self.cur_len, IPEXTransformerAtten.beam_size],
                dtype=torch.int,
                device=self.device,
            )

        do_sdp_fusion = os.environ.get("ENABLE_SDP_FUSION", "OFF").upper() in [
            "1",
            "Y",
            "ON",
            "YES",
            "TRUE",
        ]
        if self.config.sdp_fusion_enable and do_sdp_fusion:
            attn_weights = None

            # parameters
            dropout = 0.0
            alpha = 1.0 / math.sqrt(self.head_dim)
            beta = 1.0  # TODO: ignored by native
            is_causal = False
            if self.use_casual_mask is True and query.shape[2] != 1:
                is_causal = True

            blocked_attn_mask = None
            if attention_mask is not None:
                # transform the attention_mask to casual mask if the attention_mask is in bool
                if attention_mask.dtype == torch.bool:
                    blocked_attn_mask = None
                    if query.shape[2] != 1:
                        is_causal = True
                else:
                    blocked_attn_mask = self.get_blocked_attn_mask(attention_mask)

            blocked_alibi = None
            if alibi is not None:
                blocked_alibi = self.get_blocked_alibi(alibi)
            if first_token or IPEXTransformerAtten.beam_size == 1:
                # 1st token
                # query/key/value shape [bs*beam, head, q_seq, dim], layout [bs*beam, q_seq, head, dim]
                # attn_output shape [bs*beam, head, q_seq, dim], layout [bs*beam, q_seq, head, dim]
                # attention_mask= None
                # is_causal=True
                seq_first = False
                if IPEXTransformerAtten.beam_size == 1:
                    seq_first = True
                attn_output = torch.xpu.IpexSDP(
                    query,
                    key,
                    value,
                    blocked_alibi,
                    blocked_attn_mask,
                    head_mask,
                    alpha,
                    beta,
                    dropout,
                    is_causal,
                    seq_first,
                )
            else:
                # 2nd to the last token
                # key_prompt shape [bs, head, prompt_seq, dim], layout [bs, prompt_seq, head, dim]
                # value_prompt shape [bs, head, prompt_seq, dim], layout [bs, prompt_seq, head, dim]
                # query shape [bs*beam, head, q_seq, dim], layout [q_seq, bs*beam, head, dim]
                # key shape [bs*beam, head, kv_seq, dim], layout [kv_seq, bs*beam, head, dim]
                # value shape [bs*beam, head, kv_seq, dim], layout [kv_seq, bs*beam, head, dim]
                # attn_output shape [bs*beam, head, seq, dim], layout [seq, bs*beam, head, dim]
                # attention_mask= None
                # is_causal=False
                # beam_idx shape [kv_len, beam], layout [kv_len, beam], dtype=int32
                # self.cur_len = kv_len
                attn_output = torch.xpu.IpexSDP_Index(
                    query,
                    key_prompt,
                    value_prompt,
                    key,
                    value,
                    IPEXTransformerAtten.beam_index,
                    blocked_alibi,
                    blocked_attn_mask,
                    head_mask,
                    self.cur_len,
                    alpha,
                    beta,
                    dropout,
                    is_causal,
                )
        else:
            if not first_token and IPEXTransformerAtten.beam_size > 1:
                key, value = self.reorder_cache(
                    key_prompt,
                    value_prompt,
                    key,
                    value,
                    IPEXTransformerAtten.beam_index,
                )
            attn_output, attn_weights = self.naive_self_attention(
                query,
                key,
                value,
                attention_mask=attention_mask,
                head_mask=head_mask,
                alibi=alibi,
                first_token=first_token,
            )

        if self.row_major:
            if first_token and IPEXTransformerAtten.beam_size > 1:
                # 1st token
                # reshape the attn_output shape/layour from shape [bs*beam, num_head, seq_len, head_dim],
                # layout [bs*beam, seq_len, num_head, head_dim]
                # to shape [bs*beam, seq_len, hidden_size], layout [bs*beam, seq_len, hidden_size]
                attn_output = attn_output.permute(0, 2, 1, 3)
            else:
                # 2nd to last token
                # reshape the attn_output shape/layour from shape [bs*beam, num_head, seq_len, head_dim],
                # layout [seq_len, bs*beam, num_head, head_dim]
                # to shape [seq_len, bs*beam, hidden_size], layout [seq_len, bs*beam, hidden_size]
                attn_output = attn_output.permute(2, 0, 1, 3)
            attn_output = attn_output.reshape(
                attn_output.size()[:-2] + (self.embed_dim // self.tp_size,)
            )
        else:
            # [bs*beam, head, q_seq, dim] x [bs*beam, head, dim, kv_seq] = [bs*beam, head, q_seq, kv_seq]
            # [bs*beam, head, q_seq, kv_seq] x [bs*beam, head, kv_seq, dim] = [bs*beam, head, q_seq, dim]
            attn_output = attn_output.permute(0, 2, 1, 3)
            attn_output = attn_output.reshape(
                attn_output.size()[:-2] + (self.embed_dim // self.tp_size,)
            )
        return attn_output, attn_weights

    """
    def get_beam_width(self):
        return IPEXTransformerAtten.beam_index.shape[1] // IPEXTransformerAtten.batch_size 
        if IPEXTransformerAtten.beam_index != None else 1
    """

    def expand_beam_idx(self):
        bs = IPEXTransformerAtten.batch_size
        beam_idx = IPEXTransformerAtten.beam_index
        beam = beam_idx.shape[1] // bs
        expand_beam_idx = torch.empty_like(beam_idx)
        for i in range(bs):
            expand_beam_idx[:, i * beam : (i + 1) * beam] = (
                beam_idx[:, i * beam : (i + 1) * beam] + beam * i
            )
        return expand_beam_idx

    def reorder_cache(self, key_prompt, value_prompt, key, value, beam_idx_cache):
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        key_prompt = key if key_prompt is None else key_prompt
        value_prompt = value if value_prompt is None else value_prompt
        # past_key_values: 28 decoder layers of [key, value]
        # beam_idx_cache: [kv_out_len, bs*beam]

        # key_prompt shape[bs, head, seq, dim] layout[bs, seq, head, dim]
        # key shape[bs*beam, head, kv_seq, dim] layout[kv_len, bs*beam, head, dim]
        bs = key_prompt.shape[0]
        beam = int(key.shape[0] // bs)
        _, num_head, seq_prompt, head_dim = key_prompt.shape
        prompt_shape = [bs, 1, num_head, seq_prompt, head_dim]
        expand_shape = [bs, beam, num_head, seq_prompt, head_dim]
        shape = [bs * beam, num_head, seq_prompt, head_dim]
        # expand the key_prompt/value_prompt from shape [bs, num_head, seq_prompt, head_dim]
        # to shape [bs*beam, num_head, seq_prompt, head_dim]
        key_prompt = (
            key_prompt.reshape(prompt_shape).expand(expand_shape).reshape(shape)
        )
        value_prompt = (
            value_prompt.reshape(prompt_shape).expand(expand_shape).reshape(shape)
        )
        key_list = [key_prompt]
        value_list = [value_prompt]

        beam_idx_cache = self.expand_beam_idx()
        for idx in range(beam_idx_cache.shape[0]):
            beam_idx = beam_idx_cache[idx]
            current_key = key[:, :, idx, :].view(bs * beam, num_head, 1, head_dim)
            current_key = current_key.index_select(0, beam_idx.to(key.device))
            key_list.append(current_key)
            current_value = value[:, :, idx, :].view(bs * beam, num_head, 1, head_dim)
            current_value = current_value.index_select(0, beam_idx.to(value.device))
            value_list.append(current_value)

        key = torch.cat(key_list, dim=2)
        value = torch.cat(value_list, dim=2)
        return key, value

    def get_final_output(
        self, attn_output: torch.Tensor, residual: Optional[torch.Tensor] = None
    ):
        if self.row_major:
            if residual is None:
                if self.is_int4 and attn_output.shape[0] == 1:
                    attn_output = torch.ops.torch_ipex.mm_int4(
                        attn_output,
                        self.out_qwei,
                        self.out_scl,
                        self.out_zp,
                        self.out_gs,
                    )
                else:
                    attn_output = torch.matmul(attn_output, self.out_wei)
                if self.out_bias is not None:
                    attn_output += self.out_bias
                self.all_reduce_if_necessary(attn_output)
            else:
                shape = [attn_output.shape[0], attn_output.shape[1], self.embed_dim]
                if self.out_bias is not None:
                    if self.is_int4 and attn_output.shape[0] == 1:
                        attn_output = torch.ops.torch_ipex.mm_bias_resadd_int4(
                            attn_output,
                            self.out_wei,
                            self.out_bias,
                            residual,
                            1.0 / self.tp_size,
                        )
                    else:
                        attn_output = torch.ops.torch_ipex.mm_bias_resadd(
                            attn_output,
                            self.out_wei,
                            self.out_bias,
                            1.0 / self.tp_size,
                            residual,
                            1.0 / self.tp_size,
                        )
                else:
                    attn_output = torch.addmm(
                        residual.flatten(0, -2),
                        attn_output.flatten(0, -2),
                        self.out_wei,
                        beta=1.0 / self.tp_size,
                    )
                attn_output = attn_output.view(shape)
                self.all_reduce_if_necessary(attn_output)
        else:
            attn_output = torch.matmul(attn_output, self.out_proj.weight.t())
            self.all_reduce_if_necessary(attn_output)
            if residual is not None:
                attn_output += residual
        return attn_output

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        key_value_states: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        residual: Optional[torch.Tensor] = None,
        alibi: torch.Tensor = None,
        first_token=False,
    ):
        def print_rank_x(i, content):
            if dist.get_rank() == 1:
                print(content)

        # if self.row_major:
        #     if first_token and IPEXTransformerAtten.beam_size > 1:
        #         # [bs*beam, seq, head, head_dim]
        #         kv_seq_len = hidden_states.shape[1]
        #     else:
        #         kv_seq_len = hidden_states.shape[0]
        # else:
        #     kv_seq_len = hidden_states.shape[1]
        # if layer_past is not None:
        #     kv_seq_len += layer_past[0].shape[2]

        # greedy_search
        # the shape of query, key, value, [seq, bs*beam, head*dim]
        # the layout of query, key, value, [seq, bs*beam, head*dim]
        # beam_search
        # 1st token:
        # the shape of query, key, value, [bs*beam, seq, head*dim]
        # the layout of query, key, value, [bs*beam, seq, head*dim]
        # 2nd to last:
        # the shape of query, key, value, [seq, bs*beam, head*dim]
        # the layout of query, key, value, [seq, bs*beam, head*dim]
        query, key, value = self.compute_qkv(
            hidden_states=hidden_states,
            key_value_state=key_value_states,
            layer_past=layer_past,
            first_token=first_token,
        )

        # greedy_search
        # the shape of query, key, value, [seq, bs*beam, head, dim]
        # the layout of query, key, value, [seq, bs*beam, head, dim]
        # beam_search
        # 1st token
        # the shape of query, key, value, [bs*beam, seq, head, dim]
        # the layout of query, key, value, [bs*beam, seq, head, dim]
        # 2nd to last:
        # the shape of query, key, value, [seq, bs*beam, head, dim]
        # the layout of query, key, value, [seq, bs*beam, head, dim]
        query, key = self.apply_rotary_embedding(query, key, position_ids, kv_seq_len)

        # greedy search
        # the shape of query, key, value, [seq, bs*beam, head, dim]
        # the layout of query, key, value, [seq, bs*beam, head, dim]
        # 1st token
        # the shape of query, key, value, [bs*beam, seq, head, dim]
        # the layout of query, key, value, [bs*beam, seq, head, dim]
        # 2nd to last:
        # the shape of query, key, value, [seq, bs*beam, head, dim]
        # the layout of query, key, value, [seq, bs*beam, head, dim]
        query, key, value = self.combine_with_cache(
            query, key, value, first_token, layer_past
        )

        if use_cache or self.is_decoder:
            if IPEXTransformerAtten.beam_size == 1:
                present = (key, value)
            else:
                if self.row_major:
                    # key, value shape [bs*beam=1, head, seq, dim]
                    seq_len = (
                        self.cur_len
                        if self.key_prompt is None
                        else self.cur_len + self.key_prompt.shape[2]
                    )
                    cache_shape = (
                        IPEXTransformerAtten.beam_size,
                        key.shape[1],
                        seq_len,
                        key.shape[3],
                    )
                    key_cache = torch.empty(
                        cache_shape, device=key.device, dtype=key.dtype
                    )
                    value_cache = key_cache
                    present = (key_cache, value_cache)
                else:
                    present = (key, value)
        else:
            present = None

        # greedy search
        # the shape of query, key, value, [bs*beam, head, seq, dim]
        # the layout of query, key, value, [seq, bs*beam, head, dim]
        # beam search
        # 1st token
        # the shape of query, key, value, [bs*beam, head, seq, dim]
        # the layout of query, key, value, [bs*beam, seq, head, dim]
        # 2nd to last:
        # the shape of query, key, value, [bs*beam, head, seq, dim]
        # the layout of query, key, value, [seq, bs*beam, head, dim]
        # GQA/MQA repeat k/v heads if n_kv_heads < n_heads
        if first_token or IPEXTransformerAtten.beam_size == 1:
            key = self.repeat_kv(key, self.num_key_value_groups, first_token)
            value = self.repeat_kv(value, self.num_key_value_groups, first_token)
            key_prompt, value_prompt = key, value
        else:
            key = self.repeat_kv(key, self.num_key_value_groups, first_token)
            value = self.repeat_kv(value, self.num_key_value_groups, first_token)
            key_prompt = self.repeat_kv(
                self.key_prompt, self.num_key_value_groups, first_token
            )
            value_prompt = self.repeat_kv(
                self.value_prompt, self.num_key_value_groups, first_token
            )

        # greedy search
        # the shape of query, key, value, [bs*beam, head, seq, dim]
        # the layout of query, key, value, [seq, bs*beam, head, dim]
        # beam search
        # 1st token
        # the shape of query, key, value, [bs*beam, head, seq, dim]
        # the layout of query, key, value, [bs*beam, seq, head, dim]
        # 2nd to last
        # the shape of query, key, value, [bs*beam, head, seq, dim]
        # the layout of query, key, value, [seq, bs*beam, head, dim]
        attn_output, attn_weight = self.self_attention(
            key_prompt,
            value_prompt,
            query,
            key,
            value,
            attention_mask,
            head_mask,
            alibi,
            first_token,
        )

        # greedy search
        # the shape of attn_output [seq, bs*beam, hidden_size]
        # the layout of attn_output [seq, bs*beam, hidden_size]
        # beam search
        # 1st token
        # the shape of attn_output [bs*beam, seq, hidden_size]
        # the layout of attn_output [bs*beam, seq, hidden_size]
        # 2nd to last:
        # the shape of attn_output [seq, bs*beam, hidden_size]
        # the layout of attn_output [seq, bs*beam, hidden_size]
        attn_output = self.get_final_output(attn_output=attn_output, residual=residual)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weight,)
        else:
            outputs += (None,)

        return outputs  # return as (attn_output, present, output_atten)


class IPEXTransformerMLP(nn.Module):
    batch_size = 1

    def __init__(self, config: IPEXTransformerConfig, is_int4=False):
        super().__init__()
        self.fc_in = IPEXEmptyLinear() if not is_int4 else IPEXEmptyINT4Linear()
        self.fc_out = IPEXEmptyLinear() if not is_int4 else IPEXEmptyINT4Linear()
        self.act = ACT2FN[config.activation_function]
        self.drop_out = (
            nn.Dropout(config.residual_pdrop)
            if config.residual_pdrop is not None
            else nn.Identity()
        )

        self.tp_size = config.tp_size
        self.tp_group = config.tp_group
        col_major = os.environ.get("COL_MAJOR", "OFF").upper() in [
            "1",
            "Y",
            "ON",
            "YES",
            "TRUE",
        ]
        self.row_major = not col_major
        self.fc_in_bias = None
        self.fc_out_bias = None
        self.is_int4 = is_int4

        if is_int4:
            self.fc_in_qwei = None
            self.fc_out_qwei = None
            self.fc_in_scl = None
            self.fc_in_zp = None
            self.fc_out_scl = None
            self.fc_out_zp = None
        else:
            self.fc_in_wei = None
            self.fc_out_wei = None

    @staticmethod
    def release_resources():
        pass

    def all_reduce_if_necessary(self, target):
        if self.tp_group is not None:
            dist.all_reduce(target, group=self.tp_group)
        return target

    def forward(self, hidden_states: Optional[torch.Tensor]):
        if self.row_major:
            if isinstance(self.act, nn.GELU):
                hidden_states = torch.ops.torch_ipex.matmul_gelu(
                    hidden_states,
                    self.fc_in_wei,
                    self.fc_in.bias,
                    1.0,
                    self.act.approximate,
                )
            else:
                hidden_states = torch.ops.torch_ipex.matmul_bias_out(
                    hidden_states, self.fc_in_wei, self.fc_in.bias
                )
                hidden_states = self.act(hidden_states)
            hidden_states = torch.ops.torch_ipex.matmul_bias_out(
                hidden_states, self.fc_out_wei, self.fc_out.bias
            )
        else:
            hidden_states = self.fc_in(hidden_states)
            hidden_states = self.act(hidden_states)
            hidden_states = self.fc_out(hidden_states)
        return self.drop_out(hidden_states)


class IPEXTransformerConverter:
    tp_group = None
    tp_size = 1

    def __init__(
        self, module, config, device="cpu", dtype=torch.float, name=""
    ) -> None:
        self.module = module
        self.config = config
        self.dtype = dtype
        self.device = device
        col_major = os.environ.get("COL_MAJOR", "OFF").upper() in [
            "1",
            "Y",
            "ON",
            "YES",
            "TRUE",
        ]
        self.row_major = not col_major
        self.module_name = name

    def construct_transformer_config(self):
        pass

    def construct_ipex_optimized_module(self):
        pass

    def port_attn_parameters(self):
        pass

    def port_mlp_parameters(self):
        pass

    def port_layer_norm_parameters(self):
        pass

    def port_block_parameters(self):
        pass

    def port_all_parameters_to_new_module(self):
        pass

    def get_transformed_model_to_run(self):
        pass

    @staticmethod
    def update_tp_data(tp_size, tp_group):
        IPEXTransformerConverter.tp_size = tp_size
        IPEXTransformerConverter.tp_group = tp_group
