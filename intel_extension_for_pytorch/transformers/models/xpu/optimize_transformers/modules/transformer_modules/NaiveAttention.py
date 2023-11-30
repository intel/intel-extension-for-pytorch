import torch
import torch.nn as nn
import torch.distributed as dist

# from intel_extension_for_pytorch.nn.utils._transformer_configuration import IPEXTransformerConfig
from .._transformer_configuration import IPEXTransformerConfig
from .Linear import IPEXTransformerLinear, IPEXTransformerQLinear  # noqa
from .BaseAttention import IPEXTransformerAttn, IPEXRuntimeAttnCache
import os

enable_naive_path = os.environ.get("ENABLE_NAIVE_PATH", "OFF").upper() in [
    "1",
    "Y",
    "ON",
    "YES",
    "TRUE",
]


class IPEXTransformerAttnNaive(IPEXTransformerAttn):
    beam_idx = None
    expanded_beam_idx_cache = None
    timestep = 0
    attention_mask = None

    def __init__(self, config: IPEXTransformerConfig) -> None:
        super().__init__(config)
        self.config = config
        self.embed_dim = config.embedding_dim
        self.num_attn_head = config.num_attention_head
        self.tp_size = config.tp_size
        self.tp_group = config.tp_group
        self.head_dim = self.embed_dim // self.num_attn_head
        self.num_attn_head = self.num_attn_head // self.tp_size
        self.max_position = config.max_positions
        self.max_out_position = config.max_out_positions
        self.seq_len = 0
        self.prev_seq_len = 0
        self.is_decoder = config.is_decoder
        self.use_casual_mask = config.use_casual_mask

        if self.config.scale_attention:
            self.scale_attn = torch.sqrt(
                torch.tensor(self.head_dim, device=self.config.device)
            )
            self.scale_attn_scalar = self.scale_attn.item()
        else:
            self.scale_attn = None

        if self.use_casual_mask:
            mask = torch.ones((self.max_position, self.max_position), dtype=torch.float)
            mask = (
                1 - torch.tril(mask).view(1, 1, self.max_position, self.max_position)
            ) * (-66504.0)
            IPEXTransformerAttnNaive.attention_mask = mask.to(self.config.device)

        self.qkv_proj = IPEXTransformerLinear()

        self.q_proj = IPEXTransformerLinear()
        self.k_proj = IPEXTransformerLinear()
        self.v_proj = IPEXTransformerLinear()

        self.out_proj = IPEXTransformerLinear()

        self.attn_drop = (
            nn.Dropout(self.config.attn_dropout)
            if self.config.attn_dropout is not None
            else nn.Identity()
        )

        self.runtime_cache = IPEXRuntimeAttnCache()

    @staticmethod
    def release_resources(self):
        self.runtime_cache.clear_cache()

    @staticmethod
    def update_beam_idx(beam_idx):
        IPEXTransformerAttnNaive.beam_idx = beam_idx

    def load_parameter(self, q_proj, k_proj, v_proj, out_proj):
        self.q_proj.weight = q_proj.weight
        self.k_proj.weight = k_proj.weight
        self.v_proj.weight = v_proj.weight
        self.out_proj.weight = out_proj.weight

        self.q_proj.bias = q_proj.bias
        self.k_proj.bias = k_proj.bias
        self.v_proj.bias = v_proj.bias
        self.out_proj.bias = out_proj.bias
        self.position_embed = self.config.rotary_embedding_class(
            self.config, q_proj.weight.dtype
        )

    def transpose_parameter(self):
        self.q_proj.weight.data = self.q_proj.weight.transpose(0, 1).contiguous()
        self.k_proj.weight.data = self.k_proj.weight.transpose(0, 1).contiguous()
        self.v_proj.weight.data = self.v_proj.weight.transpose(0, 1).contiguous()
        self.out_proj.weight.data = self.out_proj.weight.transpose(0, 1).contiguous()
        # Note: synchronize to ensure the completion of contiguous
        torch.xpu.synchronize()

    def cat_qkv(self):
        shape = [3, -1, self.q_proj.weight.shape[-1]]
        self.qkv_proj.weight = (
            torch.stack([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight])
            .contiguous()
            .view(shape)
        )
        # Note: synchronize to ensure the completion of contiguous
        torch.xpu.synchronize()

        self.q_proj.weight.data = self.qkv_proj.weight[0, :, :]
        self.k_proj.weight.data = self.qkv_proj.weight[1, :, :]
        self.v_proj.weight.data = self.qkv_proj.weight[2, :, :]

        if self.q_proj.bias is not None:
            bias_shape = [3, -1]
            self.qkv_proj.bias = (
                torch.stack([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias])
                .contiguous()
                .view(bias_shape)
            )
            # Note: synchronize to ensure the completion of contiguous
            torch.xpu.synchronize()

            self.q_proj.bias.data = self.qkv_proj.bias[0]
            self.k_proj.bias.data = self.qkv_proj.bias[1]
            self.v_proj.bias.data = self.qkv_proj.bias[2]

    # ################################ pre_qkv ######################################################
    def pre_qkv(self, hidden_states, layer_past, **kwargs):
        if self.is_beam_search():
            self.prepare_cache_for_beam_search(hidden_states, layer_past)

    def prepare_cache_for_beam_search(self, hidden_states, layer_past):
        self.prepare_kv_prompt(hidden_states)
        self.prepare_kv_cache(hidden_states)
        if self.is_1st_token_beam_search():
            self.prev_seq_len = 0
            self.seq_len = 0
        else:
            self.seq_len = self.prev_seq_len + 1

    def prepare_kv_prompt(self, hidden_states):
        bs_beam, seq_len, embed_dim = self.get_runtime_shape(hidden_states)
        if (
            self.runtime_cache.key_prompt is None
            or self.runtime_cache.value_prompt is None
            or IPEXTransformerAttn.timestamp == 0
        ):
            out_shape = [bs_beam, seq_len, self.head_dim * self.num_attn_head]
            self.runtime_cache.key_prompt = torch.empty(
                out_shape, device=hidden_states.device, dtype=hidden_states.dtype
            )
            self.runtime_cache.value_prompt = torch.empty(
                out_shape, device=hidden_states.device, dtype=hidden_states.dtype
            )

    def prepare_kv_cache(self, hidden_states):
        bs_beam, seq_len, embed_dim = self.get_runtime_shape(hidden_states)
        if (
            self.runtime_cache.key_cache is None
            or self.runtime_cache.key_cache.shape[1] != bs_beam
        ):
            cache_shape = [
                self.max_position,
                bs_beam,
                self.num_attn_head,
                self.head_dim,
            ]
            self.runtime_cache.key_cache = torch.empty(
                cache_shape, device=hidden_states.device, dtype=hidden_states.dtype
            )
            self.runtime_cache.value_cache = torch.empty(
                cache_shape, device=hidden_states.device, dtype=hidden_states.dtype
            )

    # ################################ qkv_gemm ################################################
    def qkv_gemm(self, hidden_states, key_value_states, layer_past, **kwargs):
        if self.is_beam_search():
            query, key, value = self.prepare_qkv_input(hidden_states)
            query, key, value = self.compute_qkv_gemm(hidden_states, query, key, value)
            query, key, value = self.process_qkv_output(
                hidden_states, query, key, value
            )
        else:
            query = self.q_proj(hidden_states)
            key = self.k_proj(hidden_states)
            value = self.v_proj(hidden_states)

            out_shape = hidden_states.size()[:-1] + (self.num_attn_head, self.head_dim)
            query = query.view(out_shape)
            key = key.view(out_shape)
            value = value.view(out_shape)

        return query, key, value

    # ################################ prepare_qkv_input ###########################################

    def prepare_qkv_input(self, hidden_states, **kwargs):
        # assert False, "prepare_qkv_input() in Attention.py have not been properly dispatched during runtime"
        if self.is_1st_token_beam_search():
            return self.prepare_qkv_input_1st_token_beam_search(hidden_states)
        else:
            return self.prepare_qkv_input_2nd2last(hidden_states)

    def prepare_qkv_input_1st_token_beam_search(self, hidden_states, **kwargs):
        bs_beam, seq_len, embed_dim = self.get_runtime_shape(hidden_states)
        out_shape = [bs_beam, seq_len, self.head_dim * self.num_attn_head]
        query = torch.empty(
            out_shape, device=hidden_states.device, dtype=hidden_states.dtype
        )
        return query, self.runtime_cache.key_prompt, self.runtime_cache.value_prompt

    def prepare_qkv_input_2nd2last(self, hidden_states, **kwargs):
        bs_beam, seq_len, embed_dim = self.get_runtime_shape(hidden_states)
        out_shape = [seq_len, bs_beam, self.head_dim * self.num_attn_head]
        query = torch.empty(
            out_shape, device=hidden_states.device, dtype=hidden_states.dtype
        )
        return query, self.runtime_cache.key_cache, self.runtime_cache.value_cache

    # ################################# compute_qkv #################################################

    def compute_qkv_gemm(self, hidden_states, query, key, value):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        if self.is_1st_token():
            self.runtime_cache.key_prompt = key
            self.runtime_cache.value_prompt = value
        else:
            self.runtime_cache.key_cache = key
            self.runtime_cache.value_cache = value
        return query, key, value

    # ################################# process_qkv_output ###########################################
    def process_qkv_output(self, hidden_states, query, key, value):
        out_shape = hidden_states.size()[:-1] + (self.num_attn_head, self.head_dim)
        if self.is_1st_token_beam_search():
            self.runtime_cache.key_prompt = self.runtime_cache.key_prompt.view(
                out_shape
            )
            self.runtime_cache.value_prompt = self.runtime_cache.value_prompt.view(
                out_shape
            )
        if self.is_beam_search():
            self.prev_seq_len = self.seq_len
        query = query.view(out_shape)
        key = key.view(out_shape)
        value = value.view(out_shape)
        return query, key, value

    # ################################## post qkv #######################################################
    def post_qkv(self, query, key, value, position_ids, layer_past=None):
        bs_beam, seq, _ = self.get_runtime_shape(query)
        key, query = self.position_embed(
            key, query, position_ids, self.layer_id, self.beam_size, seq
        )
        if self.is_beam_search():
            query, key, value = self.combine_kv_cache_interface(
                query, key, value, layer_past
            )
        else:
            query = query.permute(1, 2, 0, 3)
            key = key.permute(1, 2, 0, 3)
            value = value.permute(1, 2, 0, 3)
            key, value = self.cat_past_kv(key, value, layer_past)
        return query, key, value

    def cat_past_kv(self, key, value, layer_past=None):
        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        self.seq_len = key.shape[-2]
        return key, value

    # ################################## combine_kv_cache ################################################################
    def combine_kv_cache_interface(self, query, key, value, layer_past=None):
        if self.is_1st_token_beam_search():
            return self.combine_kv_cache_1st_token_beam_search(
                query, key, value, layer_past
            )
        else:
            return self.combine_kv_cache_2nd2last(query, key, value, layer_past)

    def combine_kv_cache_1st_token_beam_search(
        self, query, key, value, layer_past=None
    ):
        self.runtime_cache.key_prompt = self.runtime_cache.key_prompt.permute(
            0, 2, 1, 3
        )
        self.runtime_cache.value_prompt = self.runtime_cache.value_prompt.permute(
            0, 2, 1, 3
        )
        query = query.permute(0, 2, 1, 3)
        return query, self.runtime_cache.key_prompt, self.runtime_cache.value_prompt

    def combine_kv_cache_2nd2last(self, query, key, value, layer_past=None):
        query = query.permute(1, 2, 0, 3)
        key = key.permute(1, 2, 0, 3)
        value = value.permute(1, 2, 0, 3)
        if IPEXTransformerAttn.timestamp != 1:
            key, value = self.cat_past_kv(key, value, layer_past)
        return query, key, value

    # ##################################### get present ###############################################
    def get_present(self, query, key, value, use_cache):
        present = None
        if use_cache or self.is_decoder:
            present = (key, value)
        return present

    # ##################################### pre sdp ####################################################
    def pre_sdp(self, key, value):
        return key, value

    def sdp_kv_preprocess(self, key, value):
        if self.is_1st_token_beam_search():
            return self.sdp_kv_preprocess_1st_token_beam_search(key, value)
        else:
            return self.sdp_kv_preprocess_2nd2last(key, value)

    def sdp_kv_preprocess_1st_token_beam_search(self, key, value):
        key = self.repeat_kv(key, 1)
        value = self.repeat_kv(value, 1)
        key_prompt, value_prompt = key, value
        return key, value, key_prompt, value_prompt

    def sdp_kv_preprocess_2nd2last(self, key, value):
        key = self.repeat_kv(key, 1)
        value = self.repeat_kv(value, 1)
        key_prompt = self.repeat_kv(self.runtime_cache.key_prompt, 1)
        value_prompt = self.repeat_kv(self.runtime_cache.value_prompt, 1)
        return key, value, key_prompt, value_prompt

    # ################################################################ sdp ##########################################
    def sdp(self, query, key, value, attention_mask, head_mask, alibi):
        if self.is_beam_search():
            key, value, key_prompt, value_prompt = self.sdp_kv_preprocess(key, value)
            if not self.is_1st_token():
                key, value = self.reorder_cache(
                    key_prompt, value_prompt, key, value, self.beam_idx
                )
            attn_output, attn_weights = self.naive_self_attention(
                query,
                key,
                value,
                attention_mask=attention_mask,
                head_mask=head_mask,
                alibi=alibi,
                first_token=self.is_1st_token(),
            )
            attn_output = self.process_sdp_output(attn_output)
            attn_output = attn_output.reshape(
                attn_output.size()[:-2] + (self.head_dim * self.num_attn_head,)
            )
        else:
            attn_output, attn_weights = self.naive_self_attention(
                query,
                key,
                value,
                attention_mask=attention_mask,
                head_mask=head_mask,
                alibi=alibi,
                first_token=self.is_1st_token(),
            )

            # if self.is_1st_token_beam_search():
            #     attn_output = attn_output.permute(0, 2, 1, 3)
            # else:
            #     attn_output = attn_output.permute(2, 0, 1, 3)
            attn_output = attn_output.permute(0, 2, 1, 3)
            attn_output = attn_output.reshape(
                attn_output.size()[:-2] + (self.embed_dim // self.tp_size,)
            )
        return attn_output, attn_weights

    def process_sdp_output(self, attention_output):
        if self.is_1st_token_beam_search():
            return self.process_sdp_output_1st_token_beam_search(attention_output)
        else:
            return self.process_sdp_output_general(attention_output)

    def process_sdp_output_1st_token_beam_search(self, attention_output):
        return attention_output.permute(0, 2, 1, 3)

    def process_sdp_output_general(self, attention_output):
        return attention_output.permute(2, 0, 1, 3)

    def expand_beam_idx(self):
        bs = IPEXTransformerAttn.batch_size
        beam_idx = IPEXTransformerAttnNaive.beam_idx
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

        beam_idx_cache = self.expand_beam_idx()
        current_timestep = beam_idx_cache.shape[0]

        if current_timestep != IPEXTransformerAttnNaive.timestep:
            # update expanded_beam_idx_cache
            IPEXTransformerAttnNaive.timestep = current_timestep
            beam_idx_cache = beam_idx_cache.transpose(0, 1)
            expanded_beam_idx_cache = (
                beam_idx_cache.repeat_interleave(head_dim, dim=1)
                .reshape((-1, current_timestep, head_dim))
                .unsqueeze(1)
            )
            # gather only support LongTensor as index
            expanded_beam_idx_cache = torch.cat(
                [expanded_beam_idx_cache] * num_head, dim=1
            ).long()
            IPEXTransformerAttnNaive.expanded_beam_idx_cache = expanded_beam_idx_cache
        reordered_past_key = torch.gather(
            key, dim=0, index=IPEXTransformerAttnNaive.expanded_beam_idx_cache
        )
        reordered_past_value = torch.gather(
            value, dim=0, index=IPEXTransformerAttnNaive.expanded_beam_idx_cache
        )

        key = torch.cat([key_prompt, reordered_past_key], dim=2)
        value = torch.cat([value_prompt, reordered_past_value], dim=2)
        return key, value

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
            if first_token and IPEXTransformerAttn.beam_size > 1:
                shape = [
                    IPEXTransformerAttn.batch_size,
                    IPEXTransformerAttn.beam_size,
                    num_heads,
                    -1,
                    kv_length,
                ]
                alibi = alibi.view(shape)[:, 0, :, :, :].reshape(
                    [IPEXTransformerAttn.batch_size * num_heads, -1, kv_length]
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
                IPEXTransformerAttnNaive.attention_mask.to(attn_weights.dtype)
                query_length, key_length = query.size(-2), key.size(-2)
                casual_mask = IPEXTransformerAttnNaive.attention_mask[
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

    # ########################################################################### post sdp #############################

    def post_sdp(self, attn_output, residual=None):
        attn_output = self.out_proj(attn_output)
        self.all_reduce_if_necessary(attn_output)
        if residual is not None:
            attn_output += residual
        return attn_output

    def all_reduce_if_necessary(self, reduce_target):
        if self.tp_group is not None:
            dist.all_reduce(reduce_target, group=self.tp_group)
        return reduce_target

    def repeat_kv(self, kv, n_rep):
        if n_rep == 1:
            return kv
        bs_beam, num_kv_heads, seq_len, head_dim = kv.shape
        if self.timestamp == 0 and self.beam_num > 1:
            kv = kv.permute(0, 2, 1, 3)
            kv = kv[:, :, :, None, :].expand(
                bs_beam, seq_len, num_kv_heads, n_rep, head_dim
            )
            kv = kv.reshape(bs_beam, seq_len, num_kv_heads * n_rep, head_dim)
            kv = kv.permute(1, 2, 0, 3)
        else:
            kv = kv.permute(2, 0, 1, 3)
            kv = kv[:, :, :, None, :].expand(
                seq_len, bs_beam, num_kv_heads, n_rep, head_dim
            )
            kv = kv.reshape(seq_len, bs_beam, num_kv_heads * n_rep, head_dim)
            kv = kv.permute(1, 2, 0, 3)
        return kv

    def get_runtime_shape(self, hidden_states):
        # This api should always return the shape attr in [bs * beam, seq_len, num_head, head_dim]
        if self.timestamp == 0:
            return (
                hidden_states.shape[0],
                hidden_states.shape[1],
                hidden_states.shape[2],
            )
        else:
            return hidden_states.shape[1], hidden_states.shape[0], hidden_states[2:]

    def is_naive_implementation():  # noqa
        return True
