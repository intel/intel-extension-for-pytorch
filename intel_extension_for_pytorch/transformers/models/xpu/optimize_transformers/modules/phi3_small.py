import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from functools import partial

from .transformer_modules.RoPE import Phi3SmallRotaryEmbedding

from ._transformers import MAX_OUT_SEQ_LEN
from .transformer_modules.XPUAttentionfp16 import IPEXAttention
from .transformer_modules.XPUAttentionInt4 import (
    IPEXAttentionInt4,
    IPEXAttentionInt4OneDNN,
)

from .transformer_modules.QuantizedMlp import *  # noqa
from .transformer_modules.BaseAttention import IPEXTransformerAttn
from ._transformer_configuration import (
    IPEXTransformerConfigPhi3Small,
    SupportedActivation,
)
from .transformer_modules.DecoderBlock import IPEXTransformerBlock
from .transformer_modules.model_utils import (
    load_attn_fused_qkv_params,
    transpose_attn_fused_qkv_params,
    xpu_gemm_use_xetla,
)
from .transformer_modules.CacheUtils import IPEXStaticCache, CacheFormat


def _cdiv(x, y):
    return (x + y - 1) // y


def _get_mask_dense(query, key, block_size, vert_stride, local_blocks, device):
    bs, num_attn_head, q_len, head_dim = query.shape
    kv_len = key.shape[-2]

    N_BLOCK = _cdiv(kv_len, block_size)
    q_pos = torch.arange(N_BLOCK)[None, :, None]
    k_pos = torch.arange(N_BLOCK)[None, None]
    head_sliding_step = max(
        1, int(vert_stride / num_attn_head)
    )  # if vert_stride <= n_heads, rotating the heads
    mask_vert_strided = [
        (torch.arange(N_BLOCK) + h * head_sliding_step + 1) % vert_stride == 0
        for h in range(num_attn_head)
    ]
    mask_vert_strided = torch.vstack(mask_vert_strided).unsqueeze(1)
    block_mask_dense = (
        ((q_pos >= k_pos) & ((q_pos - k_pos < local_blocks) | mask_vert_strided))
        .to(device)
        .to(dtype=torch.uint8)
    )
    mask_dense = torch.kron(
        block_mask_dense, block_mask_dense.new_ones((block_size, block_size))
    )
    causal_mask = torch.tril(torch.ones(kv_len, kv_len)).type_as(mask_dense)
    mask_dense = (mask_dense[..., -kv_len:, :kv_len] * causal_mask[None])[None, :, :, :]

    return mask_dense


def _phi3small_sdp(
    self,
    query,
    key,
    value,
    past_key_value,
    attention_mask,
    head_mask,
    alibi,
    scale,
    use_causal,
):
    """
    q: (bs_beam, num_attn_head, q_len, head_dim), kv: (bs_beam, num_kv_head, kv_seq_len, head_dim)
    esimd fmha expects [b, n, f, h] contiguous layout

    Phi3-small uses block-sparse attention and dense attention in a interleaving order,
    controlled by dense_attention_every_n_layers
    """
    if self.config.dense_attention_every_n_layers and (
        (self.layer_id + 1) % self.config.dense_attention_every_n_layers == 0
    ):
        # dense attention
        causal_mask = None
        if query.size(2) == key.size(2):
            seq_len = query.size(2)
            causal_mask = torch.ones((seq_len, seq_len), dtype=torch.uint8)
            causal_mask = (
                torch.tril(causal_mask)
                .view(1, 1, seq_len, seq_len)
                .repeat(1, self.num_heads, 1, 1)
                .contiguous()
            )
            causal_mask = causal_mask.to(self.config.device)
        else:
            bs, num_head, q_len, _ = query.shape
            kv_len = key.size(2)
            causal_mask = torch.ones(
                (bs, num_head, q_len, kv_len), dtype=torch.uint8
            ).to(self.config.device)
        attention_output = torch.ops.torch_ipex.fmha_esimd(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            causal_mask,
            True,
        )

    else:
        # block sparse attention
        block_mask_dense = _get_mask_dense(
            query,
            key,
            self.config.blocksparse_block_size,
            self.config.blocksparse_vert_stride,
            self.config.blocksparse_num_local_blocks,
            self.config.device,
        )
        if query.shape[-2] == 1:
            block_mask_dense = block_mask_dense[:, :, -1:, :].contiguous()
        attention_output = torch.ops.torch_ipex.fmha_esimd(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            block_mask_dense,
            True,
        )
    return attention_output, None


def phi3small_prepare_inputs_for_generation(
    self,
    input_ids: torch.LongTensor,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    **kwargs,
):
    cache_position = kwargs.get("cache_position", None)
    if past_key_values:
        if inputs_embeds is not None:
            input_ids = input_ids[:, -cache_position.shape[0] :]
        elif input_ids.shape[1] != cache_position.shape[0]:
            input_ids = input_ids[:, cache_position]

    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]
    else:
        position_ids = None

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
    )
    return model_inputs


class NewIPEXPhi3SmallDecoderLayer(IPEXTransformerBlock):
    def __init__(
        self,
        module,
        config,
        layer_idx,
        dtype="fp16",
        device="xpu",
        module_name="",
        impl_mode=None,
        tp_size=1,
        tp_group=None,
    ):
        super().__init__(module, config, dtype, device, module_name)
        self.ipex_config = self.build_ipex_transformer_config(
            config, device, dtype, impl_mode, tp_size, tp_group
        )
        self.layer_idx = layer_idx

        if dtype == "fp16":
            self.self_attn = IPEXAttention(self.ipex_config, module.self_attn.layer_idx)
        elif dtype == "int4" and xpu_gemm_use_xetla():
            self.self_attn = IPEXAttentionInt4(
                self.ipex_config, module.self_attn.layer_idx
            )
        elif dtype == "int4" and not xpu_gemm_use_xetla():
            self.self_attn = IPEXAttentionInt4OneDNN(
                self.ipex_config, module.self_attn.layer_idx
            )
        else:
            raise NotImplementedError(
                "IPEXAttention dose not support this modelType {} !".format(dtype)
            )

        self.self_attn.sdp = partial(_phi3small_sdp, self.self_attn)
        self.self_attn.position_embed = self.ipex_config.rotary_embedding_class(
            self.ipex_config, torch.float16
        )
        self.mlp = self.build_mlp_from_config("phi3small")
        self.input_layernorm = nn.LayerNorm(
            self.ipex_config.embedding_dim, self.ipex_config.norm_eps
        )
        self.post_attention_layernorm = nn.LayerNorm(
            self.ipex_config.embedding_dim, self.ipex_config.norm_eps
        )
        self.port_all_parameters_to_new_module()

    def build_ipex_transformer_config(
        self, config, device, dtype, impl_mode, tp_size, tp_group
    ) -> IPEXTransformerConfigPhi3Small:
        activation_function = self.config.hidden_act
        ipex_activation = None
        for act in SupportedActivation:
            if activation_function in act.value:
                ipex_activation = act
                break
        assert ipex_activation is not None, (
            "found unrecognized activation function,"
            "can not build ipex config from {}".format(activation_function)
        )

        assert dtype in [
            "fp16",
            "int4",
        ], "dtype tag {} passed to optimized_transformers is not supported!".format(
            dtype
        )

        return IPEXTransformerConfigPhi3Small(
            embedding_dim=self.config.hidden_size,
            intermediate_dim=self.config.intermediate_size,
            num_attention_head=self.config.num_attention_heads,
            num_key_value_head=self.config.num_key_value_heads,
            max_positions=self.config.max_position_embeddings,
            max_out_positions=MAX_OUT_SEQ_LEN,
            # ========== rope config =====================
            rotary_embedding_class=Phi3SmallRotaryEmbedding,
            positional_embedding_base=self.config.rope_embedding_base,
            rope_position_scale=self.config.rope_position_scale,
            rope_scaling=self.config.rope_scaling,
            # rotary_dim=self.config.rotary_dim,
            # ========== Block Sparse Attention Pattern ==
            blocksparse_homo_head_pattern=self.config.blocksparse_homo_head_pattern,
            blocksparse_block_size=self.config.blocksparse_block_size,
            blocksparse_num_local_blocks=self.config.blocksparse_num_local_blocks,
            blocksparse_vert_stride=self.config.blocksparse_vert_stride,
            blocksparse_triton_kernel_block_size=self.config.blocksparse_triton_kernel_block_size,
            dense_attention_every_n_layers=self.config.dense_attention_every_n_layers,
            # ============================================
            use_causal_mask=False,
            activation_function=activation_function,
            ipex_act=ipex_activation,
            ffn_dropout_prob=self.config.ffn_dropout_prob,
            gegelu_limit=self.config.gegelu_limit,
            norm_eps=self.config.layer_norm_epsilon,
            # attn_dropout=self.config.attention_dropout,
            scale_attention=self.config.mup_use_scaling,
            mup_attn_multiplier=self.config.mup_attn_multiplier,
            dtype=dtype,
            impl=impl_mode,
            tp_size=tp_size,
            tp_group=tp_group,
        )

    def port_attn_parameter(self):
        self.self_attn.load_parameter(
            self.module.self_attn.query_key_value,
            self.module.self_attn.dense,
            dtype=self.ipex_config.dtype,
            grouped=True,
        )

    def port_mlp_parameter(self):
        self.mlp.load_parameter(self.module.mlp.up_proj, self.module.mlp.down_proj)

    def port_norm_parameter(self):
        self.input_layernorm.weight = self.module.input_layernorm.weight
        self.input_layernorm.bias = self.module.input_layernorm.bias
        self.post_attention_layernorm.weight = (
            self.module.post_attention_layernorm.weight
        )
        self.post_attention_layernorm.bias = self.module.post_attention_layernorm.bias

    def transpose_parameter(self):
        self.self_attn.transpose_parameter(dtype=self.ipex_config.dtype, grouped=True)
        self.mlp.transpose_parameter()

    def port_all_parameters_to_new_module(self):
        self.self_attn.load_parameter = partial(
            load_attn_fused_qkv_params, self.self_attn
        )
        self.self_attn.transpose_parameter = partial(
            transpose_attn_fused_qkv_params, self.self_attn
        )
        super().port_all_parameters_to_new_module()
        if self.ipex_config.transpose:
            self.transpose_parameter()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[
        Tuple[torch.Tensor],
        Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]],
    ]:
        bs = IPEXTransformerAttn.batch_size
        seq_len = hidden_states.size(1)
        IPEXTransformerAttn.beam_size = hidden_states.shape[0] // bs

        residual = hidden_states
        hidden_states = torch.ops.torch_ipex.fast_layer_norm(
            hidden_states,
            self.input_layernorm.normalized_shape,
            self.input_layernorm.weight,
            self.input_layernorm.bias,
            self.input_layernorm.eps,
        )

        if (
            isinstance(past_key_values, IPEXStaticCache)
            and past_key_values.cache_format == CacheFormat.FBNH
            and not self.self_attn.beam_search_first_iter(seq_len)
            and cache_position is None
        ):
            cache_position = torch.arange(seq_len, device=self.device)

        hidden_states, present_key_value, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            residual=residual,
        )

        residual = hidden_states
        hidden_states = torch.ops.torch_ipex.fast_layer_norm(
            hidden_states,
            self.post_attention_layernorm.normalized_shape,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.bias,
            self.post_attention_layernorm.eps,
        )
        hidden_states = self.mlp(hidden_states, residual)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (past_key_values,)
        return outputs
