import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import DynamicCache
from .transformer_modules.RoPE import GPTJRotaryEmbedding
from ._transformers import MAX_OUT_SEQ_LEN
from .transformer_modules.QuantizedAttention import (  # noqa F401
    IPEXTransformerAttnOptimizedFp16,
    IPEXTransformerAttnOptimizedInt4,
)  # noqa
from .transformer_modules.NaiveAttention import IPEXTransformerAttnNaive  # noqa
from .transformer_modules.BaseAttention import IPEXTransformerAttn
from .transformer_modules.Mlp import *  # noqa
from .transformer_modules.QuantizedMlp import *  # noqa
from ._transformer_configuration import IPEXTransformerConfig, SupportedActivation
from .transformer_modules.DecoderBlock import IPEXTransformerBlock
import os
from .transformer_modules.XPUAttentionfp16 import (
    IPEXAttention,
)
from .transformer_modules.XPUAttentionInt4 import (
    IPEXAttentionInt4,
    IPEXAttentionInt4OneDNN,
)

enable_naive_path = os.environ.get("ENABLE_NAIVE_PATH", "OFF").upper() in [
    "1",
    "Y",
    "ON",
    "YES",
    "TRUE",
]


def GPTJ_prepare_inputs_for_generation(
    self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
):
    if past_key_values is None:
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
        else:
            past_key_values = DynamicCache()
    token_type_ids = kwargs.get("token_type_ids", None)
    # Omit tokens covered by past_key_values
    if past_key_values.get_seq_length() != 0:
        past_length = past_key_values.get_seq_length()
        # Some generation methods already pass only the last input ID
        if input_ids.shape[1] > past_length:
            remove_prefix_length = past_length
        else:
            # Default to old behavior: keep only final ID
            remove_prefix_length = input_ids.shape[1] - 1

        input_ids = input_ids[:, remove_prefix_length:]
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values.get_seq_length() != 0:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values.get_seq_length() != 0:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
    )

    return model_inputs


def GPTJModel_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, input_shape[-1])

    if past_key_values is None:
        past_length = 0
        past_key_values = tuple([None] * len(self.h))
    else:
        past_length = past_key_values.get_seq_length()

    cache_position = torch.arange(
        past_length, past_length + input_ids.shape[1], device=device
    )

    if position_ids is None:
        position_ids = torch.arange(
            past_length, input_shape[-1] + past_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)

    if not hasattr(self, "_use_flash_attention_") or not self._use_flash_attention_2:
        # Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x num_attention_heads x N x N
    # head_mask has shape n_layer x batch x num_attention_heads x N x N
    head_mask = self.get_head_mask(head_mask, self.config.n_layer)

    if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)

    hidden_states = inputs_embeds

    if token_type_ids is not None:
        token_type_embeds = self.wte(token_type_ids)
        hidden_states = hidden_states + token_type_embeds

    hidden_states = self.drop(hidden_states)

    output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    presents = () if use_cache else None
    all_self_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None
    for i, block in enumerate(self.h):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(hidden_states.device)
            # Ensure layer_past is on same device as hidden_states (might not be correct)
            if layer_past is not None:
                layer_past = tuple(
                    past_state.to(hidden_states.device) for past_state in layer_past
                )
            # Ensure that attention_mask is always on the same device as hidden_states
            if attention_mask is not None:
                attention_mask = attention_mask.to(hidden_states.device)
            if isinstance(head_mask, torch.Tensor):
                head_mask = head_mask.to(hidden_states.device)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:
            outputs = self._gradient_checkpointing_func(
                block.__call__,
                hidden_states,
                None,
                attention_mask,
                position_ids,
                head_mask[i],
                use_cache,
                output_attentions,
            )
        else:
            outputs = block(
                hidden_states=hidden_states,
                layer_past=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
                layer_idx=i,
                cache_position=cache_position,
            )
        hidden_states = outputs[0]
        if use_cache is True:
            presents = outputs[1]

        if output_attentions:
            all_self_attentions = all_self_attentions + (
                outputs[2 if use_cache else 1],
            )

        # Model Parallel: If it's the last layer for that device, put things on the next device
        if self.model_parallel:
            for k, v in self.device_map.items():
                if i == v[-1] and "cuda:" + str(k) != self.last_device:
                    hidden_states = hidden_states.to("cuda:" + str(k + 1))

    hidden_states = self.ln_f(hidden_states)

    hidden_states = hidden_states.view(output_shape)
    # Add last hidden state
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, presents, all_hidden_states, all_self_attentions]
            if v is not None
        )

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


class NewIPEXGPTJBlock(IPEXTransformerBlock):
    def __init__(
        self,
        module,
        config,
        dtype="fp16",
        device="xpu",
        module_name="",
        impl_mode=None,
        tp_size=1,
        tp_group=None,
        **kwargs,
    ):
        super().__init__(module, config, dtype, device, module_name)
        self.ipex_config = self.build_ipex_transformer_config(
            config, device, dtype, impl_mode, tp_size, tp_group
        )
        # self.self_attn = self.build_attention_from_config(grouped=grouped)

        if dtype == "fp16":
            self.attn = IPEXAttention(self.ipex_config)
        elif dtype == "int4" and xpu_gemm_use_xetla():
            self.attn = IPEXAttentionInt4(self.ipex_config)
        elif dtype == "int4" and not xpu_gemm_use_xetla():
            self.attn = IPEXAttentionInt4OneDNN(self.ipex_config)
        else:
            raise NotImplementedError(
                "IPEXAttention dose not support this modelType {} !".format(dtype)
            )

        self.mlp = self.build_mlp_from_config("gptj")
        self.ln = nn.LayerNorm(
            self.ipex_config.embedding_dim, eps=self.ipex_config.norm_eps
        )
        # self.ln = LlamaRMSNorm(self.ipex_config.embedding_dim, eps=self.ipex_config.norm_eps)
        self.port_all_parameters_to_new_module()
        # self.mlp = IPEXGPTJMLP(config)

    def build_ipex_transformer_config(
        self, config, device, dtype, impl_mode, tp_size, tp_group
    ) -> IPEXTransformerConfig:
        activation_function = self.config.activation_function
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

        return IPEXTransformerConfig(
            embedding_dim=self.config.n_embd,
            intermediate_dim=self.config.n_inner,
            num_attention_head=self.config.n_head,
            num_key_value_head=self.config.n_head,
            max_positions=self.config.n_positions,
            max_out_positions=MAX_OUT_SEQ_LEN,
            rotary_embedding_class=GPTJRotaryEmbedding,
            rotary_dim=self.config.rotary_dim,
            use_causal_mask=True,
            activation_function=activation_function,
            ipex_act=ipex_activation,
            norm_eps=self.config.layer_norm_epsilon,
            residual_dropout=self.config.resid_pdrop,
            attn_dropout=self.config.attn_pdrop,
            residual_pdrop=self.config.resid_pdrop,
            scale_attention=True,
            dtype=dtype,
            impl=impl_mode,
            tp_size=tp_size,
            tp_group=tp_group,
        )

    def port_attn_parameter(self):
        self.attn.load_parameter(
            self.module.attn.q_proj,
            self.module.attn.k_proj,
            self.module.attn.v_proj,
            self.module.attn.out_proj,
        )

    def port_mlp_parameter(self):
        self.mlp.load_parameter(self.module.mlp.fc_in, self.module.mlp.fc_out)

    def port_norm_parameter(self):
        self.ln.weight = self.module.ln_1.weight
        self.ln.bias = self.module.ln_1.bias

    def transpose_parameter(self):
        self.attn.transpose_parameter()
        self.mlp.transpose_parameter()

    def port_all_parameters_to_new_module(self):
        super().port_all_parameters_to_new_module()
        if self.ipex_config.transpose:
            self.transpose_parameter()
        self.attn.cat_qkv()

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        position_embeddings: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Union[
        Tuple[torch.Tensor],
        Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]],
    ]:
        # hidden_states:  [bs*beam, seq, hidden_size]
        # position_ids:   [bs*beam, seq]
        # attention_mask: [bs*beam, head, q_seq, kv_seq]
        bs = IPEXTransformerAttn.batch_size
        dim = hidden_states.dim()
        IPEXTransformerAttn.beam_size = hidden_states.shape[0] // bs
        IPEXTransformerMLP.beam_size = hidden_states.shape[0] // bs

        _, seq, hidden_size = hidden_states.shape
        beam = IPEXTransformerAttn.beam_size
        first_token = True if seq > 1 else False
        if first_token and beam > 1:
            hidden_states = hidden_states.view(bs, beam, seq, hidden_size)[
                :, 0, :, :
            ].contiguous()
            if position_ids is not None:
                position_ids = position_ids.view(bs, beam, position_ids.shape[1])[
                    :, 0, :
                ].view(bs, position_ids.shape[1])
            if attention_mask is not None:
                attention_mask = attention_mask.view(
                    bs,
                    beam,
                    attention_mask.shape[1],
                    attention_mask.shape[2],
                    attention_mask.shape[3],
                )[:, 0, :, :, :].view(
                    bs,
                    attention_mask.shape[1],
                    attention_mask.shape[2],
                    attention_mask.shape[3],
                )
        if hasattr(layer_past, "max_batch_size") and layer_past.max_batch_size < beam:
            repeat_cnt = beam // layer_past.max_batch_size
            for i in range(len(layer_past.key_cache)):
                layer_past.key_cache[i] = layer_past.key_cache[i].repeat(
                    1, repeat_cnt, 1, 1
                )
                layer_past.value_cache[i] = layer_past.value_cache[i].repeat(
                    1, repeat_cnt, 1, 1
                )
            layer_past.max_batch_size = beam

        residual = hidden_states
        hidden_states = torch.ops.torch_ipex.fast_layer_norm(
            hidden_states,
            self.ln.normalized_shape,
            self.ln.weight,
            self.ln.bias,
            self.ln.eps,
        )
        self.attn.layer_idx = layer_idx
        attn_outputs = self.attn(
            hidden_states=hidden_states,
            past_key_value=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            cache_position=cache_position,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]

        hidden_states = self.mlp(hidden_states, attn_output, residual)

        if first_token and beam > 1:
            hidden_states = hidden_states.view(bs, 1, seq, hidden_size)
            hidden_states = hidden_states.expand([bs, beam, seq, hidden_size]).view(
                bs * beam, seq, hidden_size
            )

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        return outputs
