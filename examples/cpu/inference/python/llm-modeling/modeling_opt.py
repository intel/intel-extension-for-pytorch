from typing import List, Optional, Tuple, Union

import torch
import intel_extension_for_pytorch as ipex
import torch.utils.checkpoint
from torch import nn

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.opt.configuration_opt import OPTConfig
from transformers.models.opt.modeling_opt import OPTPreTrainedModel


class OPTAttention(nn.Module):
    def __init__(
        self,
        config: OPTConfig,
        is_decoder: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.config = config

        def _handle_deprecated_argument(config_arg_name, config, fn_arg_name, kwargs):
            val = None
            if fn_arg_name in kwargs:
                val = kwargs.pop(fn_arg_name)
            else:
                val = getattr(config, config_arg_name)
            return val

        self.embed_dim = _handle_deprecated_argument(
            "hidden_size", config, "embed_dim", kwargs
        )
        self.num_heads = _handle_deprecated_argument(
            "num_attention_heads", config, "num_heads", kwargs
        )
        self.dropout = _handle_deprecated_argument(
            "attention_dropout", config, "dropout", kwargs
        )
        self.enable_bias = _handle_deprecated_argument(
            "enable_bias", config, "bias", kwargs
        )

        self.head_dim = self.embed_dim // self.num_heads
        self.is_causal = True

        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        # ==================== Changes to apply ipex.llm layers  ====================
        self._IPEXIndirectAccessKVCacheAttention = ipex.llm.modules.IndirectAccessKVCacheAttention(
            config.max_position_embeddings
        )
        # ==========================================================================

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        if is_cross_attention and past_key_value is not None:
            key = (
                past_key_value[0]
                .view(bsz, tgt_len, self.num_heads, self.head_dim)
                .contiguous()
            )
            value = (
                past_key_value[1]
                .view(bsz, tgt_len, self.num_heads, self.head_dim)
                .contiguous()
            )
        elif is_cross_attention:
            key = (
                self.k_proj(key_value_states)
                .view(bsz, tgt_len, self.num_heads, self.head_dim)
                .contiguous()
            )
            value = (
                self.v_proj(key_value_states)
                .view(bsz, tgt_len, self.num_heads, self.head_dim)
                .contiguous()
            )
        else:
            key = (
                self.k_proj(hidden_states)
                .view(bsz, tgt_len, self.num_heads, self.head_dim)
                .contiguous()
            )
            value = (
                self.v_proj(hidden_states)
                .view(bsz, tgt_len, self.num_heads, self.head_dim)
                .contiguous()
            )
        query = (
            self.q_proj(hidden_states)
            .view(bsz, tgt_len, self.num_heads, self.head_dim)
            .contiguous()
        )
        # ==================== Changes to apply ipex.llm layers  ====================
        (
            attn_output,
            attn_weights,
            past_key_value_decoder,
        ) = self._IPEXIndirectAccessKVCacheAttention(
            query,
            key,
            value,
            1 / self.scaling,
            past_key_value,
            layer_head_mask,
            attention_mask,
        )
        # ==========================================================================
        if self.is_decoder:
            past_key_value = past_key_value_decoder

        if not output_attentions:
            attn_weights_reshaped = None
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)

        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        # move to OPTDecoderLayer to enable linear+add fusion
        # attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class OPTDecoderLayer(nn.Module):
    def __init__(self, config: OPTConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = OPTAttention(config=config, is_decoder=True)
        self.do_layer_norm_before = config.do_layer_norm_before
        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)
        self.final_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states

        if self.do_layer_norm_before:
            # ==================== orignal path  ====================
            # hidden_states = self.self_attn_layer_norm(hidden_states)
            # ==================== Changes to apply ipex.llm layers  ====================
            # option 1 : replace module
            # if not hasattr(self, "ipex_layernorm_1"):
            #     self.ipex_layernorm_1 = ipex.llm.modules.FastLayerNorm(
            #         self.embed_dim,
            #         eps=self.eps,
            #         weight=self.self_attn_layer_norm.weight,
            #         bias=self.self_attn_layer_norm.bias,
            #     )
            #     del self.self_attn_layer_norm
            # hidden_states = self.ipex_layernorm_1(hidden_states)
            #
            # option 2 : use function call
            hidden_states = ipex.llm.functional.fast_layer_norm(
                hidden_states,
                [self.embed_dim],
                self.self_attn_layer_norm.weight,
                self.self_attn_layer_norm.bias,
                1e-05,
            )
            # ==========================================================================

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # ==================== orignal path  ====================
        # hidden_states = self.self_attn.out_proj(hidden_states) +residual
        # ==================== Changes to apply ipex.llm layers  ====================
        if not hasattr(self, "ipex_fusion_0"):
            self.ipex_fusion_0 = ipex.llm.modules.LinearAdd(self.self_attn.out_proj)
            del self.__dict__["_modules"]["self_attn"].out_proj
        hidden_states = self.ipex_fusion_0(hidden_states, residual)
        # ==========================================================================

        if not self.do_layer_norm_before:
            # ==================== orignal path  ====================
            # hidden_states = self.self_attn_layer_norm(hidden_states)
            # ==================== Changes to apply ipex.llm layers  ====================
            # option 1 : replace module
            # if not hasattr(self, "ipex_layernorm_1"):
            #     self.ipex_layernorm_1 = ipex.llm.modules.FastLayerNorm(
            #         self.embed_dim,
            #         eps=self.eps,
            #         weight=self.self_attn_layer_norm.weight,
            #         bias=self.self_attn_layer_norm.bias,
            #     )
            #     del self.self_attn_layer_norm
            # hidden_states = self.ipex_layernorm_1(hidden_states)
            #
            # option 2 : use function call
            hidden_states = ipex.llm.functional.fast_layer_norm(
                hidden_states,
                [self.embed_dim],
                self.self_attn_layer_norm.weight,
                self.self_attn_layer_norm.bias,
                1e-05,
            )
            # ==========================================================================

        hidden_states_shape = hidden_states.shape
        residual = hidden_states

        if self.do_layer_norm_before:
            # ==================== orignal path  ====================
            # hidden_states = self.final_layer_norm(hidden_states)
            # ==================== Changes to apply ipex.llm layers  ====================
            # option 1 : replace module
            # if not hasattr(self, "ipex_layernorm_2"):
            #     self.ipex_layernorm_2 = ipex.llm.modules.FastLayerNorm(
            #         self.embed_dim,
            #         eps=self.eps,
            #         weight=self.final_layer_norm.weight,
            #         bias=self.final_layer_norm.bias,
            #     )
            #     del self.final_layer_norm
            # hidden_states = self.ipex_layernorm_2(hidden_states)
            #
            # option 2 : use function call
            hidden_states = ipex.llm.functional.fast_layer_norm(
                hidden_states,
                [self.embed_dim],
                self.final_layer_norm.weight,
                self.final_layer_norm.bias,
                1e-05,
            )
            # ==========================================================================

        # ==================== orignal path  ====================
        # hidden_states = torch.nn.functional.relu(self.fc1(hidden_states))
        # ==================== Changes to apply ipex.llm layers  ====================
        if not hasattr(self, "ipex_fusion_1"):
            self.ipex_fusion_1 = ipex.llm.modules.LinearRelu(self.fc1)
            del self.__dict__["_modules"]["fc1"]
        hidden_states = self.ipex_fusion_1(hidden_states)
        # ==========================================================================

        # ==================== orignal path  ====================
        # hidden_states = self.fc2(hidden_states) + residual
        # ==================== Changes to apply ipex.llm layers  ====================
        if not hasattr(self, "ipex_fusion_2"):
            self.ipex_fusion_2 = ipex.llm.modules.LinearAdd(self.fc2)
            del self.__dict__["_modules"]["fc2"]
        hidden_states = self.ipex_fusion_2(hidden_states, residual)
        # ==========================================================================

        hidden_states = hidden_states.view(hidden_states_shape)

        if not self.do_layer_norm_before:
            # ==================== orignal path  ====================
            # hidden_states = self.final_layer_norm(hidden_states)
            # ==================== Changes to apply ipex.llm layers  ====================
            # option 1 : replace module
            # if not hasattr(self, "ipex_layernorm_2"):
            #     self.ipex_layernorm_2 = ipex.llm.modules.FastLayerNorm(
            #         self.embed_dim,
            #         eps=self.eps,
            #         weight=self.final_layer_norm.weight,
            #         bias=self.final_layer_norm.bias,
            #     )
            #     del self.final_layer_norm
            # hidden_states = self.ipex_layernorm_2(hidden_states)
            #
            # option 2 : use function call
            hidden_states = ipex.llm.functional.fast_layer_norm(
                hidden_states,
                [self.embed_dim],
                self.final_layer_norm.weight,
                self.final_layer_norm.bias,
                1e-05,
            )
            # ==========================================================================

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        # if use_cache:
        # use cache always to be true for generation
        outputs += (present_key_value,)

        return outputs


class OPTLearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(
        self, attention_mask: torch.LongTensor, past_key_values_length: int = 0
    ):
        attention_mask = attention_mask.long()
        positions = (
            torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask
        ).long() - 1
        positions = positions[:, past_key_values_length:]
        return super().forward(positions + self.offset)


class OPTDecoder(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.word_embed_proj_dim, self.padding_idx
        )
        self.embed_positions = OPTLearnedPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size
        )

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(
                config.hidden_size, config.word_embed_proj_dim, bias=False
            )
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(
                config.word_embed_proj_dim, config.hidden_size, bias=False
            )
        else:
            self.project_in = None

        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size,
                elementwise_affine=config.layer_norm_elementwise_affine,
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList(
            [OPTDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
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
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        mask_seq_length = past_key_values_length + seq_length

        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, mask_seq_length, device=inputs_embeds.device
            )
        elif attention_mask.shape[1] != mask_seq_length:
            raise ValueError(
                f"The provided attention mask has length {attention_mask.shape[1]}, but its length should be "
                f"{mask_seq_length} (sum of the lengths of current and past inputs)"
            )
        causal_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            # if use_cache:
            # use cache always to be true for generation
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        # use cache always to be true for generation
        next_cache = next_decoder_cache  # if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class OPTModel(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.decoder = OPTDecoder(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
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

        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs

        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )


class IPEXOPTForCausalLM(OPTPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = OPTModel(config)

        self.lm_head = nn.Linear(
            config.word_embed_proj_dim, config.vocab_size, bias=False
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
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
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        # ==================== for generation, lm head only needs last token as input  ====================
        if (
            hasattr(self, "config")
            and hasattr(self.config, "lm_head_generation")
            and self.config.lm_head_generation
            and hidden_states.size(1) != 1
        ):
            hidden_states = hidden_states[:, -1:, :]

        logits = self.lm_head(hidden_states).contiguous()

        loss = None
        if (
            hasattr(self, "config")
            and hasattr(self.config, "use_ipex_optimize")
            and self.config.use_ipex_optimize
        ):
            # return dict is handled by ipex._set_optimized_model_for_generation
            output = (logits,) + outputs[1:]
            return output

        if not return_dict:
            output = (logits,) + outputs[1:]
            return output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    # ==================== rewrite to _reorder_cache to work with ipex.llm.modules.IndirectAccessKVCacheAttention  ====================
    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        if (
            len(past_key_values[0]) == 4 and past_key_values[0][0].shape[-1] == 1
        ):  # discrete kv_cache
            for layer_past in past_key_values:
                layer_past[3][layer_past[0].size(-2) - 1] = beam_idx
            return past_key_values
        else:
            return tuple(
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                )
                for layer_past in past_key_values
            )
