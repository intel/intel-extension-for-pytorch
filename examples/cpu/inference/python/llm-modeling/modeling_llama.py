import math
from typing import List, Optional, Tuple, Union

import torch
import intel_extension_for_pytorch as ipex
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel


from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
)


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        # ==================== orignal path  ====================
        # down_proj = self.down_proj(torch.nn.functional.silu(self.gate_proj(x))*self.up_proj))
        # ==================== Changes to apply ipex.llm layers  ====================
        if not hasattr(self, "ipex_fusion"):
            self.ipex_fusion = ipex.llm.modules.Linear2SiluMul(
                self.gate_proj, self.up_proj
            )
            del self.__dict__["_modules"]["gate_proj"]
            del self.__dict__["_modules"]["up_proj"]
        down_proj = self.ipex_fusion(x)
        # self.down_proj is move to LlamaDecoderLayer to enable linear+add fusion
        # ==========================================================================
        return down_proj


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # ==================== Changes to apply ipex.llm layers  ====================
        self._IPEXIndirectAccessKVCacheAttention = ipex.llm.modules.IndirectAccessKVCacheAttention(
            self.max_position_embeddings
        )
        self.ipex_rotary_emb = ipex.llm.modules.RotaryEmbedding(
            self.max_position_embeddings,
            self.head_dim,
            self.rope_theta,
            self.config.architectures[0],
        )
        # ==========================================================================

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[List[torch.FloatTensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )

        kv_seq_len = (
            q_len + past_key_value[0].size(-2) if past_key_value is not None else q_len
        )
        # ==================== Changes to apply ipex.llm layers  ====================
        key_states = self.ipex_rotary_emb(
            key_states,
            position_ids,
            self.num_key_value_heads,
            self.head_dim,
            self.head_dim // 2,
            self.head_dim,
            kv_seq_len,
        )
        query_states = self.ipex_rotary_emb(
            query_states,
            position_ids,
            self.num_heads,
            self.head_dim,
            self.head_dim // 2,
            self.head_dim,
            kv_seq_len,
        )

        (attn_output, attn_weights, past_key_value) = self._IPEXIndirectAccessKVCacheAttention(
            query_states,
            key_states,
            value_states,
            math.sqrt(self.head_dim),
            past_key_value,
            None,
            attention_mask,
        )
        # ==========================================================================

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # move to LlamaDecoderLayer to enable linear+add fusion
        # attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = ipex.llm.modules.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = ipex.llm.modules.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        # ==================== origal path  ====================
        # hidden_states = self.self_attn.o_proj(hidden_states)
        # hidden_states = residual + hidden_states
        # ==================== Changes to apply ipex.llm layers  ====================
        if not hasattr(self, "ipex_fusion_1"):
            self.ipex_fusion_1 = ipex.llm.modules.LinearAdd(self.self_attn.o_proj)
            del self.__dict__["_modules"]["self_attn"].o_proj
        hidden_states = self.ipex_fusion_1(hidden_states, residual)
        # ==========================================================================

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # ==================== origal path  ====================
        # hidden_states = self.mlp.down_proj(hidden_states)
        # hidden_states = residual + hidden_states
        # ==================== Changes to apply ipex.llm layers  ====================
        if not hasattr(self, "ipex_fusion_2"):
            self.ipex_fusion_2 = ipex.llm.modules.LinearAdd(self.mlp.down_proj)
            del self.__dict__["_modules"]["mlp"].down_proj
        hidden_states = self.ipex_fusion_2(hidden_states, residual)
        # ==========================================================================

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        # if use_cache:
        # use cache always to be true for generation
        outputs += (present_key_value,)

        return outputs


class LlamaModel(PreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # ==================== Changes to apply ipex.llm layers  ====================
        self.norm = ipex.llm.modules.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # ==========================================================================

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
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
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if hasattr(self, "_prepare_decoder_attention_mask"):
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        hidden_states = inputs_embeds

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
                attention_mask=attention_mask,
                position_ids=position_ids,
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

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache  # if use_cache else None
        # use cache always to be true for generation

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


class IPEXLlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
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

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
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

        logits = self.lm_head(hidden_states)
        logits = logits.float()

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

    # ======== rewrite to prepare_inputs_for_generation to work with ipex.llm.modules.IndirectAccessKVCacheAttention  =========
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
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
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
