from typing import Optional, Tuple, Union

import torch
import intel_extension_for_pytorch as ipex
import torch.fx
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.gptj.modeling_gptj import GPTJPreTrainedModel


class GPTJAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        max_positions = config.max_position_embeddings

        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads
        self.scale_attn = torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32)
        ).to(torch.get_default_dtype())

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.rotary_dim = config.rotary_dim
        pos_embd_dim = self.rotary_dim or self.embed_dim

        # ==================== Changes to apply ipex.llm layers  ====================
        self.ipex_rotary_emb = ipex.llm.modules.RotaryEmbedding(
            max_positions,
            pos_embd_dim,
            backbone=config.architectures[0],
        )
        self._IPEXIndirectAccessKVCacheAttention = ipex.llm.modules.IndirectAccessKVCacheAttention(
            max_positions
        )
        # ==========================================================================

    def _split_heads(self, tensor, num_attention_heads, attn_head_size, rotary):
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        if rotary:
            return tensor
        if len(tensor.shape) == 5:
            return tensor.permute(0, 1, 3, 2, 4)
        elif len(tensor.shape) == 4:
            return tensor.permute(0, 2, 1, 3)
        else:
            raise ValueError(
                f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}"
            )

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        if len(tensor.shape) == 5:
            tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()
        elif len(tensor.shape) == 4:
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
        else:
            raise ValueError(
                f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}"
            )
        new_shape = tensor.size()[:-2] + (num_attention_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor, Tuple[torch.Tensor]],
        Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],
    ]:
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_attention_heads, self.head_dim, True)
        key = self._split_heads(key, self.num_attention_heads, self.head_dim, True)

        # ==================== Changes to apply ipex.llm layers  ====================
        key = self.ipex_rotary_emb(
            key,
            position_ids.contiguous(),
            self.num_attention_heads,
            self.head_dim,
            1,
            64,
        )
        query = self.ipex_rotary_emb(
            query,
            position_ids.contiguous(),
            self.num_attention_heads,
            self.head_dim,
            1,
            64,
        )
        value = self._split_heads(value, self.num_attention_heads, self.head_dim, True)

        (
            attn_output,
            attn_weights,
            present,
        ) = self._IPEXIndirectAccessKVCacheAttention(
            query,
            key,
            value,
            self.scale_attn,
            layer_past,
            head_mask,
            attention_mask,
        )
        # ==========================================================================

        attn_output = self._merge_heads(
            attn_output, self.num_attention_heads, self.head_dim
        )
        attn_output = self.out_proj(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class GPTJMLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.n_embd
        self.fc_in = nn.Linear(embed_dim, intermediate_size)
        self.fc_out = nn.Linear(intermediate_size, embed_dim)

    def forward(self, hidden_states: Optional[torch.FloatTensor]) -> torch.FloatTensor:
        # ==================== orignal path  ====================
        # hidden_states = NewGelu(self.fc_in(hidden_states))
        # ==================== Changes to apply ipex.llm layers  ====================
        if not hasattr(self, "ipex_fusion"):
            self.ipex_fusion = ipex.llm.modules.LinearNewGelu(self.fc_in)
            del self.__dict__["_modules"]["fc_in"]
        hidden_states = self.ipex_fusion(hidden_states)
        # move self.fc_out to GPTJBlock to enable linear+add+add fusion
        # hidden_states = self.fc_out(hidden_states)
        # ==========================================================================

        return hidden_states


class GPTJBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.n_embd = config.n_embd
        self.eps = config.layer_norm_epsilon
        self.ln_1 = nn.LayerNorm(self.n_embd, eps=self.eps)
        self.attn = GPTJAttention(config)
        self.mlp = GPTJMLP(inner_dim, config)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor],
        Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]],
    ]:
        residual = hidden_states

        # ==================== orignal path  ====================
        # hidden_states = self.ln_1(hidden_states)
        # ==================== Changes to apply ipex.llm layers  ====================
        # option 1 : replace module
        # if not hasattr(self, "ipex_layernorm"):
        #     self.ipex_layernorm = ipex.llm.modules.FastLayerNorm(
        #         self.n_embd,
        #         eps=self.eps,
        #         weight=self.ln_1.weight,
        #         bias=self.ln_1.bias if hasattr(self, "ln_1") else None,
        #     )
        #     del self.ln_1
        # hidden_states = self.ipex_layernorm(hidden_states)
        #
        # option 2 : use function call
        hidden_states = ipex.llm.functional.fast_layer_norm(
            hidden_states, [self.n_embd], self.ln_1.weight, self.ln_1.bias, self.eps
        )
        # ==========================================================================

        attn_outputs = self.attn(
            hidden_states=hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]

        feed_forward_hidden_states = self.mlp(hidden_states)

        # ==================== orignal path  ====================
        # hidden_states = attn_output + feed_forward_hidden_states + residual
        # ==================== Changes to apply ipex.llm layers  ====================
        if not hasattr(self, "ipex_fusion"):
            self.ipex_fusion = ipex.llm.modules.LinearAddAdd(self.mlp.fc_out)
            del self.__dict__["_modules"]["mlp"].fc_out
        hidden_states = self.ipex_fusion(
            feed_forward_hidden_states, residual, attn_output
        )
        # ==========================================================================

        # use cache always to be true for generation
        # if use_cache:
        outputs = (hidden_states,) + outputs
        # else:
        #     outputs = (hidden_states,) + outputs[1:]

        return outputs


class GPTJModel(GPTJPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.h = nn.ModuleList([GPTJBlock(config) for _ in range(config.n_layer)])
        self.eps = config.layer_norm_epsilon
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=self.eps)
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(
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
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        hidden_states = inputs_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states=hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]
            # use cache always to be true for generation
            # if use_cache is True:
            presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1],
                )

        # ==================== orignal path  ====================
        # hidden_states = self.ln_f(hidden_states)

        # ==================== Changes to apply ipex.llm layers  ====================
        # option 1 : replace module
        # if not hasattr(self, "ipex_layernorm"):
        #     self.ipex_layernorm = ipex.llm.modules.FastLayerNorm(
        #         self.embed_dim,
        #         eps=self.eps,
        #         weight=self.ln_f.weight,
        #         bias=self.ln_f.bias,
        #     )
        #     del self.ln_f
        # hidden_states = self.ipex_layernorm(hidden_states)
        #
        # option 2 : use a function call
        hidden_states = ipex.llm.functional.fast_layer_norm(
            hidden_states, [self.embed_dim], self.ln_f.weight, self.ln_f.bias, self.eps
        )
        # ==========================================================================

        hidden_states = hidden_states.view(output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class IPEXGPTJForCausalLM(GPTJPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPTJModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        token_type_ids = kwargs.get("token_type_ids", None)
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        attention_mask = kwargs.get("attention_mask", None)
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
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )

        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        # ==================== for generation, lm head only needs last token as input  ====================
        if (
            hasattr(self, "config")
            and hasattr(self.config, "lm_head_generation")
            and self.config.lm_head_generation
            and hidden_states.size(1) != 1
        ):
            hidden_states = hidden_states[:, -1:, :]
        lm_logits = self.lm_head(hidden_states).to(torch.float32)

        loss = None
        if (
            hasattr(self, "config")
            and hasattr(self.config, "use_ipex_optimize")
            and self.config.use_ipex_optimize
        ):
            # return dict is handled by ipex._set_optimized_model_for_generation
            output = (lm_logits,) + transformer_outputs[1:]
            return output

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    # ==================== rewrite to _reorder_cache to work with ipex.llm.modules.IndirectAccessKVCacheAttention  ====================
    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        if len(past_key_values[0]) == 4 and past_key_values[0][0].shape[-1] == 1:
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
