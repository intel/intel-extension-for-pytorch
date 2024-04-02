import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class NewIPEXBertSelfAttention(nn.Module):
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
    ):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.attn_query_weight = module.query.weight
        self.attn_key_weight = module.key.weight
        self.attn_value_weight = module.value.weight
        self.attn_query_bias = module.query.bias
        self.attn_key_bias = module.key.bias
        self.attn_value_bias = module.value.bias

        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob

        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )

        if (
            self.position_embedding_type is not None
            and self.position_embedding_type != "absolute"
        ):
            raise ValueError(
                f"Positional embedding type {self.position_embedding_type} not "
                f"supported for `optimize_transformers` integration"
            )

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        if head_mask is not None:
            raise ValueError(
                "head_mask not supported for `optimize_transformers` integration"
            )
        if output_attentions:
            raise ValueError(
                "output_attentions=True can not be supported with BetterTransformer."
            )

        query_layer = self.transpose_for_scores(
            F.linear(
                hidden_states, weight=self.attn_query_weight, bias=self.attn_query_bias
            )
        )
        key_layer = self.transpose_for_scores(
            F.linear(
                hidden_states, weight=self.attn_key_weight, bias=self.attn_key_bias
            )
        )
        value_layer = self.transpose_for_scores(
            F.linear(
                hidden_states, weight=self.attn_value_weight, bias=self.attn_value_bias
            )
        )

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        context_layer = torch.xpu.IpexSDP_dropout(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            is_causal=False,
            dropout_p=self.attention_probs_dropout_prob if self.training else 0.0,
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
