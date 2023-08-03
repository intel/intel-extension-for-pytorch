import torch
import torch.nn as nn
from typing import Optional, Tuple

from intel_extension_for_pytorch.nn.utils._transformer_configuration import IPEXTransformerConfig
from ._transformers import IPEXTransformerAtten, IPEXTransformerMLP, IPEXEmptyLinear
from ._transformer_configuration import IPEXTransformerConfig
from ._transformer_converter import IPEXTransformerConverter, MAX_SEQ_LEN, MAX_OUT_SEQ_LEN
from .RoPE import LlamaRotaryEmbedding
from .Norm import LlamaRMSNorm
import math

class IPEXLlamaAttn(IPEXTransformerAtten):
    def __init__(self, config) -> None:
        super().__init__(config)


class IPEXLlamaMLP(IPEXTransformerMLP):
    def __init__(self,
                 config: IPEXTransformerConfig):
        super().__init__(config)
        self.up_proj = IPEXEmptyLinear()
        self.up_wei = None

    def forward(self, hidden_states, residual):
        if self.row_major:
            if isinstance(self.act, nn.SiLU):
                hidden_states1 = torch.ops.torch_ipex.mm_silu(hidden_states, self.fc_in_wei)
            else:
                hidden_states1 = torch.matmul(hidden_states, self.fc_in_wei)
                hidden_states1 = self.act(hidden_states1)
            hidden_states = torch.ops.torch_ipex.mm_resmul(hidden_states, self.up_wei, hidden_states1)
            shape = list(hidden_states.size())
            shape[-1] = self.fc_out_wei.shape[-1]
            hidden_states = torch.addmm(residual.flatten(0, -2), hidden_states.flatten(0, -2), self.fc_out_wei).view(shape)
        else:
            hidden_states = self.fc_out(self.act(self.fc_in(hidden_states)) * self.up_proj(hidden_states))
            hidden_states += residual
        return hidden_states


class IPEXLlamaBlock(nn.Module):
    def __init__(self, 
                 config: IPEXTransformerConfig):
        super().__init__()
        self.attn = IPEXLlamaAttn(config=config)
        self.mlp = IPEXLlamaMLP(config=config)
        self.input_layernorm = LlamaRMSNorm(config)
        self.post_attn_layernorm = LlamaRMSNorm(config)

    def release_resources(self):
        self.attn.release_resources()
        self.mlp.release_resources()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # hidden_states:  [bs*beam, seq, hidden_size]
        # position_ids:   [bs*beam, seq]
        # attention_mask: [bs*beam, head, q_seq, kv_seq]
        bs = IPEXTransformerAtten.batch_size
        beam = hidden_states.shape[0] // bs
        hidden_shape = [bs, beam, hidden_states.shape[1], hidden_states.shape[2]]
        if hidden_states.shape[1] > 1:
            hidden_states = hidden_states.view(hidden_shape)[:, 0, :, :]        # [bs, seq, hidden_size]
            if position_ids is not None:
                position_ids = position_ids.view(bs, beam, position_ids.shape[1])[:,0,:].view(bs, position_ids.shape[1])
            if attention_mask is not None:
                attention_mask = attention_mask.view(bs, beam, attention_mask.shape[1], attention_mask.shape[2], attention_mask.shape[3])[:,0,:,:,:].view(bs, attention_mask.shape[1], attention_mask.shape[2], attention_mask.shape[3])
        # convert layout form [bs, seq, hidden_size] to [seq, bs, hidden_size]
        hidden_states = hidden_states.transpose(0, 1).contiguous()

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, present_key_value, self_attn_weights = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            residual=residual
        )

        residual = hidden_states
        hidden_states = self.post_attn_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, residual)

        # convert hidden_states form [seq, beam, hidden_size] back to [beam, seq, hidden_size]
        hidden_states = hidden_states.transpose(0, 1)
        if hidden_states.shape[1] > 1:
            hidden_states = hidden_states.view(bs, 1, hidden_states.shape[1], hidden_states.shape[2]).expand([bs, beam, hidden_states.shape[1], hidden_states.shape[2]])
            hidden_states = hidden_states.reshape(bs*beam, hidden_states.shape[2], hidden_states.shape[3])

        outputs = (hidden_states, )
        if output_attentions:
            outputs += (self_attn_weights, )

        if use_cache:
            outputs += (present_key_value, )

        return outputs


class IPEXLlamaConverter(IPEXTransformerConverter):
    def __init__(self,
                 module,
                 config = None,
                 device = "cpu",
                 dtype = torch.float):
        super().__init__(module, config, device=device, dtype=dtype)
        from transformers.models.llama.configuration_llama import LlamaConfig
        self.config = config if config is not None else LlamaConfig()
        self.ipex_transformers_config = self.construct_transformer_config()
        self.ipex_optimized_module = self.construct_ipex_optimized_module()
        self.port_all_parameters_to_new_module()

    def construct_transformer_config(self):
        n_positions = max(self.config.max_position_embeddings, MAX_SEQ_LEN)
        embed_dim = self.config.hidden_size
        num_head = self.config.num_attention_heads
        activate_function = self.config.hidden_act
        norm_eps = self.config.rms_norm_eps
        use_cache = self.config.use_cache
        intermediate_size = self.config.intermediate_size
        return IPEXTransformerConfig(
            embed_dim=embed_dim,
            intermediate_size=intermediate_size,
            num_attention_heads=num_head,
            max_positions=n_positions,
            max_out_positions=MAX_OUT_SEQ_LEN,
            rotary_embedding_class=LlamaRotaryEmbedding,
            rotary_dim=None,
            rotate_half=True,
            rotate_every_two=False,
            use_casual_mask=False,
            activation_function=activate_function,
            norm_eps=norm_eps,
            residual_dropout=None,
            attn_dropout=None,
            enable_bias=False,
            residual_pdrop=None,
            scale_attention=True,
            is_decoder=False,
            do_norm_before=None,
            ln_elementwise_affine=None,
            seq_first=True,
            kv_cache_optimize=True,
            positional_embedding_base=10000,
            sdp_fusion_enable=True,
            device=self.device,
            dtype=self.dtype,
            tp_size=IPEXTransformerConverter.tp_size,
            tp_group=IPEXTransformerConverter.tp_group
        )

    def construct_ipex_optimized_module(self):
        return IPEXLlamaBlock(self.ipex_transformers_config)

    def port_attn_parameters(self):
        if self.row_major:
            self.module.self_attn.q_proj.weight.data = self.module.self_attn.q_proj.weight.t().contiguous()
            self.module.self_attn.k_proj.weight.data = self.module.self_attn.k_proj.weight.t().contiguous()
            self.module.self_attn.v_proj.weight.data = self.module.self_attn.v_proj.weight.t().contiguous()

            shape = [3, -1, self.module.self_attn.q_proj.weight.shape[-1]]
            self.ipex_optimized_module.attn.qkv_wei = torch.cat([self.module.self_attn.q_proj.weight.data, 
                                                                 self.module.self_attn.k_proj.weight.data, 
                                                                 self.module.self_attn.v_proj.weight.data], dim=0).view(shape)
            self.module.self_attn.q_proj.weight.data = self.ipex_optimized_module.attn.qkv_wei[0, :, :]
            self.module.self_attn.k_proj.weight.data = self.ipex_optimized_module.attn.qkv_wei[1, :, :]
            self.module.self_attn.v_proj.weight.data = self.ipex_optimized_module.attn.qkv_wei[2, :, :]
            self.ipex_optimized_module.attn.q_wei = self.module.self_attn.q_proj.weight
            self.ipex_optimized_module.attn.k_wei = self.module.self_attn.k_proj.weight
            self.ipex_optimized_module.attn.v_wei = self.module.self_attn.v_proj.weight

            self.ipex_optimized_module.attn.bias = self.module.self_attn.q_proj.bias
            self.ipex_optimized_module.attn.bias = self.module.self_attn.k_proj.bias
            self.ipex_optimized_module.attn.bias = self.module.self_attn.v_proj.bias

            self.module.self_attn.o_proj.weight.data = self.module.self_attn.o_proj.weight.t().contiguous()
            self.ipex_optimized_module.attn.out_wei = self.module.self_attn.o_proj.weight
            self.ipex_optimized_module.attn.out_bias = self.module.self_attn.o_proj.bias
        else:
            self.ipex_optimized_module.attn.k_proj.weight = self.module.self_attn.k_proj.weight
            self.ipex_optimized_module.attn.k_proj.bias = self.module.self_attn.k_proj.bias
            self.ipex_optimized_module.attn.q_proj.weight = self.module.self_attn.q_proj.weight
            self.ipex_optimized_module.attn.q_proj.bias = self.module.self_attn.q_proj.bias
            self.ipex_optimized_module.attn.v_proj.weight = self.module.self_attn.v_proj.weight
            self.ipex_optimized_module.attn.v_proj.bias = self.module.self_attn.v_proj.bias
            self.ipex_optimized_module.attn.out_proj.weight = self.module.self_attn.o_proj.weight
            self.ipex_optimized_module.attn.out_proj.bias = self.module.self_attn.o_proj.bias


    def port_mlp_parameters(self):
        if self.row_major:
            self.module.mlp.gate_proj.weight.data = self.module.mlp.gate_proj.weight.t().contiguous()
            self.ipex_optimized_module.mlp.fc_in_wei = self.module.mlp.gate_proj.weight
            self.ipex_optimized_module.mlp.fc_in.bias = self.module.mlp.gate_proj.bias

            self.module.mlp.down_proj.weight.data = self.module.mlp.down_proj.weight.t().contiguous()
            self.ipex_optimized_module.mlp.fc_out_wei = self.module.mlp.down_proj.weight
            self.ipex_optimized_module.mlp.fc_out.bias = self.module.mlp.down_proj.bias

            self.module.mlp.up_proj.weight.data = self.module.mlp.up_proj.weight.t().contiguous()
            self.ipex_optimized_module.mlp.up_wei = self.module.mlp.up_proj.weight
            self.ipex_optimized_module.mlp.up_proj.bias = self.module.mlp.up_proj.bias
        else:
            self.ipex_optimized_module.mlp.fc_in.weight = self.module.mlp.gate_proj.weight
            self.ipex_optimized_module.mlp.fc_in.bias = self.module.mlp.gate_proj.bias
            self.ipex_optimized_module.mlp.fc_out.weight = self.module.mlp.down_proj.weight
            self.ipex_optimized_module.mlp.fc_out.bias = self.module.mlp.down_proj.bias
            self.ipex_optimized_module.mlp.up_proj.weight = self.module.mlp.up_proj.weight
            self.ipex_optimized_module.mlp.up_proj.bias = self.module.mlp.up_proj.bias

    def port_layer_norm_parameters(self):
        self.ipex_optimized_module.input_layernorm.weight = self.module.input_layernorm.weight
        self.ipex_optimized_module.post_attn_layernorm.weight = self.module.post_attention_layernorm.weight

    def port_all_parameters_to_new_module(self):
        self.port_attn_parameters()
        self.port_mlp_parameters()
        self.port_layer_norm_parameters()

    def get_transformed_module(self):
        return self.ipex_optimized_module