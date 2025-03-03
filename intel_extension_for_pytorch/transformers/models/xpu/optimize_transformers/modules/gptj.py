import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
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
            self.attn = IPEXAttention(self.ipex_config, module.attn.layer_idx)
        elif dtype == "int4" and xpu_gemm_use_xetla():
            self.attn = IPEXAttentionInt4(self.ipex_config, module.attn.layer_idx)
        elif dtype == "int4" and not xpu_gemm_use_xetla():
            self.attn = IPEXAttentionInt4OneDNN(self.ipex_config, module.attn.layer_idx)
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
        self.layer_idx = kwargs.get("layer_idx", None)

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
