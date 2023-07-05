import torch 
import os
from ._transformers import IPEXGPTJBlock, IPEXLlamaBlock, IPEXOptBlock, IPEXBloomBlock

# from transformers.models.gptj.modeling_gptj import GPTJBlock

# from transformers.models.opt.modeling_opt import OPTDecoderLayer
# from transformers.configuration_utils import PretrainedConfig
from .RoPE import GPTJRotaryEmbedding, LlamaRotaryEmbedding, PositionalEmbedding
from ._transformer_configuration import IPEXTransformerConfig
from typing import Optional, Tuple, Union
import torch.nn as nn
# from transformers.models.llama.configuration_llama import 


class IPEXTransformerConverter:
    tp_group = None
    tp_size = 1

    def __init__(self, module, config, device = "cpu", dtype = torch.float) -> None:
        self.module = module
        self.config = config
        self.dtype = dtype
        self.device = device
        col_major = os.environ.get("COL_MAJOR", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]
        self.row_major = not col_major

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


class IPEXGPTJConverter(IPEXTransformerConverter):
    def __init__(self,
                 module,
                 config = None,
                 device = "cpu",
                 dtype = torch.float):
        from transformers.models.gptj.configuration_gptj import GPTJConfig
        super().__init__(module, config, device=device, dtype=dtype)
        self.config = config if config is not None else GPTJConfig()
        self.ipex_transformers_config = self.construct_transformer_config()
        # print(self.ipex_transformers_config.__dict__)
        self.ipex_optimized_module = self.construct_ipex_optimized_module()
        self.port_all_parameters_to_new_module()

    def construct_transformer_config(self):
        n_positions = self.config.n_positions
        embed_dim = self.config.n_embd
        num_head = self.config.n_head
        rotary_dim = self.config.rotary_dim
        activate_function = self.config.activation_function
        resid_pdrop = self.config.resid_pdrop
        attn_pdrop = self.config.attn_pdrop
        layer_norm_eps = self.config.layer_norm_epsilon
        use_cache = self.config.use_cache
        intermediate_size = self.config.n_inner
        return IPEXTransformerConfig(
            embed_dim=embed_dim,
            intermediate_size=intermediate_size,
            num_attention_heads=num_head,
            max_positions=n_positions,
            rotary_embedding_class=GPTJRotaryEmbedding,
            rotary_dim=rotary_dim,
            rotate_half=False,
            rotate_every_two=True,
            use_casual_mask=True,
            activation_function=activate_function,
            norm_eps=layer_norm_eps,
            residual_dropout=resid_pdrop,
            attn_dropout=attn_pdrop,
            enable_bias=False,
            residual_pdrop=resid_pdrop,
            scale_attention=True,
            is_decoder=False,
            do_norm_before=False,
            ln_elementwise_affine=False,
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
        return IPEXGPTJBlock(self.ipex_transformers_config)

    def port_attn_parameters(self):
        self.ipex_optimized_module.attn.k_proj.weight = self.module.attn.k_proj.weight
        self.ipex_optimized_module.attn.k_proj.bias = self.module.attn.k_proj.bias
        self.ipex_optimized_module.attn.q_proj.weight = self.module.attn.q_proj.weight
        self.ipex_optimized_module.attn.q_proj.bias = self.module.attn.q_proj.bias
        self.ipex_optimized_module.attn.v_proj.weight = self.module.attn.v_proj.weight
        self.ipex_optimized_module.attn.v_proj.bias = self.module.attn.v_proj.bias
        self.ipex_optimized_module.attn.out_proj.weight = self.module.attn.out_proj.weight
        self.ipex_optimized_module.attn.out_proj.bias = self.module.attn.out_proj.bias

        if self.row_major:
            self.ipex_optimized_module.attn.q_wei = self.ipex_optimized_module.attn.q_proj.weight.transpose(0, 1).contiguous()
            del self.ipex_optimized_module.attn.q_proj.weight
            self.ipex_optimized_module.attn.k_wei = self.ipex_optimized_module.attn.k_proj.weight.transpose(0, 1).contiguous()
            del self.ipex_optimized_module.attn.k_proj.weight
            self.ipex_optimized_module.attn.v_wei = self.ipex_optimized_module.attn.v_proj.weight.transpose(0, 1).contiguous()
            del self.ipex_optimized_module.attn.v_proj.weight
            self.ipex_optimized_module.attn.out_wei = self.ipex_optimized_module.attn.out_proj.weight.transpose(0, 1).contiguous()
            del self.ipex_optimized_module.attn.out_proj.weight
            self.ipex_optimized_module.attn.qkv_wei = torch.stack([self.ipex_optimized_module.attn.q_wei, self.ipex_optimized_module.attn.k_wei, self.ipex_optimized_module.attn.v_wei]).contiguous()
            del self.ipex_optimized_module.attn.q_wei
            del self.ipex_optimized_module.attn.k_wei
            del self.ipex_optimized_module.attn.v_wei


    def port_mlp_parameters(self):
        self.ipex_optimized_module.mlp.fc_in.weight = self.module.mlp.fc_in.weight
        self.ipex_optimized_module.mlp.fc_in.bias = self.module.mlp.fc_in.bias
        self.ipex_optimized_module.mlp.fc_out.weight = self.module.mlp.fc_out.weight
        self.ipex_optimized_module.mlp.fc_out.bias = self.module.mlp.fc_out.bias

        if self.row_major:
            self.ipex_optimized_module.mlp.fc_in_wei = self.ipex_optimized_module.mlp.fc_in.weight.transpose(0, 1).contiguous()
            del self.ipex_optimized_module.mlp.fc_in.weight
            self.ipex_optimized_module.mlp.fc_out_wei = self.ipex_optimized_module.mlp.fc_out.weight.transpose(0, 1).contiguous()
            del self.ipex_optimized_module.mlp.fc_out.weight

    def port_layer_norm_parameters(self):
        self.ipex_optimized_module.ln.weight = self.module.ln_1.weight
        self.ipex_optimized_module.ln.bias = self.module.ln_1.bias

    def port_all_parameters_to_new_module(self):
        self.port_attn_parameters()
        self.port_mlp_parameters()
        self.port_layer_norm_parameters()

    def get_transformed_module(self):
        return self.ipex_optimized_module

class IPEXOptConverter(IPEXTransformerConverter):
    def __init__(self,
                 module,
                 config = None,
                 device = "cpu",
                 dtype = torch.float):
        from transformers.models.opt.configuration_opt import OPTConfig
        super().__init__(module, config, device=device, dtype=dtype)
        self.config = config if config is not None else OPTConfig()
        self.ipex_transformers_config = self.construct_transformer_config()
        self.ipex_optimized_module = self.construct_ipex_optimized_module()
        self.port_all_parameters_to_new_module()

    def construct_transformer_config(self):
        n_positions = self.config.max_position_embeddings
        embed_dim = self.config.hidden_size
        num_head = self.config.num_attention_heads
        activate_function = self.config.activation_function
        resid_pdrop = self.config.dropout
        use_cache = self.config.use_cache
        intermediate_size = self.config.ffn_dim
        do_layer_norm_before = self.config.do_layer_norm_before
        enable_bias = self.config.enable_bias
        layer_norm_eltwise_affine = self.config.layer_norm_elementwise_affine
        # is_decoder = self.config.is_decoder
        return IPEXTransformerConfig(
            embed_dim=embed_dim,
            intermediate_size=intermediate_size,
            num_attention_heads=num_head,
            max_positions=n_positions,
            rotary_embedding_class=PositionalEmbedding,
            rotary_dim=None,
            rotate_half=False,
            rotate_every_two=False,
            use_casual_mask=False,
            activation_function=activate_function,
            norm_eps=None,
            residual_dropout=resid_pdrop,
            attn_dropout=None,
            enable_bias=enable_bias,
            residual_pdrop=resid_pdrop,
            scale_attention=False,
            is_decoder=True,
            do_norm_before=do_layer_norm_before,
            ln_elementwise_affine=layer_norm_eltwise_affine,
            seq_first=False,
            kv_cache_optimize=False,
            positional_embedding_base=10000,
            sdp_fusion_enable=False,
            device=self.device,
            dtype=self.dtype,
            tp_size=IPEXTransformerConverter.tp_size,
            tp_group=IPEXTransformerConverter.tp_group
        )

    def construct_ipex_optimized_module(self):
        return IPEXOptBlock(self.ipex_transformers_config)

    def port_attn_parameters(self):
        self.ipex_optimized_module.attn.k_proj.weight = self.module.self_attn.k_proj.weight
        self.ipex_optimized_module.attn.k_proj.bias = self.module.self_attn.k_proj.bias
        self.ipex_optimized_module.attn.q_proj.weight = self.module.self_attn.q_proj.weight
        self.ipex_optimized_module.attn.q_proj.bias = self.module.self_attn.q_proj.bias
        self.ipex_optimized_module.attn.v_proj.weight = self.module.self_attn.v_proj.weight
        self.ipex_optimized_module.attn.v_proj.bias = self.module.self_attn.v_proj.bias
        self.ipex_optimized_module.attn.out_proj.weight = self.module.self_attn.out_proj.weight
        self.ipex_optimized_module.attn.out_proj.bias = self.module.self_attn.out_proj.bias

        if self.row_major:
            self.ipex_optimized_module.attn.q_wei = self.ipex_optimized_module.attn.q_proj.weight.transpose(0, 1).contiguous()
            del self.ipex_optimized_module.attn.q_proj.weight
            self.ipex_optimized_module.attn.k_wei = self.ipex_optimized_module.attn.k_proj.weight.transpose(0, 1).contiguous()
            del self.ipex_optimized_module.attn.k_proj.weight
            self.ipex_optimized_module.attn.v_wei = self.ipex_optimized_module.attn.v_proj.weight.transpose(0, 1).contiguous()
            del self.ipex_optimized_module.attn.v_proj.weight
            self.ipex_optimized_module.attn.out_wei = self.ipex_optimized_module.attn.out_proj.weight.transpose(0, 1).contiguous()
            del self.ipex_optimized_module.attn.out_proj.weight 
            self.ipex_optimized_module.attn.qkv_wei = torch.stack([self.ipex_optimized_module.attn.q_wei, self.ipex_optimized_module.attn.k_wei, self.ipex_optimized_module.attn.v_wei]).contiguous()
            del self.ipex_optimized_module.attn.q_wei
            del self.ipex_optimized_module.attn.k_wei
            del self.ipex_optimized_module.attn.v_wei

    def port_mlp_parameters(self):
        self.ipex_optimized_module.mlp.fc_in.weight = self.module.fc1.weight
        self.ipex_optimized_module.mlp.fc_in.bias = self.module.fc1.bias
        self.ipex_optimized_module.mlp.fc_out.weight = self.module.fc2.weight
        self.ipex_optimized_module.mlp.fc_out.bias = self.module.fc2.bias

        if self.row_major:
            self.ipex_optimized_module.mlp.fc_in_wei = self.ipex_optimized_module.mlp.fc_in.weight.transpose(0, 1)
            del self.ipex_optimized_module.mlp.fc_in.weight
            self.ipex_optimized_module.mlp.fc_out_wei = self.ipex_optimized_module.mlp.fc_out.weight.transpose(0, 1)
            del self.ipex_optimized_module.mlp.fc_out.weight

    def port_layer_norm_parameters(self):
        self.ipex_optimized_module.self_attn_layer_norm.weight = self.module.self_attn_layer_norm.weight
        self.ipex_optimized_module.self_attn_layer_norm.bias = self.module.self_attn_layer_norm.bias
        self.ipex_optimized_module.final_layer_norm.weight = self.module.final_layer_norm.weight
        self.ipex_optimized_module.final_layer_norm.bias = self.module.final_layer_norm.bias

    def port_all_parameters_to_new_module(self):
        self.port_attn_parameters()
        self.port_mlp_parameters()
        self.port_layer_norm_parameters()

    def get_transformed_module(self):
        return self.ipex_optimized_module

class IPEXLlamaConverter(IPEXTransformerConverter):
    def __init__(self,
                 module,
                 config = None,
                 device = "cpu",
                 dtype = torch.float):
        super().__init__(module, config, device=device, dtype=dtype)
        self.config = config if config is not None else transformers.models.llama.configuration_llama.LlamaConfig()
        self.ipex_transformers_config = self.construct_transformer_config()
        self.ipex_optimized_module = self.construct_ipex_optimized_module()
        self.port_all_parameters_to_new_module()

    def construct_transformer_config(self):
        n_positions = self.config.max_position_embeddings
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
        self.ipex_optimized_module.attn.k_proj.weight = self.module.self_attn.k_proj.weight
        self.ipex_optimized_module.attn.k_proj.bias = self.module.self_attn.k_proj.bias
        self.ipex_optimized_module.attn.q_proj.weight = self.module.self_attn.q_proj.weight
        self.ipex_optimized_module.attn.q_proj.bias = self.module.self_attn.q_proj.bias
        self.ipex_optimized_module.attn.v_proj.weight = self.module.self_attn.v_proj.weight
        self.ipex_optimized_module.attn.v_proj.bias = self.module.self_attn.v_proj.bias
        self.ipex_optimized_module.attn.out_proj.weight = self.module.self_attn.o_proj.weight
        self.ipex_optimized_module.attn.out_proj.bias = self.module.self_attn.o_proj.bias

        if self.row_major:
            self.ipex_optimized_module.attn.q_wei = self.ipex_optimized_module.attn.q_proj.weight.t().contiguous()
            del self.module.self_attn.q_proj.weight
            self.ipex_optimized_module.attn.k_wei = self.ipex_optimized_module.attn.k_proj.weight.t().contiguous()
            del self.module.self_attn.k_proj.weight
            self.ipex_optimized_module.attn.v_wei = self.ipex_optimized_module.attn.v_proj.weight.t().contiguous()
            del self.module.self_attn.v_proj.weight
            shape = [3, -1, self.ipex_optimized_module.attn.q_wei.shape[-1]]
            self.ipex_optimized_module.attn.qkv_wei = torch.cat([self.ipex_optimized_module.attn.q_wei, self.ipex_optimized_module.attn.k_wei, self.ipex_optimized_module.attn.v_wei], dim=0).view(shape)
            del self.ipex_optimized_module.attn.q_wei
            del self.ipex_optimized_module.attn.k_wei
            del self.ipex_optimized_module.attn.v_wei

            self.ipex_optimized_module.attn.out_wei = self.ipex_optimized_module.attn.out_proj.weight.t().contiguous()
            del self.ipex_optimized_module.attn.out_proj.weight


    def port_mlp_parameters(self):
        self.ipex_optimized_module.mlp.fc_in.weight = self.module.mlp.gate_proj.weight
        self.ipex_optimized_module.mlp.fc_in.bias = self.module.mlp.gate_proj.bias
        self.ipex_optimized_module.mlp.fc_out.weight = self.module.mlp.down_proj.weight
        self.ipex_optimized_module.mlp.fc_out.bias = self.module.mlp.down_proj.bias
        self.ipex_optimized_module.mlp.up_proj.weight = self.module.mlp.up_proj.weight
        self.ipex_optimized_module.mlp.up_proj.bias = self.module.mlp.up_proj.bias

        if self.row_major:
            self.ipex_optimized_module.mlp.fc_in_wei = self.ipex_optimized_module.mlp.fc_in.weight.t().contiguous()
            del self.ipex_optimized_module.mlp.fc_in.weight
            self.ipex_optimized_module.mlp.fc_out_wei = self.ipex_optimized_module.mlp.fc_out.weight.t().contiguous()
            del self.ipex_optimized_module.mlp.fc_out.weight
            self.ipex_optimized_module.mlp.up_wei = self.ipex_optimized_module.mlp.up_proj.weight.t().contiguous()
            del self.ipex_optimized_module.mlp.up_proj.weight

    def port_layer_norm_parameters(self):
        self.ipex_optimized_module.input_layernorm.weight = self.module.input_layernorm.weight
        self.ipex_optimized_module.post_attn_layernorm.weight = self.module.post_attention_layernorm.weight

    def port_all_parameters_to_new_module(self):
        self.port_attn_parameters()
        self.port_mlp_parameters()
        self.port_layer_norm_parameters()

    def get_transformed_module(self):
        return self.ipex_optimized_module

class IPEXBloomConverter(IPEXTransformerConverter):
    def __init__(self, module, config, device="cpu", dtype=torch.float) -> None:
        from transformers.models.bloom.configuration_bloom import BloomConfig
        super().__init__(module, config, device, dtype)
        self.config: BloomConfig = config if config is not None else BloomConfig()
        self.ipex_transformers_config = self.construct_transformer_config()
        self.ipex_optimized_module = self.construct_ipex_optimized_module()
        self.port_all_parameters_to_new_module()

    def construct_transformer_config(self):
        # bloom don't have n_position attribute, we set it as 2048 just like other LLM models.
        n_positions = 2048
        embed_dim = self.config.hidden_size
        num_head = self.config.n_head
        hidden_dropout = self.config.hidden_dropout
        attention_dropout = self.config.attention_dropout
        before_norm = self.config.apply_residual_connection_post_layernorm
        # activate_function = self.config.hidden_act
        norm_eps = self.config.layer_norm_epsilon
        use_cache = self.config.use_cache
        intermediate_size = 4 * embed_dim
        return IPEXTransformerConfig(
            embed_dim=embed_dim,
            intermediate_size=intermediate_size,
            num_attention_heads=num_head,
            max_positions=n_positions,
            rotary_embedding_class=PositionalEmbedding,
            rotary_dim=None,
            rotate_half=False,
            rotate_every_two=False,
            use_casual_mask=False,
            activation_function="bloom_gelu",
            norm_eps=norm_eps,
            residual_dropout=None,
            attn_dropout=attention_dropout,
            enable_bias=False,
            residual_pdrop=None,
            scale_attention=True,
            is_decoder=False,
            do_norm_before=before_norm,
            ln_elementwise_affine=None,
            seq_first=False,
            kv_cache_optimize=False,
            positional_embedding_base=10000,
            sdp_fusion_enable=False,
            device=self.device,
            dtype=self.dtype,
            tp_size=IPEXTransformerConverter.tp_size,
            tp_group=IPEXTransformerConverter.tp_group
        )
    def construct_ipex_optimized_module(self):
        return IPEXBloomBlock(self.ipex_transformers_config)

    def port_attn_parameters(self):
        self.ipex_optimized_module.self_attention.query_key_value.weight = nn.Parameter(self.module.self_attention.query_key_value.weight)
        self.ipex_optimized_module.self_attention.query_key_value.bias = nn.Parameter(self.module.self_attention.query_key_value.bias)
        self.ipex_optimized_module.self_attention.out_proj.weight = nn.Parameter(self.module.self_attention.dense.weight)
        self.ipex_optimized_module.self_attention.out_proj.bias = nn.Parameter(self.module.self_attention.dense.bias)

        if self.row_major:
            self.ipex_optimized_module.self_attention.qkv_wei = self.ipex_optimized_module.self_attention.query_key_value.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.self_attention.qkv_bias = self.ipex_optimized_module.self_attention.query_key_value.bias 
            del self.ipex_optimized_module.self_attention.query_key_value.weight
            del self.ipex_optimized_module.self_attention.query_key_value.bias
            self.ipex_optimized_module.self_attention.out_wei = self.ipex_optimized_module.self_attention.out_proj.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.self_attention.out_bias = self.ipex_optimized_module.self_attention.out_proj.bias
            del self.ipex_optimized_module.self_attention.out_proj.weight
            del self.ipex_optimized_module.self_attention.out_proj.bias

    def port_mlp_parameters(self):
        self.ipex_optimized_module.mlp.fc_in.weight = nn.Parameter(self.module.mlp.dense_h_to_4h.weight)
        self.ipex_optimized_module.mlp.fc_in.bias = nn.Parameter(self.module.mlp.dense_h_to_4h.bias)
        self.ipex_optimized_module.mlp.fc_out.weight = nn.Parameter(self.module.mlp.dense_4h_to_h.weight)
        self.ipex_optimized_module.mlp.fc_out.bias = nn.Parameter(self.module.mlp.dense_4h_to_h.bias)
        
        if self.row_major:
            self.ipex_optimized_module.mlp.fc_in_wei = self.ipex_optimized_module.mlp.fc_in.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.mlp.fc_in_bias = self.ipex_optimized_module.mlp.fc_in.bias
            del self.ipex_optimized_module.mlp.fc_in.weight
            del self.ipex_optimized_module.mlp.fc_in.bias
            self.ipex_optimized_module.mlp.fc_out_wei = self.ipex_optimized_module.mlp.fc_out.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.mlp.fc_out_bias = self.ipex_optimized_module.mlp.fc_out.bias
            del self.ipex_optimized_module.mlp.fc_out.weight
            del self.ipex_optimized_module.mlp.fc_out.bias

    def port_layer_norm_parameters(self):
        self.ipex_optimized_module.input_layernorm.weight = nn.Parameter(self.module.input_layernorm.weight)
        self.ipex_optimized_module.input_layernorm.bias = nn.Parameter(self.module.input_layernorm.bias)
        self.ipex_optimized_module.post_attention_layernorm.weight = nn.Parameter(self.module.post_attention_layernorm.weight)
        self.ipex_optimized_module.post_attention_layernorm.bias = nn.Parameter(self.module.post_attention_layernorm.bias)

    def port_all_parameters_to_new_module(self):
        self.port_attn_parameters()
        self.port_mlp_parameters()
        self.port_layer_norm_parameters()

    def get_transformed_module(self):
        return self.ipex_optimized_module

def _convert_to_bloom_cache_ipex(
        past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Converts the cache to the format expected by Bloom, i.e. to tuple(tuple([batch_size * num_heads, ...]))
        """
        batch_size, num_heads, seq_length, head_dim = past_key_value[0][0].shape

        return tuple(
            (
                layer_past[0].view(batch_size, num_heads, seq_length, head_dim),
                layer_past[1].view(batch_size, num_heads, seq_length, head_dim),
            )
            for layer_past in past_key_value
        )
def transformer_frontend_replace(model, config = None, dtype = torch.float):
    import transformers
    try:
        import deepspeed
    except ImportError as e:
        print("Warning: we didn't find Deepspeed in your env, multi-tile optimization will be closed")
    else:
        if isinstance(model, deepspeed.InferenceEngine):
            IPEXTransformerConverter.update_tp_data(model._config.tensor_parallel.tp_size, model._config.tensor_parallel.tp_group)

    transformers_converter = {
        transformers.models.gptj.modeling_gptj.GPTJBlock: IPEXGPTJConverter,
        transformers.models.llama.modeling_llama.LlamaDecoderLayer: IPEXLlamaConverter,
        transformers.models.opt.modeling_opt.OPTDecoderLayer: IPEXOptConverter,
        transformers.models.bloom.modeling_bloom.BloomBlock: IPEXBloomConverter
    }

    if config is None and hasattr(model, "config"):
        config = model.config
        config.dtype = dtype
        config.device = model.device
    if hasattr(model, "_convert_to_bloom_cache"):
        setattr(model, "_convert_to_bloom_cache", _convert_to_bloom_cache_ipex)
    for name, module in model.named_children():
        for m, converter in transformers_converter.items():
            if isinstance(module, m):
                module_converter = converter(module, config, dtype=dtype, device=config.device)
                module_transformed = module_converter.get_transformed_module()
                setattr(model, name, module_transformed)
                continue
            else:
                transformer_frontend_replace(module, config, dtype=dtype)
    return model
