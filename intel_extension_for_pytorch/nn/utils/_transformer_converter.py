import torch 
import os
from ._transformers import IPEXGPTJBlock, IPEXLlamaBlock, IPEXOptBlock, IPEXBloomBlock, IPEXEmptyLinearWithPadding, IPEXEmptyINT4LinearWithPadding

# from transformers.models.gptj.modeling_gptj import GPTJBlock

# from transformers.models.opt.modeling_opt import OPTDecoderLayer
# from transformers.configuration_utils import PretrainedConfig
from .RoPE import GPTJRotaryEmbedding, LlamaRotaryEmbedding, PositionalEmbedding
from ._transformer_configuration import IPEXTransformerConfig
from typing import Optional, Tuple, Union
import torch.nn as nn

from functools import partial
from ._utils import ipex_beam_search, _ipex_prepare_model_inputs, ipex_beam_search_without_optimize
from ._inference_ops import OpConverter
# from transformers.models.llama.configuration_llama import 

MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "0"))
MAX_OUT_SEQ_LEN = max(128, int(os.environ.get("MAX_OUT_SEQ_LEN", "0")))

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
                 dtype = torch.float,
                 is_int4 = False):
        from transformers.models.gptj.configuration_gptj import GPTJConfig
        super().__init__(module, config, device=device, dtype=dtype)
        self.is_int4 = is_int4
        self.config = config if config is not None else GPTJConfig()
        self.ipex_transformers_config = self.construct_transformer_config()
        # print(self.ipex_transformers_config.__dict__)
        self.ipex_optimized_module = self.construct_ipex_optimized_module()
        self.port_all_parameters_to_new_module()

    def construct_transformer_config(self):
        n_positions = max(self.config.n_positions, MAX_SEQ_LEN)
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
            max_out_positions=MAX_OUT_SEQ_LEN,
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
        return IPEXGPTJBlock(self.ipex_transformers_config, self.is_int4)

    def port_attn_parameters(self):
        if self.is_int4:
            if self.row_major:
                self.ipex_optimized_module.attn.q_wei = self.module.attn.q_proj.qweight
                self.ipex_optimized_module.attn.q_scl = self.module.attn.q_proj.scales
                self.ipex_optimized_module.attn.q_zp = self.module.attn.q_proj.qzeros
                self.ipex_optimized_module.attn.q_gs = self.module.attn.q_proj.group_size
                self.ipex_optimized_module.attn.q_proj.bias = self.module.attn.q_proj.bias
                self.ipex_optimized_module.attn.k_wei = self.module.attn.k_proj.qweight
                self.ipex_optimized_module.attn.k_scl = self.module.attn.k_proj.scales
                self.ipex_optimized_module.attn.k_zp = self.module.attn.k_proj.qzeros
                self.ipex_optimized_module.attn.k_gs = self.module.attn.k_proj.group_size
                self.ipex_optimized_module.attn.k_proj.bias = self.module.attn.k_proj.bias
                self.ipex_optimized_module.attn.v_wei = self.module.attn.v_proj.qweight
                self.ipex_optimized_module.attn.v_scl = self.module.attn.v_proj.scales
                self.ipex_optimized_module.attn.v_zp = self.module.attn.v_proj.qzeros
                self.ipex_optimized_module.attn.v_gs = self.module.attn.v_proj.group_size
                self.ipex_optimized_module.attn.v_proj.bias = self.module.attn.v_proj.bias
                self.ipex_optimized_module.attn.out_wei = self.module.attn.out_proj.qweight
                self.ipex_optimized_module.attn.out_scl = self.module.attn.out_proj.scales
                self.ipex_optimized_module.attn.out_zp = self.module.attn.out_proj.qzeros
                self.ipex_optimized_module.attn.out_gs = self.module.attn.out_proj.group_size
                self.ipex_optimized_module.attn.out_proj.bias = self.module.attn.out_proj.bias

                shape = [3, -1, self.module.attn.q_proj.qweight.shape[-1]]
                self.ipex_optimized_module.attn.qkv_wei = torch.stack([self.ipex_optimized_module.attn.q_wei, self.ipex_optimized_module.attn.k_wei, self.ipex_optimized_module.attn.v_wei]).contiguous().view(shape)
                self.ipex_optimized_module.attn.qkv_scl = torch.stack([self.ipex_optimized_module.attn.q_scl, self.ipex_optimized_module.attn.k_scl, self.ipex_optimized_module.attn.v_scl]).contiguous().view(shape)
                self.ipex_optimized_module.attn.qkv_zp = torch.stack([self.ipex_optimized_module.attn.q_zp, self.ipex_optimized_module.attn.k_zp, self.ipex_optimized_module.attn.v_zp]).contiguous().view(shape)
                self.ipex_optimized_module.attn.qkv_gs = self.ipex_optimized_module.attn.q_gs

                self.ipex_optimized_module.attn.q_wei.data = self.ipex_optimized_module.attn.qkv_wei[0, :, :]
                self.ipex_optimized_module.attn.q_scl.data = self.ipex_optimized_module.attn.qkv_scl[0, :, :]
                self.ipex_optimized_module.attn.q_zp.data = self.ipex_optimized_module.attn.qkv_zp[0, :, :]
                self.ipex_optimized_module.attn.k_wei.data = self.ipex_optimized_module.attn.qkv_wei[1, :, :]
                self.ipex_optimized_module.attn.k_scl.data = self.ipex_optimized_module.attn.qkv_scl[1, :, :]
                self.ipex_optimized_module.attn.k_zp.data = self.ipex_optimized_module.attn.qkv_zp[1, :, :]
                self.ipex_optimized_module.attn.v_wei.data = self.ipex_optimized_module.attn.qkv_wei[2, :, :]
                self.ipex_optimized_module.attn.v_scl.data = self.ipex_optimized_module.attn.qkv_scl[2, :, :]
                self.ipex_optimized_module.attn.v_zp.data = self.ipex_optimized_module.attn.qkv_zp[2, :, :]
                self.ipex_optimized_module.attn.qkv_bias = None
            else:
                self.ipex_optimized_module.attn.k_proj.qweight = self.module.attn.k_proj.qweight
                self.ipex_optimized_module.attn.k_proj.scales = self.module.attn.k_proj.scales
                self.ipex_optimized_module.attn.k_proj.qzeros = self.module.attn.k_proj.qzeros
                self.ipex_optimized_module.attn.k_proj.group_size = self.module.attn.k_proj.group_size
                self.ipex_optimized_module.attn.k_proj.bias = self.module.attn.k_proj.bias
                self.ipex_optimized_module.attn.q_proj.qweight = self.module.attn.q_proj.qweight
                self.ipex_optimized_module.attn.q_proj.scales = self.module.attn.q_proj.scales
                self.ipex_optimized_module.attn.q_proj.qzeros = self.module.attn.q_proj.qzeros
                self.ipex_optimized_module.attn.q_proj.group_size = self.module.attn.q_proj.group_size
                self.ipex_optimized_module.attn.q_proj.bias = self.module.attn.q_proj.bias
                self.ipex_optimized_module.attn.v_proj.qweight = self.module.attn.v_proj.qweight
                self.ipex_optimized_module.attn.v_proj.scales = self.module.attn.v_proj.scales
                self.ipex_optimized_module.attn.v_proj.qzeros = self.module.attn.v_proj.qzeros
                self.ipex_optimized_module.attn.v_proj.group_size = self.module.attn.v_proj.group_size
                self.ipex_optimized_module.attn.v_proj.bias = self.module.attn.v_proj.bias
                self.ipex_optimized_module.attn.out_proj.qweight = self.module.attn.out_proj.qweight
                self.ipex_optimized_module.attn.out_proj.scales = self.module.attn.out_proj.scales
                self.ipex_optimized_module.attn.out_proj.qzeros = self.module.attn.out_proj.qzeros
                self.ipex_optimized_module.attn.out_proj.group_size = self.module.attn.out_proj.group_size
                self.ipex_optimized_module.attn.out_proj.bias = self.module.attn.out_proj.bias


        else: 
            if self.row_major:
                self.module.attn.q_proj.weight.data = self.module.attn.q_proj.weight.transpose(0, 1).contiguous()
                self.ipex_optimized_module.attn.q_wei = self.module.attn.q_proj.weight
                self.ipex_optimized_module.attn.q_proj.bias = self.module.attn.q_proj.bias
                self.module.attn.k_proj.weight.data = self.module.attn.k_proj.weight.transpose(0, 1).contiguous()
                self.ipex_optimized_module.attn.k_wei = self.module.attn.k_proj.weight
                self.ipex_optimized_module.attn.k_proj.bias = self.module.attn.k_proj.bias
                self.module.attn.v_proj.weight.data = self.module.attn.v_proj.weight.transpose(0, 1).contiguous()
                self.ipex_optimized_module.attn.v_wei = self.module.attn.v_proj.weight
                self.ipex_optimized_module.attn.v_proj.bias = self.module.attn.v_proj.bias
                self.module.attn.out_proj.weight.data = self.module.attn.out_proj.weight.transpose(0, 1).contiguous()
                self.ipex_optimized_module.attn.out_wei = self.module.attn.out_proj.weight
                self.ipex_optimized_module.attn.out_proj.bias = self.module.attn.out_proj.bias
                shape = [3, -1, self.module.attn.q_proj.weight.shape[-1]]
                self.ipex_optimized_module.attn.qkv_wei = torch.stack([self.ipex_optimized_module.attn.q_wei, self.ipex_optimized_module.attn.k_wei, self.ipex_optimized_module.attn.v_wei]).contiguous().view(shape)
                self.ipex_optimized_module.attn.q_wei.data = self.ipex_optimized_module.attn.qkv_wei[0, :, :]
                self.ipex_optimized_module.attn.k_wei.data = self.ipex_optimized_module.attn.qkv_wei[1, :, :]
                self.ipex_optimized_module.attn.v_wei.data = self.ipex_optimized_module.attn.qkv_wei[2, :, :]
                self.ipex_optimized_module.attn.qkv_bias = None
            else:
                self.ipex_optimized_module.attn.k_proj.weight = self.module.attn.k_proj.weight
                self.ipex_optimized_module.attn.k_proj.bias = self.module.attn.k_proj.bias
                self.ipex_optimized_module.attn.q_proj.weight = self.module.attn.q_proj.weight
                self.ipex_optimized_module.attn.q_proj.bias = self.module.attn.q_proj.bias
                self.ipex_optimized_module.attn.v_proj.weight = self.module.attn.v_proj.weight
                self.ipex_optimized_module.attn.v_proj.bias = self.module.attn.v_proj.bias
                self.ipex_optimized_module.attn.out_proj.weight = self.module.attn.out_proj.weight
                self.ipex_optimized_module.attn.out_proj.bias = self.module.attn.out_proj.bias

    def port_mlp_parameters(self):
        if self.is_int4:
            if self.row_major:
                self.ipex_optimized_module.mlp.fc_in_wei = self.module.mlp.fc_in.qweight
                self.ipex_optimized_module.mlp.fc_in_scl = self.module.mlp.fc_in.scales
                self.ipex_optimized_module.mlp.fc_in_zp = self.module.mlp.fc_in.qzeros
                self.ipex_optimized_module.mlp.fc_in_gs = self.module.mlp.fc_in.group_size
                self.ipex_optimized_module.mlp.fc_in.bias = self.module.mlp.fc_in.bias
                self.ipex_optimized_module.mlp.fc_out_wei = self.module.mlp.fc_out.qweight
                self.ipex_optimized_module.mlp.fc_out_scl = self.module.mlp.fc_out.scales
                self.ipex_optimized_module.mlp.fc_out_zp = self.module.mlp.fc_out.qzeros
                self.ipex_optimized_module.mlp.fc_out_gs = self.module.mlp.fc_out.group_size
                self.ipex_optimized_module.mlp.fc_out.bias = self.module.mlp.fc_out.bias
            else:
                self.ipex_optimized_module.mlp.fc_in.qweight = self.module.mlp.fc_in.qweight
                self.ipex_optimized_module.mlp.fc_in.scales = self.module.mlp.fc_in.scales
                self.ipex_optimized_module.mlp.fc_in.qzeros = self.module.mlp.fc_in.qzeros
                self.ipex_optimized_module.mlp.fc_in.group_size = self.module.mlp.fc_in.group_size
                self.ipex_optimized_module.mlp.fc_in.bias = self.module.mlp.fc_in.bias
                self.ipex_optimized_module.mlp.fc_out.qweight = self.module.mlp.fc_out.qweight
                self.ipex_optimized_module.mlp.fc_out.scales = self.module.mlp.fc_out.scales
                self.ipex_optimized_module.mlp.fc_out.qzeros = self.module.mlp.fc_out.qzeros
                self.ipex_optimized_module.mlp.fc_out.group_size = self.module.mlp.fc_out.group_size
                self.ipex_optimized_module.mlp.fc_out.bias = self.module.mlp.fc_out.bias

        else:
            if self.row_major:
                self.module.mlp.fc_in.weight.data = self.module.mlp.fc_in.weight.transpose(0, 1).contiguous()
                self.ipex_optimized_module.mlp.fc_in_wei = self.module.mlp.fc_in.weight
                self.ipex_optimized_module.mlp.fc_in.bias = self.module.mlp.fc_in.bias
                self.module.mlp.fc_out.weight.data = self.module.mlp.fc_out.weight.transpose(0, 1).contiguous()
                self.ipex_optimized_module.mlp.fc_out_wei = self.module.mlp.fc_out.weight
                self.ipex_optimized_module.mlp.fc_out.bias = self.module.mlp.fc_out.bias
            else:
                self.ipex_optimized_module.mlp.fc_in.weight = self.module.mlp.fc_in.weight
                self.ipex_optimized_module.mlp.fc_in.bias = self.module.mlp.fc_in.bias
                self.ipex_optimized_module.mlp.fc_out.weight = self.module.mlp.fc_out.weight
                self.ipex_optimized_module.mlp.fc_out.bias = self.module.mlp.fc_out.bias

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
        n_positions = max(self.config.max_position_embeddings, MAX_SEQ_LEN)
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
            max_out_positions=MAX_OUT_SEQ_LEN,
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
            seq_first=True,
            kv_cache_optimize=False,
            positional_embedding_base=10000,
            sdp_fusion_enable=True,
            device=self.device,
            dtype=self.dtype,
            tp_size=IPEXTransformerConverter.tp_size,
            tp_group=IPEXTransformerConverter.tp_group
        )

    def construct_ipex_optimized_module(self):
        return IPEXOptBlock(self.ipex_transformers_config)

    def port_attn_parameters(self):
        if self.row_major:
            self.module.self_attn.q_proj.weight.data = self.module.self_attn.q_proj.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.attn.q_wei = self.module.self_attn.q_proj.weight
            self.ipex_optimized_module.attn.q_proj.bias = self.module.self_attn.q_proj.bias

            self.module.self_attn.k_proj.weight.data = self.module.self_attn.k_proj.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.attn.k_wei = self.module.self_attn.k_proj.weight
            self.ipex_optimized_module.attn.k_proj.bias = self.module.self_attn.k_proj.bias

            self.module.self_attn.v_proj.weight.data = self.module.self_attn.v_proj.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.attn.v_wei = self.module.self_attn.v_proj.weight
            self.ipex_optimized_module.attn.v_proj.bias = self.module.self_attn.v_proj.bias

            self.module.self_attn.out_proj.weight.data = self.module.self_attn.out_proj.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.attn.out_wei = self.module.self_attn.out_proj.weight
            self.ipex_optimized_module.attn.out_proj.bias = self.module.self_attn.out_proj.bias
        else:
            self.ipex_optimized_module.attn.k_proj.weight = self.module.self_attn.k_proj.weight
            self.ipex_optimized_module.attn.k_proj.bias = self.module.self_attn.k_proj.bias
            self.ipex_optimized_module.attn.q_proj.weight = self.module.self_attn.q_proj.weight
            self.ipex_optimized_module.attn.q_proj.bias = self.module.self_attn.q_proj.bias
            self.ipex_optimized_module.attn.v_proj.weight = self.module.self_attn.v_proj.weight
            self.ipex_optimized_module.attn.v_proj.bias = self.module.self_attn.v_proj.bias
            self.ipex_optimized_module.attn.out_proj.weight = self.module.self_attn.out_proj.weight
            self.ipex_optimized_module.attn.out_proj.bias = self.module.self_attn.out_proj.bias

    def port_mlp_parameters(self):
        if self.row_major:
            self.module.fc1.weight.data = self.module.fc1.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.mlp.fc_in_wei = self.module.fc1.weight.data
            self.ipex_optimized_module.mlp.fc_in_bias = self.module.fc1.bias

            self.module.fc2.weight.data = self.module.fc2.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.mlp.fc_out_wei = self.module.fc2.weight
            self.ipex_optimized_module.mlp.fc_out_bias = self.module.fc2.bias
        else:
            self.ipex_optimized_module.mlp.fc_in.weight = self.module.fc1.weight
            self.ipex_optimized_module.mlp.fc_in.bias = self.module.fc1.bias
            self.ipex_optimized_module.mlp.fc_out.weight = self.module.fc2.weight
            self.ipex_optimized_module.mlp.fc_out.bias = self.module.fc2.bias

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
        n_positions = max(2048, MAX_SEQ_LEN)
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
            max_out_positions=MAX_OUT_SEQ_LEN,
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
        return IPEXBloomBlock(self.ipex_transformers_config)

    def port_attn_parameters(self):
        tp_size = self.ipex_transformers_config.tp_size
        if self.row_major:
            embed_dim = self.config.hidden_size
            num_head = self.config.n_head // tp_size
            shape = [num_head, 3, -1, embed_dim]

            self.module.self_attention.query_key_value.weight.data = \
                self.module.self_attention.query_key_value.weight.view(shape).contiguous().transpose(0, 1).contiguous().view([3, -1, embed_dim]).transpose(1, 2).contiguous()
            self.ipex_optimized_module.self_attention.qkv_wei = \
                self.module.self_attention.query_key_value.weight

            self.module.self_attention.query_key_value.bias.data = \
                self.module.self_attention.query_key_value.bias.view([num_head, 3, -1]).transpose(0, 1).contiguous().view([3, -1])
            self.ipex_optimized_module.self_attention.qkv_bias = \
                self.module.self_attention.query_key_value.bias

            self.module.self_attention.dense.weight.data = \
                self.module.self_attention.dense.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.self_attention.out_wei = self.module.self_attention.dense.weight
            self.module.self_attention.dense.bias.data = self.module.self_attention.dense.bias.data / self.tp_size
            self.ipex_optimized_module.self_attention.out_bias = self.module.self_attention.dense.bias
        else:
            self.ipex_optimized_module.self_attention.query_key_value.weight = nn.Parameter(self.module.self_attention.query_key_value.weight)
            self.ipex_optimized_module.self_attention.query_key_value.bias = nn.Parameter(self.module.self_attention.query_key_value.bias)
            self.ipex_optimized_module.self_attention.out_proj.weight = nn.Parameter(self.module.self_attention.dense.weight)
            self.ipex_optimized_module.self_attention.out_proj.bias = nn.Parameter(self.module.self_attention.dense.bias)


    def port_mlp_parameters(self):
        if self.row_major:
            self.module.mlp.dense_h_to_4h.weight.data = self.module.mlp.dense_h_to_4h.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.mlp.fc_in_wei = self.module.mlp.dense_h_to_4h.weight
            self.ipex_optimized_module.mlp.fc_in.bias = nn.Parameter(self.module.mlp.dense_h_to_4h.bias)

            self.module.mlp.dense_4h_to_h.weight.data = self.module.mlp.dense_4h_to_h.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.mlp.fc_out_wei = self.module.mlp.dense_4h_to_h.weight
            self.module.mlp.dense_4h_to_h.bias.data = self.module.mlp.dense_4h_to_h.bias.data / self.tp_size
            self.ipex_optimized_module.mlp.fc_out_bias = self.module.mlp.dense_4h_to_h.bias
        else:
            self.ipex_optimized_module.mlp.fc_in.weight = nn.Parameter(self.module.mlp.dense_h_to_4h.weight)
            self.ipex_optimized_module.mlp.fc_in.bias = nn.Parameter(self.module.mlp.dense_h_to_4h.bias)
            self.ipex_optimized_module.mlp.fc_out.weight = nn.Parameter(self.module.mlp.dense_4h_to_h.weight)
            self.ipex_optimized_module.mlp.fc_out.bias = nn.Parameter(self.module.mlp.dense_4h_to_h.bias)

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

def int4_gemm_padding(qdata):
    k, n = qdata.shape
    if n % 8 != 0:
        padded_n = (n + 8 - 1) // 8 * 8
        padded_qdata = torch.empty(k, padded_n, dtype=qdata.dtype, device=qdata.device)
        padded_qdata[:, :n] = qdata
        return padded_qdata
    else:
        return qdata
		
def int4_gemm_bias_padding(qdata):
    n = qdata.shape[0]
    if n % 4 != 0:
        padded_n = (n + 4 - 1) // 4 * 4
        padded_qdata = torch.empty(padded_n, dtype=qdata.dtype, device=qdata.device)
        padded_qdata[:n] = qdata
        return padded_qdata
    else:
        return qdata

def int4_gemm_scale_padding(scale):
    k, n = scale.shape
    if n % 4 != 0:
        padded_n = (n + 4 - 1) // 4 * 4
        padded_scale = torch.empty(k, padded_n, dtype=scale.dtype, device=scale.device)
        padded_scale[:, :n] = scale
        return padded_scale
    else:
        return scale

def gemm_padding(weight, bias=None):
    n, k = weight.shape
    if n % 4 != 0:
        padded_n = (n + 4 - 1) // 4 * 4
        padded_weight = torch.zeros(padded_n, k, dtype=weight.dtype, device=weight.device)
        padded_weight[:n, :] = weight
        if bias is not None:
            padded_bias = torch.zeros(padded_n, dtype=bias.dtype, device=bias.device)
            padded_bias[:n] = bias
        else:
            padded_bias = None
        return padded_weight, padded_bias
    else:
        return weight, bias

def pad_for_gptj_lm_head(model, is_int4):
    if is_int4:
        n = model.lm_head.qweight.shape[1] * 2 - 1 #specific for 50401(25201) int4 weight

        lm_head_new = IPEXEmptyINT4LinearWithPadding(n)
        lm_head_new.qweight = model.lm_head.qweight
        lm_head_new.bias = model.lm_head.bias if model.lm_head.bias is not None else None
        lm_head_new.scales = model.lm_head.scales
        lm_head_new.qzeros = model.lm_head.qzeros
        lm_head_new.group_size = model.lm_head.group_size
        model.lm_head = lm_head_new 

        model.lm_head.qweight.data = int4_gemm_padding(model.lm_head.qweight)
        model.lm_head.scales.data = int4_gemm_scale_padding(model.lm_head.scales)
        model.lm_head.qzeros.data = int4_gemm_padding(model.lm_head.qzeros)

        if model.lm_head.bias is not None:
            model.lm_head.bias.data = int4_gemm_bias_padding(model.lm_head.bias)

    else:
        n = model.lm_head.weight.shape[0] #[n, k]

        lm_head_new = IPEXEmptyLinearWithPadding(n)
        lm_head_new.weight = model.lm_head.weight
        lm_head_new.bias = model.lm_head.bias
        model.lm_head = lm_head_new

        if model.lm_head.bias is not None:
            model.lm_head.weight.data, model.lm_head.bias.data = gemm_padding(model.lm_head.weight, model.lm_head.bias)
        else:
            model.lm_head.weight.data, _ = gemm_padding(model.lm_head.weight)

def transformer_frontend_replace(model, config = None, dtype = torch.float, is_int4=False):
    import transformers
    enable_ds = False
    try:
        import deepspeed
    except ImportError as e:
        print("Warning: we didn't find Deepspeed in your env, multi-tile optimization will be closed")
    else:
        enable_ds = True
        OpConverter.update_deepspeed_supported_ops()
        if isinstance(model, deepspeed.InferenceEngine):
            IPEXTransformerConverter.update_tp_data(model._config.tensor_parallel.tp_size, model._config.tensor_parallel.tp_group)

    transformers_converter = {
        transformers.models.gptj.modeling_gptj.GPTJBlock: IPEXGPTJConverter,
        transformers.models.llama.modeling_llama.LlamaDecoderLayer: IPEXLlamaConverter,
        transformers.models.opt.modeling_opt.OPTDecoderLayer: IPEXOptConverter,
        transformers.models.bloom.modeling_bloom.BloomBlock: IPEXBloomConverter
    }

    def recursive_module_replace(module, config, dtype, enable_deepspeed=False, is_int4=is_int4):
        not_deepspeed_engine = not enable_deepspeed or not isinstance(module, deepspeed.InferenceEngine)
        if config is None and hasattr(module, "config") and not_deepspeed_engine:
            config = module.config
            config.dtype = dtype
            config.device = module.device

        if hasattr(module, "_convert_to_bloom_cache"):
            setattr(module, "_convert_to_bloom_cache", _convert_to_bloom_cache_ipex)
        
        if hasattr(module, "_prepare_model_inputs"):
            setattr(module, "_prepare_model_inputs", partial(_ipex_prepare_model_inputs, module))

        if os.environ.get("DISABLE_KV_CACHE", "OFF") not in ["1", "Y", "YES", "TRUE", "ON"]:
            if hasattr(module, "beam_search"):
                setattr(module, "beam_search", partial(ipex_beam_search, module))


        for name, named_module in module.named_children():
            if type(named_module) in transformers_converter.keys():
                module_converter = transformers_converter[type(named_module)](named_module, config, dtype=dtype, device=config.device, is_int4=is_int4)
                module_transformed = module_converter.get_transformed_module()
                setattr(module, name, module_transformed)
            # elif OpConverter.valid_op_for_convert(named_module):
            #     op_transformed = OpConverter.convert_op(named_module)
            #     setattr(module, name, op_transformed)
            else:
                recursive_module_replace(named_module, config, dtype=dtype, is_int4=is_int4)
        return model

    replaced_model = recursive_module_replace(model, None, dtype=dtype, enable_deepspeed=enable_ds, is_int4=is_int4)

    return replaced_model
