import torch 
from ._transformers import IPEXGPTJBlock, IPEXLlamaBlock, IPEXOptBlock
from transformers.models.gptj.configuration_gptj import GPTJConfig
from transformers.models.gptj.modeling_gptj import GPTJBlock
from transformers.models.opt.configuration_opt import OPTConfig
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.configuration_utils import PretrainedConfig
import transformers
from .RoPE import GPTJRotaryEmbedding, LlamaRotaryEmbedding, PositionalEmbedding
from ._transformer_configuration import IPEXTransformerConfig
# from transformers.models.llama.configuration_llama import 

class IPEXTransformerConverter:
    def __init__(self, module, config, device = "cpu", dtype = torch.float) -> None:
        self.module = module
        self.config = config
        self.dtype = dtype
        self.device = device

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


class IPEXGPTJConverter(IPEXTransformerConverter):
    def __init__(self,
                 module,
                 config: GPTJConfig = None,
                 device = "cpu",
                 dtype = torch.float):
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
            seq_first=False,
            kv_cache_optimize=False,
            positional_embedding_base=10000,
            sdp_fusion_enable=True,
            device=self.device,
            dtype=self.dtype
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
         
        self.ipex_optimized_module.attn.q_wei = self.ipex_optimized_module.attn.q_proj.weight.t().contiguous()
        self.ipex_optimized_module.attn.k_wei = self.ipex_optimized_module.attn.k_proj.weight.t().contiguous()
        self.ipex_optimized_module.attn.v_wei = self.ipex_optimized_module.attn.v_proj.weight.t().contiguous()
        self.ipex_optimized_module.attn.out_wei = self.ipex_optimized_module.attn.out_proj.weight.t().contiguous()
        shape = [3, -1, self.ipex_optimized_module.attn.q_wei.shape[-1]]
        self.ipex_optimized_module.attn.qkv_wei = torch.cat([self.ipex_optimized_module.attn.q_wei, self.ipex_optimized_module.attn.k_wei, self.ipex_optimized_module.attn.v_wei], dim=0).view(shape)
        

    def port_mlp_parameters(self):
        self.ipex_optimized_module.mlp.fc_in.weight = self.module.mlp.fc_in.weight
        self.ipex_optimized_module.mlp.fc_in.bias = self.module.mlp.fc_in.bias
        self.ipex_optimized_module.mlp.fc_out.weight = self.module.mlp.fc_out.weight
        self.ipex_optimized_module.mlp.fc_out.bias = self.module.mlp.fc_out.bias
        self.ipex_optimized_module.mlp.fc_in_wei = self.ipex_optimized_module.mlp.fc_in.weight.t().contiguous()
        self.ipex_optimized_module.mlp.fc_out_wei = self.ipex_optimized_module.mlp.fc_out.weight.t().contiguous()

        
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
                 config: OPTConfig = None,
                 device = "cpu",
                 dtype = torch.float):
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
            dtype=self.dtype
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
        
        self.ipex_optimized_module.attn.q_wei = self.ipex_optimized_module.attn.q_proj.weight.t().contiguous()
        self.ipex_optimized_module.attn.k_wei = self.ipex_optimized_module.attn.k_proj.weight.t().contiguous()
        self.ipex_optimized_module.attn.v_wei = self.ipex_optimized_module.attn.v_proj.weight.t().contiguous()
        self.ipex_optimized_module.attn.out_wei = self.ipex_optimized_module.attn.out_proj.weight.t().contiguous()
        shape = [3, -1, self.ipex_optimized_module.attn.q_wei.shape[-1]]
        self.ipex_optimized_module.attn.qkv_wei = torch.cat([self.ipex_optimized_module.attn.q_wei, self.ipex_optimized_module.attn.k_wei, self.ipex_optimized_module.attn.v_wei], dim=0).view(shape)

    def port_mlp_parameters(self):
        self.ipex_optimized_module.mlp.fc_in.weight = self.module.fc1.weight
        self.ipex_optimized_module.mlp.fc_in.bias = self.module.fc1.bias
        self.ipex_optimized_module.mlp.fc_out.weight = self.module.fc2.weight
        self.ipex_optimized_module.mlp.fc_out.bias = self.module.fc2.bias
        self.ipex_optimized_module.mlp.fc_in_wei = self.ipex_optimized_module.mlp.fc_in.weight.t().contiguous()
        self.ipex_optimized_module.mlp.fc_out_wei = self.ipex_optimized_module.mlp.fc_out.weight.t().contiguous()

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
            scale_attention=False,
            is_decoder=False,
            do_norm_before=None,
            ln_elementwise_affine=None,
            seq_first=False,
            kv_cache_optimize=True,
            positional_embedding_base=10000,
            sdp_fusion_enable=False,
            device=self.device,
            dtype=self.dtype
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
        
        self.ipex_optimized_module.attn.q_wei = self.ipex_optimized_module.attn.q_proj.weight.t().contiguous()
        self.ipex_optimized_module.attn.k_wei = self.ipex_optimized_module.attn.k_proj.weight.t().contiguous()
        self.ipex_optimized_module.attn.v_wei = self.ipex_optimized_module.attn.v_proj.weight.t().contiguous()
        self.ipex_optimized_module.attn.out_wei = self.ipex_optimized_module.attn.out_proj.weight.t().contiguous()
        shape = [3, -1, self.ipex_optimized_module.attn.q_wei.shape[-1]]
        self.ipex_optimized_module.attn.qkv_wei = torch.cat([self.ipex_optimized_module.attn.q_wei, self.ipex_optimized_module.attn.k_wei, self.ipex_optimized_module.attn.v_wei], dim=0).view(shape)

    def port_mlp_parameters(self):
        self.ipex_optimized_module.mlp.fc_in.weight = self.module.mlp.gate_proj.weight
        self.ipex_optimized_module.mlp.fc_in.bias = self.module.mlp.gate_proj.bias
        self.ipex_optimized_module.mlp.fc_out.weight = self.module.mlp.down_proj.weight
        self.ipex_optimized_module.mlp.fc_out.bias = self.module.mlp.down_proj.bias
        self.ipex_optimized_module.mlp.up_proj.weight = self.module.mlp.up_proj.weight
        self.ipex_optimized_module.mlp.up_proj.bias = self.module.mlp.up_proj.bias
        self.ipex_optimized_module.mlp.fc_in_wei = self.ipex_optimized_module.mlp.fc_in.weight.t().contiguous()
        self.ipex_optimized_module.mlp.fc_out_wei = self.ipex_optimized_module.mlp.fc_out.weight.t().contiguous()
        self.ipex_optimized_module.mlp.up_wei = self.ipex_optimized_module.mlp.up_proj.weight.t().contiguous()

    def port_layer_norm_parameters(self):
        self.ipex_optimized_module.input_layernorm.weight = self.module.input_layernorm.weight
        self.ipex_optimized_module.post_attn_layernorm.weight = self.module.post_attention_layernorm.weight

    def port_all_parameters_to_new_module(self):
        self.port_attn_parameters()
        self.port_mlp_parameters()
        self.port_layer_norm_parameters()

    def get_transformed_module(self):
        return self.ipex_optimized_module

def transformer_frontend_replace(model, config = None, dtype = torch.float):
    transformers_converter = {
        transformers.models.gptj.modeling_gptj.GPTJBlock: IPEXGPTJConverter,
        transformers.models.llama.modeling_llama.LlamaDecoderLayer: IPEXLlamaConverter,
        transformers.models.opt.modeling_opt.OPTDecoderLayer: IPEXOptConverter
    }
    if config is None and hasattr(model, "config"):
        config = model.config
        config.dtype = dtype
        config.device = model.device
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
