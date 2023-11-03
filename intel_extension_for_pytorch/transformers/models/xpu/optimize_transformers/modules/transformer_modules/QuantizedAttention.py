import torch
from .._transformer_configuration import IPEXTransformerConfig
from .Attention import IPEXTransformerAttnOptimizedFp16
from .Linear import IPEXTransformerQLinear


class IPEXTransformerAttnOptimizedInt4(IPEXTransformerAttnOptimizedFp16):
    def __init__(self, config: IPEXTransformerConfig) -> None:
        super().__init__(config)

        self.q_proj_quant = IPEXTransformerQLinear()
        self.k_proj_quant = IPEXTransformerQLinear()
        self.v_proj_quant = IPEXTransformerQLinear()
        self.out_proj_quant = IPEXTransformerQLinear()

        self.qkv_proj_quant = IPEXTransformerQLinear()

    def load_parameter(self, q_proj, k_proj, v_proj, out_proj):
        self.q_proj_quant.weight = q_proj.qweight
        self.k_proj_quant.weight = k_proj.qweight
        self.v_proj_quant.weight = v_proj.qweight
        self.out_proj_quant.weight = out_proj.qweight

        self.q_proj_quant.scale = q_proj.scales
        self.k_proj_quant.scale = k_proj.scales
        self.v_proj_quant.scale = v_proj.scales
        self.out_proj_quant.scale = out_proj.scales

        self.q_proj_quant.zp = q_proj.qzeros
        self.k_proj_quant.zp = k_proj.qzeros
        self.v_proj_quant.zp = v_proj.qzeros
        self.out_proj_quant.zp = out_proj.qzeros

        self.q_proj_quant.gs = q_proj.group_size
        self.k_proj_quant.gs = k_proj.group_size
        self.v_proj_quant.gs = v_proj.group_size
        self.out_proj_quant.gs = out_proj.group_size

        self.position_embed = self.config.rotary_embedding_class(
            self.config, torch.float16
        )

    def transpose_parameter(self):
        pass

    def cat_qkv(self):
        shape = [3, -1, self.q_proj_quant.weight.shape[-1]]
        self.qkv_proj_quant.weight = (
            torch.stack(
                [
                    self.q_proj_quant.weight,
                    self.k_proj_quant.weight,
                    self.v_proj_quant.weight,
                ]
            )
            .contiguous()
            .view(shape)
        )
        self.qkv_proj_quant.scale = (
            torch.stack(
                [
                    self.q_proj_quant.scale,
                    self.k_proj_quant.scale,
                    self.v_proj_quant.scale,
                ]
            )
            .contiguous()
            .view(shape)
        )
        self.qkv_proj_quant.zp = (
            torch.stack(
                [self.q_proj_quant.zp, self.k_proj_quant.zp, self.v_proj_quant.zp]
            )
            .contiguous()
            .view(shape)
        )
        self.qkv_proj_quant.gs = self.q_proj_quant.gs
        self.qkv_proj_quant.bias = None

    def compute_qkv_gemm(self, hidden_states, query, key, value):
        torch.ops.torch_ipex.mm_qkv_out_int4(
            hidden_states,
            self.qkv_proj_quant.weight,
            self.qkv_proj_quant.scale,
            self.qkv_proj_quant.zp,
            self.qkv_proj_quant.bias,
            query,
            key,
            value,
            self.qkv_proj_quant.gs,
        )
        return query, key, value

    def out_proj_compute(self, attn_output, residual=None):
        if residual is None:
            attn_output = torch.ops.torch_ipex.mm_int4(
                attn_output,
                self.out_proj_quant.weight,
                self.out_proj_quant.scale,
                self.out_proj_quant.zp,
                self.out_proj_quant.gs,
            )
            if self.out_proj.bias is not None:
                attn_output += self.out_proj.bias
        else:
            shape = [attn_output.shape[0], attn_output.shape[1], self.embed_dim]
            if self.out_proj.bias is not None:
                attn_output = torch.ops.torch_ipex.mm_bias_int4(
                    attn_output,
                    self.out_proj_quant.weight,
                    self.out_proj_quant.bias,
                    self.out_proj_quant.scale,
                    self.out_proj_quant.zp,
                    self.out_proj_quant.gs,
                )
            else:
                attn_output = torch.ops.torch_ipex.mm_int4(
                    attn_output,
                    self.out_proj_quant.weight,
                    self.out_proj_quant.scale,
                    self.out_proj_quant.zp,
                    self.out_proj_quant.gs,
                )
            attn_output = attn_output + residual
            attn_output = attn_output.view(shape)
        return attn_output
