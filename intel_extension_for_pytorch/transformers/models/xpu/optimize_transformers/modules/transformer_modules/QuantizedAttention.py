import torch
import intel_extension_for_pytorch as ipex
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
        self.q_proj_quant.weight = q_proj.qweight.byte()
        self.k_proj_quant.weight = k_proj.qweight.byte()
        self.v_proj_quant.weight = v_proj.qweight.byte()
        self.out_proj_quant.weight = out_proj.qweight.byte()
        # print("self.q_proj_quant.weight dtype: ", self.q_proj_quant.weight.dtype)

        self.q_proj_quant.scale = q_proj.scales
        self.k_proj_quant.scale = k_proj.scales
        self.v_proj_quant.scale = v_proj.scales
        self.out_proj_quant.scale = out_proj.scales

        has_qzeros = hasattr(q_proj, "qzeros")
        # q, k, v, out should have the same attributes
        if has_qzeros:
            self.q_proj_quant.zp = q_proj.qzeros.byte()
            self.k_proj_quant.zp = k_proj.qzeros.byte()
            self.v_proj_quant.zp = v_proj.qzeros.byte()
            self.out_proj_quant.zp = out_proj.qzeros.byte()
            # print("self.q_proj_quant.zp dtype: ", self.q_proj_quant.zp.dtype)

        self.q_proj_quant.bias = q_proj.bias
        self.k_proj_quant.bias = k_proj.bias
        self.v_proj_quant.bias = v_proj.bias
        self.out_proj_quant.bias = out_proj.bias

        self.q_proj_quant.gs = q_proj.blocksize
        self.k_proj_quant.gs = k_proj.blocksize
        self.v_proj_quant.gs = v_proj.blocksize
        self.out_proj_quant.gs = out_proj.blocksize

        self.position_embed = self.config.rotary_embedding_class(
            self.config, torch.float16
        )

    def transpose_parameter(self):
        self.q_proj_quant.weight.data = self.q_proj_quant.weight.transpose(
            0, 1
        ).contiguous()
        self.k_proj_quant.weight.data = self.k_proj_quant.weight.transpose(
            0, 1
        ).contiguous()
        self.v_proj_quant.weight.data = self.v_proj_quant.weight.transpose(
            0, 1
        ).contiguous()
        self.out_proj_quant.weight.data = self.out_proj_quant.weight.transpose(
            0, 1
        ).contiguous()

        self.q_proj_quant.scale.data = self.q_proj_quant.scale.transpose(
            0, 1
        ).contiguous()
        self.k_proj_quant.scale.data = self.k_proj_quant.scale.transpose(
            0, 1
        ).contiguous()
        self.v_proj_quant.scale.data = self.v_proj_quant.scale.transpose(
            0, 1
        ).contiguous()
        self.out_proj_quant.scale.data = self.out_proj_quant.scale.transpose(
            0, 1
        ).contiguous()

        has_qzeros = hasattr(self.q_proj_quant, "zp")
        if has_qzeros:
            self.q_proj_quant.zp.data = self.q_proj_quant.zp.transpose(
                0, 1
            ).contiguous()
            self.k_proj_quant.zp.data = self.k_proj_quant.zp.transpose(
                0, 1
            ).contiguous()
            self.v_proj_quant.zp.data = self.v_proj_quant.zp.transpose(
                0, 1
            ).contiguous()
            self.out_proj_quant.zp.data = self.out_proj_quant.zp.transpose(
                0, 1
            ).contiguous()

        torch.xpu.synchronize()

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
        has_qzeros = hasattr(self.q_proj_quant, "zp")
        if has_qzeros:
            self.qkv_proj_quant.zp = (
                torch.stack(
                    [self.q_proj_quant.zp, self.k_proj_quant.zp, self.v_proj_quant.zp]
                )
                .contiguous()
                .view(shape)
            )
        self.qkv_proj_quant.gs = self.q_proj_quant.gs
        if self.q_proj_quant.bias is not None:
            bias_shape = [3, -1]
            self.qkv_proj_quant.bias = (
                torch.stack(
                    [
                        self.q_proj_quant.bias,
                        self.k_proj_quant.bias,
                        self.v_proj_quant.bias,
                    ]
                )
                .contiguous()
                .view(bias_shape)
            )

    def compute_qkv_gemm(self, hidden_states, query, key, value):
        # print("self.qkv_proj_quant.weight.dtype: ", self.qkv_proj_quant.weight.dtype)
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
        arch = 1 if ipex._C._has_2d_block_array(0) else 0
        if residual is None:
            if self.out_proj.bias is not None:
                attn_output = torch.ops.torch_ipex.mm_bias_int4(
                    attn_output,
                    self.out_proj_quant.weight,
                    self.out_proj.bias,
                    self.out_proj_quant.scale,
                    self.out_proj_quant.zp,
                    self.out_proj_quant.gs,
                    arch,
                )
            else:
                attn_output = torch.ops.torch_ipex.mm_int4(
                    attn_output,
                    self.out_proj_quant.weight,
                    self.out_proj_quant.scale,
                    self.out_proj_quant.zp,
                    self.out_proj_quant.gs,
                    arch,
                )
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
                    arch,
                )
            else:
                attn_output = torch.ops.torch_ipex.mm_int4(
                    attn_output,
                    self.out_proj_quant.weight,
                    self.out_proj_quant.scale,
                    self.out_proj_quant.zp,
                    self.out_proj_quant.gs,
                    arch,
                )
            attn_output = attn_output + residual
            attn_output = attn_output.view(shape)
        return attn_output
