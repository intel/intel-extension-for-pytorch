import torch
import intel_extension_for_pytorch as ipex
from .._transformer_configuration import IPEXTransformerConfig
from .Attention import IPEXTransformerAttnOptimizedFp16
from intel_extension_for_pytorch.nn.utils._quantize_convert import (
    WeightOnlyQuantizedLinear,
)


class IPEXTransformerAttnOptimizedInt4(IPEXTransformerAttnOptimizedFp16):
    def __init__(self, config: IPEXTransformerConfig) -> None:
        super().__init__(config)

        self.q_proj_quant = WeightOnlyQuantizedLinear(
            in_features=4096, out_features=4096
        )
        self.k_proj_quant = WeightOnlyQuantizedLinear(
            in_features=4096, out_features=4096
        )
        self.v_proj_quant = WeightOnlyQuantizedLinear(
            in_features=4096, out_features=4096
        )
        self.out_proj_quant = WeightOnlyQuantizedLinear(
            in_features=4096, out_features=4096
        )

        self.qkv_proj_quant = WeightOnlyQuantizedLinear(
            in_features=4096, out_features=4096
        )

    def load_parameter(self, q_proj, k_proj, v_proj, out_proj):
        self.q_proj_quant.set_weights_bias(q_proj.qweight, q_proj.bias)
        self.q_proj_quant.set_scales_zps_gidx(q_proj.scales, q_proj.qzeros)
        self.q_proj_quant.blocksize = q_proj.blocksize

        self.k_proj_quant.set_weights_bias(k_proj.qweight, k_proj.bias)
        self.k_proj_quant.set_scales_zps_gidx(k_proj.scales, k_proj.qzeros)
        self.k_proj_quant.blocksize = k_proj.blocksize

        self.v_proj_quant.set_weights_bias(v_proj.qweight, v_proj.bias)
        self.v_proj_quant.set_scales_zps_gidx(v_proj.scales, v_proj.qzeros)
        self.v_proj_quant.blocksize = v_proj.blocksize

        self.out_proj_quant.set_weights_bias(out_proj.qweight, out_proj.bias)
        self.out_proj_quant.set_scales_zps_gidx(out_proj.scales, out_proj.qzeros)
        self.out_proj_quant.blocksize = out_proj.blocksize

        q_proj.qweight = None
        q_proj.bias = None
        q_proj.scales = None
        q_proj.qzeros = None

        k_proj.qweight = None
        k_proj.bias = None
        k_proj.scales = None
        k_proj.qzeros = None

        v_proj.qweight = None
        v_proj.bias = None
        v_proj.scales = None
        v_proj.qzeros = None

        out_proj.qweight = None
        out_proj.bias = None
        out_proj.scales = None
        out_proj.qzeros = None

        self.position_embed = self.config.rotary_embedding_class(
            self.config, torch.float16
        )

    def transpose_parameter(self):
        self.q_proj_quant.qweight.data = self.q_proj_quant.qweight.transpose(
            0, 1
        ).contiguous()
        self.k_proj_quant.qweight.data = self.k_proj_quant.qweight.transpose(
            0, 1
        ).contiguous()
        self.v_proj_quant.qweight.data = self.v_proj_quant.qweight.transpose(
            0, 1
        ).contiguous()
        self.out_proj_quant.qweight.data = self.out_proj_quant.qweight.transpose(
            0, 1
        ).contiguous()

        self.q_proj_quant.scales.data = self.q_proj_quant.scales.transpose(
            0, 1
        ).contiguous()
        self.k_proj_quant.scales.data = self.k_proj_quant.scales.transpose(
            0, 1
        ).contiguous()
        self.v_proj_quant.scales.data = self.v_proj_quant.scales.transpose(
            0, 1
        ).contiguous()
        self.out_proj_quant.scales.data = self.out_proj_quant.scales.transpose(
            0, 1
        ).contiguous()

        if self.q_proj_quant.qzeros:
            self.q_proj_quant.qzeros.data = self.q_proj_quant.qzeros.transpose(
                0, 1
            ).contiguous()
            self.k_proj_quant.qzeros.data = self.k_proj_quant.qzeros.transpose(
                0, 1
            ).contiguous()
            self.v_proj_quant.qzeros.data = self.v_proj_quant.qzeros.transpose(
                0, 1
            ).contiguous()
            self.out_proj_quant.qzeros.data = self.out_proj_quant.qzeros.transpose(
                0, 1
            ).contiguous()

        torch.xpu.synchronize()

    def cat_qkv(self):
        shape = [3, -1, self.q_proj_quant.qweight.shape[-1]]
        qkv_proj_quant_qweight = (
            torch.stack(
                [
                    self.q_proj_quant.qweight,
                    self.k_proj_quant.qweight,
                    self.v_proj_quant.qweight,
                ]
            )
            .contiguous()
            .view([3, *self.q_proj_quant.qweight.shape])
        )
        qkv_proj_quant_scales = (
            torch.stack(
                [
                    self.q_proj_quant.scales,
                    self.k_proj_quant.scales,
                    self.v_proj_quant.scales,
                ]
            )
            .contiguous()
            .view([3, *self.q_proj_quant.scales.shape])
        )
        qkv_proj_quant_qzeros = None
        if self.q_proj_quant.qzeros:
            qkv_proj_quant_qzeros = (
                torch.stack(
                    [
                        self.q_proj_quant.qzeros,
                        self.k_proj_quant.qzeros,
                        self.v_proj_quant.qzeros,
                    ]
                )
                .contiguous()
                .view([3, *self.q_proj_quant.qzeros.shape])
            )

        qkv_proj_quant_bias = None
        if self.q_proj_quant.bias is not None:
            bias_shape = [3, -1]
            qkv_proj_quant_bias = (
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
        self.qkv_proj_quant.set_weights_bias(
            qkv_proj_quant_qweight, qkv_proj_quant_bias
        )
        self.qkv_proj_quant.set_scales_zps_gidx(
            qkv_proj_quant_scales, qkv_proj_quant_qzeros
        )
        self.qkv_proj_quant.blocksize = self.q_proj_quant.blocksize

        # Note: synchronize to ensure the completion of contiguous
        torch.xpu.synchronize()

        if self.q_proj_quant.qzeros:
            self.q_proj_quant.qzeros.data = self.qkv_proj_quant.qzeros[0, :, :]
            self.k_proj_quant.qzeros.data = self.qkv_proj_quant.qzeros[1, :, :]
            self.v_proj_quant.qzeros.data = self.qkv_proj_quant.qzeros[2, :, :]

        if self.qkv_proj_quant.bias is not None:
            self.q_proj_quant.bias.data = self.qkv_proj_quant.bias[0, :]
            self.k_proj_quant.bias.data = self.qkv_proj_quant.bias[1, :]
            self.v_proj_quant.bias.data = self.qkv_proj_quant.bias[2, :]

    def compute_qkv_gemm(self, hidden_states, query, key, value):
        torch.ops.torch_ipex.mm_qkv_out_int4(
            hidden_states,
            self.qkv_proj_quant.qweight,
            self.qkv_proj_quant.scales,
            self.qkv_proj_quant.qzeros,
            self.qkv_proj_quant.bias,
            query,
            key,
            value,
            self.qkv_proj_quant.blocksize,
        )
        return query, key, value

    def out_proj_compute(self, attn_output, residual=None):
        arch = 1 if ipex._C._has_2d_block_array(0) else 0
        if residual is None:
            if self.out_proj.bias is not None:
                attn_output = torch.ops.torch_ipex.mm_bias_int4(
                    attn_output,
                    self.out_proj_quant.qweight,
                    self.out_proj_quant.bias,
                    self.out_proj_quant.scales,
                    self.out_proj_quant.qzeros,
                    self.out_proj_quant.blocksize,
                )
            else:
                attn_output = torch.ops.torch_ipex.mm_int4(
                    attn_output,
                    self.out_proj_quant.qweight,
                    self.out_proj_quant.scales,
                    self.out_proj_quant.qzeros,
                    self.out_proj_quant.blocksize,
                )
        else:
            shape = [attn_output.shape[0], attn_output.shape[1], self.embed_dim]
            if self.out_proj.bias is not None:
                attn_output = torch.ops.torch_ipex.mm_bias_add_int4(
                    attn_output,
                    self.out_proj_quant.qweight,
                    self.out_proj_quant.bias,
                    self.out_proj_quant.scales,
                    self.out_proj_quant.qzeros,
                    self.out_proj_quant.blocksize,
                    residual,
                )
            else:
                attn_output = torch.ops.torch_ipex.mm_add_int4(
                    attn_output,
                    self.out_proj_quant.qweight,
                    self.out_proj_quant.scales,
                    self.out_proj_quant.qzeros,
                    self.out_proj_quant.blocksize,
                    residual,
                )
            attn_output = attn_output.view(shape)
        return attn_output
