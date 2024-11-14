import torch
from .._transformer_configuration import IPEXTransformerConfig
from typing import Optional
from .XPUAttentionfp16 import (
    IPEXAttention,
)
from intel_extension_for_pytorch.nn.utils._quantize_convert import (
    WeightOnlyQuantizedLinear,
    dequant_gemm_block,
)
from .model_utils import xpu_gemm_use_xetla
from .CacheUtils import CacheFormat


class IPEXAttentionInt4(IPEXAttention):
    def __init__(
        self, config: IPEXTransformerConfig, layer_idx: Optional[int] = None
    ) -> None:
        super().__init__(config)
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.embedding_dim
        self.use_causal_mask = config.use_causal_mask
        self.num_heads = config.num_attention_head
        self.num_kv_heads = config.num_key_value_head
        self.head_dim = self.hidden_size // self.num_heads

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
        self.q_proj_quant.set_scales_zps_gidx(
            q_proj.scales, q_proj.qzeros, q_proj.g_idx
        )
        self.q_proj_quant.blocksize = q_proj.blocksize

        self.k_proj_quant.set_weights_bias(k_proj.qweight, k_proj.bias)
        self.k_proj_quant.set_scales_zps_gidx(
            k_proj.scales, k_proj.qzeros, k_proj.g_idx
        )
        self.k_proj_quant.blocksize = k_proj.blocksize

        self.v_proj_quant.set_weights_bias(v_proj.qweight, v_proj.bias)
        self.v_proj_quant.set_scales_zps_gidx(
            v_proj.scales, v_proj.qzeros, v_proj.g_idx
        )
        self.v_proj_quant.blocksize = v_proj.blocksize

        self.out_proj_quant.set_weights_bias(out_proj.qweight, out_proj.bias)
        self.out_proj_quant.set_scales_zps_gidx(
            out_proj.scales, out_proj.qzeros, out_proj.g_idx
        )
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
        # Xetla path requires column major layout.
        self.q_proj_quant.qweight.data = self.q_proj_quant.qweight.transpose(
            0, 1
        ).contiguous()
        self.q_proj_quant.use_optimum_format = False
        self.k_proj_quant.qweight.data = self.k_proj_quant.qweight.transpose(
            0, 1
        ).contiguous()
        self.k_proj_quant.use_optimum_format = False
        self.v_proj_quant.qweight.data = self.v_proj_quant.qweight.transpose(
            0, 1
        ).contiguous()
        self.v_proj_quant.use_optimum_format = False
        self.out_proj_quant.qweight.data = self.out_proj_quant.qweight.transpose(
            0, 1
        ).contiguous()
        self.out_proj_quant.use_optimum_format = False

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

        if self.q_proj_quant.qzeros is not None:
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
        qkv_proj_quant_qweight = torch.cat(
            [
                self.q_proj_quant.qweight,
                self.k_proj_quant.qweight,
                self.v_proj_quant.qweight,
            ]
        ).contiguous()
        qkv_proj_quant_scales = torch.cat(
            [
                self.q_proj_quant.scales,
                self.k_proj_quant.scales,
                self.v_proj_quant.scales,
            ]
        ).contiguous()
        qkv_proj_quant_qzeros = None
        if self.q_proj_quant.qzeros is not None:
            qkv_proj_quant_qzeros = torch.cat(
                [
                    self.q_proj_quant.qzeros,
                    self.k_proj_quant.qzeros,
                    self.v_proj_quant.qzeros,
                ]
            ).contiguous()

        qkv_proj_quant_bias = None
        if self.q_proj_quant.bias is not None:
            qkv_proj_quant_bias = torch.cat(
                [
                    self.q_proj_quant.bias,
                    self.k_proj_quant.bias,
                    self.v_proj_quant.bias,
                ]
            ).contiguous()
        self.qkv_proj_quant.set_weights_bias(
            qkv_proj_quant_qweight, qkv_proj_quant_bias
        )
        self.qkv_proj_quant.set_scales_zps_gidx(
            qkv_proj_quant_scales, qkv_proj_quant_qzeros
        )
        self.qkv_proj_quant.blocksize = self.q_proj_quant.blocksize

        # Note: synchronize to ensure the completion of contiguous
        torch.xpu.synchronize()

    def compute_qkv_gemm(self, hidden_states, query, key, value):
        if hidden_states.shape[1] > 1 and xpu_gemm_use_xetla():
            # dequantize+gemm kernel can improve IPEX int4 linear performance of first token
            if (
                self.q_proj_quant.qweight is None
                and self.qkv_proj_quant.qweight is not None
            ):
                qkv_out = dequant_gemm_block(hidden_states, self.qkv_proj_quant)
                mq = query.shape[-1]
                mk = key.shape[-1]
                if IPEXAttention.cache_type == "static":
                    if (
                        IPEXAttention.cache_format == CacheFormat.FBNH
                        and not self.beam_search_first_iter(hidden_states.shape[1])
                    ):
                        query = qkv_out[:, :, :mq].transpose(0, 1)
                        key.copy_(
                            qkv_out[:, :, mq : mq + mk].transpose(0, 1)
                        ).contiguous()
                        value.copy_(
                            qkv_out[:, :, mq + mk :].transpose(0, 1)
                        ).contiguous()
                    else:
                        query = qkv_out[:, :, :mq]
                        key.copy_(qkv_out[:, :, mq : mq + mk]).contiguous()
                        value.copy_(qkv_out[:, :, mq + mk :]).contiguous()
                else:
                    query = qkv_out[:, :, :mq]
                    key = qkv_out[:, :, mq : mq + mk]
                    value = qkv_out[:, :, mq + mk :]
            else:
                dequant_gemm_block(hidden_states, self.q_proj_quant, output=query)
                dequant_gemm_block(hidden_states, self.k_proj_quant, output=key)
                dequant_gemm_block(hidden_states, self.v_proj_quant, output=value)
            return query, key, value
        if self.q_proj_quant.g_idx is not None:
            # qkv fusion cannot apply actorder
            return self.compute_qkv_gemm_separate(hidden_states, query, key, value)
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

    def compute_qkv_gemm_separate(self, hidden_states, query, key, value):
        if self.q_proj.bias is None:
            torch.ops.torch_ipex.mm_int4_out(
                hidden_states,
                self.q_proj_quant.qweight,
                self.q_proj_quant.scales,
                self.q_proj_quant.qzeros,
                query,
                self.q_proj_quant.blocksize,
                self.q_proj_quant.g_idx,
            )
        else:
            query = torch.ops.torch_ipex.mm_int4(
                hidden_states,
                self.q_proj_quant.qweight,
                self.q_proj_quant.bias,
                self.q_proj_quant.scales,
                self.q_proj_quant.qzeros,
                self.q_proj_quant.blocksize,
                self.q_proj_quant.g_idx,
            )

        if self.k_proj.bias is None:
            torch.ops.torch_ipex.mm_int4_out(
                hidden_states,
                self.k_proj_quant.qweight,
                self.k_proj_quant.scales,
                self.k_proj_quant.qzeros,
                key,
                self.k_proj_quant.blocksize,
                self.k_proj_quant.g_idx,
            )
        else:
            key.copy_(
                torch.ops.torch_ipex.mm_int4(
                    hidden_states,
                    self.k_proj_quant.qweight,
                    self.k_proj_quant.bias,
                    self.k_proj_quant.scales,
                    self.k_proj_quant.qzeros,
                    self.k_proj_quant.blocksize,
                    self.k_proj_quant.g_idx,
                )
            )

        if self.v_proj.bias is None:
            torch.ops.torch_ipex.mm_int4_out(
                hidden_states,
                self.v_proj_quant.qweight,
                self.v_proj_quant.scales,
                self.v_proj_quant.qzeros,
                value,
                self.v_proj_quant.blocksize,
                self.v_proj_quant.g_idx,
            )
        else:
            value.copy_(
                torch.ops.torch_ipex.mm_int4(
                    hidden_states,
                    self.v_proj_quant.qweight,
                    self.v_proj_quant.bias,
                    self.v_proj_quant.scales,
                    self.v_proj_quant.qzeros,
                    self.v_proj_quant.blocksize,
                    self.v_proj_quant.g_idx,
                )
            )
        return query, key, value

    def out_proj_compute(self, attn_output, residual=None):
        if attn_output.shape[1] > 1 and xpu_gemm_use_xetla():
            # dequantize+gemm kernel can improve IPEX int4 linear performance of first token
            attn_output = dequant_gemm_block(attn_output, self.out_proj_quant)
            if residual is not None:
                attn_output += residual
            return attn_output
        if residual is None:
            if self.out_proj.bias is not None:
                attn_output = torch.ops.torch_ipex.mm_bias_int4(
                    attn_output,
                    self.out_proj_quant.qweight,
                    self.out_proj_quant.bias,
                    self.out_proj_quant.scales,
                    self.out_proj_quant.qzeros,
                    self.out_proj_quant.blocksize,
                    self.out_proj_quant.g_idx,
                )
            else:
                attn_output = torch.ops.torch_ipex.mm_int4(
                    attn_output,
                    self.out_proj_quant.qweight,
                    self.out_proj_quant.scales,
                    self.out_proj_quant.qzeros,
                    self.out_proj_quant.blocksize,
                    self.out_proj_quant.g_idx,
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
                    self.out_proj_quant.g_idx,
                )
            else:
                attn_output = torch.ops.torch_ipex.mm_add_int4(
                    attn_output,
                    self.out_proj_quant.qweight,
                    self.out_proj_quant.scales,
                    self.out_proj_quant.qzeros,
                    self.out_proj_quant.blocksize,
                    residual,
                    self.out_proj_quant.g_idx,
                )
            attn_output = attn_output.view(shape)
        return attn_output


class IPEXAttentionInt4OneDNN(IPEXAttentionInt4):
    def __init__(
        self, config: IPEXTransformerConfig, layer_idx: Optional[int] = None
    ) -> None:
        super().__init__(config, layer_idx)

    def transpose_parameter(self):
        # OneDNN path requires column major layout under row major shape.
        self.q_proj_quant.qweight.data = (
            self.q_proj_quant.qweight.transpose(0, 1).contiguous().transpose(0, 1)
        )
        self.q_proj_quant.use_optimum_format = False
        self.k_proj_quant.qweight.data = (
            self.k_proj_quant.qweight.transpose(0, 1).contiguous().transpose(0, 1)
        )
        self.k_proj_quant.use_optimum_format = False
        self.v_proj_quant.qweight.data = (
            self.v_proj_quant.qweight.transpose(0, 1).contiguous().transpose(0, 1)
        )
        self.v_proj_quant.use_optimum_format = False
        self.out_proj_quant.qweight.data = (
            self.out_proj_quant.qweight.transpose(0, 1).contiguous().transpose(0, 1)
        )
        self.out_proj_quant.use_optimum_format = False

        self.q_proj_quant.scales.data = self.q_proj_quant.scales
        self.k_proj_quant.scales.data = self.k_proj_quant.scales
        self.v_proj_quant.scales.data = self.v_proj_quant.scales
        self.out_proj_quant.scales.data = self.out_proj_quant.scales
        # Assign zero point and OneDNN will broadcast it automatically.
        self.q_proj_quant.qzeros = torch.Tensor([8]).to(torch.int8).to("xpu")

        self.k_proj_quant.qzeros = torch.Tensor([8]).to(torch.int8).to("xpu")
        self.v_proj_quant.qzeros = torch.Tensor([8]).to(torch.int8).to("xpu")
        self.out_proj_quant.qzeros = torch.Tensor([8]).to(torch.int8).to("xpu")

        torch.xpu.synchronize()

    def cat_qkv(self):
        pass

    def compute_qkv_gemm(self, hidden_states, query, key, value):
        if (
            self.q_proj_quant.qweight is None
            and self.qkv_proj_quant.qweight is not None
        ):
            qkv_out = torch.ops.torch_ipex.mm_bias_int4(
                hidden_states,
                self.qkv_proj_quant.qweight,
                self.qkv_proj_quant.bias,
                self.qkv_proj_quant.scales,
                self.qkv_proj_quant.qzeros,
                self.qkv_proj_quant.blocksize,
                self.qkv_proj_quant.g_idx,
            )
            mq = query.shape[-1]
            mk = key.shape[-1]
            # Statice Cache needs to store the key and value in the applied space.
            # Dynamic Cache will cat the new key and value, so that does not need inplace operation.
            if IPEXAttention.cache_type == "static":
                if (
                    IPEXAttention.cache_format == CacheFormat.FBNH
                    and not self.beam_search_first_iter(hidden_states.shape[1])
                ):
                    query = qkv_out[:, :, :mq].transpose(0, 1).contiguous()
                    key.copy_(qkv_out[:, :, mq : mq + mk].transpose(0, 1))
                    value.copy_(qkv_out[:, :, mq + mk :].transpose(0, 1))
                else:
                    query = qkv_out[:, :, :mq].contiguous()
                    key.copy_(qkv_out[:, :, mq : mq + mk])
                    value.copy_(qkv_out[:, :, mq + mk :])
            elif IPEXAttention.cache_type == "dynamic":
                query = qkv_out[:, :, :mq].contiguous()
                key = qkv_out[:, :, mq : mq + mk].contiguous()
                value = qkv_out[:, :, mq + mk :].contiguous()
            return query, key, value
        else:
            query = torch.ops.torch_ipex.mm_bias_int4(
                hidden_states,
                self.q_proj_quant.qweight,
                self.q_proj_quant.bias,
                self.q_proj_quant.scales,
                self.q_proj_quant.qzeros,
                self.q_proj_quant.blocksize,
                self.q_proj_quant.g_idx,
            )
            key_out = torch.ops.torch_ipex.mm_bias_int4(
                hidden_states,
                self.k_proj_quant.qweight,
                self.k_proj_quant.bias,
                self.k_proj_quant.scales,
                self.k_proj_quant.qzeros,
                self.k_proj_quant.blocksize,
                self.k_proj_quant.g_idx,
            )
            value_out = torch.ops.torch_ipex.mm_bias_int4(
                hidden_states,
                self.v_proj_quant.qweight,
                self.v_proj_quant.bias,
                self.v_proj_quant.scales,
                self.v_proj_quant.qzeros,
                self.v_proj_quant.blocksize,
                self.v_proj_quant.g_idx,
            )
            if IPEXAttention.cache_type == "static":
                if self.beam_idx is None or self.beam_search_next_token(
                    hidden_states.size(1)
                ):
                    key.copy_(key_out.transpose(0, 1))
                    value.copy_(value_out.transpose(0, 1))
                    return query.transpose(0, 1).contiguous(), key, value
                key.copy_(key_out)
                value.copy_(value_out)
                return query, key, value
            elif IPEXAttention.cache_type == "dynamic":
                return query, key_out, value_out
