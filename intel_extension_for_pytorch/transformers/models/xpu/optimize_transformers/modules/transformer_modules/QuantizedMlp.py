import torch
import intel_extension_for_pytorch as ipex
from .Mlp import IPEXTransformerMLP
from intel_extension_for_pytorch.nn.utils._quantize_convert import (
    WeightOnlyQuantizedLinear,
)
from .model_utils import xpu_gemm_use_xetla


class IPEXTransformerMLPOptimizedInt4(IPEXTransformerMLP):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.fc_in_quant = WeightOnlyQuantizedLinear(
            in_features=4096, out_features=4096
        )
        self.fc_out_quant = WeightOnlyQuantizedLinear(
            in_features=4096, out_features=4096
        )

    def load_parameter(self, fc_in, fc_out):
        self.fc_in_quant.set_weights_bias(fc_in.qweight, fc_in.bias)
        self.fc_in_quant.set_scales_zps_gidx(fc_in.scales, fc_in.qzeros)
        self.fc_in_quant.blocksize = fc_in.blocksize

        self.fc_out_quant.set_weights_bias(fc_out.qweight, fc_out.bias)
        self.fc_out_quant.set_scales_zps_gidx(fc_out.scales, fc_out.qzeros)
        self.fc_out_quant.blocksize = fc_out.blocksize

        fc_in.qweight = None
        fc_in.bias = None
        fc_in.scales = None
        fc_in.qzeros = None

        fc_out.qweight = None
        fc_out.bias = None
        fc_out.scales = None
        fc_out.qzeros = None

    def transpose_parameter(self):
        if xpu_gemm_use_xetla():
            self.fc_in_quant.qweight.data = self.fc_in_quant.qweight.transpose(
                0, 1
            ).contiguous()
            self.fc_out_quant.qweight.data = self.fc_out_quant.qweight.transpose(
                0, 1
            ).contiguous()

            self.fc_in_quant.scales.data = self.fc_in_quant.scales.transpose(
                0, 1
            ).contiguous()
            self.fc_out_quant.scales.data = self.fc_out_quant.scales.transpose(
                0, 1
            ).contiguous()

            if self.fc_in_quant.qzeros is not None:
                self.fc_in_quant.qzeros.data = self.fc_in_quant.qzeros.transpose(
                    0, 1
                ).contiguous()
                self.fc_out_quant.qzeros.data = self.fc_out_quant.qzeros.transpose(
                    0, 1
                ).contiguous()
        else:
            self.fc_in_quant.qweight.data = (
                self.fc_in_quant.qweight.transpose(0, 1).contiguous().transpose(0, 1)
            )
            self.fc_out_quant.qweight.data = (
                self.fc_out_quant.qweight.transpose(0, 1).contiguous().transpose(0, 1)
            )

            self.fc_in_quant.scales.data = self.fc_in_quant.scales
            self.fc_out_quant.scales.data = self.fc_out_quant.scales

            self.fc_in_quant.qzeros = torch.ones(
                [
                    self.fc_in_quant.qweight.size()[-2] // self.fc_in_quant.blocksize,
                    self.fc_in_quant.qweight.size()[-1] // 8,
                ],
                dtype=torch.int32,
                device="xpu",
            )
            self.fc_in_quant.qzeros = torch.fill(
                self.fc_in_quant.qzeros, int(-2004318072)
            )

            self.fc_out_quant.qzeros = torch.ones(
                [
                    self.fc_out_quant.qweight.size()[-2] // self.fc_out_quant.blocksize,
                    self.fc_out_quant.qweight.size()[-1] // 8,
                ],
                dtype=torch.int32,
                device="xpu",
            )
            self.fc_out_quant.qzeros = torch.fill(
                self.fc_out_quant.qzeros, int(-2004318072)
            )

        torch.xpu.synchronize()

    def inter_mm(self, hidden_states):
        if self.fc_in_quant.bias is None:
            hidden_states = torch.ops.torch_ipex.mm_int4(
                hidden_states,
                self.fc_in_quant.qweight,
                self.fc_in_quant.scales,
                self.fc_in_quant.qzeros,
                self.fc_in_quant.blocksize,
            )
        else:
            hidden_states = torch.ops.torch_ipex.mm_bias_int4(
                hidden_states,
                self.fc_in_quant.qweight,
                self.fc_in_quant.bias,
                self.fc_in_quant.scales,
                self.fc_in_quant.qzeros,
                self.fc_in_quant.blocksize,
            )

        return self.act(hidden_states)

    def out_mm(self, hidden_states, residual=None):
        if self.fc_out_quant.bias is None:
            hidden_states = torch.ops.torch_ipex.mm_int4(
                hidden_states,
                self.fc_out_quant.qweight,
                self.fc_out_quant.scales,
                self.fc_out_quant.qzeros,
                self.fc_out_quant.blocksize,
            )
        else:
            hidden_states = torch.ops.torch_ipex.mm_bias_int4(
                hidden_states,
                self.fc_out_quant.qweight,
                self.fc_out_quant.bias,
                self.fc_out_quant.scales,
                self.fc_out_quant.qzeros,
                self.fc_out_quant.blocksize,
            )

        if residual is not None:
            hidden_states += residual

        hidden_states = self.all_reduce_if_necessary(hidden_states)
        return hidden_states


class IPEXTransformerMLPOptimizedInt4Gelu(IPEXTransformerMLPOptimizedInt4):
    def __init__(self, config) -> None:
        super().__init__(config)

    def inter_mm(self, hidden_states):
        if self.fc_in_quant.bias is None:
            hidden_states = torch.ops.torch_ipex.mm_int4(
                hidden_states,
                self.fc_in_quant.qweight,
                self.fc_in_quant.scales,
                self.fc_in_quant.qzeros,
                self.fc_in_quant.blocksize,
            )
            hidden_states = self.act(hidden_states)
        else:
            hidden_states = torch.ops.torch_ipex.mm_bias_gelu_int4(
                hidden_states,
                self.fc_in_quant.qweight,
                self.fc_in_quant.scales,
                self.fc_in_quant.qzeros,
                self.fc_in_quant.bias,
                self.fc_in_quant.blocksize,
                "tanh",
            )
        return hidden_states


class IPEXTransformerMLPOptimizedInt4GeluGptj(IPEXTransformerMLPOptimizedInt4Gelu):
    def __init__(self, config) -> None:
        super().__init__(config)

    def forward(self, hidden_states, attn_output, residual):
        inter_out = self.inter_mm(hidden_states)
        out = self.out_mm(inter_out, attn_output, residual)
        return out

    def out_mm(self, inter_out, attn_out, residual):
        hidden_states = torch.ops.torch_ipex.mm_bias_resadd_resadd_int4(
            inter_out,
            self.fc_out_quant.qweight,
            self.fc_out_quant.bias,
            attn_out,
            residual,
            self.fc_out_quant.scales,
            self.fc_out_quant.qzeros,
            self.fc_out_quant.blocksize,
        )
        hidden_states = self.all_reduce_if_necessary(hidden_states)
        return hidden_states


class IPEXTransformerMLPOptimizedInt4Silu(IPEXTransformerMLPOptimizedInt4):
    def __init__(self, config):
        super().__init__(config)


class IPEXTransformerMLPOptimizedInt4SiluQwenInnerMMMixin:
    def transpose_inner(self):
        pass

    def inter_mm(self, hidden_states):
        assert self.fc_in_quant.blocksize == self.fc_out_quant.blocksize
        if self.fc_in_quant.bias is None:
            hidden_states1 = torch.ops.torch_ipex.mm_int4(
                hidden_states,
                self.fc_in_quant.qweight,
                self.fc_in_quant.scales,
                self.fc_in_quant.qzeros,
                self.fc_in_quant.blocksize,
            )
        else:
            hidden_states1 = torch.ops.torch_ipex.mm_bias_int4(
                hidden_states,
                self.fc_in_quant.qweight,
                self.fc_in_quant.bias,
                self.fc_in_quant.scales,
                self.fc_in_quant.qzeros,
                self.fc_in_quant.blocksize,
            )

        if self.fc_out_quant.bias is None:
            return torch.ops.torch_ipex.mm_silu_mul_int4(
                hidden_states,
                self.fc_out_quant.qweight,
                self.fc_out_quant.scales,
                self.fc_out_quant.qzeros,
                self.fc_out_quant.blocksize,
                hidden_states1,
            )
        else:
            return torch.ops.torch_ipex.mm_bias_silu_mul_int4(
                hidden_states,
                self.fc_out_quant.qweight,
                self.fc_out_quant.bias,
                self.fc_out_quant.scales,
                self.fc_out_quant.qzeros,
                self.fc_out_quant.blocksize,
                hidden_states1,
            )


class IPEXTransformerMLPOptimizedInt4SiluQwenInnerMLPMixin:
    def transpose_inner(self):
        self.mlp_silu_qweight = torch.stack(
            (self.fc_out_quant.qweight, self.fc_in_quant.qweight)
        ).contiguous()
        del self.fc_out_quant.qweight
        del self.fc_in_quant.qweight
        self.mlp_silu_scales = torch.stack(
            (self.fc_out_quant.scales, self.fc_in_quant.scales)
        ).contiguous()
        del self.fc_out_quant.scales
        del self.fc_in_quant.scales
        self.mlp_silu_qzeros = None
        if self.fc_out_quant.qzeros is not None:
            self.mlp_silu_qzeros = torch.stack(
                (self.fc_out_quant.qzeros, self.fc_in_quant.qzeros)
            ).contiguous()
            del self.fc_out_quant.qzeros
            del self.fc_in_quant.qzeros

    def inter_mm(self, hidden_states):
        assert self.fc_in_quant.blocksize == self.fc_out_quant.blocksize
        has_gate_bias = self.fc_out_quant.bias is not None
        has_up_bias = self.fc_in_quant.bias is not None
        common_args = (
            hidden_states,
            self.mlp_silu_qweight,
            self.mlp_silu_scales,
            self.mlp_silu_qzeros,
        )
        if not has_gate_bias and not has_up_bias:
            return torch.ops.torch_ipex.mlp_silu_mul_int4(
                *common_args,
                self.fc_out_quant.blocksize,
            )
        elif has_gate_bias and not has_up_bias:
            return torch.ops.torch_ipex.mlp_bias_silu_mul_int4(
                *common_args,
                self.fc_out_quant.bias,
                self.fc_out_quant.blocksize,
            )
        elif not has_gate_bias and has_up_bias:
            return torch.ops.torch_ipex.mlp_silu_mul_bias_int4(
                *common_args,
                self.fc_in_quant.bias,
                self.fc_out_quant.blocksize,
            )
        elif has_gate_bias and has_up_bias:
            return torch.ops.torch_ipex.mlp_bias_silu_mul_bias_int4(
                *common_args,
                self.fc_out_quant.bias,
                self.fc_in_quant.bias,
                self.fc_out_quant.blocksize,
            )


class IPEXTransformerMLPOptimizedInt4SiluQwen(
    IPEXTransformerMLPOptimizedInt4SiluQwenInnerMLPMixin,
    IPEXTransformerMLPOptimizedInt4Silu,
):
    def __init__(self, config):
        super().__init__(config)
        self.c_proj_quant = WeightOnlyQuantizedLinear(
            in_features=4096, out_features=4096
        )
        self.arch = 1 if ipex._C._has_2d_block_array(0) else 0

    def load_parameter(self, fc_in, fc_out, c_proj):
        super().load_parameter(fc_in, fc_out)
        self.c_proj_quant.set_weights_bias(c_proj.qweight, c_proj.bias)
        self.c_proj_quant.set_scales_zps_gidx(c_proj.scales, c_proj.qzeros)
        self.c_proj_quant.blocksize = c_proj.blocksize

        c_proj.qweight = None
        c_proj.bias = None
        c_proj.scales = None
        c_proj.qzeros = None

    def transpose_parameter(self):
        super().transpose_parameter()
        if xpu_gemm_use_xetla():
            super().transpose_inner()
            self.c_proj_quant.qweight.data = self.c_proj_quant.qweight.transpose(
                0, 1
            ).contiguous()
            self.c_proj_quant.scales.data = self.c_proj_quant.scales.transpose(
                0, 1
            ).contiguous()
            if self.c_proj_quant.qzeros is not None:
                self.c_proj_quant.qzeros.data = self.c_proj_quant.qzeros.transpose(
                    0, 1
                ).contiguous()
        else:
            self.c_proj_quant.qweight.data = (
                self.c_proj_quant.qweight.transpose(0, 1).contiguous().transpose(0, 1)
            )
            self.c_proj_quant.scales.data = self.c_proj_quant.scales
            self.c_proj_quant.qzeros = torch.ones(
                [
                    self.c_proj_quant.qweight.size()[-2] // self.c_proj_quant.blocksize,
                    self.c_proj_quant.qweight.size()[-1] // 8,
                ],
                dtype=torch.int32,
                device="xpu",
            )
            self.c_proj_quant.qzeros = torch.fill(
                self.c_proj_quant.qzeros, int(-2004318072)
            )
        torch.xpu.synchronize()

    def out_mm(self, hidden_states, residual=None):
        if self.c_proj_quant.bias is None:
            hidden_states = torch.ops.torch_ipex.mm_add_int4(
                hidden_states,
                self.c_proj_quant.qweight,
                self.c_proj_quant.scales,
                self.c_proj_quant.qzeros,
                self.c_proj_quant.blocksize,
                residual,
            )
        else:
            hidden_states = torch.ops.torch_ipex.mm_bias_add_int4(
                hidden_states,
                self.c_proj_quant.qweight,
                self.c_proj_quant.bias,
                self.c_proj_quant.scales,
                self.c_proj_quant.qzeros,
                self.c_proj_quant.blocksize,
                residual,
            )
        return hidden_states

    def inter_mm(self, hidden_states):
        if xpu_gemm_use_xetla():
            return super().inter_mm(hidden_states)
        else:
            if self.fc_in_quant.bias is None:
                hidden_states1 = torch.ops.torch_ipex.mm_int4(
                    hidden_states,
                    self.fc_in_quant.qweight,
                    self.fc_in_quant.scales,
                    self.fc_in_quant.qzeros,
                    self.fc_in_quant.blocksize,
                )
            else:
                hidden_states1 = torch.ops.torch_ipex.mm_bias_int4(
                    hidden_states,
                    self.fc_in_quant.qweight,
                    self.fc_in_quant.bias,
                    self.fc_in_quant.scales,
                    self.fc_in_quant.qzeros,
                    self.fc_in_quant.blocksize,
                )
            if self.fc_out_quant.bias is None:
                return torch.ops.torch_ipex.mm_silu_mul_int4(
                    hidden_states,
                    self.fc_out_quant.qweight,
                    self.fc_out_quant.scales,
                    self.fc_out_quant.qzeros,
                    self.fc_out_quant.blocksize,
                    hidden_states1,
                )
            else:
                return torch.ops.torch_ipex.mm_bias_silu_mul_int4(
                    hidden_states,
                    self.fc_out_quant.qweight,
                    self.fc_out_quant.bias,
                    self.fc_out_quant.scales,
                    self.fc_out_quant.qzeros,
                    self.fc_out_quant.blocksize,
                    hidden_states1,
                )


class IPEXTransformerMLPOptimizedInt4SiluLlama(IPEXTransformerMLPOptimizedInt4SiluQwen):
    def __init__(self, config):
        super().__init__(config)

    def load_parameter(self, fc_in, fc_out, c_proj):
        # gate_proj, down_proj, up_proj
        return super().load_parameter(c_proj, fc_in, fc_out)


class IPEXTransformerMLPOptimizedInt4SiluPhi3(IPEXTransformerMLPOptimizedInt4):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, hidden_states, residual=None):
        # fc_in ==> gate_up_proj
        # fc_out ==> down_proj
        if self.fc_in_quant.bias is None:
            up_states = torch.ops.torch_ipex.mm_int4(
                hidden_states,
                self.fc_in_quant.qweight,
                self.fc_in_quant.scales,
                self.fc_in_quant.qzeros,
                self.fc_in_quant.blocksize,
            )
        else:
            up_states = torch.ops.torch_ipex.mm_bias_int4(
                hidden_states,
                self.fc_in_quant.qweight,
                self.fc_in_quant.bias,
                self.fc_in_quant.scales,
                self.fc_in_quant.qzeros,
                self.fc_in_quant.blocksize,
            )

        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.act(gate)

        if self.fc_out_quant.bias is None:
            output = torch.ops.torch_ipex.mm_int4(
                up_states,
                self.fc_out_quant.qweight,
                self.fc_out_quant.scales,
                self.fc_out_quant.qzeros,
                self.fc_out_quant.blocksize,
            )
        else:
            output = torch.ops.torch_ipex.mm_bias_int4(
                up_states,
                self.fc_out_quant.qweight,
                self.fc_out_quant.bias,
                self.fc_out_quant.scales,
                self.fc_out_quant.qzeros,
                self.fc_out_quant.blocksize,
            )
        return output
