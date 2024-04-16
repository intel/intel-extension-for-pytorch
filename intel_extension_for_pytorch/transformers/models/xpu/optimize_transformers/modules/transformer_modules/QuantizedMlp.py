import torch
import intel_extension_for_pytorch as ipex
from .Mlp import IPEXTransformerMLP
from intel_extension_for_pytorch.nn.utils._quantize_convert import (
    WeightOnlyQuantizedLinear,
)
from .Activation import ACT2FN


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

        self.fc_in_quant.qzeros.data = self.fc_in_quant.qzeros.transpose(
            0, 1
        ).contiguous()
        self.fc_out_quant.qzeros.data = self.fc_out_quant.qzeros.transpose(
            0, 1
        ).contiguous()

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


class IPEXTransformerMLPOptimizedInt4SiluQwen(IPEXTransformerMLPOptimizedInt4Silu):
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
        self.c_proj_quant.qweight.data = self.c_proj_quant.qweight.transpose(
            0, 1
        ).contiguous()
        self.c_proj_quant.scales.data = self.c_proj_quant.scales.transpose(
            0, 1
        ).contiguous()
        self.c_proj_quant.qzeros.data = self.c_proj_quant.qzeros.transpose(
            0, 1
        ).contiguous()
        torch.xpu.synchronize()
        try_linear_gate_reorder_input = torch.empty(
            1, 1, 4096, dtype=torch.float16, device="xpu"
        )
        try_linear_down_reorder_input = torch.empty(
            1, 1, 11008, dtype=torch.float16, device="xpu"
        )
        try_linear_mlp_reorder = torch.ops.torch_ipex.mm_esimd_int4(
            try_linear_gate_reorder_input,
            self.fc_out_quant.qweight,
            self.fc_out_quant.scales,
            self.fc_out_quant.qzeros,
            self.fc_out_quant.blocksize,
            True,
        )
        try_linear_mlp_reorder = torch.ops.torch_ipex.mm_esimd_int4(
            try_linear_down_reorder_input,
            self.c_proj_quant.qweight,
            self.c_proj_quant.scales,
            self.c_proj_quant.qzeros,
            self.c_proj_quant.blocksize,
            True,
        )
        try_linear_mlp_reorder = torch.ops.torch_ipex.mm_esimd_int4(
            try_linear_gate_reorder_input,
            self.fc_in_quant.qweight,
            self.fc_in_quant.scales,
            self.fc_in_quant.qzeros,
            self.fc_in_quant.blocksize,
            True,
        )

    def inter_mm(self, hidden_states):
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


class IPEXTransformerMLPOptimizedInt4SiluLlama(IPEXTransformerMLPOptimizedInt4SiluQwen):
    def __init__(self, config):
        super().__init__(config)
        self.act_fn = ACT2FN[config.activation_function]

    def load_parameter(self, fc_in, fc_out, c_proj):
        # gate down up
        # gate_proj fc_out_quant
        # down_proj c_proj_quant
        # up_proj fc_in_quant
        return super().load_parameter(c_proj, fc_in, fc_out)

    def forward(self, x, residual=None):
        # down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        hidden_states_gate = torch.ops.torch_ipex.mm_esimd_int4(
            x,
            self.fc_out_quant.qweight,
            self.fc_out_quant.scales,
            self.fc_out_quant.qzeros,
            self.fc_out_quant.blocksize,
            False,
        )

        # synchronize for the 1st token to WA 1024in/windows hang issue
        if hidden_states_gate.shape[0] != 1:
            torch.xpu.synchronize()
            torch.xpu.empty_cache()

        hidden_states_act = self.act_fn(hidden_states_gate)
        hidden_states_up = torch.ops.torch_ipex.mm_esimd_int4(
            x,
            self.fc_in_quant.qweight,
            self.fc_in_quant.scales,
            self.fc_in_quant.qzeros,
            self.fc_in_quant.blocksize,
            False,
        )
        hidden_states = hidden_states_act * hidden_states_up

        down_proj = torch.ops.torch_ipex.mm_esimd_int4(
            hidden_states,
            self.c_proj_quant.qweight,
            self.c_proj_quant.scales,
            self.c_proj_quant.qzeros,
            self.c_proj_quant.blocksize,
            False,
        )

        return down_proj
