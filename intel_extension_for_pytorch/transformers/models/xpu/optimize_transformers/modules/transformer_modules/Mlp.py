import torch
import torch.nn as nn
import torch.distributed as dist

from .Activation import ACT2FN
from .Linear import IPEXTransformerLinear, IPEXTransformerQLinear, matmul_add_add


class IPEXTransformerBaseMLP(nn.Module):
    beam_size = 1

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.tp_size = config.tp_size
        self.tp_group = config.tp_group
        self.module = None

    @staticmethod
    def release_resources():
        pass

    def load_parameter(self, module):
        self.module = module

    def transpose_parameter(self, module):
        raise NotImplementedError

    def all_reduce_if_necessary(self, target):
        if self.tp_group is not None:
            dist.all_reduce(target, group=self.tp_group)
        return target

    def inter_mm(self, hidden_states):
        raise NotImplementedError

    def out_mm(self, hidden_states, residual=None):
        raise NotImplementedError

    def forward(self, hidden_states, residual=None):
        inter_output = self.inter_mm(hidden_states)
        output = self.out_mm(inter_output, residual)
        return output


class IPEXTransformerMLP(IPEXTransformerBaseMLP):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.fc_in = IPEXTransformerLinear()
        self.fc_out = IPEXTransformerLinear()
        self.act = ACT2FN[config.activation_function]

    def load_parameter(self, fc_in, fc_out):
        self.fc_in.weight = fc_in.weight
        self.fc_out.weight = fc_out.weight

        self.fc_in.bias = fc_in.bias
        self.fc_out.bias = fc_out.bias

    def transpose_parameter(self):
        self.fc_in.weight.data = self.fc_in.weight.transpose(0, 1).contiguous()
        self.fc_out.weight.data = self.fc_out.weight.transpose(0, 1).contiguous()

    def inter_mm(self, hidden_states):
        hidden_states = self.fc_in(hidden_states)
        return self.act(hidden_states)

    def out_mm(self, hidden_states, residual=None):
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.all_reduce_if_necessary(hidden_states)
        if self.fc_out.bias is not None:
            hidden_states += self.fc_out.bias
        if residual is not None:
            hidden_states += residual
        return hidden_states


class IPEXTransformerMLPNaiveFp16GeluGptj(IPEXTransformerMLP):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.dropout = nn.Dropout(config.residual_dropout)

    def forward(self, hidden_states, attn_output=None, residual=None):
        if self.beam_size == 1:
            residual = residual.transpose(0, 1).contiguous()
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        feed_forward_hidden_states = self.dropout(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual
        return hidden_states


class IPEXTransformerMLPOptimizedFp16(IPEXTransformerMLP):
    def __init__(self, config) -> None:
        super().__init__(config)

    def inter_mm(self, hidden_states):
        hidden_states = torch.ops.torch_ipex.matmul_bias_out(
            hidden_states, self.fc_in.weight, self.fc_in.bias
        )
        return self.act(hidden_states)

    def out_mm(self, hidden_states, residual=None):
        hidden_states = matmul_add_add(
            hidden_states, self.fc_out.weight, self.tp_size, self.fc_out.bias, residual
        )
        hidden_states = self.all_reduce_if_necessary(hidden_states)
        return hidden_states


class IPEXTransformerMLPOptimizedFp16ReluOpt(IPEXTransformerMLPOptimizedFp16):
    def __init__(self, config) -> None:
        super().__init__(config)

    def inter_mm(self, hidden_states):
        hidden_states = torch.ops.torch_ipex.matmul_relu(
            hidden_states, self.fc_in.weight, self.fc_in.bias, 1.0
        )
        return hidden_states


class IPEXTransformerMLPOptimizedFp16Gelu(IPEXTransformerMLPOptimizedFp16):
    def __init__(self, config) -> None:
        super().__init__(config)

    def inter_mm(self, hidden_states):
        hidden_states = torch.ops.torch_ipex.matmul_gelu(
            hidden_states, self.fc_in.weight, self.fc_in.bias, 1.0, self.act.approximate
        )
        return hidden_states


class IPEXTransformerMLPOptimizedFp16GeluGptj(IPEXTransformerMLPOptimizedFp16Gelu):
    def __init__(self, config) -> None:
        super().__init__(config)

    def forward(self, hidden_states, attn_output, residual):
        inter_out = self.inter_mm(hidden_states)
        out = self.out_mm(inter_out, attn_output, residual)
        return out

    def out_mm(self, inter_out, attn_out, residual):
        hidden_states = torch.ops.torch_ipex.mm_bias_resadd_resadd(
            inter_out,
            self.fc_out.weight,
            self.fc_out.bias,
            1.0 / self.tp_size,
            attn_out,
            1.0 / self.tp_size,
            residual,
            1.0 / self.tp_size,
        )
        hidden_states = self.all_reduce_if_necessary(hidden_states)
        return hidden_states


class IPEXTransformerMLPOptimizedInt4(IPEXTransformerMLP):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.fc_in_quant = IPEXTransformerQLinear()
        self.fc_out_quant = IPEXTransformerQLinear()

    # TODO(ganyi): int4 load data
    def load_parameter(self, fc_in, fc_out):
        self.fc_in_quant.weight = fc_in.qweight
        self.fc_out_quant.weight = fc_out.qweight

        self.fc_in_quant.bias = fc_in.bias
        self.fc_out_quant.bias = fc_out.bias

        self.fc_in_quant.scale = fc_in.scales
        self.fc_out_quant.scale = fc_out.scales

        self.fc_in_quant.zp = fc_in.qzeros
        self.fc_out_quant.zp = fc_out.qzeros

        self.fc_in_quant.gs = fc_in.group_size
        self.fc_out_quant.gs = fc_out.group_size

    def transpose_parameter(self):
        pass

    def inter_mm(self, hidden_states):
        if self.fc_in_quant.bias is None:
            hidden_states = torch.ops.torch_ipex.mm_int4(
                hidden_states,
                self.fc_in_quant.weight,
                self.fc_in_quant.scale,
                self.fc_in_quant.zp,
                self.fc_in_quant.gs,
            )
        else:
            hidden_states = torch.ops.torch_ipex.mm_bias_int4(
                hidden_states,
                self.fc_in_quant.weight,
                self.fc_in_quant.bias,
                self.fc_in_quant.scale,
                self.fc_in_quant.zp,
                self.fc_in_quant.gs,
            )

        return self.act(hidden_states)

    def out_mm(self, hidden_states, residual=None):
        if self.fc_out_quant.bias is None:
            hidden_states = torch.ops.torch_ipex.mm_int4(
                hidden_states,
                self.fc_out_quant.weight,
                self.fc_out_quant.scale,
                self.fc_out_quant.zp,
                self.fc_out_quant.gs,
            )
        else:
            hidden_states = torch.ops.torch_ipex.mm_bias_int4(
                hidden_states,
                self.fc_out_quant.weight,
                self.fc_out_quant.bias,
                self.fc_out_quant.scale,
                self.fc_out_quant.zp,
                self.fc_out_quant.gs,
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
                self.fc_in_quant.weight,
                self.fc_in_quant.scale,
                self.fc_in_quant.zp,
                self.fc_in_quant.gs,
            )
            hidden_states = self.act(hidden_states)
        else:
            hidden_states = torch.ops.torch_ipex.mm_bias_gelu_int4(
                hidden_states,
                self.fc_in_quant.weight,
                self.fc_in_quant.scale,
                self.fc_in_quant.zp,
                self.fc_in_quant.bias,
                self.fc_in_quant.gs,
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
            self.fc_out_quant.weight,
            self.fc_out_quant.bias,
            attn_out,
            residual,
            self.fc_out_quant.scale,
            self.fc_out_quant.zp,
            self.fc_out_quant.gs,
        )
        hidden_states = self.all_reduce_if_necessary(hidden_states)
        return hidden_states


class IPEXTransformerMLPOptimizedFp16Silu(IPEXTransformerMLPOptimizedFp16):
    def __init__(self, config):
        super().__init__(config)

    def inter_mm(self, hidden_states):
        if self.fc_in.bias is None:
            hidden_states = torch.ops.torch_ipex.mm_silu(
                hidden_states, self.fc_in.weight
            )
        else:
            hidden_states = self.fc_in(hidden_states)
        return hidden_states


class IPEXTransformerMLPOptimizedFp16SiluLlama(IPEXTransformerMLPOptimizedFp16Silu):
    def __init__(self, config):
        super().__init__(config)
        self.up_proj = IPEXTransformerLinear()

    def load_parameter(self, fc_in, fc_out, up_proj):
        super().load_parameter(fc_in, fc_out)
        self.up_proj.weight = up_proj.weight
        self.up_proj.bias = up_proj.bias

    def transpose_parameter(self):
        super().transpose_parameter()
        self.up_proj.weight.data = self.up_proj.weight.transpose(0, 1).contiguous()
        # Note: synchronize to ensure the completion of contiguous
        torch.xpu.synchronize()

    def inter_mm(self, hidden_states):
        hidden_states1 = torch.ops.torch_ipex.mm_silu(hidden_states, self.fc_in.weight)
        hidden_states = torch.ops.torch_ipex.mm_resmul(
            hidden_states, self.up_proj.weight, hidden_states1
        )
        return hidden_states


class IPEXTransformerMLPOptimizedFp16SiluBaichuan(
    IPEXTransformerMLPOptimizedFp16SiluLlama
):
    def __init__(self, config):
        super().__init__(config)


class IPEXTransformerMLPOptimizedFp16SiluQwen(IPEXTransformerMLPOptimizedFp16Silu):
    def __init__(self, config):
        super().__init__(config)
        self.c_proj = IPEXTransformerLinear()

    def load_parameter(self, fc_in, fc_out, c_proj):
        #            QWenMLP
        # fc_in   |    w1
        # fc_out  |    w2
        # c_proj  |    c_proj
        super().load_parameter(fc_in, fc_out)
        self.c_proj.weight = c_proj.weight
        self.c_proj.bias = c_proj.bias

    def transpose_parameter(self):
        super().transpose_parameter()
        self.c_proj.weight.data = self.c_proj.weight.transpose(0, 1).contiguous()
        # Note: synchronize to ensure the completion of contiguous
        torch.xpu.synchronize()

    def inter_mm(self, hidden_states):
        # hidden_states = self.fc_in(hidden_states) * self.act(self.fc_out(hidden_states))
        hidden_states = self.fc_in(hidden_states) * torch.ops.torch_ipex.mm_silu(
            hidden_states, self.fc_out.weight
        )
        return hidden_states

    def out_mm(self, hidden_states, residual=None):
        if self.c_proj.bias is None:
            return torch.ops.torch_ipex.mm_resadd(
                hidden_states, self.c_proj.weight, residual, 1.0
            )
        else:
            return torch.ops.torch_ipex.mm_bias_resadd(
                hidden_states, self.c_proj.weight, self.c_proj.bias, 1.0, residual, 1.0
            )
