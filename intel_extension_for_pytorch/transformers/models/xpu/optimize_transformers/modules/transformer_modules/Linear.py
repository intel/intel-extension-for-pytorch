import torch
import torch.nn as nn
from enum import Enum
from abc import abstractmethod


class GemmDtype(Enum):
    FP16 = 0
    W4A16_GPTQ = 1


class IPEXTransformerLinear(nn.Module):
    def __init__(self, weight=None, bias=None) -> None:
        super().__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, input):
        return torch.ops.torch_ipex.matmul_bias_out(input, self.weight, self.bias)


class IPEXLowbitGemmBase(nn.Module):
    def __init__(
        self,
        dtype: GemmDtype,
        weight=None,
        bias=None,
        qweight=None,
        scales=None,
        qzeros=None,
        blocksize=None,
        tp_size=1,
        tp_group=None,
    ):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.dtype = dtype
        self.qweight = qweight
        self.scales = scales
        self.qzeros = qzeros
        self.blocksize = blocksize
        self.tp_size = tp_size
        self.tp_group = tp_group

    @abstractmethod
    def load_parameter(self, *args, **kwargs):
        raise NotImplementedError(
            "This class dose not implement load_parameter method !"
        )

    @abstractmethod
    def forward_fp16(self, *args, **kwargs):
        raise NotImplementedError("This class dose not implement forward_fp16 method !")

    @abstractmethod
    def forward_w4a16(self, *args, **kwargs):
        raise NotImplementedError(
            "This class dose not implement forward_w4a16 method !"
        )

    def forward(self, *args, **kwargs):
        if self.dtype == GemmDtype.FP16:
            return self.forward_fp16(*args, **kwargs)
        elif self.dtype == GemmDtype.W4A16_GPTQ:
            return self.forward_w4a16(*args, **kwargs)
        else:
            raise NotImplementedError(
                "QKVFusionGemm dose not support this GemmType {} !".format(self.dtype)
            )


class IPEXQKVFusedGemm(IPEXLowbitGemmBase):
    def __init__(
        self,
        num_head,
        num_kv_head,
        dtype: GemmDtype = GemmDtype.FP16,
        weight=None,
        bias=None,
    ):
        super().__init__(dtype, weight, bias)
        self.num_head = num_head
        self.num_kv_head = num_kv_head

    def load_parameter(self, q_proj, k_proj, v_proj):
        if self.num_head == self.num_kv_head:  # MHA path
            q_proj_trans = q_proj.weight.transpose(0, 1).contiguous()
            k_proj_trans = k_proj.weight.transpose(0, 1).contiguous()
            v_proj_trans = v_proj.weight.transpose(0, 1).contiguous()
            shape = [3, -1, q_proj_trans.shape[-1]]
            self.weight = (
                torch.stack([q_proj_trans, k_proj_trans, v_proj_trans])
                .contiguous()
                .view(shape)
            )
            torch.xpu.synchronize()

            q_proj.weight.data = self.weight[0, :, :]
            k_proj.weight.data = self.weight[1, :, :]
            v_proj.weight.data = self.weight[2, :, :]

            if q_proj.bias is not None:
                bias_shape = [3, -1]
                self.bias = (
                    torch.stack([q_proj.bias, k_proj.bias, v_proj.bias])
                    .contiguous()
                    .view(bias_shape)
                )
                torch.xpu.synchronize()

                q_proj.bias.data = self.bias[0, :]
                k_proj.bias.data = self.bias[1, :]
                v_proj.bias.data = self.bias[2, :]
        else:  # GQA path
            embed_dim = q_proj.weight.shape[-1]
            head_dim = q_proj.weight.shape[0] // self.num_head
            group = self.num_head // self.num_kv_head + 2
            q_proj_w = q_proj.weight.view(
                self.num_kv_head, group - 2, head_dim, embed_dim
            )
            k_proj_w = k_proj.weight.view(self.num_kv_head, 1, head_dim, embed_dim)
            v_proj_w = v_proj.weight.view(self.num_kv_head, 1, head_dim, embed_dim)
            self.weight = (
                torch.cat([q_proj_w, k_proj_w, v_proj_w], dim=1)
                .view(self.num_kv_head, group, head_dim, embed_dim)
                .permute(3, 0, 1, 2)
                .contiguous()
            )
            torch.xpu.synchronize()

            if q_proj.bias is not None:
                q_b = q_proj.bias.view(self.num_kv_head, group - 2, head_dim)
                k_b = k_proj.bias.view(self.num_kv_head, 1, head_dim)
                v_b = v_proj.bias.view(self.num_kv_head, 1, head_dim)
                self.bias = torch.cat([q_b, k_b, v_b], dim=1).view(
                    self.num_kv_head, group, head_dim
                )
                torch.xpu.synchronize()

    def forward_fp16(self, input, q, k, v):
        if self.num_head == self.num_kv_head:
            torch.ops.torch_ipex.mm_qkv_out(input, self.weight, self.bias, q, k, v)
        else:
            torch.ops.torch_ipex.mm_qkv_group_out(
                input, self.weight, self.bias, q, k, v
            )
        return q, k, v


class IPEXQKVFusedGemmInt4(IPEXLowbitGemmBase):
    def __init__(
        self,
        num_head,
        num_kv_head,
        dtype: GemmDtype = GemmDtype.W4A16_GPTQ,
        qweight=None,
        scales=None,
        qzeros=None,
        bias=None,
        blocksize=None,
    ):
        super().__init__(
            dtype,
            qweight=qweight,
            bias=bias,
            scales=scales,
            qzeros=qzeros,
            blocksize=blocksize,
        )
        self.num_head = num_head
        self.num_kv_head = num_kv_head
        self.qweight = qweight
        self.scales = scales
        self.qzeros = qzeros
        self.bias = bias
        self.blocksize = blocksize

    def forward_w4a16(self, input, q, k, v):
        if self.num_head == self.num_kv_head:
            torch.ops.torch_ipex.mm_qkv_out_int4(
                input,
                self.qweight,
                self.scales,
                self.qzeros,
                self.bias,
                q,
                k,
                v,
                self.blocksize,
            )
            return q, k, v
        else:
            torch.ops.torch_ipex.mm_qkv_group_out(
                input, self.weight, self.bias, q, k, v
            )
        return q, k, v


class IPEXQKVFusedGemmInt4OneDNN(IPEXLowbitGemmBase):
    def __init__(
        self,
        num_head,
        num_kv_head,
        dtype: GemmDtype = GemmDtype.W4A16_GPTQ,
        qweight=None,
        scales=None,
        qzeros=None,
        bias=None,
        blocksize=None,
    ):
        super().__init__(
            dtype,
            qweight=qweight,
            bias=bias,
            scales=scales,
            qzeros=qzeros,
            blocksize=blocksize,
        )
        self.num_head = num_head
        self.num_kv_head = num_kv_head
        self.qweight = qweight
        self.scales = scales
        self.qzeros = qzeros
        self.bias = bias
        self.blocksize = blocksize

    def forward_w4a16(self, input, q, k, v):
        if self.num_head == self.num_kv_head:
            if (
                self.q_proj_quant.qweight is None
                and self.qkv_proj_quant.qweight is not None
            ):
                qkv_out = torch.ops.torch_ipex.mm_bias_int4(
                    input,
                    self.qkv_proj_quant.qweight,
                    self.qkv_proj_quant.bias,
                    self.qkv_proj_quant.scales,
                    self.qkv_proj_quant.qzeros,
                    self.qkv_proj_quant.blocksize,
                ).contiguous()
                m = q.shape[-1]
                q = qkv_out[:, :, :m]
                k = qkv_out[:, :, m : 2 * m]
                v = qkv_out[:, :, 2 * m :]
            else:
                q = torch.ops.torch_ipex.mm_bias_int4(
                    input,
                    self.q_proj_quant.qweight,
                    self.q_proj_quant.bias,
                    self.q_proj_quant.scales,
                    self.q_proj_quant.qzeros,
                    self.q_proj_quant.blocksize,
                )
                k = torch.ops.torch_ipex.mm_bias_int4(
                    input,
                    self.k_proj_quant.qweight,
                    self.k_proj_quant.bias,
                    self.k_proj_quant.scales,
                    self.k_proj_quant.qzeros,
                    self.k_proj_quant.blocksize,
                )
                v = torch.ops.torch_ipex.mm_bias_int4(
                    input,
                    self.v_proj_quant.qweight,
                    self.v_proj_quant.bias,
                    self.v_proj_quant.scales,
                    self.v_proj_quant.qzeros,
                    self.v_proj_quant.blocksize,
                )
            return q, k, v
        else:
            torch.ops.torch_ipex.mm_qkv_group_out(
                input, self.weight, self.bias, q, k, v
            )
        return q, k, v


class IPEXLowbitGemm(IPEXLowbitGemmBase):

    def forward_fp16(self, input):
        return torch.ops.torch_ipex.matmul_bias_out(input, self.weight, self.bias)


class IPEXLowbitGemmAdd(IPEXLowbitGemmBase):
    def __init__(
        self, tp_size=1, dtype: GemmDtype = GemmDtype.FP16, weight=None, bias=None
    ) -> None:
        super().__init__(dtype, weight, bias)
        # for tensor parallel case, we need to scale the residual input with 1 / tp_size before allreduce.
        self.tp_size_scale = 1.0 / tp_size

    def load_parameter(self, proj):
        self.weight = proj.weight.transpose(0, 1).contiguous()
        if proj.bias is not None:
            self.bias = proj.bias

    def forward_fp16(self, input, residual):
        if residual is None:
            attn_output = torch.matmul(input, self.weight)
            if self.bias is not None:
                attn_output += self.bias
            return attn_output
        if self.bias is None:
            return torch.addmm(
                residual.flatten(0, -2),
                input.flatten(0, -2),
                self.weight,
                beta=self.tp_size_scale,
            )
        return torch.ops.torch_ipex.mm_bias_resadd(
            input,
            self.weight,
            self.bias,
            self.tp_size_scale,
            residual,
            self.tp_size_scale,
        )


class IPEXLowbitGemmAddInt4(IPEXLowbitGemmBase):
    def __init__(
        self,
        tp_size=1,
        dtype: GemmDtype = GemmDtype.W4A16_GPTQ,
        qweight=None,
        scales=None,
        qzeros=None,
        bias=None,
        blocksize=None,
    ) -> None:
        super().__init__(
            dtype,
            qweight=qweight,
            bias=bias,
            scales=scales,
            qzeros=qzeros,
            blocksize=blocksize,
        )

    def forward_w4a16(self, input, residual):
        if residual is None:
            if self.out_proj.bias is not None:
                attn_output = torch.ops.torch_ipex.mm_bias_int4(
                    input,
                    self.out_proj_quant.qweight,
                    self.out_proj_quant.bias,
                    self.out_proj_quant.scales,
                    self.out_proj_quant.qzeros,
                    self.out_proj_quant.blocksize,
                )
            else:
                attn_output = torch.ops.torch_ipex.mm_int4(
                    input,
                    self.out_proj_quant.qweight,
                    self.out_proj_quant.scales,
                    self.out_proj_quant.qzeros,
                    self.out_proj_quant.blocksize,
                )
        else:
            shape = [input.shape[0], input.shape[1], self.embed_dim]
            if self.out_proj.bias is not None:
                attn_output = torch.ops.torch_ipex.mm_bias_add_int4(
                    input,
                    self.out_proj_quant.qweight,
                    self.out_proj_quant.bias,
                    self.out_proj_quant.scales,
                    self.out_proj_quant.qzeros,
                    self.out_proj_quant.blocksize,
                    residual,
                )
            else:
                attn_output = torch.ops.torch_ipex.mm_add_int4(
                    input,
                    self.out_proj_quant.qweight,
                    self.out_proj_quant.scales,
                    self.out_proj_quant.qzeros,
                    self.out_proj_quant.blocksize,
                    residual,
                )
            attn_output = attn_output.view(shape)
        return attn_output


def matmul_add_add(attn_output, weight, tp_size=1, bias=None, residual=None):
    seq_len, bs, _ = attn_output.size()
    if residual is None:
        attn_output = torch.matmul(attn_output, weight)
        if bias is not None:
            attn_output += bias
    else:
        if bias is not None:
            attn_output = torch.ops.torch_ipex.mm_bias_resadd(
                attn_output, weight, bias, 1.0 / tp_size, residual, 1.0 / tp_size
            )
        else:
            attn_output = torch.addmm(
                residual.flatten(0, -2),
                attn_output.flatten(0, -2),
                weight,
                beta=1.0 / tp_size,
            )
    attn_output = attn_output.view(seq_len, bs, -1)
    return attn_output
