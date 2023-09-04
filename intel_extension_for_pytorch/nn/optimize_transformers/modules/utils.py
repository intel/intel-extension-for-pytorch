import torch 
import torch.distributed as dist
from ._transformers import IPEXEmptyLinearWithPadding, IPEXEmptyINT4Linear, IPEXEmptyINT4LinearWithPadding


def is_int4(model):
    pass

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
    if n % 16 != 0:
        padded_n = (n + 16 - 1) // 16 * 16
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

def print_rank_x(x, content):
    if dist.get_rank() == 1:
        print(content)

def pad_for_gptj_lm_head(model, is_int4=False):
    if is_int4:
        n = model.lm_head.qweight.shape[1] * 2 - 1 #specific for 50401(25201) int4 weight

        lm_head_new = IPEXEmptyINT4LinearWithPadding(n)
        lm_head_new.qweight = model.lm_head.qweight
        lm_head_new.weight = model.lm_head.weight
        lm_head_new.bias = model.lm_head.bias if model.lm_head.bias is not None else None
        lm_head_new.scales = model.lm_head.scales
        lm_head_new.qzeros = model.lm_head.qzeros
        lm_head_new.group_size = model.lm_head.group_size.data.item()
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