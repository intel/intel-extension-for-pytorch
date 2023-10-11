import torch 
import torch.distributed as dist
from ._transformers import IPEXEmptyLinearWithPadding, IPEXEmptyINT4Linear, IPEXEmptyINT4LinearWithPadding, IPEXTransformerConverter, IPEXLmHeadLinearAllreduceWithPadding
import os

def is_int4(model):
    if hasattr(model, "dtype_tag"):
        return model.dtype_tag == "int4"

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
    if hasattr(model, "slicing_pad"):
        return
    else:
        setattr(model, "slicing_pad", True)
    if is_int4:
        n = model.lm_head.qweight.shape[1] * 2 - 1 #specific for 50401(25201) int4 weight

        lm_head_new = IPEXEmptyINT4LinearWithPadding(n)
        lm_head_new.qweight = model.lm_head.qweight
        lm_head_new.weight = model.lm_head.weight
        lm_head_new.bias = model.lm_head.bias if model.lm_head.bias is not None else None
        lm_head_new.scales = model.lm_head.scales
        lm_head_new.qzeros = model.lm_head.qzeros
        lm_head_new.group_size = model.lm_head.group_size
        model.lm_head = lm_head_new

        model.lm_head.qweight.data = int4_gemm_padding(model.lm_head.qweight)
        model.lm_head.scales.data = int4_gemm_scale_padding(model.lm_head.scales)
        model.lm_head.qzeros.data = int4_gemm_padding(model.lm_head.qzeros)

        if model.lm_head.bias is not None:
            model.lm_head.bias.data = int4_gemm_bias_padding(model.lm_head.bias)

    else:
        n = model.lm_head.weight.shape[0] #[n, k]

        # tensor parallel for the last linear
        # TOTO: update IPEXEmptyLinearWithPadding after DeepSpeed PR (https://github.com/microsoft/DeepSpeed/pull/3962) merge or Ipex auto TP
        if hasattr(model, "lm_head") and dist.is_initialized() and False:
            lm_head_new = IPEXLmHeadLinearAllreduceWithPadding(
                n, torch.nn.parameter.Parameter(model.lm_head.weight, requires_grad=False), dist.get_rank(), dist.get_world_size(),
                model.lm_head.bias if model.lm_head.bias is None else torch.nn.parameter.Parameter(model.lm_head.bias), IPEXTransformerConverter.tp_group)
            model.lm_head = lm_head_new
        else:
            lm_head_new = IPEXEmptyLinearWithPadding(n)
            lm_head_new.weight = model.lm_head.weight
            lm_head_new.bias = model.lm_head.bias

        if model.lm_head.bias is not None:
            model.lm_head.weight.data, model.lm_head.bias.data = gemm_padding(model.lm_head.weight, model.lm_head.bias)
        else:
            model.lm_head.weight.data, _ = gemm_padding(model.lm_head.weight)


class TransformerFallbackController:
    fallback_attn = os.environ.get("FALLBACK_ATTN", "IPEXTransformerAttnNaive")
    fallback_mlp = os.environ.get("FALLBACK_MLP", "IPEXTransformerMLP")
    fallback_flag = os.environ.get("FALLBACK", False)
    acc_check_flag = os.environ.get("ACC_CHECK", False)
    acc_tolerance = os.environ.get("ACC_TOLERANCE", 1e-3)
    fallback_func = os.environ.get("FALLBACK_FUNC", [])
    
    @staticmethod
    def fallback_decorator(cls):
        # cls_name = cls.__name__
        assert not (TransformerFallbackController.fallback_flag and TransformerFallbackController.acc_check_flag), "fallback flag and acc check flag can not be set at the same time. If you are using fallback decorator please don't set the ACC_CHECK flag"
        mros = [elem.__name__ for elem in cls.__mro__[1:]]
        can_fallback = False
        if TransformerFallbackController.fallback_attn in mros or TransformerFallbackController.fallback_mlp in mros:
            can_fallback = True
        def decorator(func):
            is_fallback_func = False
            if func.__name__ in TransformerFallbackController.fallback_func:
                is_fallback_func = True
            def wrapper(*args, **kwargs):
                if can_fallback and is_fallback_func and TransformerFallbackController.fallback_flag:
                    if TransformerFallbackController.fallback_attn in mros:
                        idx = mros.index(TransformerFallbackController.fallback_attn)
                    else:
                        idx = mros.index(TransformerFallbackController.fallback_mlp)
                    running_cls = cls.__mro__[1:][idx]
                    running_func = getattr(running_cls, func.__name__)
                    return running_func(*args, **kwargs)
                elif not (can_fallback or can_fallback) and TransformerFallbackController.fallback_flag:
                    print("method {} can not fallback to target implementation for its baseclass do not have its fallback candidate, please make sure you have contructed the expected Attenion and MLP layer".format(func.__name__))
                    print("method {} will remains unchange in this model".format(func.__name__))
                return func(*args, **kwargs)
            return wrapper
        return decorator

    @staticmethod
    def acc_check(function: str, check1, check2):
        if isinstance(check1, list):
            if not len(check1) == len(check2):
                print("different shape as list")
                return
            for i, elem in enumerate(check1):
                TransformerFallbackController.acc_check(elem, check2[i])
        if isinstance(check1, dict):
            if not len(check1.keys()) == len(check2.keys()):
                print("different shape as dict")
                return
            for key in check1.keys():
                TransformerFallbackController.acc_check(check1[key], check2[key])
        if isinstance(check1, torch.Tensor) and isinstance(check2, torch.Tensor):
            diff = (check1 - check2) > TransformerFallbackController.acc_tolerance
            ratio = diff.sum() / diff.numel()
            if not ratio == 0:
                print("{} generate different result between 2 implementation, the difference ratio on tolerance {} is {}".format(function, TransformerFallbackController.acc_tolerance, ratio))
            else:
                print("{} have pass the accuracy check with tolerance {}".format(TransformerFallbackController.acc_tolerance))

    @staticmethod
    def acc_compare_decorator(cls, acc_check_fn):
        assert not (TransformerFallbackController.fallback_flag and TransformerFallbackController.acc_check_flag), "fallback flag and acc check flag can not be set at the same time. If you are using acc check decorator please don't set the FALLBACK flag"
        mros = [elem.__name__ for elem in cls.__mro__[1:]]
        can_check_acc = False
        if TransformerFallbackController.fallback_attn in mros or TransformerFallbackController.fallback_mlp in mros:
            can_check_acc = True
        def decorator(func):
            is_acc_check_func = False
            if func.__name__ in TransformerFallbackController.fallback_func:
                is_acc_check_func = True
            def wrapper(*args, **kwargs):
                if can_check_acc and is_acc_check_func and TransformerFallbackController.acc_check_flag:
                    if TransformerFallbackController.fallback_attn in mros:
                        idx = mros.index(TransformerFallbackController.fallback_attn)
                    else:
                        idx = mros.index(TransformerFallbackController.fallback_mlp)
                    running_cls = cls.__mro__[1:][idx]
                    running_func = getattr(running_cls, func.__name__)
                    fallback_func_result = running_func(*args, **kwargs)
                    func_result = func(*args, **kwargs)
                    acc_check_fn(func.__name__, func_result, fallback_func_result)
                    return func_result
                elif not (can_check_acc or is_acc_check_func) and TransformerFallbackController.fallback_flag:
                    print("method {} can not fallback to target implementation for its baseclass do not have its fallback candidate, please make sure you have contructed the expected Attenion and MLP layer".format(func.__name__))
                    print("method {} will remains unchange in this model".format(func.__name__))
                return func(*args, **kwargs)
            return wrapper
        return decorator