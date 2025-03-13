import torch
import intel_extension_for_pytorch as ipex


def has_bitsandbytes():
    return ipex._C._is_bnb_kernel_enabled()


def cpercentile_clipping_g32(g, gnorm_vec, step, n):
    return torch.ops.torch_ipex.cpercentile_clipping_g32(g, gnorm_vec, step, n)


def cpercentile_clipping_g16(g, gnorm_vec, step, n):
    return torch.ops.torch_ipex.cpercentile_clipping_g16(g, gnorm_vec, step, n)


def cadam_8bit_blockwise_grad_fp32(
    p,
    g,
    state1,
    state2,
    beta1,
    beta2,
    beta3,
    alpha,
    eps,
    step,
    lr,
    qmap1,
    qmap2,
    absmax1,
    absmax2,
    weight_decay,
    gnorm_scale,
    skip_zeros,
    g_numel,
):
    return torch.ops.torch_ipex.cadam_8bit_blockwise_grad_fp32(
        p,
        g,
        state1,
        state2,
        beta1,
        beta2,
        beta3,
        alpha,
        eps,
        step,
        lr,
        qmap1,
        qmap2,
        absmax1,
        absmax2,
        weight_decay,
        gnorm_scale,
        skip_zeros,
        g_numel,
    )


def cadam_8bit_blockwise_grad_fp16(
    p,
    g,
    state1,
    state2,
    beta1,
    beta2,
    beta3,
    alpha,
    eps,
    step,
    lr,
    qmap1,
    qmap2,
    absmax1,
    absmax2,
    weight_decay,
    gnorm_scale,
    skip_zeros,
    g_numel,
):
    return torch.ops.torch_ipex.cadam_8bit_blockwise_grad_fp16(
        p,
        g,
        state1,
        state2,
        beta1,
        beta2,
        beta3,
        alpha,
        eps,
        step,
        lr,
        qmap1,
        qmap2,
        absmax1,
        absmax2,
        weight_decay,
        gnorm_scale,
        skip_zeros,
        g_numel,
    )


def cadam_8bit_blockwise_grad_bf16(
    p,
    g,
    state1,
    state2,
    beta1,
    beta2,
    beta3,
    alpha,
    eps,
    step,
    lr,
    qmap1,
    qmap2,
    absmax1,
    absmax2,
    weight_decay,
    gnorm_scale,
    skip_zeros,
    g_numel,
):
    return torch.ops.torch_ipex.cadam_8bit_blockwise_grad_bf16(
        p,
        g,
        state1,
        state2,
        beta1,
        beta2,
        beta3,
        alpha,
        eps,
        step,
        lr,
        qmap1,
        qmap2,
        absmax1,
        absmax2,
        weight_decay,
        gnorm_scale,
        skip_zeros,
        g_numel,
    )


def cdequantize_blockwise_fp32(code, A, absmax, out, blocksize, n):
    return torch.ops.torch_ipex.cdequantize_blockwise_fp32(
        code, A, absmax, out, blocksize, n
    )


def cdequantize_blockwise_fp16(code, A, absmax, out, blocksize, n):
    return torch.ops.torch_ipex.cdequantize_blockwise_fp16(
        code, A, absmax, out, blocksize, n
    )


def cdequantize_blockwise_bf16(code, A, absmax, out, blocksize, n):
    return torch.ops.torch_ipex.cdequantize_blockwise_bf16(
        code, A, absmax, out, blocksize, n
    )
