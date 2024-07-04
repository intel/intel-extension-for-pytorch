import torch
import pytest
from inference_test_utils import allclose, get_dtypes
import intel_extension_for_pytorch as ipex

inference_module = ipex.xpu.deepspeed
ipex_device = 'xpu:0'

def ref_implementation(vals, gamma, epsilon):
    variance = vals.to(torch.float32).pow(2).mean(-1, keepdim=True)
    vals = vals * torch.rsqrt(variance + epsilon)

    if gamma.dtype in [torch.float16, torch.bfloat16]:
        vals = vals.to(gamma.dtype)

    return gamma * vals


def ds_implementation(vals, gamma, epsilon):
    return inference_module.ds_rms_norm(vals, gamma, epsilon)


@pytest.mark.skipif(not inference_module.has_deepspeed(), reason="deepspeed module is not available")
@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("seq_len", [1, 128])
@pytest.mark.parametrize("channels", [384, 512, 768, 1024, 2048, 8192, 14432])
@pytest.mark.parametrize("dtype", get_dtypes())
def test_rms_norm(batch, seq_len, channels, dtype):
    device = ipex_device
    vals = torch.randn((batch, seq_len, channels), dtype=dtype, device=device)
    gamma = torch.randn((channels), dtype=dtype, device=device)
    epsilon = 1e-5

    ref_output = ref_implementation(vals, gamma, epsilon)
    new_output = ds_implementation(vals, gamma, epsilon)

    assert allclose(new_output, ref_output)


def pre_ds_implementation(vals, residual, gamma, epsilon):
    return inference_module.ds_pre_rms_norm(vals, residual, gamma, epsilon)


def pre_ref_implementation(vals, residual, gamma, epsilon):
    residual = vals.to(torch.float32) + residual.to(torch.float32)
    vals = residual

    variance = vals.to(torch.float32).pow(2).mean(-1, keepdim=True)
    vals = vals * torch.rsqrt(variance + epsilon)

    if gamma.dtype in [torch.float16, torch.bfloat16]:
        vals = vals.to(gamma.dtype)

    return gamma * vals, residual.to(gamma.dtype)


@pytest.mark.skipif(not inference_module.has_deepspeed(), reason="deepspeed module is not available")
@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("seq_len", [1, 128])
@pytest.mark.parametrize("channels", [384, 512, 768, 1024, 2048, 8192, 14432])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_pre_norm(batch, seq_len, channels, dtype):
    device = ipex_device
    vals = torch.randn((batch, seq_len, channels), dtype=dtype, device=device)
    residual = torch.randn((batch, seq_len, channels), dtype=dtype, device=device)
    gamma = torch.randn((channels), dtype=dtype, device=device)
    epsilon = 1e-5

    ref_output = pre_ref_implementation(vals, residual, gamma, epsilon)
    new_output = pre_ds_implementation(vals, residual, gamma, epsilon)

    assert allclose(new_output[0], ref_output[0])
    #assert allclose(new_output[1], ref_output[1])
