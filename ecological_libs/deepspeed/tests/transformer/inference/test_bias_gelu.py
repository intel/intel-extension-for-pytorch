import pytest
import torch
from inference_test_utils import allclose, get_dtypes
from packaging import version as pkg_version
import intel_extension_for_pytorch as ipex

inference_module = ipex.deepspeed
ipex_device = 'xpu:0'
torch_minor_version = None


def run_bias_gelu_reference(activations, bias):
    # Expected behavior is that of casting to float32 internally and using the tanh approximation
    return torch.nn.functional.gelu(activations.to(torch.float32) + bias.to(torch.float32),
                                    approximate='tanh').to(activations.dtype)


def run_bias_gelu_ds(activations, bias):
    if activations.dtype == torch.float16:
        return inference_module.bias_gelu_fp16(activations, bias)
    elif activations.dtype == torch.bfloat16:
        return inference_module.bias_gelu_bf16(activations, bias)
    else:
        return inference_module.bias_gelu_fp32(activations, bias)


@pytest.mark.inference_ops
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("sequence", [1, 128, 255])
@pytest.mark.parametrize("channels", [512, 1232, 4096])
@pytest.mark.parametrize("dtype", get_dtypes())
def test_bias_gelu(batch, sequence, channels, dtype):
    if pkg_version.parse(torch.__version__) < pkg_version.parse("1.12"):
        pytest.skip("gelu implementation matches only after torch 1.12")

    activations_ds = torch.randn((batch, sequence, channels), dtype=dtype, device=ipex_device)
    bias_ds = torch.randn((channels), dtype=dtype, device=ipex_device)

    activations_ref = activations_ds.clone().detach()
    bias_ref = bias_ds.clone().detach()

    ds_out = run_bias_gelu_ds(activations_ds, bias_ds)
    ref_out = run_bias_gelu_reference(activations_ref, bias_ref)
    assert (allclose(ds_out, ref_out))
