import pytest
import torch
from inference_test_utils import allclose, get_dtypes
import intel_extension_for_pytorch as ipex

inference_module = ipex.deepspeed
ipex_device = 'xpu:0'
torch_minor_version = None

def run_bias_add_reference(activations, bias):
    return activations + bias


def run_bias_add_ds(activations, bias):
    if activations.dtype == torch.float16:
        return inference_module.bias_add_fp16(activations, bias)
    elif activations.dtype == torch.bfloat16:
        return inference_module.bias_add_bf16(activations, bias)
    else:
        return inference_module.bias_add_fp32(activations, bias)


@pytest.mark.inference_ops
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("sequence", [1, 128, 255])
@pytest.mark.parametrize("channels", [512, 1232, 4096])
@pytest.mark.parametrize("dtype", get_dtypes())
def test_bias_add(batch, sequence, channels, dtype):
    activations_ds = torch.randn((batch, sequence, channels), dtype=dtype, device=ipex_device)
    bias_ds = torch.randn((channels), dtype=dtype, device=ipex_device)

    activations_ref = activations_ds.clone().detach()
    bias_ref = bias_ds.clone().detach()

    ds_out = run_bias_add_ds(activations_ds, bias_ds)
    ref_out = run_bias_add_reference(activations_ref, bias_ref)
    if not allclose(ds_out, ref_out):
        print((ds_out - ref_out).abs().max())
        assert (allclose(ds_out, ref_out))
