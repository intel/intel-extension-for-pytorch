import pytest
import torch
import intel_extension_for_pytorch as ipex

inference_module = ipex.xpu.deepspeed
ipex_device = 'xpu:0'
torch_minor_version = None

def allclose(x, y):
    assert x.dtype == y.dtype
    rtol, atol = {torch.float32: (5e-4, 5e-5), torch.float16: (3e-2, 2e-3)}[x.dtype]
    return torch.allclose(x, y, rtol=rtol, atol=atol)


def version_appropriate_gelu(activations):
    global torch_minor_version
    if torch_minor_version is None:
        torch_minor_version = int(torch.__version__.split('.')[1])
    # If torch version = 1.12
    if torch_minor_version < 12:
        return torch.nn.functional.gelu(activations)
    else:
        return torch.nn.functional.gelu(activations, approximate='tanh')


def run_gelu_reference(activations):
    # Expected behavior is that of casting to float32 internally and using the tanh approximation
    return version_appropriate_gelu(activations.to(torch.float32)).to(activations.dtype)


def run_gelu_ds(activations, use_triton_ops=False):
    # if use_triton_ops:
    #     from deepspeed.ops.transformer.inference.triton import gelu
    #     return gelu(activations)

    channels = activations.shape[-1]
    bias = torch.zeros((channels), dtype=activations.dtype, device=ipex_device)
    if activations.dtype == torch.float16:
        return inference_module.bias_gelu_fp16(activations, bias)
    else:
        return inference_module.bias_gelu_fp32(activations, bias)


@pytest.mark.skipif(not inference_module.has_deepspeed(), reason="deepspeed module is not available")
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("sequence", [1, 128, 255])
@pytest.mark.parametrize("channels", [512, 1232, 4096])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("use_triton_ops", [False])
def test_gelu(batch, sequence, channels, dtype, use_triton_ops):
    activations_ds = torch.randn((batch, sequence, channels), dtype=dtype, device=ipex_device)
    activations_ref = activations_ds.clone().detach()

    if use_triton_ops:
        pytest.skip("triton has to be installed for the test")
    ds_out = run_gelu_ds(activations_ds, use_triton_ops)
    ref_out = run_gelu_reference(activations_ref)
    assert (allclose(ds_out, ref_out))
