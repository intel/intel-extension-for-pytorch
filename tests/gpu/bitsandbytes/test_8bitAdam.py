import os
from os.path import join
import shutil
import time
import uuid
import pytest
import torch

import bitsandbytes as bnb
import bitsandbytes.functional as F
k = 20

def assert_most_approx_close(a, b, rtol=1e-3, atol=1e-3, max_error_count=0):
    idx = torch.isclose(a, b, rtol=rtol, atol=atol)
    error_count = (idx == 0).sum().item()
    if error_count > max_error_count:
        print(f"Too many values not close: assert {error_count} < {max_error_count}")
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


str2optimizers = {}
str2optimizers["adam8bit_blockwise"] = (torch.optim.Adam, lambda pxx: bnb.optim.Adam8bit(pxx, block_wise=True))

str2statenames = {}
str2statenames["adam8bit_blockwise"] = [
    ("exp_avg", "state1", "qmap1", "absmax1"),
    ("exp_avg_sq", "state2", "qmap2", "absmax2"),
]

optimizer_names_8bit = [
    "adam8bit_blockwise",
]


@pytest.mark.parametrize("optim_name", optimizer_names_8bit, ids=["adam8bit_blockwise"])
@pytest.mark.parametrize("gtype", [torch.float32, torch.float16, torch.bfloat16], ids=["float", "half", "bfloat16"])
@pytest.mark.parametrize("dim2", [32, 1024, 4097], ids=["32", "1024", "4097"])
@pytest.mark.parametrize("dim1", [1024], ids=["1024"])
def test_optimizer8bit(dim1, dim2, gtype, optim_name):
    device_t = "xpu"
    torch.set_printoptions(precision=6)

    if gtype == torch.bfloat16 and "blockwise" not in optim_name:
        pytest.skip()

    if dim1 == 1 and dim2 == 1:
        return
    p1 = torch.randn(dim1, dim2, device=device_t, dtype=gtype) * 0.1
    p2 = p1.clone()
    p1 = p1.float()
    blocksize = 256

    torch_optimizer = str2optimizers[optim_name][0]([p1])
    bnb_optimizer = str2optimizers[optim_name][1]([p2])

    if gtype == torch.float32:
        atol, rtol = 3e-3, 1e-3
        patol, prtol = 1e-5, 1e-3
    elif gtype == torch.bfloat16:
        atol, rtol = 3e-3, 1e-3
        patol, prtol = 1e-4, 1e-2
    else:
        atol, rtol = 3e-3, 1e-3
        patol, prtol = 1e-5, 1e-3

    errors = []
    relerrors = []

    for i in range(50):
        g = torch.randn(dim1, dim2, device=device_t, dtype=gtype) * 0.01
        p1.grad = g.clone().float()
        p2.grad = g.clone()

        bnb_optimizer.step()
        torch_optimizer.step()

        # since Lion can have pretty noisy updates where things lie at the boundary
        assert_most_approx_close(p1, p2.float(), patol, prtol, max_error_count=0)

        dequant_states = []
        for name1, name2, qmap, max_val in str2statenames[optim_name]:
            if "blockwise" in optim_name:
                s1 = F.dequantize_blockwise(
                    code=bnb_optimizer.state[p2][qmap],
                    absmax=bnb_optimizer.state[p2][max_val],
                    A=bnb_optimizer.state[p2][name2],
                    blocksize=blocksize,
                )
            
            num_not_close = torch.isclose(torch_optimizer.state[p1][name1], s1, atol=atol, rtol=rtol) == 0
            # assert num_not_close.sum().item() < 20
            if s1 is not None:
                dequant_states.append(s1.clone())

        err = torch.abs(p1 - p2)
        relerr = err / (torch.abs(p1) + 1e-9)
        if g.dtype == torch.bfloat16:
            assert err.mean() <= 0.00017
            assert relerr.mean() <= 0.0016
        else:
            assert err.mean() < 0.00006
            assert relerr.mean() < 0.0006

        print("iter ", i, " passed")

        errors.append(err.mean().item())
        relerrors.append(relerr.mean().item())

        # the parameters diverge quickly. Here we keep them close
        # together so we can test against the Adam error
        p1.data = p1.data.to(gtype).float()
        p2.copy_(p1.data)
        torch.testing.assert_close(p1.to(gtype), p2)
        for (name1, name2, qmap, max_val), s in zip(str2statenames[optim_name], dequant_states):
            torch_optimizer.state[p1][name1].copy_(s.data)

    print(sum(errors)/len(errors))
    print(sum(relerrors)/len(relerrors))

if __name__ == '__main__':
    test_optimizer8bit(1024, 32, torch.float32, optimizer_names_8bit[0])
