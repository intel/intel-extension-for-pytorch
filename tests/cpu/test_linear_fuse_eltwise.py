import unittest
import torch
import intel_extension_for_pytorch as ipex
import intel_extension_for_pytorch._C as core
from torch.testing._internal.common_utils import TestCase
import copy


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = torch.nn.ModuleList()
        self.mlp.append(torch.nn.Linear(10, 10))
        self.mlp.append(torch.nn.ReLU())
        self.mlp.append(torch.nn.Linear(10, 10))
        self.mlp.append(torch.nn.Sigmoid())

    def forward(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x


class TestLinearFuseEltwise(TestCase):
    def test_linear_fuse_eltwise(self):
        x1 = torch.rand(5, 10).requires_grad_()
        x2 = copy.deepcopy(x1)
        dtypes = [
            torch.float,
        ]
        if core.onednn_has_bf16_support():
            dtypes.append(torch.bfloat16)
        if core.onednn_has_fp16_support():
            dtypes.append(torch.float16)
        for dtype in dtypes:
            model = MLP()
            opt = torch.optim.SGD(model.parameters(), lr=0.01)
            model, opt = ipex.optimize(
                model, optimizer=opt, dtype=dtype, auto_kernel_selection=True
            )
            with torch.cpu.amp.autocast(
                enabled=(dtype in [torch.bfloat16, torch.float16]), dtype=dtype
            ):
                ref_out = model(x1).sum()
            ref_out.backward()

            fused_model = copy.deepcopy(model)
            fused_model.mlp[0] = ipex.nn.modules.IPEXLinearEltwise(
                fused_model.mlp[0], "relu"
            )
            fused_model.mlp[1] = torch.nn.Identity()
            fused_model.mlp[2] = ipex.nn.modules.IPEXLinearEltwise(
                fused_model.mlp[2], "sigmoid"
            )
            fused_model.mlp[3] = torch.nn.Identity()
            with torch.cpu.amp.autocast(
                enabled=(dtype in [torch.bfloat16, torch.float16]), dtype=dtype
            ):
                out = fused_model(x2).sum()
            out.backward()
            atol = None
            rtol = None
            if dtype == torch.float16:
                atol = 1e-4
                rtol = 1e-3
            self.assertEqual(out, ref_out)
            self.assertEqual(x1.grad, x2.grad, atol=atol, rtol=rtol)


if __name__ == "__main__":
    test = unittest.main()
