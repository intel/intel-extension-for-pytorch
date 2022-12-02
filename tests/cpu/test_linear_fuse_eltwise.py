import unittest
import torch
import intel_extension_for_pytorch as ipex
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
        for dtype in [torch.float, torch.bfloat16]:
            model = MLP()
            opt = torch.optim.SGD(model.parameters(), lr=0.01)
            model, opt = ipex.optimize(model, optimizer=opt, dtype=dtype, auto_kernel_selection=True)
            with torch.cpu.amp.autocast(enabled=(dtype == torch.bfloat16)):
                ref_out = model(x1).sum()
            ref_out.backward()

            fused_model = copy.deepcopy(model)
            fused_model.mlp[0] = ipex.nn.modules.IPEXLinearEltwise(fused_model.mlp[0], 'relu')
            fused_model.mlp[1] = torch.nn.Identity()
            fused_model.mlp[2] = ipex.nn.modules.IPEXLinearEltwise(fused_model.mlp[2], 'sigmoid')
            fused_model.mlp[3] = torch.nn.Identity()
            with torch.cpu.amp.autocast(enabled=(dtype == torch.bfloat16)):
                out = fused_model(x2).sum()
            out.backward()
            self.assertEqual(out, ref_out)
            self.assertEqual(x1.grad, x2.grad)

if __name__ == '__main__':
    test = unittest.main()
