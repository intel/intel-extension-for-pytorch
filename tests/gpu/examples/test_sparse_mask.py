import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch


class TestTorchMethod(TestCase):
    def test_sparse_mask(self, dtype=torch.float):
        nse = 5
        dims = (5, 5, 2, 2)
        I = torch.cat([torch.randint(0, dims[0], size=(nse,)),
                      torch.randint(0, dims[1], size=(nse,))], 0).reshape(2, nse)
        V = torch.randn(nse, dims[2], dims[3])
        S = torch._sparse_coo_tensor_unsafe(I, V, dims).coalesce()
        S_xpu = S.to("xpu")
        D = torch.randn(dims)
        D_xpu = D.to("xpu")
        output = D.sparse_mask(S)
        output_xpu = D_xpu.sparse_mask(S_xpu)
        print(output)
        print(output_xpu.cpu())
        self.assertEqual(output, output_xpu.cpu())
