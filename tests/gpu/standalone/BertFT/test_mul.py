import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

shapes = [
            (2, 1, 1, 384),
        ]

class TestTorchMethod(TestCase):
    def test_mul(self, dtype=torch.float):
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            
            batch1 = torch.randn(shape, device=cpu_device)
            #batch2 = torch.randn(shape[1], device=cpu_device)

            #
            # Test mum OP.
            #
            batch1_dpcpp = batch1.to(dpcpp_device)
            #batch2_dpcpp = batch2.to(dpcpp_device)
            print("torch.mul cpu", torch.mul(batch1, batch1))
            print("torch.mul dpcpp", torch.mul(batch1_dpcpp, batch1_dpcpp).to("cpu"))
            self.assertEqual(
                torch.mul(batch1, batch1),
                torch.mul(batch1_dpcpp, batch1_dpcpp).to(cpu_device),
            )
