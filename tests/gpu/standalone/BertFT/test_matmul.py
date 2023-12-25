import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

shapes = [
            ((2, 16, 384, 64), (2, 16, 64, 384)),
            ((2, 16, 384, 384), (2, 16, 384, 64)),
            ]

class TestTorchMethod(TestCase):
    def test_matmul(self, dtype=torch.float):
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            
            batch1 = torch.randn(shape[0], device=cpu_device, dtype=dtype)
            batch2 = torch.randn(shape[1], device=cpu_device, dtype=dtype)

            batch1_dpcpp = batch1.to(dpcpp_device)
            batch2_dpcpp = batch2.to(dpcpp_device)
            print("torch.matmul cpu", torch.matmul(batch1, batch2))
            print("torch.matmul dpcpp", torch.matmul(batch1_dpcpp, batch2_dpcpp).to("cpu"))
            #print("tensor.bmm dpcpp", batch1_dpcpp.bmm(batch2_dpcpp).to("cpu"))
            self.assertEqual(
                torch.matmul(batch1, batch2),
                torch.matmul(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
            )

    def test_matmul_bfloat16(self, dtype=torch.bfloat16):
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            
            batch1 = torch.randn(shape[0], device=cpu_device, dtype=dtype)
            batch2 = torch.randn(shape[1], device=cpu_device, dtype=dtype)

            batch1_dpcpp = batch1.to(dpcpp_device)
            batch2_dpcpp = batch2.to(dpcpp_device)
            print("torch.matmul cpu", torch.matmul(batch1, batch2))
            print("torch.matmul dpcpp", torch.matmul(batch1_dpcpp, batch2_dpcpp).to("cpu"))
            #print("tensor.bmm dpcpp", batch1_dpcpp.bmm(batch2_dpcpp).to("cpu"))
            self.assertEqual(
                torch.matmul(batch1, batch2),
                torch.matmul(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
            )

    def test_matmul_float16(self, dtype=torch.bfloat16):
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            
            batch1 = torch.randn(shape[0], device=cpu_device, dtype=dtype)
            batch2 = torch.randn(shape[1], device=cpu_device, dtype=dtype)

            dtype_dpcpp = torch.float16
            batch1_dpcpp = batch1.to(dpcpp_device).to(dtype_dpcpp)
            batch2_dpcpp = batch2.to(dpcpp_device).to(dtype_dpcpp)
            print("torch.matmul cpu", torch.matmul(batch1, batch2))
            print("torch.matmul dpcpp", torch.matmul(batch1_dpcpp, batch2_dpcpp).to("cpu"))
            #print("tensor.bmm dpcpp", batch1_dpcpp.bmm(batch2_dpcpp).to("cpu"))
            self.assertEqual(
                torch.matmul(batch1, batch2),
                torch.matmul(batch1_dpcpp, batch2_dpcpp).to(cpu_device).to(torch.bfloat16),
            )
