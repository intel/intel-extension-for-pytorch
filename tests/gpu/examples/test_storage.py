import torch
from torch.testing._internal.common_utils import TestCase
import ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_storage_float(self, dtype=torch.float):
        x = torch.ones(10,device=dpcpp_device,dtype=dtype)
        print(x.storage())

        module=torch.nn.Conv2d(16,16,32,32).to(dpcpp_device)
        torch.save(module, "./temp.pt")
        module2 = torch.load("./temp.pt")


    def test_storage_bfloat(self, dtype=torch.bfloat16):
        x = torch.ones(10,device=dpcpp_device,dtype=dtype)
        print(x.storage())

    def test_storage_int(self, dtype=torch.int):
        x = torch.ones(10,device=dpcpp_device,dtype=dtype)
        print(x.storage())

    def test_storage_bool(self, dtype=torch.bool):
        x = torch.ones(10,device=dpcpp_device,dtype=dtype)
        print(x.storage())

    def test_storage_double(self, dtype=torch.double):
        x = torch.ones(10,device=dpcpp_device,dtype=dtype)
        print(x.storage())

    def test_storage_half(self, dtype=torch.half):
        x = torch.ones(10,device=dpcpp_device,dtype=dtype)
        print(x.storage())

    def test_storage_long(self, dtype=torch.long):
        x = torch.ones(10,device=dpcpp_device,dtype=dtype)
        print(x.storage())
        
