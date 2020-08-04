import torch
from torch.testing._internal.common_utils import TestCase
import torch_ipex
import time
import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")

class TestTorchMethod(TestCase):
    def test_batch_linear_algebra(self, dtype=torch.float):
        x_cpu = torch.randn(5, 5)

        x_dpcpp = x_cpu.to(dpcpp_device)
        #y_cpu1 = x_cpu.new_ones((2, 3))
        y_cpu1 = torch.randn(5, 5)
        #y_cpu2 = x_cpu.new_ones((2, 3))
        y_cpu2 = torch.randn(5, 5)

        y_dpcpp1 = y_cpu1.to(dpcpp_device)
        y_dpcpp2 = y_cpu2.to(dpcpp_device)

        print("y_cpu", torch.tril(y_cpu2))
        print("y_dpcpp", torch.tril(y_dpcpp2).to("cpu"))
        self.assertEqual(torch.tril(y_cpu2),
                         torch.tril(y_dpcpp2).to(cpu_device))

        print("y_cpu", torch.triu(y_cpu2))
        print("y_dpcpp", torch.triu(y_dpcpp2).to("cpu"))
        self.assertEqual(torch.triu(y_cpu2),
                         torch.triu(y_dpcpp2).to(cpu_device))

    @pytest.mark.skipif("not torch_ipex._onemkl_is_enabled()")
    def test_logdet(self, dtype=torch.float):
        ts = int(time.time())
        torch.manual_seed(ts)

        A = torch.randn(3, 3).to(cpu_device)
        
        a = torch.det(A)
        print("torch.det(A)", a.to(cpu_device))
        b = torch.logdet(A)
        print("torch.logdet(A)", b.to(cpu_device))

        print("A", A.to(cpu_device))
        A_det = A.det()
        print("A.det()", A_det.to(cpu_device))
        A_det_log = A.det().log()
        print("A.det().log()", A_det_log.to(cpu_device))
        
        ######
        A_dpcpp = A.to(dpcpp_device)
        
        a_dpcpp = torch.det(A_dpcpp)
        print("torch.det(A_dpcpp)", a_dpcpp.to(cpu_device))
        b_dpcpp = torch.logdet(A_dpcpp)
        print("torch.logdet(A_dpcpp)", b_dpcpp.to(cpu_device))
       
        print("A_dpcpp", A_dpcpp.to(cpu_device))
        A_dpcpp_det = A_dpcpp.det()
        print("A_dpcpp.det()", A_dpcpp_det.to(cpu_device))
        A_dpcpp_det_log = A_dpcpp.det().log()
        print("A_dpcpp.det().log()", A_dpcpp_det_log.to(cpu_device))

        ## asssert
        self.assertEqual(a, a_dpcpp.to(cpu_device))
        self.assertEqual(b, b_dpcpp.to(cpu_device))
        self.assertEqual(A.to(cpu_device), A_dpcpp.to(cpu_device))
        self.assertEqual(A_det.to(cpu_device), A_dpcpp_det.to(cpu_device))
        self.assertEqual(A_det_log.to(cpu_device), A_dpcpp_det_log.to(cpu_device))


