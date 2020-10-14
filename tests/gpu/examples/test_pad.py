import torch
from torch.nn import functional as F
from torch.testing._internal.common_utils import TestCase
import torch_ipex


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


class TestNNMethod(TestCase):
    def test_pad(self, dtype=torch.float):
        print("cpu")
        t4d = torch.empty(3, 3, 4, 2)
        p1d = (1, 1)  # pad last dim by 1 on each side
        out1 = F.pad(t4d, p1d, "constant", 0)
        # print(out.size())
        t4d = torch.empty(3, 3, 4, 2)
        p2d = (1, 1, 2, 2)  # pad last dim by (1, 1) and 2nd to last by (2, 2)
        out2 = F.pad(t4d, p2d, "constant", 0)
        # print(out.size())
        t4d = torch.empty(3, 3, 4, 2)
        p3d = (0, 1, 2, 1, 3, 3)  # pad by (0, 1), (2, 1), and (3, 3)
        out3 = F.pad(t4d, p3d, "constant", 0)
        # print(out.size())

        print("dpcpp")
        t4d = torch.empty(3, 3, 4, 2).to("dpcpp")
        p1d = (1, 1)  # pad last dim by 1 on each side
        out1_dpcpp = F.pad(t4d, p1d, "constant", 0)
        # print(out.size())
        t4d = torch.empty(3, 3, 4, 2).to("dpcpp")
        p2d = (1, 1, 2, 2)  # pad last dim by (1, 1) and 2nd to last by (2, 2)
        out2_dpcpp = F.pad(t4d, p2d, "constant", 0)
        # print(out.size())
        t4d = torch.empty(3, 3, 4, 2).to("dpcpp")
        p3d = (0, 1, 2, 1, 3, 3)  # pad by (0, 1), (2, 1), and (3, 3)
        out3_dpcpp = F.pad(t4d, p3d, "constant", 0)
        # print(out.size())
        self.assertEqual(out1.size(), out1_dpcpp.cpu().size())
        self.assertEqual(out2.size(), out2_dpcpp.cpu().size())
        self.assertEqual(out3.size(), out3_dpcpp.cpu().size())
