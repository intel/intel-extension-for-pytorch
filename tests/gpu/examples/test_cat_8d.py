import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_cat_8d(self, dtype=torch.float):
        input1 = torch.randn([256, 8, 8, 3, 3, 3, 3])
        input2 = torch.randn([256, 8, 8, 3, 3, 3, 3])
        input1_xpu = input1.to(dpcpp_device)
        input2_xpu = input2.to(dpcpp_device)
        output1 = torch.stack([input1, input2], dim=0)
        output1_xpu = torch.stack([input1_xpu, input2_xpu], dim=0)
        output2 = output1.reshape([2, 256, 8, 8, 9, 9])
        output2_xpu = output1_xpu.reshape([2, 256, 8, 8, 9, 9])
        output3 = torch.stack([output2, output2], dim=0)
        output3_xpu = torch.stack([output2_xpu, output2_xpu], dim=0)
        self.assertEqual(output3, output3.cpu())
