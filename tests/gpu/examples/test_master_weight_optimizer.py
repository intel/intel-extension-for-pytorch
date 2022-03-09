import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

import pytest


class Conv2dRelu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dRelu, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.conv2 = nn.Conv2d(out_channels, in_channels, **kwargs)

    def forward(self, x):
        y = F.relu(self.conv1(x), inplace=True)
        y = F.relu(self.conv2(y), inplace=True)
        return y


class TestNNMethod(TestCase):
    @pytest.mark.skipif(torch.xpu.using_onednn_layout(), reason="test_SGDMasterWeight does not support onednn block format")
    def test_SGDMasterWeight(self, dtype=torch.float):
        model_real = Conv2dRelu(2, 2, kernel_size=3, stride=1, padding=1, bias=True)
        model_ref = Conv2dRelu(2, 2, kernel_size=3, stride=1, padding=1, bias=True)
        p_real_list = list(model_real.parameters())
        p_ref_list = list(model_ref.parameters())
        for i in range(len(p_ref_list)):
            p_real_list[i].data = p_ref_list[i].data.clone()

        print("start ...")
        print("ref model ...")
        print(list(model_ref.parameters()))
        print("real model ...")
        print(list(model_real.parameters()))

        model_real.train()
        model_ref.train()

        optimizer_real = torch.xpu.optim.SGDMasterWeight(model_real.parameters(), 0.01, 0.9, 0.0001)
        optimizer_ref = torch.optim.SGD(model_ref.parameters(), 0.01, 0.9, 0.0001)

        model_real.bfloat16()
        model_ref.float()

        model_ref.to("cpu")
        model_real.to("xpu")

        x_refs = [torch.randn([1, 2, 3, 3]).requires_grad_(),
                  torch.randn([1, 2, 3, 3]).requires_grad_()]
        x_reals = [x_refs[0].bfloat16().to("xpu").requires_grad_(),
                   x_refs[1].bfloat16().to("xpu").requires_grad_()]
        gy_refs = [torch.randn([1, 2, 3, 3]),
                   torch.randn([1, 2, 3, 3])]
        gy_reals = [gy_refs[0].bfloat16().to("xpu"),
                    gy_refs[1].bfloat16().to("xpu")]

        print("ref training ...")
        for i in range(2):
            y = model_ref(x_refs[i])
            optimizer_ref.zero_grad(set_to_none=True)
            y.backward(gy_refs[i])
            optimizer_ref.step()
        print("ref model ...")
        print(list(model_ref.parameters()))

        print("real training ...")
        for i in range(2):
            y = model_real(x_reals[i])
            optimizer_real.zero_grad(set_to_none=True)
            y.backward(gy_reals[i])
            optimizer_real.step()
        print("real model ...")
        print(list(model_real.cpu().float().parameters()))

        p_real_list = list(model_real.parameters())
        p_ref_list = list(model_ref.parameters())
        for i in range(len(p_ref_list)):
            self.assertEqual(p_real_list[i], p_ref_list[i], atol=1e-3, rtol=1.3e-04)

    # test official torch AdamW
    @pytest.mark.skipif(torch.xpu.using_onednn_layout(), reason="test_AdamMasterWeight does not support onednn block format")
    def test_AdamMasterWeight(self, dtype=torch.float):
        model_real = Conv2dRelu(2, 2, kernel_size=3, stride=1, padding=1, bias=True)
        model_ref = Conv2dRelu(2, 2, kernel_size=3, stride=1, padding=1, bias=True)
        p_real_list = list(model_real.parameters())
        p_ref_list = list(model_ref.parameters())
        for i in range(len(p_ref_list)):
            p_real_list[i].data = p_ref_list[i].data.clone()

        print("start ...")
        print("ref model ...")
        print(list(model_ref.parameters()))
        print("real model ...")
        print(list(model_real.parameters()))

        model_real.train()
        model_ref.train()

        optimizer_real = torch.xpu.optim.AdamMasterWeight(
            model_real.parameters(),
            0.001,
            weight_decay=0.95
        )
        optimizer_ref = torch.optim.Adam(
            model_ref.parameters(),
            0.001,
            weight_decay=0.95
        )

        model_real.bfloat16()
        model_ref.float()

        model_ref.to("cpu")
        model_real.to("xpu")

        x_refs = [torch.randn([1, 2, 3, 3]).requires_grad_(),
                  torch.randn([1, 2, 3, 3]).requires_grad_()]
        x_reals = [x_refs[0].bfloat16().to("xpu").requires_grad_(),
                   x_refs[1].bfloat16().to("xpu").requires_grad_()]
        gy_refs = [torch.randn([1, 2, 3, 3]),
                   torch.randn([1, 2, 3, 3])]
        gy_reals = [gy_refs[0].bfloat16().to("xpu"),
                    gy_refs[1].bfloat16().to("xpu")]

        print("ref training ...")
        for i in range(2):
            y = model_ref(x_refs[i])
            optimizer_ref.zero_grad(set_to_none=True)
            y.backward(gy_refs[i])
            optimizer_ref.step()
        print("ref model ...")
        print(list(model_ref.parameters()))

        print("real training ...")
        for i in range(2):
            y = model_real(x_reals[i])
            optimizer_real.zero_grad(set_to_none=True)
            y.backward(gy_reals[i])
            optimizer_real.step()
        print("real model ...")
        print(list(model_real.cpu().float().parameters()))

        p_real_list = list(model_real.parameters())
        p_ref_list = list(model_ref.parameters())
        for i in range(len(p_ref_list)):
            self.assertEqual(p_real_list[i], p_ref_list[i], atol=1e-3, rtol=1.3e-04)
