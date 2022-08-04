import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

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

def base_optimizer(model_real, model_ref, mem_format, is_sgd):
    print("start ...")
    print("ref model ...")
    print(list(model_ref.parameters()))
    print("real model ...")
    print(list(model_real.parameters()))

    model_ref.to("cpu")
    model_real.to("xpu")
    model_ref = model_ref.to(memory_format=mem_format)
    model_real = model_real.to(memory_format=mem_format)

    p_real_list = list(model_real.parameters())
    p_ref_list = list(model_ref.parameters())
    if is_sgd:
        optimizer_real = torch.xpu.optim.SGDMasterWeight(
            model_real.parameters(), 0.01, 0.9, 0.0001)
        optimizer_ref = torch.optim.SGD(
            model_ref.parameters(), 0.01, 0.9, 0.0001)
    else:
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

    x = [torch.randn([1, 2, 3, 3]),
         torch.randn([1, 2, 3, 3])]
    gy = [torch.randn([1, 2, 3, 3]),
          torch.randn([1, 2, 3, 3])]
    x_refs = [x[0].clone().to(memory_format=mem_format).requires_grad_(),
              x[1].clone().to(memory_format=mem_format).requires_grad_()]
    x_reals = [x[0].clone().bfloat16().to("xpu").to(memory_format=mem_format).requires_grad_(),
               x[1].clone().bfloat16().to("xpu").to(memory_format=mem_format).requires_grad_()]
    gy_refs = [gy[0].clone().to(memory_format=mem_format),
               gy[1].clone().to(memory_format=mem_format)]
    gy_reals = [gy[0].clone().bfloat16().to("xpu").to(memory_format=mem_format),
                gy[1].clone().bfloat16().to("xpu").to(memory_format=mem_format)]

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

class TestNNMethod(TestCase):
    @pytest.mark.skipif(torch.xpu.using_onednn_layout(), reason="test_SGDMasterWeight does not support onednn block format")
    def test_MasterWeight(self, dtype=torch.float):
        model_real = Conv2dRelu(2, 2, kernel_size=3, stride=1, padding=1, bias=True)
        model_ref = Conv2dRelu(2, 2, kernel_size=3, stride=1, padding=1, bias=True)
        p_real_list = list(model_real.parameters())
        p_ref_list = list(model_ref.parameters())
        for i in range(len(p_ref_list)):
            p_real_list[i].data = p_ref_list[i].data.clone()
        model_real.train()
        model_ref.train()

        format_list = [torch.contiguous_format, torch.channels_last]
        is_sgd = [True, False]
        for i, mem_format in enumerate(format_list):
            for j, flag in enumerate(is_sgd):
                if flag:
                    print("Test optimizer SGD, format = {}".format(mem_format))
                else:
                    print("Test optimizer Adam, format = {}".format(mem_format))

                base_optimizer(model_real, model_ref, mem_format, flag)
                p_real_list = list(model_real.parameters())
                p_ref_list = list(model_ref.parameters())
                for i in range(len(p_ref_list)):
                    self.assertEqual(p_real_list[i], p_ref_list[i], atol=1e-3, rtol=1.3e-04)
