import torch
from torch.nn.functional import relu_
from torch.nn.functional import relu
from torch.testing._internal.common_utils import TestCase
import ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestNNMethod(TestCase):
    def test_SplitSGD(self):
        device='xpu'
        dtype_bf16=torch.bfloat16
        dtype_fp32=torch.float32

        # input and target
        input_cpu = torch.randn(1, 3, 224, 224, device='cpu', dtype=torch.float32, requires_grad=True)
        target_cpu = torch.randn(1, 64, 112, 112, device='cpu', dtype=torch.float32)

        # model
        m_cpu1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        m_cpu2 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        m_cpu2.weight.data = m_cpu1.weight.data.clone()

        # criterion fucntion
        c = torch.nn.MSELoss()

        # model
        m_dpcpp_fp32 = m_cpu1.to(device, dtype=dtype_fp32)
        m_dpcpp_bf16 = m_cpu2.to(device, dtype=dtype_bf16)

        # input
        input_dpcpp_bf16 = input_cpu.clone().to(device, dtype_bf16).requires_grad_()
        input_dpcpp_fp32 = input_cpu.clone().to(device, dtype_fp32).requires_grad_()

        # target
        target_dpcpp_bf16 = target_cpu.clone().to(device, dtype_bf16)
        target_dpcpp_fp32 = target_cpu.clone().to(device, dtype_fp32)

        # criterion
        c_dpcpp_bf16 = c.to(device)
        c_dpcpp_fp32 = c.to(device)

        # optim
        optim_dpcpp_bf16 = ipex.optim.SplitSGD(m_dpcpp_bf16.parameters(), lr=0.1, momentum=0, weight_decay=0)
        optim_dpcpp_fp32 = torch.optim.SGD(m_dpcpp_fp32.parameters(), lr=0.1, momentum=0, weight_decay=0)

        # forward
        output_dpcpp_bf16 = m_dpcpp_bf16(input_dpcpp_bf16)
        output_dpcpp_fp32 = m_dpcpp_fp32(input_dpcpp_fp32)

        # loss
        loss_bf16 = c_dpcpp_bf16(output_dpcpp_bf16, target_dpcpp_bf16)
        loss_fp32 = c_dpcpp_fp32(output_dpcpp_fp32, target_dpcpp_fp32)

        # optim
        optim_dpcpp_bf16.zero_grad()
        optim_dpcpp_fp32.zero_grad()

        loss_bf16.backward()
        loss_fp32.backward()

        optim_dpcpp_bf16.step()
        optim_dpcpp_fp32.step()

        self.assertEqual(m_dpcpp_bf16.weight.cpu().float(), m_dpcpp_fp32.weight.cpu(), atol=1e-3, rtol=1.3e-06)
