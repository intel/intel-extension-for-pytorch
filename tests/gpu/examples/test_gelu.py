import torch
import torch.nn.functional
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa
import copy

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestNNMethod(TestCase):
  def test_activation_gelu(self, dtype=torch.bfloat16):
      approximates = ["tanh", "none"]
      for approximate in approximates:
        #   GELU = torch.nn.GELU(approximate="none")
        GELU = torch.nn.GELU(approximate=approximate)
        GELU_dpcpp = copy.deepcopy(GELU).to("xpu")
        x_cpu = torch.tensor(
            [[-0.1, 0.2], [-0.2, 0.3], [0.4, 0.5], [0.5, -0.6]])
        x_dpcpp = x_cpu.to("xpu")
        x_cpu.requires_grad_(True)
        x_dpcpp.requires_grad_(True)
        y_cpu = GELU(x_cpu)
        y_dpcpp = GELU_dpcpp(x_dpcpp)
        print("cpu gelu ", y_cpu)
        print("dpcpp gelu ", y_dpcpp.cpu())
        self.assertEqual(y_cpu, y_dpcpp.cpu())

        # y_cpu = torch.tensor([[1, 1],[1, 1],[1, 1],[1, 1]]);
        # y_dpcpp = y_cpu.to("xpu")
        y_cpu.backward(x_cpu)
        y_dpcpp.backward(x_dpcpp)

        print("cpu gelu bwd", x_cpu.grad)
        print("dpcpp gelu bwd", x_dpcpp.grad.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())


  def test_activation_gelu_block(self, dtype=torch.float):
      to_block_cpu = torch.nn.Conv2d(4, 4, kernel_size=3, padding=1)
      # print("--------to_block_cpu------", to_block_cpu)
      to_block_dpcpp = copy.deepcopy(to_block_cpu).xpu()
      test_shape = [1, 4, 3, 3]
      with torch.xpu.onednn_layout():
          # GELU = torch.nn.GELU()
          # GELU = torch.nn.GELU(approximate="none")
          GELU = torch.nn.GELU(approximate="tanh")
          GELU_dpcpp = copy.deepcopy(GELU).to("xpu")
          x_cpu = torch.randn(test_shape)
          x_dpcpp = x_cpu.to("xpu")
          x_cpu.requires_grad_(True)
          x_dpcpp.requires_grad_(True)
          # print("=======x_cpu========", x_cpu)
          # print("=======blk conv cpu========", to_block_cpu(x_cpu))          
          # print("=======x_dpcpp========", x_dpcpp)
          # print("=======blk conv dpcpp========", to_block_dpcpp(x_dpcpp))
          input_cpu = to_block_cpu(x_cpu)
          input_dpcpp = input_cpu.to('xpu')
          y_cpu = GELU(input_cpu)
          y_dpcpp = GELU_dpcpp(input_dpcpp)
          print("cpu gelu ", y_cpu)
          print("dpcpp gelu ", y_dpcpp.cpu())
          self.assertEqual(y_cpu, y_dpcpp.cpu())
          # y_cpu.backward(x_cpu)
          # y_dpcpp.backward(x_dpcpp)
          # print("cpu gelu bwd", x_cpu.grad)
          # print("dpcpp gelu bwd", x_dpcpp.grad.cpu())
          # self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

