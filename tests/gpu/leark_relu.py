import numpy
import torch
import torch.nn as nn
import torch_ipex
from torch.autograd import Variable

dtype = torch.float32
cpu_device = torch.device("cpu")

# functionality
x_cpu = torch.randn([3, 4], device=cpu_device, dtype=dtype, requires_grad=True)
grad_x = torch.randn(3, 4, device=cpu_device, dtype=dtype, requires_grad=True)


def test_Xelu(x_cpu, grad_x, Xelu):
    y_cpu = Xelu(x_cpu)
    y_cpu.backward(grad_x)
    
    print("cpu output ", y_cpu)
    print("cpu grad ", x_cpu.grad)
    
    
    Xelu.to("dpcpp")
    Xelu.zero_grad()
    
    x_dpcpp = Variable(x_cpu.to("dpcpp"), requires_grad=True)
    grad_dpcpp = Variable(grad_x.to("dpcpp"), requires_grad=True)
    
    y_dpcpp = Xelu(x_dpcpp)
    y_dpcpp.backward(grad_dpcpp)
    
    print("dpcpp output", y_dpcpp.cpu())
    print("dpcpp grad ", x_dpcpp.grad.cpu())


test_Xelu(x_cpu, grad_x, nn.LeakyReLU(0.1))

