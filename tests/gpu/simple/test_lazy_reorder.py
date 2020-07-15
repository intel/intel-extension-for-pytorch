import torch
import torch.nn as nn
from torch.nn.functional import relu_
from torch.nn.functional import relu
import torch_ipex
from memory_profiler import profile

@profile
def test_lazy_reorder():
    cpu_device = torch.device("cpu")
    dpcpp_device = torch.device("dpcpp")

    conv1 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)
    conv2 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)
    bn1 = nn.BatchNorm2d(2)
    bn2 = nn.BatchNorm2d(2)
    relu = nn.ReLU()

    x_cpu = torch.randn([1, 2, 3, 3], device=cpu_device)
    y_cpu = relu(bn1(conv1(x_cpu)))
    z_cpu = relu_(bn2(conv2(y_cpu)))
    z_cpu.tanh_()
    print("cpu", z_cpu)

    conv1.to("dpcpp")
    conv2.to("dpcpp")
    bn1.to("dpcpp")
    bn2.to("dpcpp")
    x_dpcpp = x_cpu.to("dpcpp")
    print("iter-1 ...")
    y_dpcpp = relu(bn1(conv1(x_dpcpp)))
    z_dpcpp = relu_(bn2(conv2(y_dpcpp)))

    print("iter-2 ...")
    y_dpcpp = relu(bn1(conv1(x_dpcpp)))
    z_dpcpp = relu_(bn2(conv2(y_dpcpp)))

    z_dpcpp.tanh_()
    print("dpcpp", z_dpcpp.to("cpu"))


if __name__ == "__main__":
    test_lazy_reorder()
