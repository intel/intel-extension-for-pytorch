import torch
import torch.nn as nn
from torch.nn.functional import relu_
from torch.nn.functional import relu
import ipex

def test_lazy_reorder():
    cpu_device = torch.device("cpu")
    dpcpp_device = torch.device("xpu")

    conv1 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)
    conv2 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)
    bn1 = nn.BatchNorm2d(2)
    bn2 = nn.BatchNorm2d(2)
    relu = nn.ReLU()
    max_pool = nn.MaxPool2d(kernel_size=3, stride=1,
                            padding=1, return_indices=True)

    x_cpu = torch.randn([1, 2, 3, 3], device=cpu_device)
    y_cpu = relu(bn1(conv1(x_cpu)))
    z_cpu = relu_(bn2(conv2(y_cpu + x_cpu)))
    a_cpu = z_cpu + y_cpu
    a_cpu, _ = max_pool(a_cpu)
    a_cpu.tanh_()
    print("cpu", a_cpu)

    conv1.to("xpu")
    conv2.to("xpu")
    bn1.to("xpu")
    bn2.to("xpu")
    x_dpcpp = x_cpu.to("xpu")
    print("iter-1 ...")
    y_dpcpp = relu(bn1(conv1(x_dpcpp)))
    z_dpcpp = relu_(bn2(conv2(y_dpcpp + x_dpcpp)))
    a_dpcpp = z_dpcpp + y_dpcpp
    a_dpcpp, _ = max_pool(a_dpcpp)
    a_dpcpp.tanh_()

    print("iter-2 ...")
    y_dpcpp = relu(bn1(conv1(x_dpcpp)))
    z_dpcpp = relu_(bn2(conv2(y_dpcpp + x_dpcpp)))
    a_dpcpp = z_dpcpp + y_dpcpp
    a_dpcpp, _ = max_pool(a_dpcpp)
    a_dpcpp.tanh_()
    print("xpu", a_dpcpp.to("cpu"))


if __name__ == "__main__":
    test_lazy_reorder()
