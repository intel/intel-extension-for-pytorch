import torch
import torch.nn as nn
import torch_ipex
from memory_profiler import profile

@profile
def test_lazy_reorder():
    cpu_device = torch.device("cpu")
    dpcpp_device = torch.device("dpcpp")

    conv1 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)

    x_cpu = torch.randn([1, 2, 3, 3], device=cpu_device)
    y_cpu = conv1(x_cpu)
    z_cpu = conv1(y_cpu)
    z_cpu.tanh_()
    print("cpu", z_cpu)

    conv1.to(dpcpp_device)
    x_dpcpp = x_cpu.to(dpcpp_device)
    y_dpcpp = conv1(x_dpcpp)
    z_dpcpp = conv1(y_dpcpp)
    z_dpcpp.tanh_()
    print("dpcpp", z_dpcpp.to("cpu"))


if __name__ == "__main__":
    test_lazy_reorder()
