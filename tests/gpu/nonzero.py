import torch
import torch_ipex

dpcpp_device = torch.device("dpcpp")
cpu_device = torch.device("cpu")

#
# Test torch.nonzero on CPU device
#
a_cpu = torch.tensor([1, 1, 1, 0, 1], device = cpu_device)
b_cpu = torch.tensor([[0.6, 0.0, 0.0, 0.0],
        [0.0, 0.4, 0.0, 0.0],
        [0.0, 0.0, 1.2, 0.0],
        [0.0, 0.0, 0.0, -0.4]], device = cpu_device)

print("For Tensor:", a_cpu)
print("torch.nonzero on CPU returns", torch.nonzero(a_cpu))
print("\n")

print("For Tensor:", b_cpu)
print("torch.nonzero on CPU returns", torch.nonzero(b_cpu))
print("\n")

print("For Tensor:", a_cpu)
print("torch.nonzero with as_tuple TRUE on CPU returns", torch.nonzero(a_cpu, as_tuple=True))
print("\n")

print("For Tensor:", b_cpu)
print("torch.nonzero with as_tuple TRUE on CPU returns", torch.nonzero(b_cpu, as_tuple=True))
print("\n")


#
# Test torch.nonzero on SYCL device
#

a_dpcpp = torch.tensor([1, 1, 1, 0, 1], device = dpcpp_device)
b_dpcpp = torch.tensor([[0.6, 0.0, 0.0, 0.0],
        [0.0, 0.4, 0.0, 0.0],
        [0.0, 0.0, 1.2, 0.0],
        [0.0, 0.0, 0.0, -0.4]], device = dpcpp_device)

print("For Tensor:", a_dpcpp.to("cpu"))
print("torch.nonzero on SYCL returns", torch.nonzero(a_dpcpp).cpu())
print("\n")

print("For Tensor:", b_dpcpp.to("cpu"))
print("torch.nonzero on SYCL returns", torch.nonzero(b_dpcpp).cpu())
print("\n")

print("For Tensor:", a_dpcpp.to("cpu"))
print("torch.nonzero with as_tuple TRUE on SYCL returns", torch.nonzero(a_dpcpp, as_tuple=True)[0].cpu())
print("\n")

print("For Tensor:", b_dpcpp.to("cpu"))
r1, r2 = torch.nonzero(b_dpcpp, as_tuple=True)
print("torch.nonzero with as_tuple TRUE on SYCL returns", r1.cpu(), r2.cpu())
print("\n")


