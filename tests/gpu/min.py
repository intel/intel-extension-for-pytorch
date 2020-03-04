import torch
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")

#
# Test minall OP.

y = torch.randn(3, 3)

y_dpcpp = y.to("dpcpp")
z_dpcpp = torch.min(y_dpcpp)

print("Testing minall OP!\n")
print("For Tensor:", y)
print("torch.min on cpu returns", torch.min(y))
print("torch.min on dpcpp device returns", z_dpcpp.to("cpu"))
print("\n")

#
# Test cmin OP.
#
print("Testing cmin OP!\n")
c_dpcpp = torch.randn(4).to("dpcpp")
d_dpcpp = torch.tensor([[ 0.0120, -0.9505, -0.3025, -1.4899]], device = dpcpp_device)
e_dpcpp = torch.min(c_dpcpp, d_dpcpp)

print("For Tensor:", c_dpcpp.to("cpu"))
print("For Tensor:", d_dpcpp.to("cpu"))
print("torch.min on cpu returns", torch.min(c_dpcpp.to("cpu"), d_dpcpp.to("cpu")))
print("torch.min on dpcpp device returns", e_dpcpp.to("cpu"))
print("\n")

#
# Test min OP.
#
print("Testing min OP!\n")
a_dpcpp = torch.randn((9, 5)).to("dpcpp")

print("1-test (-2) dim")
a_cpu = a_dpcpp.to("cpu")
b_dpcpp, b_dpcpp_index = a_dpcpp.min(-2)

print("For Tensor:", a_cpu)
print("torch.min on cpu returns", torch.min(a_cpu, -2))
print("torch.min on dpcpp device returns", b_dpcpp.to("cpu"), b_dpcpp_index.to("cpu"))
print("\n")

print("2-test (-1) dim")
a_cpu = a_dpcpp.to("cpu")
b_dpcpp, b_dpcpp_index = a_dpcpp.min(-1)

print("For Tensor:", a_cpu)
print("torch.min on cpu returns", torch.min(a_cpu, -1))
print("torch.min on dpcpp device returns", b_dpcpp.to("cpu"), b_dpcpp_index.to("cpu"))
print("\n")

print("3-test (0) dim")
a_cpu = a_dpcpp.to("cpu")
b_dpcpp, b_dpcpp_index = torch.min(a_dpcpp, 0)

print("For Tensor:", a_cpu)
print("torch.min on cpu returns", torch.min(a_cpu, 0))
print("torch.min on dpcpp device returns", b_dpcpp.to("cpu"), b_dpcpp_index.to("cpu"))
print("\n")


print("4-test (1) dim")
a_cpu = a_dpcpp.to("cpu")
b_dpcpp, b_dpcpp_index = torch.min(a_dpcpp, 1)

print("For Tensor:", a_cpu)
print("torch.min on cpu returns", torch.min(a_cpu, 1))
print("torch.min on dpcpp device returns", b_dpcpp.to("cpu"), b_dpcpp_index.to("cpu"))
print("\n")
