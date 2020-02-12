import torch
import torch_ipex

import torch.nn as nn
cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

#torch.bitwise_not(torch.tensor([-1, -2, 3], dtype=torch.int8))

input = torch.tensor([-1, 1], device=torch.device("cpu"), dtype=torch.int8)
result = torch.bitwise_not(input)
result2 = torch.logical_not(input)
input1=torch.randn(5)
result3 = torch.neg(input1)

print(input)
print(input1)
print("Testing bitwise_not on", input)
print("CPU result:")
print(result)
print(result2)
print(result3)

print("SYCL result:")
#print("--torch.bitwise_not--")
#input_sycl = torch.tensor([-1, 2], device=sycl_device)
#result_sycl1 = torch.bitwise_not(input_sycl)
#print(result_sycl1.to("cpu"))

#y_ref = torch.bitwise_not(input)
#print(y_ref)
x = input.to("dpcpp")
y = torch.bitwise_not(x)
y2= torch.logical_not(x)
y3= torch.neg(input1.to("dpcpp"))
print(y.to("cpu"))
print(y2.to("cpu"))
print(y3.to("cpu"))