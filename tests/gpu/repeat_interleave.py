import torch
import torch_ipex

a = torch.tensor([1, 2, 3])
print("For tensor:")
print(a)

print("CPU result:")
print(a.repeat_interleave(2))

a_dpcpp = a.to("dpcpp")
b_dpcpp = a_dpcpp.repeat_interleave(2)
print("[1] SYCL result using tensor.repeat_interleave:")
print(b_dpcpp.to("cpu"))




a = torch.tensor([2, 1, 1])
print("For tensor:")
print(a)

print("CPU result:")
print(torch.repeat_interleave(a, 2))

a_dpcpp = a.to("dpcpp")
b_dpcpp = torch.repeat_interleave(a_dpcpp, 2)
print("[2] SYCL result using torch.repeat_interleave:")
print(b_dpcpp.to("cpu"))
