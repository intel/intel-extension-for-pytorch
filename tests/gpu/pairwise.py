import torch
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")
x_cpu = torch.tensor([[1,2,3,4,5]], dtype=torch.int, device = cpu_device)
y_cpu = torch.tensor([[1,1,3,3,5]], dtype=torch.int, device = cpu_device)
x_dpcpp = torch.tensor([[1,2,3,4,5]], dtype=torch.int, device = dpcpp_device)
y_dpcpp = torch.tensor([[1,1,3,3,5]], dtype=torch.int, device = dpcpp_device)

print("__and__ y_cpu", x_cpu.__and__(3))
print("__and__ y_dpcpp", x_dpcpp.__and__(3).to("cpu"))

print("__iand__ y_cpu", x_cpu.__iand__(3))
print("__iand__ y_dpcpp", x_dpcpp.__iand__(3).to("cpu"))

print("__or__ y_cpu", x_cpu.__or__(3))
print("__or__ y_dpcpp", x_dpcpp.__or__(3).to("cpu"))

print("__ior__ y_cpu", x_cpu.__ior__(3))
print("__ior__ y_dpcpp", x_dpcpp.__ior__(3).to("cpu"))
