import torch
import torch_ipex
cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")
t = torch.tensor([[1,2],[3,4]], device = cpu_device)
t_cpu = torch.gather(t, 1, torch.tensor([[0,0],[1,0]], device = cpu_device))

print("cpu")
print(t_cpu)

t2 = torch.tensor([[1,2],[3,4]], device = sycl_device)
t_sycl = torch.gather(t2, 1, torch.tensor([[0,0],[1,0]], device = sycl_device))


print("dpcpp")
print(t_sycl.cpu())

