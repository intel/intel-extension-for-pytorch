import torch
import torch_ipex
x_cpu = torch.tensor([[-0.2911, -1.3204,  -2.6425,  -2.4644,  -0.6018, -0.0839, -0.1322, -0.4713, -0.3586, -0.8882]], device=torch.device("cpu"), dtype=torch.float)
x_sycl = torch.tensor([[-0.2911, -1.3204,  -2.6425,  -2.4644,  -0.6018, -0.0839, -0.1322, -0.4713, -0.3586, -0.8882]], device=torch.device("dpcpp"), dtype=torch.float)

print("abs(cpu)", x_cpu.abs())
print("abs(sycl)", x_sycl.abs().to("cpu"))

