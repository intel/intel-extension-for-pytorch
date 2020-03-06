import torch
import torch_ipex

x_cpu = torch.tensor([[-0.2911, -1.3204,  -2.6425,  -2.4644,  -0.6018, -0.0839, -0.1322, -0.4713, -0.3586, -0.8882]], device=torch.device("cpu"), dtype=torch.float)
x_dpcpp = torch.tensor([[-0.2911, -1.3204,  -2.6425,  -2.4644,  -0.6018, -0.0839, -0.1322, -0.4713, -0.3586, -0.8882]], device=torch.device("dpcpp"), dtype=torch.float)

print("y = ", torch.kthvalue(x_cpu, 4))
y = torch.kthvalue(x_dpcpp, 4)
print("y_dpcpp = ", y[0].to("cpu"), y[1].to("cpu"))

print("y = ", torch.kthvalue(x_cpu.resize_(2, 5), 1))
y = torch.kthvalue(x_dpcpp.resize_(2, 5), 1)
print("y_dpcpp = ", y[0].to("cpu"), y[1].to("cpu"))
