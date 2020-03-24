import torch
import torch.nn.functional
import torch_ipex

relu_ = torch.nn.functional.relu_
relu = torch.nn.functional.relu
x_cpu = torch.tensor([[-0.1, 0.2],[-0.2, 0.3],[0.4, 0.5],[0.5, -0.6]]);
x_dpcpp = x_cpu.to("dpcpp")

print("cpu relu ", relu(x_cpu))
print("dpcpp relu ", relu(x_dpcpp).cpu())

relu_(x_cpu)
relu_(x_dpcpp)
print("cpu relu_ ", x_cpu)
print("dpcpp relu_ ", x_dpcpp.cpu())
