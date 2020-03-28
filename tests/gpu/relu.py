import torch
import torch.nn.functional
import torch_ipex

relu_ = torch.nn.functional.relu_
relu = torch.nn.functional.relu
x_cpu = torch.tensor([[-0.1, 0.2],[-0.2, 0.3],[0.4, 0.5],[0.5, -0.6]]);
x_dpcpp = x_cpu.to("dpcpp")

relu_(x_cpu)
relu_(x_dpcpp)
print("cpu relu_ ", x_cpu)
print("dpcpp relu_ ", x_dpcpp.cpu())

x_cpu.requires_grad_(True)
x_dpcpp.requires_grad_(True)
y_cpu = relu(x_cpu)
y_dpcpp = relu(x_dpcpp)
print("cpu relu ", y_cpu)
print("dpcpp relu ", y_dpcpp.cpu())

y_cpu.backward(x_cpu)
y_dpcpp.backward(y_dpcpp)

print("cpu relu bwd", x_cpu.grad)
print("dpcpp relu bwd", x_dpcpp.grad.cpu())

