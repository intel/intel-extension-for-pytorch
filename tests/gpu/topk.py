import numpy
import torch
import torch_ipex

#x_cpu = torch.arange(-6., 6.) 
x_cpu = torch.tensor([[-0.2911, -1.3204,  -2.6425,  -2.4644,  -0.6018, -0.0839, -0.1322, -0.4713, -0.3586, -0.8882]], device=torch.device("cpu"), dtype=torch.float)


#x_cpu = torch.ones([2], dtype=torch.float)
#x_cpu[0] = 1.0
#x_cpu[1] = -1.0
x_sycl = x_cpu.to("dpcpp")

y_cpu = torch.topk(x_cpu, 2)
 
print("x: ", x_cpu)
print("y: ", y_cpu)
print("x_sycl.dim", x_sycl.dim())
y_sycl, y_sycl_idx = torch.topk(x_sycl, 2)
print("y_sycl: ", y_sycl.cpu(), y_sycl_idx.cpu())

print("==================================")

#x_cpu1 = torch.randn([1, 10], device=torch.device("cpu"), dtype=torch.float)
x_cpu1 =  torch.tensor([[-0.2911, -1.3204,  -2.6425,  -2.4644,  -0.6018, -0.0839, -0.1322, -0.4713, -0.3586, -0.8882]], device=torch.device("cpu"), dtype=torch.float)
x_sycl1 = x_cpu1.to("dpcpp")

print("x_cpu1=", x_cpu1)
y_cpu0, y_cpu1 = x_cpu1.topk(5, 1, True, True)

y_sycl0, y_sycl1 = x_sycl1.topk(5, 1, True, True)

print("y_cpu0 = ", y_cpu0, "y_cpu1 = ", y_cpu1)
print("y_sycl0 = ", y_sycl0.to("cpu"), "y_sycl1 = ", y_sycl1.to("cpu"))


x_cpu1 = torch.randn([3000, 3000], device=torch.device("cpu"), dtype=torch.float)
x_sycl1 = x_cpu1.to("dpcpp")

y_sycl0, y_sycl1 = x_sycl1.topk(5, 1, True, True)

print("y_sycl0 = ", y_sycl0.to("cpu"), "y_sycl1 = ", y_sycl1.to("cpu"))
