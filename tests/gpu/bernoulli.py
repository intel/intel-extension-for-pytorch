import numpy
import torch
import torch.nn as nn
import torch_ipex

user_cpu = torch.empty(3,3).uniform_(0,1)
cpu_res = torch.bernoulli(user_cpu, 0.5)
print("Raw Array")
print(user_cpu)
print("CPU Res")
print(cpu_res)
dpcpp_res = torch.bernoulli(user_cpu.to("dpcpp"))
print("dpcpp Res")
print(dpcpp_res.cpu())
dpcpp_res = torch.bernoulli(user_cpu.to("dpcpp"), 0.5)
print("dpcpp Res")
print(dpcpp_res.cpu())



