import torch
import torch.nn as nn

import torch_ipex

cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

input = torch.randn(3, 5)
target = torch.LongTensor(3).random_(0, 5)

input_cpu = input
target_cpu = target

input_sycl = input.to("dpcpp")
target_sycl = target.to("dpcpp")

def test_cpu(input, target, reduc):
    loss = nn.MultiMarginLoss(reduction=reduc)
    input.requires_grad = True
    output = loss(input, target)
    print(output)
    if(reduc == "none"):
        output.backward(torch.ones_like(target, dtype=torch.float))
    else:
        output.backward(torch.tensor((1.0), dtype=torch.float))
    print(input.grad)
    input.grad.zero_()

def test_sycl(input, target, reduc):
    loss = nn.MultiMarginLoss(reduction=reduc)
    input.requires_grad = True
    output = loss(input, target)
    print(output.cpu())
    if(reduc == "none"):
        output.backward(torch.ones_like(target, dtype=torch.float).to("dpcpp"))
    else:
        output.backward(torch.tensor((1.0), dtype=torch.float).to("dpcpp"))
    print(input.grad.cpu())
    input.grad.zero_()

print('none')
print("cpu")
test_cpu(input_cpu, target_cpu, "none")
print("sycl")
test_sycl(input_sycl, target_sycl, "none")

print('sum')
print("cpu")
test_cpu(input_cpu, target_cpu, "sum")
print("sycl")
test_sycl(input_sycl, target_sycl, "sum")

print('mean')
print("cpu")
test_cpu(input_cpu, target_cpu, "mean")
print("sycl")
test_sycl(input_sycl, target_sycl, "mean")

print('sum-1024*1024')
input = torch.randn(1024, 1024)
target = torch.LongTensor(1024).random_(0, 1024)

input_cpu = input
target_cpu = target

input_sycl = input.to("dpcpp")
target_sycl = target.to("dpcpp")

print('cpu')
test_cpu(input_cpu, target_cpu, "sum")
print('sycl')
test_sycl(input_sycl, target_sycl, "sum")
