import torch
import torch.nn as nn

import torch_ipex

cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

input1 = torch.randn(1, 5)
input2 = torch.randn(1, 5)
target = torch.tensor([[1], [-1]])

input1_cpu = input1
input2_cpu = input2
target_cpu = target

input1_sycl = input1.to("dpcpp")
input2_sycl = input2.to("dpcpp")
target_sycl = target.to("dpcpp")

def test_cpu(input1, input2, target, reduc):
    loss = nn.CosineEmbeddingLoss(reduction=reduc)
    input1.requires_grad = True
    input2.requires_grad = True
    output = loss(input1, input2, target)
    print(output)
    if(reduc == "none"):
        output.backward(torch.ones((2, 1), dtype=torch.float))
    else:
        output.backward(torch.tensor((1.0), dtype=torch.float))
    print(input1.grad)
    print(input2.grad)
    input1.grad.zero_()
    input2.grad.zero_()

def test_sycl(input1, input2, target, reduc):
    loss = nn.CosineEmbeddingLoss(reduction=reduc)
    input1.requires_grad = True
    input2.requires_grad = True
    output = loss(input1, input2, target)
    print(output.cpu())
    if(reduc == "none"):
        output.backward(torch.ones((2, 1), dtype=torch.float).to("dpcpp"))
    else:
        output.backward(torch.tensor((1.0), dtype=torch.float).to("dpcpp"))
    print(input1.grad.cpu())
    print(input2.grad.cpu())
    input1.grad.zero_()
    input2.grad.zero_()

print('none')
print("cpu")
test_cpu(input1_cpu, input2_cpu, target_cpu, "none")
print("sycl")
test_sycl(input1_sycl, input2_sycl, target_sycl, "none")

print('sum')
print("cpu")
test_cpu(input1_cpu, input2_cpu, target_cpu, "sum")
print("sycl")
test_sycl(input1_sycl, input2_sycl, target_sycl, "sum")

print('mean')
print("cpu")
test_cpu(input1_cpu, input2_cpu, target_cpu, "mean")
print("sycl")
test_sycl(input1_sycl, input2_sycl, target_sycl, "mean")

