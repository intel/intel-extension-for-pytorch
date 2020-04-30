import torch
import torch.nn as nn

import torch_ipex

cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

input = torch.randn(5, 6)
positive = torch.randn(5, 6)
negative = torch.randn(5, 6)

input_cpu = input
posit_cpu = positive
negat_cpu = negative

input_sycl = input.to("dpcpp")
posit_sycl = positive.to("dpcpp")
negat_sycl = negative.to("dpcpp")

def test_cpu(input, positive, negative, reduc):
    loss = nn.TripletMarginLoss(reduction=reduc)
    input.requires_grad = True
    output = loss(input, positive, negative)
    print(output)
    if(reduc == "none"):
        output.backward(torch.ones(5, dtype=torch.float))
    else:
        output.backward(torch.tensor((1.0), dtype=torch.float))
    print(input.grad)
    input.grad.zero_()

def test_sycl(input, positive, negative, reduc):
    loss = nn.TripletMarginLoss(reduction=reduc)
    input.requires_grad = True
    output = loss(input, positive, negative)
    print(output.cpu())
    if(reduc == "none"):
        output.backward(torch.ones(5, dtype=torch.float).to("dpcpp"))
    else:
        output.backward(torch.tensor((1.0), dtype=torch.float).to("dpcpp"))
    print(input.grad.cpu())
    input.grad.zero_()

print('none')
print("cpu")
test_cpu(input_cpu, posit_cpu, negat_cpu, "none")
print("sycl")
test_sycl(input_sycl, posit_sycl, negat_sycl, "none")

print('sum')
print("cpu")
test_cpu(input_cpu, posit_cpu, negat_cpu, "sum")
print("sycl")
test_sycl(input_sycl, posit_sycl, negat_sycl, "sum")

print('mean')
print("cpu")
test_cpu(input_cpu, posit_cpu, negat_cpu, "mean")
print("sycl")
test_sycl(input_sycl, posit_sycl, negat_sycl, "mean")

