import torch
import torch.nn as nn

import torch_ipex

cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

print("sum cpu")
embedding_sum = nn.EmbeddingBag(10, 3, mode='sum', scale_grad_by_freq=False)
input = torch.LongTensor([1,2,4,5,4,3,2,9], device = cpu_device)
offsets = torch.LongTensor([0,4], device = cpu_device)
weights = torch.Tensor([0.3, 0.2, 0.1, 0.3, 0.2, 0.1, 0.1, 0.1])
output = embedding_sum(input, offsets)
print(output)
grad_cpu = torch.ones(output.shape, device=cpu_device, dtype=torch.float)
grad_cpu = grad_cpu + grad_cpu
embedding_sum.zero_grad()

grad_weight = output.backward(grad_cpu)
for param in embedding_sum._parameters.values():
    print(param._grad)

print("sum sycl")
input_sycl = input.to("dpcpp")
offsets_sycl = offsets.to("dpcpp")
embedding_sum.to("dpcpp")
grad_sycl = grad_cpu.to("dpcpp")
weights_sycl = weights.to("dpcpp")
output_sycl = embedding_sum(input_sycl, offsets_sycl)
print(output_sycl.to("cpu"))

embedding_sum.zero_grad()
grad_weight = output_sycl.backward(grad_sycl)
for param in embedding_sum._parameters.values():
    print(param._grad.to("cpu"))

print("mean cpu")
embedding_mean = nn.EmbeddingBag(10, 3, mode='mean')
output = embedding_mean(input, offsets)
print(output)

embedding_mean.zero_grad()
grad_weight = output.backward(grad_cpu)
for param in embedding_mean._parameters.values():
    print(param._grad)

print("mean sycl")
embedding_mean.to("dpcpp")
output_sycl = embedding_mean(input_sycl, offsets_sycl)
print(output_sycl.to("cpu"))

embedding_mean.zero_grad()
grad_weight = output_sycl.backward(grad_sycl)
for param in embedding_mean._parameters.values():
    print(param._grad.to("cpu"))

print("max cpu")
embedding_max = nn.EmbeddingBag(10, 3, mode='max')
output = embedding_max(input, offsets)
print(output)

embedding_max.zero_grad()
grad_weight = output.backward(grad_cpu)
for param in embedding_max._parameters.values():
    print(param._grad)

print("max sycl")
embedding_max.to("dpcpp")
output_sycl = embedding_max(input_sycl, offsets_sycl)
print(output_sycl.to("cpu"))

embedding_max.zero_grad()
grad_weight = output_sycl.backward(grad_sycl)
for param in embedding_max._parameters.values():
    print(param._grad.to("cpu"))