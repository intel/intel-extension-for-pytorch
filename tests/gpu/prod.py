import torch
import torch_ipex

input=torch.randn(4, dtype=torch.float32, device = torch.device("cpu"))
print("cpu input:", input)
print("cpu output:", torch.prod(input))

input_dpcpp=input.to("dpcpp")
print("gpu input:", input_dpcpp.cpu())
print("gpu output:", torch.prod(input_dpcpp).cpu())
