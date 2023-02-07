import torch
#################### code changes ####################
import intel_extension_for_pytorch as ipex
######################################################

model = torch.jit.load('quantized_model.pt')
model.eval()
model = torch.jit.freeze(model)
data = torch.rand(1, 3, 224, 224)

with torch.no_grad():
  model(data)