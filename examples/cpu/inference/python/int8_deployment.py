import torch
#################### code changes ####################  # noqa F401
import intel_extension_for_pytorch as ipex              # noqa F401
######################################################  # noqa F401

model = torch.jit.load('static_quantized_model.pt')
model.eval()
model = torch.jit.freeze(model)
data = torch.rand(128, 3, 224, 224)

with torch.no_grad():
    model(data)

print("Execution finished")