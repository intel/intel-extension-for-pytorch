import torch
import torchvision.models as models

model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
model.eval()
data = torch.rand(128, 3, 224, 224)

# Beta Feature
#################### code changes ####################  # noqa F401
import intel_extension_for_pytorch as ipex
model = ipex.optimize(model, dtype=torch.bfloat16, weights_prepack=False)
model = torch.compile(model, backend="ipex")
######################################################  # noqa F401

with torch.no_grad(), torch.cpu.amp.autocast():
    model(data)

print("Execution finished")
