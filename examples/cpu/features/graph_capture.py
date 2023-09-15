import torch
import torchvision.models as models

model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
model.eval()
data = torch.rand(1, 3, 224, 224)

#################### code changes ####################  # noqa F401
import intel_extension_for_pytorch as ipex
model = ipex.optimize(model, graph_mode=True)
######################################################  # noqa F401

with torch.no_grad():
    model(data)

print("Execution finished")
