import torch
import torchvision.models as models

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()
data = torch.rand(1, 3, 224, 224)

# Experimental Feature
#################### code changes ####################
import intel_extension_for_pytorch as ipex
model = torch.compile(model, backend="ipex")
######################################################

with torch.no_grad():
    model(data)