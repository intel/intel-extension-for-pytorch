import torch
import torchvision.models as models

model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
model.eval()
data = torch.rand(1, 3, 224, 224)

#################### code changes ####################
import intel_extension_for_pytorch as ipex

model = ipex.optimize(model, graph_mode=True)
######################################################

with torch.no_grad():
    model(data)
