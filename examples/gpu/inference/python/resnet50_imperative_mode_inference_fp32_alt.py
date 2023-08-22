import torch
import torchvision.models as models

############# code changes #########
import intel_extension_for_pytorch

############# code changes #########

model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
model.eval()
data = torch.rand(1, 3, 224, 224)

model = model.to(memory_format=torch.channels_last)
data = data.to(memory_format=torch.channels_last)

########## code changes #########
model = model.to("xpu")
data = data.to("xpu")
model = torch.xpu.optimize(model)
########## code changes #########

with torch.no_grad():
    model(data)

print("Execution finished")
