import torch
import intel_extension_for_pytorch
import microbench
import torchvision.models as models

microbench.enable_verbose()
model = models.resnet50().xpu()
input = torch.randn(2, 3, 64, 64).xpu()
output = model(input)
microbench.disable_verbose()
