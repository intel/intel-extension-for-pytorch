import os
import torch
from torch.jit._recursive import wrap_cpp_module
from torch.quantization.quantize_jit import (
  convert_jit,
  prepare_jit,
)
#################### code changes ####################
import intel_extension_for_pytorch as ipex
######################################################

##### Example Model #####
import torchvision.models as models
model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
model.eval()
model = model.to('xpu')

with torch.no_grad():
  data = torch.rand(1, 3, 224, 224)
  data = data.to('xpu')
  modelJit = torch.jit.trace(model, data)
#########################

qconfig = torch.quantization.QConfig(
  activation=torch.quantization.observer.MinMaxObserver.with_args(
    qscheme=torch.per_tensor_symmetric,
    reduce_range=False,
    dtype=torch.quint8
  ),
  weight=torch.quantization.default_weight_observer
)
modelJit = prepare_jit(modelJit, {'': qconfig}, True)

##### Example Dataloader #####
import torchvision
DOWNLOAD = True
DATA = 'datasets/cifar10/'

transform = torchvision.transforms.Compose([
  torchvision.transforms.Resize((224, 224)),
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = torchvision.datasets.CIFAR10(
  root=DATA,
  train=True,
  transform=transform,
  download=DOWNLOAD,
)
calibration_data_loader = torch.utils.data.DataLoader(
  dataset=train_dataset,
  batch_size=128
)

for batch_idx, (d, target) in enumerate(calibration_data_loader):
  print(f'calibrated on batch {batch_idx} out of {len(calibration_data_loader)}')
  d = d.to('xpu')
  modelJit(d)
##############################

modelJit = convert_jit(modelJit, True)

data = torch.rand(1, 3, 224, 224)
data = data.to('xpu')
modelJit(data)
