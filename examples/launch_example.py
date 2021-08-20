import time
import torch
import torchvision.models as models
import intel_pytorch_extension as ipex

model = models.vgg13(pretrained=False)
model.eval()
input_batch = torch.rand(1, 3, 224, 224)

device = 'xpu'
model.to(device)
input_batch = input_batch.to(device)

with torch.no_grad():
  model = torch.jit.trace(model, input_batch)
  for i in range(100):
    model(input_batch)
  start = time.time()
  for i in range(100):
    output = model(input_batch)
  print((time.time() - start)*1000/100)
