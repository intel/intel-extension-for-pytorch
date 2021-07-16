import torch
import torchvision.models as models

model = models.vgg13(pretrained=False)
model.eval()
input_batch = torch.rand(1, 3, 224, 224)

device = 'cpu'
model.to(device)
input_batch = input_batch.to(device)

import intel_pytorch_extension as ipex
model = ipex.optimize(model, dtype=torch.float32, level='O1')
input_batch = input_batch.to(memory_format=torch.channels_last)

with torch.no_grad():
  model = torch.jit.trace(model, input_batch)
  model = torch.jit.freeze(model)
  import time
  for i in range(100):
    model(input_batch)
  start = time.time()
  for i in range(100):
    output = model(input_batch)
  print((time.time() - start)*1000/100)
