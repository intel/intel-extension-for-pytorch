#!/usr/bin/env python
# encoding: utf-8

import torch
import torchvision

model = torchvision.models.resnet50(pretrained=True)
model.eval()

input = torch.rand(1, 3, 224, 224)
model = torch.jit.trace(model, input, check_trace=False)

model.save('resnet50.pt')
print("save mode to: resnet50.pt")
