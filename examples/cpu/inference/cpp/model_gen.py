#!/usr/bin/env python
# encoding: utf-8

import torch
import torchvision

model = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
model.eval()

input = torch.rand(1, 3, 224, 224)
model = torch.jit.trace(model, input, check_trace=False)

model.save('resnet50.pt')
print("Saved model to: resnet50.pt")