Graph Capture (Experimental)
============================

### Feature Description

This feature automatically applies a combination of TorchScript trace technique and TorchDynamo to try to generate a graph model, for providing a good user experience while keep execution fast. Specifically, the process tries to generate a graph with TorchScript trace functionality first. In case of generation failure or incorrect results detected, it changes to TorchDynamo with TorchScript backend. Failure of the graph generation with TorchDynamo triggers a warning message. Meanwhile the generated graph model falls back to the original one. I.e. the inference workload runs in eager mode. Users can take advantage of this feature through a new knob `--graph_mode` of the `ipex.optimize()` function to automatically run into graph mode.

### Usage Example

```python
import torch
import torchvision.models as models

model = models.resnet50(pretrained=True)
model.eval()
data = torch.rand(1, 3, 224, 224)

model = model.to(memory_format=torch.channels_last)
data = data.to(memory_format=torch.channels_last)

#################### code changes ####################
import intel_extension_for_pytorch as ipex
model = ipex.optimize(model, graph_mode=True)
######################################################

with torch.no_grad():
    model(data)
```
