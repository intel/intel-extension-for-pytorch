import torch
import intel_extension_for_pytorch as ipex
import torchvision.models as models

# load the model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = model.eval()

# define dummy input tensor to use for the model's forward call to record operations in the model for tracing
N, C, H, W = 1, 3, 224, 224
dummy_tensor = torch.randn(N, C, H, W)

from intel_extension_for_pytorch.quantization import prepare, convert

# ipex supports two quantization schemes: static and dynamic
# default static qconfig
qconfig = ipex.quantization.default_static_qconfig_mapping

# prepare and calibrate
model = prepare(model, qconfig, example_inputs=dummy_tensor, inplace=False)

n_iter = 100
for i in range(n_iter):
    model(dummy_tensor)
 
# convert and deploy
model = convert(model)

with torch.no_grad():
    model = torch.jit.trace(model, dummy_tensor)
    model = torch.jit.freeze(model)

torch.jit.save(model, './rn50_int8_jit.pt')
