import torch
import torchvision.models as models

# Import the Intel Extension for PyTorch
import intel_extension_for_pytorch as ipex

model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
model.eval()

# Apply some fusions at the front end
model = ipex.optimize(model, dtype=torch.float32)

x = torch.randn(4, 3, 224, 224)
with torch.no_grad():
    model = torch.jit.trace(model, x, check_trace=False).eval()
    # Fold the BatchNormalization and propagate constant
    torch.jit.freeze(model)
    # Print the graph
    print(model.graph_for(x))

print("Execution finished")
