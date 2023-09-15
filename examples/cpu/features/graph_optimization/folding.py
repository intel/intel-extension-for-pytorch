import torch
import torchvision.models as models

model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
model.eval()
x = torch.randn(4, 3, 224, 224)

with torch.no_grad():
    model = torch.jit.trace(model, x, check_trace=False).eval()
    # Fold the BatchNormalization and propagate constant
    torch.jit.freeze(model)
    # Print the graph
    print(model.graph_for(x))

print("Execution finished")
