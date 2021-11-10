Graph Optimization
==================

Most Deep Learning models could be described as DAG(directed acyclic graph). Therefore, how to optimize a deep learning model from graph perspective is a nature thinking. Compared to the operator optimization and algorithm optimization, the graph optimization is at more high level. It convers not only the graph self but also the runtime. From the operator perspective, the graph optimization contains the operator fusing, the constant folding. From the runtime perspective, the graph optimization contains the operator scheduling, the computation resources management, the memory mangement.

Currently, the Intel Extension for PyTorch focuses on the operator related graph optimizations. Regarding the runtime related optimization, the extension also provides some experiment features. Please refer to the runtime extension for more details about runtime optimization.

Back to the operator realted graph optimizations, the extension extends PyTorch fusion pattern to cover more models and obtained obivous performance improvement. The fusion patterns are as the follows.
- Conv2D + ReLU
- Conv2D + SUM
- Conv2D + SUM + ReLU
- Conv2D + Sigmoid
- Conv2D + Sigmoid + MUL
- Conv2D + HardTanh
- Conv2D + SiLU
- Conv2D + ELU
- Conv3D + ReLU
- Conv3D + SUM
- Conv3D + SUM + ReLU
- Conv3D + SiLU
- Linear + ReLU
- Linear + GELU
- Add + LayerNorm
- Div + Add + Softmax
- Linear + Linear + Linear
- View + Transpose + Contiguous + View

In additon, the stock PyTorch has provided the constant propagation and BatchNormalization folding. And these optimizations will be automatically applied to the jit model by invoking `torch.jit.freeze`. Take the Resnet50 as the example.
```python
import torch
import torchvision.models as models
model = models.__dict__["resnet50 "](pretrained=True)
model.eval()
x = torch.randn(args.batch_size, 3, 224, 224)
with torch.no_grad():
    model = torch.jit.trace(model, x, check_trace=False).eval()
    # Fold the BatchNormalization and propagate constant
    torch.jit.freeze(model)
    # Print the graph
    print(model.graph_for(x))
```
If the model owner does not invoke the `torch.jit.freeze`, the `BatchNormalization` still exists on the graph. Otheriwse, the `BatchNormalization` will be folded on the graph to save the compuation and then improve the performance. Please refer to the https://en.wikipedia.org/wiki/Constant_folding for more details.

On the top of the stock PyTorch graph optimization, the model will be apllied more fusions if import the extension. The code is as the follows.
```python
import torch
import torchvision.models as models

# Import the Intel Extension for PyTorch
import intel_extension_for_pytorch as ipex

model = models.__dict__["resnet50 "](pretrained=True)
model.eval()

# Apply some fusions at the front end
model = ipex.optimize(model, dtype=torch.float32)

x = torch.randn(args.batch_size, 3, 224, 224)
with torch.no_grad():
    model = torch.jit.trace(model, x, check_trace=False).eval()
    # Fold the BatchNormalization and propagate constant
    torch.jit.freeze(model)
    # Print the graph
    print(model.graph_for(x))
```
Compared the original code, the model launcher just needs to add few lines of code, the extension will automatically acceletate the  model. Regarding the RN50, the extension will automatically fuse the Conv + ReLU and Conv + Sum + ReLU as ConvReLU and ConvSumReLU. If you check the output of `graph_for`, you will observe the fused operators.


