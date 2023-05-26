# Intel® Extension for PyTorch with oneDNN graph for INT8 integration

The quantization in Intel® Extension for PyTorch\* integrates [oneDNN graph API](https://spec.oneapi.io/onednn-graph/latest/introduction.html) in the TorchScript graph of PyTorch.

## Integration
The integration is mainly composed of the Graph Optimization part and the Graph Executor part:

### Graph Optimization
We have registered quantization-related optimization passes in the Custom Pre-passes set of PyTorch:

1. Alias and mutation reduction

    The operators of oneDNN graph are pure functional while PyTorch has operators in in-place forms or create views for buffer sharing.
    Due to the semantic gaps between the backend operators and the PyTorch operators, we have a pass to reduce mutation with best effort at the beginning.

2. Graph passing

    With a PyTorch TorchScript graph, the integration maps PyTorch operators on the graph to the corresponding backend operators to form a backend graph.

3. Partitioning

    The backend selects regions to be fused in the graph and return a list of partitions. Each partition corresponds to a fusion operator.

4. Graph rewriting

    The original PyTorch graph will be re-written based on the partitions returned from the backend. The operators in one partition will be grouped together to form a JIT operator.

5. Layout propagation

    This pass is to eliminate unnecessary layout conversions at boundaries. We set different formats to the output of a partition so that the backend could perform layout conversion internally. When `ANY` is set, the layout at boundaries will be fully decided by the backend. Otherwise, the backend should follow the layout set by the Framework.

### Graph Executor
During runtime execution of a PyTorch TorchScript graph, oneDNN graph partition will be dispatched to the oneDNN graph JIT variadic Operator. 
Inside the oneDNN graph JIT Op, input PyTorch tensors of each partition will be mapped to oneDNN graph tensors. The partition will then be [compiled](https://spec.oneapi.io/onednn-graph/latest/programming_model.html#partition) and [executed](https://spec.oneapi.io/onednn-graph/latest/programming_model.html#compiled-partition). The output oneDNN graph tensor will be mapped back to PyTorch tensors to be fed to the next operator on the TorchScript graph.

## Supported int8 fusion patterns
The `ipex.quantization.convert(model, conf, inputs)` API will convert an FP32 `torch.nn.Module` to a quantized JIT ScriptModule according to the given quantization recipes.

Here listed all the currently supported int8 patterns in Intel® Extension for PyTorch\* using oneDNN graph backend:
1. Patterns with int8 as input and fp32 as output:
- dequant -> conv
- dequant -> linear
- dequant -> conv -> relu
- dequant -> conv -> sum
- dequant -> conv -> sum -> relu
- dequant -> linear -> relu
- dequant -> linear -> gelu
- dequant -> linear -> sigmoid
- dequant -> linear -> sum
- dequant -> bmm
- dequant -> bmm -> div

2. Patterns with int8 as input and int8 as output:
- dequant -> conv -> quant
- dequant -> linear -> quant
- dequant -> conv -> relu -> quant
- dequant -> conv -> sum -> dequant
- dequant -> conv -> sum -> relu -> quant
- dequant -> linear -> relu -> quant
- dequant -> linear -> gelu -> quant
- dequant -> linear -> sigmoid -> quant
- dequant -> linear -> sum -> quant
- dequant -> bmm -> quant
- dequant -> bmm -> div -> quant
- dequant -> max_pool2d -> quant

## Tests

```bash
pytest tests/cpu/test_jit_llga_quantization_fuser.py
```

## Code structure
Most of the source code are placed in

```bash
torch_ipex/csrc/jit/codegen/onednn/*
```

Tensor related code are located at

```bash
torch_ipex/csrc/LlgaTensorImpl.h
torch_ipex/csrc/LlgaTensorImpl.cpp
```

## Limitations
### Support for dynamic shapes
The support for dynamic shapes in Intel® Extension for PyTorch\* int8 integration is still working in progress.
For the use cases where the input shapes are dynamic, for example inputs of variable image sizes in an object detection task or of variable sequence lengths in NLP tasks, the Intel® Extension for PyTorch\* int8 path may slow down the model inference.