Intel® Extension for PyTorch\* optimizations for quantization (Experimental)
=============

## Integration
The quantization in Intel® Extension for PyTorch\* integrates [oneDNN graph API](https://spec.oneapi.io/onednn-graph/latest/introduction.html) in the TorchScript graph of PyTorch.

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

The below diagram demonstrates the process of `Graph passing - Partitioning - Graph rewriting`:

![image](../../../images/int8/integration_diagram.PNG)


5. Layout propagation

    This pass is to eliminate unnecessary layout conversions at boundaries. We set different formats to the output of a partition so that the backend could perform layout conversion internally. When `ANY` is set, the layout at boundaries will be fully decided by the backend. Otherwise, the backend should follow the layout set by the Framework.
    
![image](../../../images/int8/layout_propagation.png)

### Graph Executor
During runtime execution of a PyTorch TorchScript graph, oneDNN graph partition will be dispatched to the oneDNN graph JIT variadic Operator. 
Inside the oneDNN graph JIT Op, input PyTorch tensors of each partition will be mapped to oneDNN graph tensors. The partition will then be [compiled](https://spec.oneapi.io/onednn-graph/latest/programming_model.html#partition) and [executed](https://spec.oneapi.io/onednn-graph/latest/programming_model.html#compiled-partition). The output oneDNN graph tensor will be mapped back to PyTorch tensors to be fed to the next operator on the TorchScript graph.

## Supported int8 fusion patterns
The `ipex.quantization.convert(model, conf, inputs)` API will convert an FP32 `torch.nn.Module` to a quantized JIT ScriptModule according to the given quantization recipes.

For example, for a FP32 model of one single convolution, the graph before and after conversion will be:
![image](../../../images/int8/int8_pattern.png)
 
The oneDNN graph backend will select `dequantize` and `convolution` into one partition. During execution, this partition will execute a convolution with int8 as input and fp32 as output. 

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

## Advanced features
### Cache of the integration meta-data

The profiling graph executor (the default executor) of PyTorch will record type information of tensors during runtime, including:
- data type
- shapes
- strides
- requires_grad
- device

The integration will leverage the type information on the JIT graph to compile the partition and save the meta-data including the compiled partition on the graph. During execution time, if the type of input tensors to a partition is the same as what is recorded on the graph, the cached compiled partition will be executed directly. If the type of input tensors has changed, the graph executor will enter a fallback graph which is the un-optimized PyTorch TorchScript graph.

### Weight cache

During the compilation of the oneDNN graph partition, the backend will find an optimal formats of weight tensors (of convolution or linear, etc.) to use during the execution to achieve better performance. The backend has implemented the weight cache to save the weight in the optimal formats to use during the execution, so that there's no need to do this transformation each time. The weight cache is turned on by default.

## Limitations
### Support for dynamic shapes
The support for dynamic shapes in Intel® Extension for PyTorch\* int8 integration is still working in progress.
For the use cases where the input shapes are dynamic, for example inputs of variable image sizes in an object detection task or of variable sequence lengths in NLP tasks, the Intel® Extension for PyTorch\* int8 path may slow down the model inference.
