Graph Optimization
==================

Most Deep Learning models could be described as a DAG (directed acyclic graph). Optimizing a deep learning model from a graph perspective is straight forward. Compared to the operator optimization and algorithm optimization, the graph optimization is at a higher level. It covers not only the graph but also the runtime. From the operator perspective, the graph optimization contains the operator fusing and constant folding. From the runtime perspective, the graph optimization contains the operator scheduling, computation resources management, and memory management.

The Intel® Extension for PyTorch\* focuses on operator related graph optimizations. The extension also provides some prototype features for the related runtime optimizations. Refer to the runtime extension for more details about runtime optimization.

## Ease-of-use graph optimization API
The graph optimizations of Intel® Extension for PyTorch\* are enabled by default. Users can disable it by calling:
```
ipex.enable_onednn_fusion(False)
```

### FP32 and BF16 models

[//]: # (marker_feature_graph_optimization_fp32_bf16)
[//]: # (marker_feature_graph_optimization_fp32_bf16)

Compared to the original code, the model launcher needs to add a few lines of code and the extension will automatically accelerate the model. Regarding the RN50, the extension will automatically fuse the Conv + ReLU and Conv + Sum + ReLU as ConvReLU and ConvSumReLU. If you check the output of `graph_for`, you will observe the fused operators.

### INT8 models

[//]: # (marker_feature_graph_optimization_int8)
[//]: # (marker_feature_graph_optimization_int8)

## Methodology
### Fusion
#### FP32 and BF16 fusion patterns
- Conv1D/Conv2D/Conv3D/Linear/ConvTranspose2D/ConvTranspose3D + Abs/Clamp/Elu/Exp/GELU/HardTanh/HardSwish/Log/Mish/Sigmoid/Pow/ReLU/Round/Sqrt/Square/Tanh/Leaky_ReLU/SiLU
- Conv1D/Conv2D/Conv3D/Linear/ConvTranspose2D/ConvTranspose3D + Sigmoid + MUL
- Conv1D/Conv2D/Conv3D/Linear + SUM
- Conv1D/Conv2D/Conv3D + SUM + ReLU
- Add + LayerNorm
- Div + Add + Softmax
- Linear + Linear + Linear
- View + Transpose + Contiguous + View

#### INT8 fusion patterns
The `ipex.quantization.convert(model, conf, inputs)` API will convert an FP32 `torch.nn.Module` to a quantized JIT ScriptModule according to the given quantization recipes.

For example, for a FP32 model of one single convolution, the graph before and after conversion will be:
![image](../../../images/graph_optimization/int8_pattern.png)

The oneDNN graph backend will select `dequantize` and `convolution` into one partition. During execution, this partition will execute a convolution with int8 as input and fp32 as output.

Here listed all the currently supported int8 patterns in Intel® Extension for PyTorch\* using oneDNN graph backend:

1. Conv/Linear/Matmul related fusion patterns
   ```
                                            |
                                        [Quantize]*
                   |                        |
              Dequantize                Dequantize
                   \                      /
              Conv1D/Conv2D/Conv3D/Linear/MatMul
                                |
            [Abs/Elu/GELU/HardTanh/Leaky_ReLU/Sigmoid/
       ReLU/Sqrt/Square/Tanh/[Dequantize+Add]*[0,1] ]*[0,3]
                                |
                            [Quantize]*
                                |
   ```

   ```
        |              |
      Dequantize   Dequantize
         \___      ___/
             MatMul
                \    /
                Divide
                   \   /
                   [Add]*
                     |
   ```

2. Non-Conv/Linear/Matmul related fusion patterns
   ```
              |
          Dequantize
              |
          MaxPool2D
              |
           Quantize
   ```
3. INT8-BF16 mixed-precision fusion patterns
   ```
        |              |
      Dequantize   Dequantize
        |              |
       To             To
         \___      ___/
             MatMul
                \      /
                [Divide]*
                    \     /
                     [Add]*
                       |
   ```

   ```
        |              |
      Dequantize   Dequantize
        |              |
       To             To
         \___      ___/
             MatMul
               |
             [GeLU]*
               |
              To
               |
            Quantize
               |
   ```

   ```
        |              |
      Dequantize   Dequantize
        |              |
        To            To     Dequantize
         \___      ___/          |
             MatMul              To
                \_____        ___/
                       [Add]*
                         |
   ```


### Folding
Stock PyTorch provids constant propagation and BatchNormalization folding. These optimizations are automatically applied to the jit model by invoking `torch.jit.freeze`. Take the Resnet50 as an example:

[//]: # (marker_feature_graph_optimization_folding)
[//]: # (marker_feature_graph_optimization_folding)

If the model owner does not invoke the `torch.jit.freeze`, the `BatchNormalization` still exists on the graph. Otheriwse, the `BatchNormalization` will be folded on the graph to save the compuation and then improve the performance. Refer to the [Constant Folding Wikipedia page](https://en.wikipedia.org/wiki/Constant_folding) for more details.
