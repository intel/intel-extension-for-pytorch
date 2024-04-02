Troubleshooting
===============

## General Usage

- **Problem**: Issues with the `+cpu` PyTorch package.
  - **Cause**: Certain Python packages may have PyTorch as a hard dependency. If you installed the `+cpu` version of PyTorch, installation of these packages might replace the `+cpu` version with the default version released on Pypi.org.
  - **Solution**: Reinstall the `+cpu` version back.
- **Problem**: The workload running with Intel® Extension for PyTorch\* occupies a remarkably large amount of memory.
  - **Solution**: Try to reduce the occupied memory size by setting the `--weights_prepack` parameter of the `ipex.optimize()` function to `False`.
- **Problem**: The `conv+bn` folding feature of the `ipex.optimize()` function does not work if inference is done with a custom function:
   
  ```
  import torch
  import intel_pytorch_extension as ipex

  class Module(torch.nn.Module):
      def __init__(self):
          super(Module, self).__init__()
          self.conv = torch.nn.Conv2d(1, 10, 5, 1)
          self.bn = torch.nn.BatchNorm2d(10)
          self.relu = torch.nn.ReLU()

      def forward(self, x):
          x = self.conv(x)
          x = self.bn(x)
          x = self.relu(x)
          return x

      def inference(self, x):
          return self.forward(x)

  if __name__ == '__main__':
      m = Module()
      m.eval()
      m = ipex.optimize(m, dtype=torch.float32, level="O0")
      d = torch.rand(1, 1, 112, 112)
      with torch.no_grad():
        m.inference(d)
  ```

  - **Cause**: PyTorch FX limitation.
  - **Solution**: You can avoid this error by calling `m = ipex.optimize(m, level="O0")`, which doesn't apply ipex optimization, or disable `conv+bn` folding by calling `m = ipex.optimize(m, level="O1", conv_bn_folding=False)`.

## Performance Regression

- Some models may experience performance regression comparing to 2.0.x due to deprecation of the NNC feature in PyTorch\*.

## TorchDynamo

- **Problem**: A workload that uses `torch.compile()` fails to run or demonstrates poor performance. 
  - **Cause**: The support of `torch.compile()` with `ipex` as the backend is still an beta feature. Currently, the following HuggingFace models fail to run using `torch.compile()` with `ipex` backend due to memory issues:
    - masked-language-modeling+xlm-roberta-base
    - casual-language-modeling+gpt2
    - casual-language-modeling+xlm-roberta-base
    - summarization+t5-base
    - text-classification+allenai-longformer-base-409
  - **Solution**: Use the `torch.jit` APIs and graph optimization APIs of the Intel® Extension for PyTorch\*.

## Dynamic Shape 

- **Problem**: When working with an NLP model inference with dynamic input data length using TorchScript (either `torch.jit.trace` or `torch.jit.script`), performance with Intel® Extension for PyTorch\* may be less than that without Intel® 
  Extension for PyTorch\*.
  - **Solution**: Use the workaround below: 

    - Python interface
      ```python
      torch._C._jit_set_texpr_fuser_enabled(False)
      ```
    - C++ interface
      ```c++
      #include <torch/csrc/jit/passes/tensorexpr_fuser.h>
      torch::jit::setTensorExprFuserEnabled(false);
      ```

## INT8

- **Problem**: Limitations of dynamic shapes support of static quantization:
  - When an input shape is provided in runtime for the first time, execution could take longer time to compile a new kernel for this shape. Specifically, the new kernel compilation time could be long for complicated kernels.
  - Channels Last format won't take effect with dynamic input shapes for CNN models at this time. Optimizations are undergoing.
- **Problem**: `RuntimeError: Overflow when unpacking long` when a tensor's min max value exceeds int range while performing int8 calibration.
  - **Solution**: Customize `QConfig` to use min-max calibration method.
- **Problem**: Models get large accuracy loss with the default quantization recipe.
  - **Solution**: Try using the [the INT8 Recipe Tuning API](./features/int8_recipe_tuning_api.md) to tune a recipe with satisfied accuracy loss.
- **Problem**: Incorrect results with large tensors when calibrating with `quantize_per_tensor`, when benchmarking with 1 OpenMP\* thread (find more detailed info [here](https://github.com/pytorch/pytorch/issues/80501).
  - **Solution**: Editing your code following the pseudocode below can workaround this issue, if you do need to explicitly set `OMP_NUM_THREAEDS=1` for benchmarking. However, there could be a performance regression if oneDNN graph compiler prototype feature is used.

    Workaround pseudocode:
    ```
    # perform convert/trace/freeze with omp_num_threads > 1(N)
    torch.set_num_threads(N)
    prepared_model = prepare(model, input)
    converted_model = convert(prepared_model)
    traced_model = torch.jit.trace(converted_model, input)
    freezed_model = torch.jit.freeze(traced_model)
    # run freezed model to apply optimization pass
    freezed_model(input)
  
    # benchmarking with omp_num_threads = 1
    torch.set_num_threads(1)
    run_benchmark(freezed_model, input)
    ```
- For models with dynamic control flow, please try dynamic quantization. Users are likely to get performance gain for GEMM models.
- Support for `EmbeddingBag` with INT8 when bag size > 1 is work in progress.

## BFloat16

- **Problem**: BF16 AMP(auto-mixed-precision) runs abnormally with the extension on the AVX2-only machine if the topology contains `Conv`, `Matmul`, `Linear`, and `BatchNormalization`.
  - **Solution**: TBD

## Runtime Extension

The following limitations currently exist:

- Runtime extension of `MultiStreamModule` does not support DLRM inference, since the input of DLRM (EmbeddingBag specifically) cannot be simply batch split.
- Runtime extension of `MultiStreamModule` has poor performance of RNNT Inference comparing with native throughput mode. Only part of the RNNT models (`joint_net` specifically) can be jit traced into graph. However, in one batch inference, `joint_net` is invoked multiple times. It increases the overhead of `MultiStreamModule` as input batch split, thread synchronization and output concat.

## Result Correctness

- **Problem**: Incorrect Conv and Linear result if the number of OMP threads is changed at runtime.
  - **Cause**: The oneDNN memory layout depends on the number of OMP threads, which requires the caller to detect the changes for the # of OMP threads while this release has not implemented it yet.

## Float32 Training

- **Problem**: Low throughput with DLRM FP32 Train.
  - **Solution**: A 'Sparse Add' [PR](https://github.com/pytorch/pytorch/pull/23057) is pending on review. The issue will be fixed when the PR is merged.
