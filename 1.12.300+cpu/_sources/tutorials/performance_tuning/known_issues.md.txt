Known Issues
============

- Supporting of EmbeddingBag with INT8 when bag size > 1 is working in progress.

- Compiling with gcc 11 might result in `illegal instruction` error.

- `RuntimeError: Overflow when unpacking long` when a tensor's min max value exceeds int range while performing int8 calibration. Please customize QConfig to use min-max calibration method.

- For models with dynamic control flow, please try dynamic quantization. Users are likely to get performance gain for GEMM models.

- Calibrating with quantize_per_tensor, when benchmarking with 1 OpenMP\* thread, results might be incorrect with large tensors (find more detailed info [here](https://github.com/pytorch/pytorch/issues/80501). Editing your code following the pseudocode below can workaround this issue, if you do need to explicitly set OMP_NUM_THREAEDS=1 for benchmarking. However, there could be a performance regression if oneDNN graph compiler prototype feature is utilized.

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

- BF16 AMP(auto-mixed-precision) runs abnormally with the extension on the AVX2-only machine if the topology contains `Conv`, `Matmul`, `Linear`, and `BatchNormalization`

- Runtime extension of MultiStreamModule doesn't support DLRM inference, since the input of DLRM (EmbeddingBag specifically) can't be simplely batch split.

- Runtime extension of MultiStreamModule has poor performance of RNNT Inference comparing with native throughput mode. Only part of the RNNT models (joint_net specifically) can be jit traced into graph. However, in one batch inference, `joint_net` is invoked multi times. It increases the overhead of MultiStreamModule as input batch split, thread synchronization and output concat.

- Incorrect Conv and Linear result if the number of OMP threads is changed at runtime

  The oneDNN memory layout depends on the number of OMP threads, which requires the caller to detect the changes for the # of OMP threads while this release has not implemented it yet.

- Low performance with INT8 support for dynamic shapes

  The support for dynamic shapes in Intel® Extension for PyTorch\* INT8 integration is still work in progress. When the input shapes are dynamic, for example inputs of variable image sizes in an object detection task or of variable sequence lengths in NLP tasks, the Intel® Extension for PyTorch\* INT8 path may slow down the model inference. In this case, use stock PyTorch INT8 functionality.

  **Note**: Using Runtime Extension feature if batch size cannot be divided by number of streams, because mini batch size on each stream are not equivalent, scripts run into this issues.

- Low throughput with DLRM FP32 Train

  A 'Sparse Add' [PR](https://github.com/pytorch/pytorch/pull/23057) is pending on review. The issue will be fixed when the PR is merged.

- If inference is done with a custom function, `conv+bn` folding feature of the `ipex.optimize()` function doesn't work.

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

  This is a PyTorch FX limitation. You can avoid this error by calling `m = ipex.optimize(m, level="O0")`, which doesn't apply ipex optimization, or disable `conv+bn` folding by calling `m = ipex.optimize(m, level="O1", conv_bn_folding=False)`.
