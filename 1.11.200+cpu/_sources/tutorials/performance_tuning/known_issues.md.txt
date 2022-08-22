Known Issues
============

- BF16 AMP(auto-mixed-precision) runs abnormally with the extension on the AVX2-only machine if the topology contains `Conv`, `Matmul`, `Linear`, and `BatchNormalization`

- Runtime extension does not support the scenario that the BS is not divisible by the stream number

- Incorrect Conv and Linear result if the number of OMP threads is changed at runtime

  The oneDNN memory layout depends on the number of OMP threads, which requires the caller to detect the changes for the # of OMP threads while this release has not implemented it yet.

- INT8 performance of EfficientNet and DenseNet with Intel® Extension for PyTorch\* is slower than that of FP32

- Low performance with INT8 support for dynamic shapes

  The support for dynamic shapes in Intel® Extension for PyTorch\* INT8 integration is still working in progress. For the use cases where the input shapes are dynamic, for example inputs of variable image sizes in an object detection task or of variable sequence lengths in NLP tasks, the Intel® Extension for PyTorch\* INT8 path may slow down the model inference. In this case, please utilize stock PyTorch INT8 functionality.

- Low throughtput with DLRM FP32 Train

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

  This is PyTorch FX limitation, user can avoid this error by calling `m = ipex.optimize(m, level="O0")`, which doesn't apply ipex optimization, or disable `conv+bn` folding by calling `m = ipex.optimize(m, level="O1", conv_bn_folding=False)`.
