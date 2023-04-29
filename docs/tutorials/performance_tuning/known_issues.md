Known Issues
============

## Known Issues Specific to GPU

### Usage

- FP64 data type is unsupported on current platform

  FP64 is not natively supported by the [Intel® Data Center GPU Flex Series](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/data-center-gpu/flex-series/overview.html) platform. If you run any AI workload on that platform and receive this error message, it means a kernel requiring FP64 instructions but not supported and the execution is stopped.

- MaxPool2d operator only supports 4D input for ceil mode

  If 3D input is detected, MaxPool2d will throw unsupported error message and stop execution.
  
- Runtime error `invalid device pointer` if `import horovod.torch as hvd` before `import intel_extension_for_pytorch`

  Intel® Optimization for Horovod\* need use utilities provided by Intel® Extension for PyTorch\*. The improper import order will cause Intel® Extension for PyTorch\* be unloaded before Intel® Optimization for Horovod\* at the end of the execution and trigger this error. The recommended usage is to `import intel_extension_for_pytorch` before `import horovod.torch as hvd`.

- RuntimeError: Number of dpcpp devices should be greater than zero!

  - Scenario 1: Running some AI models (e.g. 3D-Unet inference) on Ubuntu22.04 may trigger this runtime error, as oneAPI Base Toolkit 2023.1 fails to return available GPU device on ubuntu22.04 in such scenario. The workaround solution is to update the model script to make sure `import torch` and `import intel_extension_for_pytorch` happen before importing other libraries.

  - Scenario 2: If you use Intel® Extension for PyTorch\*  in a conda environment, this error might occur. Conda also ships with a libstdc++.so dynamic library file. It may conflict with the one shipped in the OS. Exporting the libstdc++.so file path in OS to an environment variable `LD_PRELOAD` could workaround this issue.

- symbol undefined caused by `_GLIBCXX_USE_CXX11_ABI`

  ```bash
  ImportError: undefined symbol: _ZNK5torch8autograd4Node4nameB5cxx11Ev
  ```

  DPC++ does not support `_GLIBCXX_USE_CXX11_ABI=0`, Intel® Extension for PyTorch\* is always compiled with `_GLIBCXX_USE_CXX11_ABI=1`. This symbol undefined issue appears when PyTorch\* is compiled with `_GLIBCXX_USE_CXX11_ABI=0`. Pass `export GLIBCXX_USE_CXX11_ABI=1` and compile PyTorch\* with particular compiler which supports `_GLIBCXX_USE_CXX11_ABI=1`. We recommend using prebuilt wheels in [download server](https://developer.intel.com/ipex-whl-stable-xpu) to avoid this issue.

### Dependency Libraries

- Can't find oneMKL library when build Intel® Extension for PyTorch\* without oneMKL

  ```bash
  /usr/bin/ld: cannot find -lmkl_sycl
  /usr/bin/ld: cannot find -lmkl_intel_ilp64
  /usr/bin/ld: cannot find -lmkl_core
  /usr/bin/ld: cannot find -lmkl_tbb_thread
  dpcpp: error: linker command failed with exit code 1 (use -v to see invocation)
  ```

  When PyTorch\* is built with oneMKL library and Intel® Extension for PyTorch\* is built without oneMKL library, this linker issue may occur. Resolve it by setting:

  ```bash
  export USE_ONEMKL=OFF
  export MKL_DPCPP_ROOT=${HOME}/intel/oneapi/mkl/latest
  ```

  Then clean build Intel® Extension for PyTorch\*.

- undefined symbol: `mkl_lapack_dspevd`. Intel MKL FATAL ERROR: cannot load `libmkl_vml_avx512.so.2` or `libmkl_vml_def.so.2`

  This issue may occur when Intel® Extension for PyTorch\* is built with oneMKL library and PyTorch\* is not build with any MKL library. The oneMKL kernel may run into CPU backend incorrectly and trigger this issue. Resolve it by installing MKL library from conda:

  ```bash
  conda install mkl
  conda install mkl-include
  ```

  then clean build PyTorch\*.

- OSError: `libmkl_intel_lp64.so.2`: cannot open shared object file: No such file or directory

  Wrong MKL library is used when multiple MKL libraries exist in system. Preload oneMKL by:

  ```bash
  export LD_PRELOAD=${MKL_DPCPP_ROOT}/lib/intel64/libmkl_intel_lp64.so.2:${MKL_DPCPP_ROOT}/lib/intel64/libmkl_intel_ilp64.so.2:${MKL_DPCPP_ROOT}/lib/intel64/libmkl_gnu_thread.so.2:${MKL_DPCPP_ROOT}/lib/intel64/libmkl_core.so.2:${MKL_DPCPP_ROOT}/lib/intel64/libmkl_sycl.so.2
  ```

  If you continue seeing similar issues for other shared object files, add the corresponding files under `${MKL_DPCPP_ROOT}/lib/intel64/` by `LD_PRELOAD`. Note that the suffix of the libraries may change (e.g. from .1 to .2), if more than one oneMKL library is installed on the system.

- OpenMP library could not be found

  Build Intel® Extension for PyTorch\* on SLES15 SP3 using default GCC 7.5 and CentOS8 using default GCC 8.5 may trigger this build error.

  ```bash
  Make Error at third_party/ideep/mkl-dnn/third_party/oneDNN/cmake/OpenMP.cmake:118 (message):
    OpenMP library could not be found.  Proceeding might lead to highly
    sub-optimal performance.
  Call Stack (most recent call first):
    third_party/ideep/mkl-dnn/third_party/oneDNN/CMakeLists.txt:117 (include)
  ```

  The root cause is GCC 7.5 or 8.5 does not support `-Wno-error=redundant-move` option. Uplift to GCC version >=9 can solve this issue.

### UnitTest

- Unit test failures on Intel® Data Center GPU Flex Series 170

  The following unit tests fail on Intel® Data Center GPU Flex Series 170.
    - test_linalg.py::TestTorchMethod::test_tensorinv_empty
    - test_distributions.py::TestDistributions::test_dirichlet_mean_var
    - test_adaptive_avg_pool2d.py::TestNNMethod::test_adaptive_avg_pool2d
    - test_multilabel_margin_loss.py::TestNNMethod::test_multiabel_margin_loss

  The same test cases pass on Intel® Data Center GPU Max Series. The root cause of the failures is under investigation.

- Unit test failures on Intel® Data Center GPU Max Series

  The following unit tests randomly fail on Intel® Data Center GPU Flex Max Series.
     - test_nn.py::TestNNDeviceTypeXPU::test_activations_bfloat16_xpu
     - test_lstm.py::TestNNMethod::test_lstm_rnnt_onednn
     - test_eigh.py::TestTorchMethod::test_linalg_eigh
     
  The test cases rarely fail if running with other test cases together using `pytest -v`. These cases pass if run individually on the same environment. The root cause of the failures is under investigation.

## Known Issues Specific to CPU

### Usage

- There might be Python packages having PyTorch as their hard dependency. If you installed `+cpu` version of PyTorch, installation of these packages might replace the `+cpu` version with the default version released on Pypi.org. If anything goes wrong, please reinstall the `+cpu` version back.

- If you found the workload runs with Intel® Extension for PyTorch\* occupies a remarkably large amount of memory, you can try to reduce the occupied memory size by setting the `--weights_prepack` parameter of the `ipex.optimize()` function to `False`.

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

### TorchDynamo

- The support of torch.compile() with ipex as the backend is still an experimental feature. If the workload fails to run or demonstrates poor performance, you can use the `torch.jit` APIs and graph optimization APIs of ipex. Currently, the below HuggingFace models fail to run using torch.compile() with ipex backend due to memory issues:
  - masked-language-modeling+xlm-roberta-base
  - casual-language-modeling+gpt2
  - casual-language-modeling+xlm-roberta-base
  - summarization+t5-base
  - text-classification+allenai-longformer-base-409

### Dynamic Shape

- When working with an NLP model inference with dynamic input data length appling with TorchScript (either `torch.jit.trace` or `torch.jit.script`), performance with Intel® Extension for PyTorch\* is possible to be less than that without Intel® Extension for PyTorch\*. In this case, adding the workarounds below would help solve this issue.
  - Python interface
    ```python
	torch._C._jit_set_texpr_fuser_enabled(False)
	```
  - C++ interface
    ```c++
    #include <torch/csrc/jit/passes/tensorexpr_fuser.h>
	torch::jit::setTensorExprFuserEnabled(false);
	```

### INT8

- Low performance with INT8 support for dynamic shapes

  The support for dynamic shapes in Intel® Extension for PyTorch\* INT8 integration is still work in progress. When the input shapes are dynamic, for example inputs of variable image sizes in an object detection task or of variable sequence lengths in NLP tasks, the Intel® Extension for PyTorch\* INT8 path may slow down the model inference. In this case, use stock PyTorch INT8 functionality.

  **Note**: Using Runtime Extension feature if batch size cannot be divided by number of streams, because mini batch size on each stream are not equivalent, scripts run into this issues.

- Supporting of EmbeddingBag with INT8 when bag size > 1 is working in progress.

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

### BFloat16

- BF16 AMP(auto-mixed-precision) runs abnormally with the extension on the AVX2-only machine if the topology contains `Conv`, `Matmul`, `Linear`, and `BatchNormalization`

### Runtime Extension

- Runtime extension of MultiStreamModule doesn't support DLRM inference, since the input of DLRM (EmbeddingBag specifically) can't be simplely batch split.

- Runtime extension of MultiStreamModule has poor performance of RNNT Inference comparing with native throughput mode. Only part of the RNNT models (joint_net specifically) can be jit traced into graph. However, in one batch inference, `joint_net` is invoked multi times. It increases the overhead of MultiStreamModule as input batch split, thread synchronization and output concat.

### Correctness

- Incorrect Conv and Linear result if the number of OMP threads is changed at runtime

  The oneDNN memory layout depends on the number of OMP threads, which requires the caller to detect the changes for the # of OMP threads while this release has not implemented it yet.

### Float32 Training

- Low throughput with DLRM FP32 Train

  A 'Sparse Add' [PR](https://github.com/pytorch/pytorch/pull/23057) is pending on review. The issue will be fixed when the PR is merged.
