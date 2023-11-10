Troubleshooting
===============

## GPU-specific Issues

### General Usage

- **Problem**: FP64 data type is unsupported on current platform.
  - **Cause**: FP64 is not natively supported by the [Intel® Data Center GPU Flex Series](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/data-center-gpu/flex-series/overview.html) platform. 
    If you run any AI workload on that platform and receive this error message, it means a kernel requires FP64 instructions that are not supported and the execution is stopped.
- **Problem**: Runtime error `invalid device pointer` if `import horovod.torch as hvd` before `import intel_extension_for_pytorch`
  - **Cause**: Intel® Optimization for Horovod\* uses utilities provided by Intel® Extension for PyTorch\*. The improper import order causes Intel® Extension for PyTorch\* to be unloaded before Intel®
    Optimization for Horovod\* at the end of the execution and triggers this error.
  - **Solution**: Do `import intel_extension_for_pytorch` before `import horovod.torch as hvd`.
- **Problem**: Number of dpcpp devices should be greater than zero.
  - **Cause**: If you use Intel® Extension for PyTorch* in a conda environment, you might encounter this error. Conda also ships the libstdc++.so dynamic library file that may conflict with the one shipped
    in the OS. 
  - **Solution**: Export the `libstdc++.so` file path in the OS to an environment variable `LD_PRELOAD`.
- **Problem**: Symbol undefined caused by `_GLIBCXX_USE_CXX11_ABI`.
    ```bash
    ImportError: undefined symbol: _ZNK5torch8autograd4Node4nameB5cxx11Ev
    ```
  - **Cause**: DPC++ does not support `_GLIBCXX_USE_CXX11_ABI=0`, Intel® Extension for PyTorch\* is always compiled with `_GLIBCXX_USE_CXX11_ABI=1`. This symbol undefined issue appears when PyTorch\* is
    compiled with `_GLIBCXX_USE_CXX11_ABI=0`.
  - **Solution**: Pass `export GLIBCXX_USE_CXX11_ABI=1` and compile PyTorch\* with particular compiler which supports `_GLIBCXX_USE_CXX11_ABI=1`. We recommend using prebuilt wheels 
    in [download server](https:// developer.intel.com/ipex-whl-stable-xpu) to avoid this issue.
- **Problem**: Bad termination after AI model execution finishes when using Intel MPI.
  - **Cause**: This is a random issue when the AI model (e.g. RN50 training) execution finishes in an Intel MPI environment. It is not user-friendly as the model execution ends ungracefully.
  - **Solution**: Add `dist.destroy_process_group()` during the cleanup stage in the model script, as described 
    in [Getting Started with Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).
- **Problem**: `-997 runtime error` when running some AI models on Intel® Arc™ A-Series GPUs.
  - **Cause**:  Some of the `-997 runtime error` are actually out-of-memory errors. As Intel® Arc™ A-Series GPUs have less device memory than Intel® Data Center GPU Flex Series 170 and Intel® Data Center GPU 
    Max  Series, running some AI models on them may trigger out-of-memory errors and cause them to report failure such as `-997 runtime error` most likely. This is expected. Memory usage optimization is a work in progress to allow Intel® Arc™ A-Series GPUs to support more AI models.
- **Problem**: Building from source for Intel® Arc™ A-Series GPUs fails on WSL2 without any error thrown.
  - **Cause**: Your system probably does not have enough RAM, so Linux kernel's Out-of-memory killer was invoked. You can verify this by running `dmesg` on bash (WSL2 terminal).
  - **Solution**: If the OOM killer had indeed killed the build process, then you can try increasing the swap-size of WSL2, and/or decreasing the number of parallel build jobs with the environment 
    variable `MAX_JOBS` (by default, it's equal to the number of logical CPU cores. So, setting `MAX_JOBS` to 1 is a very conservative approach that would slow things down a lot).
- **Problem**: Some workloads terminate with an error `CL_DEVICE_NOT_FOUND` after some time on WSL2.
  - **Cause**:  This issue is due to the [TDR feature](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys#tdrdelay) on Windows.
  - **Solution**: Try increasing TDRDelay in your Windows Registry to a large value, such as 20 (it is 2 seconds, by default), and reboot.
  
### Library Dependencies

- **Problem**: Cannot find oneMKL library when building Intel® Extension for PyTorch\* without oneMKL.

  ```bash
  /usr/bin/ld: cannot find -lmkl_sycl
  /usr/bin/ld: cannot find -lmkl_intel_ilp64
  /usr/bin/ld: cannot find -lmkl_core
  /usr/bin/ld: cannot find -lmkl_tbb_thread
  dpcpp: error: linker command failed with exit code 1 (use -v to see invocation)
  ```
  
  - **Cause**: When PyTorch\* is built with oneMKL library and Intel® Extension for PyTorch\* is built without MKL library, this linker issue may occur.
  - **Solution**: Resolve the issue by setting:

    ```bash
    export USE_ONEMKL=OFF
    export MKL_DPCPP_ROOT=${HOME}/intel/oneapi/mkl/latest
    ```

   Then clean build Intel® Extension for PyTorch\*.

- **Problem**: Undefined symbol: `mkl_lapack_dspevd`. Intel MKL FATAL ERROR: cannot load `libmkl_vml_avx512.so.2` or `libmkl_vml_def.so.2.
  - **Cause**: This issue may occur when Intel® Extension for PyTorch\* is built with oneMKL library and PyTorch\* is not build with any MKL library. The oneMKL kernel may run into CPU backend incorrectly 
    and trigger this issue. 
  - **Solution**: Resolve the issue by installing the oneMKL library from conda:

    ```bash
    conda install mkl
    conda install mkl-include
    ```

   Then clean build PyTorch\*.

- **Problem**: OSError: `libmkl_intel_lp64.so.2`: cannot open shared object file: No such file or directory.
  - **Cause**: Wrong MKL library is used when multiple MKL libraries exist in system.
  - **Solution**: Preload oneMKL by:

    ```bash
    export LD_PRELOAD=${MKL_DPCPP_ROOT}/lib/intel64/libmkl_intel_lp64.so.2:${MKL_DPCPP_ROOT}/lib/intel64/libmkl_intel_ilp64.so.2:${MKL_DPCPP_ROOT}/lib/intel64/libmkl_gnu_thread.so.2:${MKL_DPCPP_ROOT}/lib/intel64/libmkl_core.so.2:${MKL_DPCPP_ROOT}/lib/intel64/libmkl_sycl.so.2
    ```

    If you continue seeing similar issues for other shared object files, add the corresponding files under `${MKL_DPCPP_ROOT}/lib/intel64/` by `LD_PRELOAD`. Note that the suffix of the libraries may change (e.g. from .1 to .2), if more than one oneMKL library is installed on the system.

### Unit Test

- Unit test failures on Intel® Data Center GPU Flex Series 170

  The following unit tests fail on Intel® Data Center GPU Flex Series 170 but the same test cases pass on Intel® Data Center GPU Max Series. The root cause of the failures is under investigation.
    - `test_multilabel_margin_loss.py::TestNNMethod::test_multiabel_margin_loss`
    - `test_weight_norm.py::TestNNMethod::test_weight_norm_differnt_type`
      
- Unit test failures on Intel® Data Center GPU Max Series

  The following unit tests randomly fail on Intel® Data Center GPU Max Series if running with other test cases together using `pytest -v`. These cases pass if run individually on the same environment. The root cause of the failures is under investigation.
  
     - `test_nn.py::TestNNDeviceTypeXPU::test_activations_bfloat16_xpu`
     - `test_eigh.py::TestTorchMethod::test_linalg_eigh`
     - `test_baddbmm.py::TestTorchMethod::test_baddbmm_scale`

  The following unit tests fail on Intel® Data Center GPU Max Series. The root cause of the failures is under investigation with oneDNN as the operators under test use oneDNN primitives.
  
     - `test_lstm.py::TestNNMethod::test_lstm_rnnt_onednn`
     - `test_conv_transposed.py::TestTorchMethod::test_deconv3d_bias`

- Unit test failures on CPU (ICX, CPX, SPR).

  The following unit test fails on CPU if using latest transformers versoin (4.31.0). The workaround solution is to use old version transformers by pip `install transformers==4.30.0` instead.
  
     - `test_tpp_ops.py::TPPOPsTester::test_tpp_bert_embeddings`

## CPU-specific issues

### General Usage

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

### TorchDynamo

- **Problem**: A workload that uses `torch.compile()` fails to run or demonstrates poor performance. 
  - **Cause**: The support of `torch.compile()` with `ipex` as the backend is still an experimental feature. Currently, the following HuggingFace models fail to run using `torch.compile()` with `ipex` backend due to memory issues:
    - masked-language-modeling+xlm-roberta-base
    - casual-language-modeling+gpt2
    - casual-language-modeling+xlm-roberta-base
    - summarization+t5-base
    - text-classification+allenai-longformer-base-409
  - **Solution**: Use the `torch.jit` APIs and graph optimization APIs of the Intel® Extension for PyTorch\*.

### Dynamic Shape

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

### INT8

- **Problem**: Low performance with INT8 support for dynamic shapes.
  - **Cause**: The support for dynamic shapes in Intel® Extension for PyTorch\* INT8 integration is still work in progress. When the input shapes are dynamic, for example inputs of variable image sizes in an object detection task or of variable sequence lengths in NLP tasks, the 
    Intel® Extension for PyTorch\* INT8 path may slow down the model inference.
  - **Solution**: Use stock PyTorch INT8 functionality.
  **Note**: Using Runtime Extension feature if batch size cannot be divided by number of streams, because mini batch size on each stream are not equivalent, scripts run into this issue.
- **Problem**: `RuntimeError: Overflow when unpacking long` when a tensor's min max value exceeds int range while performing int8 calibration.
  - **Solution**: Customize QConfig to use min-max calibration method.
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

### BFloat16

- BF16 AMP(auto-mixed-precision) runs abnormally with the extension on the AVX2-only machine if the topology contains `Conv`, `Matmul`, `Linear`, and `BatchNormalization`

### Runtime Extension

The following limitations currently exist:

- Runtime extension of `MultiStreamModule` does not support DLRM inference, since the input of DLRM (EmbeddingBag specifically) cannot be simply batch split.
- Runtime extension of `MultiStreamModule` has poor performance of RNNT Inference comparing with native throughput mode. Only part of the RNNT models (`joint_net` specifically) can be jit traced into graph. However, in one batch inference, `joint_net` is invoked multiple times. 
  It increases the   overhead of `MultiStreamModule` as input batch split, thread synchronization and output concat.

### Result Correctness

- **Problem**: Incorrect Conv and Linear result if the number of OMP threads is changed at runtime.
  - **Cause**: The oneDNN memory layout depends on the number of OMP threads, which requires the caller to detect the changes for the # of OMP threads while this release has not implemented it yet.

### Float32 Training

- **Problem**: Low throughput with DLRM FP32 Train.
  - **Solution**: A 'Sparse Add' [PR](https://github.com/pytorch/pytorch/pull/23057) is pending on review. The issue will be fixed when the PR is merged.
