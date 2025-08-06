Troubleshooting
===============

## General Usage

- **Problem**: FP64 data type is unsupported on current platform.
  - **Cause**: FP64 is not natively supported by the [Intel® Arc™ A-Series Graphics](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/arc.html) platforms.
    If you run any AI workload on that platform and receive this error message, it means a kernel requires FP64 instructions that are not supported and the execution is stopped.
- **Problem**: Runtime error `invalid device pointer` if `import horovod.torch as hvd` before `import intel_extension_for_pytorch`.
  - **Cause**: Intel® Optimization for Horovod\* uses utilities provided by Intel® Extension for PyTorch\*. The improper import order causes Intel® Extension for PyTorch\* to be unloaded before Intel®
    Optimization for Horovod\* at the end of the execution and triggers this error.
  - **Solution**: Do `import intel_extension_for_pytorch` before `import horovod.torch as hvd`.
- **Problem**: Number of dpcpp devices should be greater than zero.
  - **Cause**: If you use Intel® Extension for PyTorch\* in a conda environment, you might encounter this error. Conda also ships the libstdc++.so dynamic library file that may conflict with the one shipped in the OS.
  - **Solution**: Export the `libstdc++.so` file path in the OS to an environment variable `LD_PRELOAD`.
- **Problem**: `-997 runtime error` when running some AI models on Intel® Arc™ Graphics family.
  - **Cause**:  Some of the `-997 runtime error` are actually out-of-memory errors. As Intel® Arc™ Graphics GPUs have less device memory than Intel® Data Center GPU Flex Series 170 and Intel® Data Center GPU
    Max  Series, running some AI models on them may trigger out-of-memory errors and cause them to report failure such as `-997 runtime error` most likely. This is expected. Memory usage optimization is working in progress to allow Intel® Arc™ Graphics GPUs to support more AI models.
- **Problem**: Building from source for Intel® Arc™ A-Series GPUs fails on WSL2 without any error thrown.
  - **Cause**: Your system probably does not have enough RAM, so Linux kernel's Out-of-memory killer was invoked. You can verify this by running `dmesg` on bash (WSL2 terminal).
  - **Solution**: If the OOM killer had indeed killed the build process, then you can try increasing the swap-size of WSL2, and/or decreasing the number of parallel build jobs with the environment
    variable `MAX_JOBS` (by default, it's equal to the number of logical CPU cores. So, setting `MAX_JOBS` to 1 is a very conservative approach that would slow things down a lot).
- **Problem**: Some workloads terminate with an error `CL_DEVICE_NOT_FOUND` after some time on WSL2.
  - **Cause**:  This issue is due to the [TDR feature](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys#tdrdelay) on Windows.
  - **Solution**: Try increasing TDRDelay in your Windows Registry to a large value, such as 20 (it is 2 seconds, by default), and reboot.

## Library Dependencies

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

- **Problem**: If you encounter issues related to MPI environment variable configuration when running distributed tasks.
  - **Cause**: MPI environment variable configuration not correct.
  - **Solution**: `conda deactivate` and then `conda activate` to activate the correct MPI environment variable automatically.

    ```
    conda deactivate
    conda activate
    ```

- **Problem**: If you encounter issues Runtime error related to C++ compiler with `torch.compile`. Runtime Error: Failed to find C++ compiler. Please specify via CXX environment variable.
  - **Cause**: Not activate C++ compiler. `torch.compile` need to find correct `cl.exe` path.
  - **Solution**: One could open "Developer Command Prompt for VS 2022" or follow [Visual Studio Developer Command Prompt and Developer PowerShell](https://learn.microsoft.com/en-us/visualstudio/ide/reference/command-prompt-powershell?view=vs-2022#developer-command-prompt) to activate visual studio environment.

- **Problem**: If you encounter a system hang issue when executing llama3-8b and phi3-mini FSDP fine-tuning cases based on XCCL backend on Intel® Data Center GPU Max platform, and the hang occurs after workload completion and before the process exits.
  - **Cause**: Compatibility issue between accelerate v1.8.1 and transformers v4.51.3.
  - **Solution**: Use torch-ccl to replace XCCL to workaround.

## Performance Issue

- **Problem**: Extended durations for data transfers from the host system to the device (H2D) and from the device back to the host system (D2H).
  - **Cause**: Absence of certain Dynamic Kernel Module Support (DKMS) packages on Ubuntu 22.04 or earlier versions.
  - **Solution**: For those running Ubuntu 22.04 or below, it's crucial to follow all the recommended installation procedures, including those labeled as [optional](https://dgpu-docs.intel.com/driver/client/overview.html#optional-out-of-tree-kernel-mode-driver-install). These steps are likely necessary to install the missing DKMS packages and ensure your system is functioning optimally. The Kernel Mode Driver (KMD) package that addresses this issue has been integrated into the Linux kernel for Ubuntu 23.04 and subsequent releases.

