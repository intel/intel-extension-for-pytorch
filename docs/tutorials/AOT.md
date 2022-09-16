# Ahead of Time (AOT) Compilation

## Introduction

[AOT Compilation](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compilation/ahead-of-time-compilation.html) is a helpful feature for development lifecycle or distribution time, when you know beforehand what your target device is going to be at application execution time. When AOT compilation is enabled, no additional compilation time is needed when running application. It also benifits the product quality since no just-in-time (JIT) bugs encountered as JIT is skipped and final code executing on the target device can be tested as-is before deliver to end-users. The disadvantage of this feature is that the final distributed binary size will be increased a lot (e.g. from 500MB to 2.5GB for Intel® Extension for PyTorch\*).

## Use case

Intel® Extension for PyTorch\* provides build option `USE_AOT_DEVLIST` for end-users who install Intel® Extension for PyTorch\* via source compilation to configure device list for AOT compilation. The target device in device list is specified by DEVICE name of the target. Multi-target AOT compilation is supported by using comma (,) as delimiter in device list. See below table for the AOT setting targeting [Intel® Data Center GPU Flex Series](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/data-center-gpu/flex-series/overview.html).

| Supported HW | AOT Setting |
| ------------ |---------------------|
| Intel® Data Center GPU Flex Series | USE_AOT_DEVLIST='dg2-g10-c0' |

Intel® Extension for PyTorch\* enables AOT compilation for Intel® GPU target devices in prebuilt wheel files. Intel® Data Center GPU Flex Series is the enabled target device in current release. If Intel® Extension for PyTorch\* is executed on a device which is not pre-configured in `USE_AOT_DEVLIST`, this application can still run as JIT compilation will be triggered automatically for allowing execution on the current device. It causes additional compilation time during execution however.

## Requirement

[Intel® Graphics Compute Runtime for oneAPI Level Zero and OpenCL™ Driver](https://github.com/intel/compute-runtime/releases) must be installed beforehand to use AOT compilation. Once `USE_AOT_DEVLIST` is configured, Intel® Extension for PyTorch\* will provide `-fsycl-targets=spir64_gen` option and `-Xs "-device ${USE_AOT_DEVLIST}"` option for generating binaries that utilize Intel® oneAPI Level Zero backend.
