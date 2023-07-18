# Advanced Configuration

The default settings for Intel® Extension for PyTorch\* are sufficient for most use cases. However, if users want to customize Intel® Extension for PyTorch\*, advanced configuration is available at build time and runtime. 

## Build Time Configuration

The following build options are supported by Intel® Extension for PyTorch\*. Users who install Intel® Extension for PyTorch\* via source compilation could override the default configuration by explicitly setting a build option ON or OFF, and then build. 

| **Build Option** | **Default<br> Value** | **Description** |
| ------ | ------ | ------ |
| USE_ONEMKL | ON | If ON, use oneMKL BLAS library. If OFF, oneMKL is not built in. |
| USE_CHANNELS_LAST_1D | ON | If ON, channels last 1D memory format is supported. If OFF, channels last 1D memory format is not supported. |
| USE_PERSIST_STREAM | ON | If ON, persistent oneDNN stream is used. If OFF, oneDNN stream is created and destroyed on demand. |
| USE_PRIMITIVE_CACHE | OFF | If ON, use Intel® Extension for PyTorch* solution to cache oneDNN primitives. <br> If OFF, use oneDNN cache solution. |
| USE_QUEUE_BARRIER | ON | If ON, use queue submit barrier. If OFF, use dummy kernel. |
| USE_SCRATCHPAD_MODE | ON | If ON, use oneDNN user mode scratchpad. If OFF, use oneDNN library mode scratchpad. |
| USE_MULTI_CONTEXT | ON | If ON, create multiple DPC++ runtime contexts per device. If OFF, create single DPC++ runtime context per device. |
| USE_AOT_DEVLIST | "" | Device list for AOT compilation. Refer to [AOT](../AOT.md) for how to configure this build option. |
| BUILD_STATS | OFF | If ON, count statistics for each component during build process. If OFF, statistics are not counted. |
| BUILD_BY_PER_KERNEL | OFF | If ON, build with -fsycl-device-code-split=per_kernel option. If OFF, this option is not set. |
| BUILD_STRIPPED_BIN | OFF | If ON, strip all symbols when building Intel® Extension for PyTorch* libraries. If OFF, symbols are kept. |
| BUILD_SEPARATE_OPS | OFF | If ON, build each operator in separate library. If OFF, build all operators in global library. |
| BUILD_SIMPLE_TRACE | OFF | If ON, collect simple trace info for each registered operator. If OFF, simple trace is not built in. |
| BUILD_OPT_LEVEL | "" | If set to 0, build with -O0 option, if ON and set to 1, build with -O1 option. Set to other value except 0 or 1, no build option is added. |
| BUILD_NO_CLANGFORMAT | OFF | If ON, build without force clang-format check. If OFF, build with force clang-format check. |
| BUILD_INTERNAL_DEBUG | OFF | If ON, use internal debug code path. If OFF, internal debug code path is not used. |

For above build options which can be configured to ON or OFF, users can configure them to 1 or 0 also, while ON equals to 1 and OFF equals to 0.

## Runtime Configuration

The following launch options are supported in Intel® Extension for PyTorch\*. Users who execute AI models on XPU could override the default configuration by explicitly setting the option value at runtime using environment variables, and then launch the execution.

| **Launch Option** | **Default<br> Value** | **Description** |
| ------ | ------ | ------ |
| IPEX_VERBOSE | 0 | Verbose level in integer. Set to 1 to print verbose output for Intel® Extension for PyTorch* GPU customized kernel. Set to other value is not supported so far. |
| IPEX_SIMPLE_TRACE | OFF | Simple trace functionality. If set to ON, enable simple trace for all operators. Set to other value is not supported. |
| IPEX_TILE_AS_DEVICE | ON | Device partition. If set to OFF, tile partition will be disabled and map device to physical device. Set to other value is not supported. |
| IPEX_XPU_SYNC_MODE | OFF | Kernel Execution mode. If set to ON, use synchronized execution mode and perform blocking wait for the completion of submitted kernel. Set to other value is not supported. |
| IPEX_FP32_MATH_MODE | FP32 | Floating-point math mode. Set to TF32 for using TF32 math mode,  BF32 for using BF32 math mode. Set to other value is not supported. Refer to https://github.com/oneapi-src/oneDNN/tree/rfcs/rfcs/20210301-computation-datatype for the definition of TF32 and BF32 math mode. |

For above launch options which can be configured to 1 or 0, users can configure them to ON or OFF also, while ON equals to 1 and OFF equals to 0.

Examples to configure the launch options:</br>

- Set one or more options before running the model

```bash
export IPEX_VERBOSE=1
export IPEX_FP32_MATH_MODE=TF32
...
python ResNet50.py
```
- Set one option when running the model

```bash
IPEX_VERBOSE=1 python ResNet50.py
```

- Set more than one options when running the model

```bash
IPEX_VERBOSE=1 IPEX_FP32_MATH_MODE=TF32 python ResNet50.py
```
