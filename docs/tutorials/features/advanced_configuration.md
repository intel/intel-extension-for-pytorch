# Advanced Configuration

The default settings for Intel® Extension for PyTorch\* are sufficient for most use cases. However, if users want to customize Intel® Extension for PyTorch\*, advanced configuration is available at build time and runtime. 

## Build Time Configuration

The following build options are supported by Intel® Extension for PyTorch\*. Users who install Intel® Extension for PyTorch\* via source compilation could override the default configuration by explicitly setting a build option ON or OFF, and then build. 

| **Build Option** | **Default<br>Value** | **Description** |
| ------ | ------ | ------ |
| USE_ONEMKL | ON | Use oneMKL BLAS |
| USE_CHANNELS_LAST_1D | ON | Use channels last 1d |
| USE_PERSIST_STREAM | ON | Use persistent oneDNN stream |
| USE_SCRATCHPAD_MODE | ON | Use oneDNN scratchpad mode |
| USE_PRIMITIVE_CACHE | OFF | Cache oneDNN primitives by FRAMEWORK, instead of oneDNN itself |
| USE_QUEUE_BARRIER | ON | Use queue submit_barrier, otherwise use dummy kernel |
| USE_MULTI_CONTEXT | ON | Create DPC++ runtime context per device |
| USE_PROFILER | ON | USE XPU Profiler in build. |
| USE_SYCL_ASSERT | OFF | Enables assert in sycl kernel |
| USE_ITT_ANNOTATION | OFF | Enables ITT annotation in sycl kernel |
| BUILD_BY_PER_KERNEL | OFF | Build by DPC++ per_kernel option (exclusive with USE_AOT_DEVLIST) |
| BUILD_INTERNAL_DEBUG | OFF | Use internal debug code path |
| BUILD_SEPARATE_OPS | OFF | Build each operator in separate library |
| BUILD_SIMPLE_TRACE | ON | Build simple trace for each registered operator |
| BUILD_JIT_QUANTIZATION_SAVE | OFF | Support jit quantization model save and load |
| USE_AOT_DEVLIST | "" | Set device list for AOT build |
| BUILD_OPT_LEVEL | "" | Add build option -Ox, accept values: 0/1 |

For above build options which can be configured to ON or OFF, users can configure them to 1 or 0 also, while ON equals to 1 and OFF equals to 0.

## Runtime Configuration

The following launch options are supported in Intel® Extension for PyTorch\*. Users who execute AI models on XPU could override the default configuration by explicitly setting the option value at runtime using environment variables, and then launch the execution.

| **Launch Option<br>CPU, GPU** | **Default<br>Value** | **Description** |
| ------ | ------ | ------ |
| IPEX_VERBOSE | 0 | Set verbose level with synchronization execution mode |
| IPEX_FP32_MATH_MODE | 0 | Set values for FP32 math mode (0: FP32, 1: TF32, 2: BF32) |

| **Launch Option<br>GPU ONLY** | **Default<br>Value** | **Description** |
| ------ | ------ | ------ |
| IPEX_XPU_SYNC_MODE | 0 | Set 1 to enforce synchronization execution mode |
| IPEX_TILE_AS_DEVICE | 1 | Set 0 to disable tile partition and map per root device |

| **Launch Option<br>Experimental** | **Default<br>Value** | **Description** |
| ------ | ------ | ------ |
| IPEX_SIMPLE_TRACE | 0 | Set 1 to enable simple trace for all operators\* |

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
