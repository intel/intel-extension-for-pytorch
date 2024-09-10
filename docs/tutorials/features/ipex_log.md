`IPEX_LOG` (Prototype)
==========================

## Introduction

`IPEX_LOG` provides the capability to log verbose information from Intel® Extension for PyTorch\* . Please use `IPEX_LOG` to get the log information or trace the execution from Intel® Extension for PyTorch\*. Please continue using PyTorch\* macros such as `TORCH_CHECK`, `TORCH_ERROR`, etc. to get the log information from PyTorch\*.

## `IPEX_LOG` Definition
### Log Level
The supported log levels are defined as follows, default log level is `DISABLED`:

|  log level   | number  | usage |
|  :----:   | :----:   | :----: |
| DISABLED  | -1 | Disable the logging |
| TRACE  | 0 | Reserve for further usage |
| DEBUG  | 1 | Provide the whole calling stack info |
| INFO  | 2 | Record calling info to other library functions and environment variable settings |
| WARN  | 3 | Warn the second attempt of an action, such as memory reallocation |
| ERR  | 4 | Report error in try catch |
| CRITICAL  | 5 | Reserve for further usage |

### Log Component
Log component is used to specify which part from Intel® Extension for PyTorch\* does this log information belong to. The supported log components are defined as follows:

|  log component   | description |
|  :----:   | :----:   
| OPS  | Launch SYCL, oneDNN, oneMKL operators | 
| SYNGRAPH  | Syngraph related | 
| MEMORY  | Allocate/Free memory, Allocate/Free cache | 
| RUNTIME  | Device / Queue related |
| ALL  | All output log |

## Usage in C++
All the usage are defined in `utils/LogUtils.h`. Currently Intel® Extension for PyTorch\* supports:

### Simple Log
You can use `IPEX_XXX_LOG`, XXX represents the log level as mentioned above. There are four parameters defined for simple log:
- Log component, representing which part of Intel® Extension for PyTorch\* does this log belong to.
- Log sub component, input an empty string("") for general usages. For `SYNGRAPH` you can add any log sub componment.
- Log message template format string, same as fmt_string in lib fmt, `{}` is used as a place holder for format args .
- Log args for template format string, args numbers should be aligned with size of `{}`s.

Below is an example for using simple log inside abs kernel:

``` c++

IPEX_INFO_LOG("OPS", "", "Add a log for inside ops {}", "abs");

```
### Event Log
Event log is used for recording a whole event, such as an operator calculation. The whole event is identified by an unique `event_id`. You can also mark each step by using `step_id`. Use `IPEX_XXX_EVENT_END()` to complete the logging of the whole event. `XXX` represents the log level mentioned above. It will be used as the log level for all logs within one single log event.

Below is an example for using event log:

```c++
IPEX_EVENT_LOG("OPS", "", "record_avg_pool", "start", "Here record the time start with arg:{}", arg);
prepare_data();
IPEX_EVENT_LOG("OPS", "", "record_avg_pool", "data_prepare_finish", "Here record the data_prepare_finish with arg:{}", arg);
avg_pool();
IPEX_INFO_EVENT_END("OPS", "", "record_avg_pool", "finish conv", "Here record the end");
```

## Enviornment settings
Intel® Extension for PyTorch\* provides five enviornment variables for configuring log output:

- `IPEX_LOG_LEVEL`, accept integar or string, default is -1 for `DISABLED`. 
- `IPEX_LOG_COMPONENT`, accept string, used for specifying the log component and sub log component you would like to log, default is "ALL". The log component and sub log component are separated by `/`. You could also specify several log components, such as "OPS;MEMORY".
- `IPEX_LOG_OUTPUT`, accept string. If you are using `IPEX_LOG_OUTPUT`, than all the logs will recorded inside a file rather than the console. Example: export IPEX_LOG_OUTPUT="./ipex.log".
- `IPEX_LOG_ROTATE_SIZE`, accept integar, default is 10. Can be used only with `IPEX_LOG_OUTPUT`, for specifing how large file will be used when rotating this log, size is MB.
- `IPEX_LOG_SPLIT_SIZE`, accept integar, default = null. Can be used only with `IPEX_LOG_OUTPUT`, for specifing how large file will be used when splitting the logs, size is MB.

## Usage in python
- `torch.xpu.set_log_level(log_level)` and `torch.xpu.get_log_level()`, these two functions are used for getting and setting the log level.
- `torch.xpu.set_log_output_file_path(log_path)` and `torch.xpu.get_log_output_file_path()`, these two functions are used for getting and setting the log output file path, once log output file path is set, logs will be recorded in file only.
- `torch.xpu.set_log_rotate_file_size(file size)` and `torch.xpu.get_log_rotate_file_size()`, these two functions are used for getting and setting the log rotate file size. Can be used when output file path is set.
- `torch.xpu.set_log_split_file_size(file size)` and `torch.xpu.get_log_split_file_size()`, these two functions are used for getting and setting the log split file size. Can be used when output file path is set.
- `torch.xpu.set_log_component(log_component)`, and `torch.xpu.get_log_component()`, these two functions are used for getting and setting the log component. The log component string are the same as defined in enviornment settings.

## Replace `IPEX_SIMPLE_TRACE`
Use `torch.xpu.set_log_level(0)` to get logs to replace the previous usage in `IPEX_SIMPLE_TRACE`.

## Replace `IPEX_VERBOSE`
Use `torch.xpu.set_log_level(1)` to get logs to replace the previous usage in `IPEX_VERBOSE`.

