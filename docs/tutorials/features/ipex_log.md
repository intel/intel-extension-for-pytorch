IPEX Logging usage
===============================================
<style>
table {
    margin: auto;
}

</style>


## Introduction

IPEX_LOGGING provides the capacity to log IPEX internal information. If you would like to use torch-style log, that is, the log/verbose is introduced by torch, and refer to cuda code, pls still use torch macro to show the log. For example, TORCH_CHECK, TORCH_ERROR. If the log is IPEX specific, or is going to trace IPEX execution, pls use IPEX_LOGGING. For some part of usage are still discussed with habana side, if has change some feature will update here.

## Feature for IPEX Log
### Log level
Currently supported log level and usage are as follow, default using log level is `WARN`:

|  log level   | number  | usage |
|  :----:   | :----:   | :----: |
| TRACE  | 0 | reserve it for further usage extension|
| DEBUG  | 1 | We would like to insert DEBUG inside each host function, when log level is debug, we can get the whole calling stack |
| INFO  | 2 | Record calls to other library functions and environment variable settings, such as onemkl calling and set verbose level|
| WARN  | 3 | On the second attempt of the program, such as memory reallocation |
| ERR  | 4 | Found error in try catch |
| CRITICAL  | 5 | reserve it for further usage extension |

### Log component
Log component is for specify which part of IPEX does this log belongs to, currently we have seprate IPEX into four parts, shown as table below.

|  log component   | description |
|  :----:   | :----:   
| OPS  | Intergrate/Launch sycl onednn, onemkl ops | 
| SYNGRAPH  | Habana Syngraph related | 
| MEMORY  | Allocate/Free memory, Allocate/Free cache | 
| RUNTIME  | Device / Queue related |

For `SYNGRAPH` you can also add log sub componment which is no restriction on categories.


## How to add log in IPEX
All the usage are inside file `utils/LogUtils.h`. IPEX Log support two types of log usage, the first one is simple log, you can use IPEX_XXX_LOG, XXX represents the log level, including six log level mentioned above. There are for params for simple log, the first one is log component, representing which part of IPEX does this log belongs to. The second on is log sub component, for most of the IPEX usage, just input a empty string("") here. For the third param, it is an log message template, you can use it as python format string, or you can also refer to fmt lib https://github.com/fmtlib/fmt. Here is an example for simple log, for add a log inside abs kernel:
``` c++

IPEX_INFO_LOG("OPS", "", "Add a log for inside ops {}", "abs");

```

For the second log level is event log, which is used for recording a whole event, such as a ops calculation. A whole event is identify through a event_id, for the whole event this event_id should be the same, but cannot be duplicate with other event_id, or it will met an undefined behaviour. You can also mark each step by using a step_id, there are no limitation for step_id. For the end of the whole event, should use IPEX_XXX_EVENT_END(), XXX is the log level mention aboved.

Below is an example for ipex event log:
```c++
IPEX_EVENT_END("OPS", "", "record_avg_pool", "start", "Here record the time start with arg:{}", arg);
prepare_data();
IPEX_EVENT_END("OPS", "", "record_avg_pool", "data_prepare_finish", "Here record the data_prepare_finish with arg:{}", arg);
avg_pool();
IPEX_INFO_EVENT_END("OPS", "", "record_avg_pool", "finish conv", "Here record the end");
```

## Enviornment settings
IPEX privide some enviornment setting used for log output settings, currently, we support below five settings.

1. IPEX_LOGGING_LEVEL, accept int or string, default is 3 for `WARN`. Currently you can choose seven different log level within ipex, including -1 `DISABLED` under this setting, all the usage related with IPEX_LOGGING will be disabled. Another six log levels are we mentioned above.
2. IPEX_LOG_COMPONENT, accept a string, sepreated by `/` first part is log component and the second part is for log sub component, used for state which component and sub log component you would like to log, default is "ALL". Currently we supoort 5 different log component,, `ALL` is for all the output in IPEX, the other four log component are we mentioned above. You could also specify several log component, sepreating using `,` such as "OPS;MEMORY".For log sub component, it still discussed with habana side, pls don't use it first.
3. IPEX_LOG_OUTPUT, accept a string. If you are using IPEX_LOG_OUTPUT, than all the logs will log inside the file rather than log into the console, you can use it like export IPEX_LOG_OUTPUT="./ipex.log", all the log will log inside ipex.log in current work folder.
4. IPEX_LOG_ROTATE_SIZE, accept a int, default =10. Only validate when export IPEX_LOG_OUTPUT, specifing how large file will be used when rotating this log, size is MB.
5. IPEX_LOG_SPLIT_SIZE, accept a int, default = null. Only validate when export IPEX_LOG_OUTPUT, specifing how large file will be used when split this log, size is MB.

## Usage in python
1. torch.xpu.set_log_level(log_level) and torch.xpu.get_log_level(), these two functions are used for get and set the log level.
2. torch.xpu.set_log_output_file_path(log_path) and torch.xpu.get_log_output_file_path(), these two functions are used for get and set the log output file path, once log output file path is set, logs will not be print on the console, will only output in the file.
3. torch.xpu.set_log_rotate_file_size(file size) and torch.xpu.get_log_rotate_file_size(), these two functions are used for get and set the log rotate file size, only validate when output file path is set.
4. torch.xpu.set_log_split_file_size(file size) and torch.xpu.get_log_split_file_size(), these two functions are used for get and set the log split file size, only validate when output file path is set.
5. torch.xpu.set_log_component(log_component), and torch.xpu.get_log_component(), these two functions are used for get and set the log component, log component string are same with enviornment settings.

## Use IPEX log for simple trace
For now, IPEX_SIMPLE_TRACE is depre deprecated, and pls use torch.xpu.set_log_level(0), it will show logs like previous IPEX_SIMPLE_TRACE.

## Use IPEX log for verbose
For now, IPEX_VERBOSE is deprecated, pls use torch.xpu.set_log_level(1), it will show logs like previous IPEX_VERBOSE.