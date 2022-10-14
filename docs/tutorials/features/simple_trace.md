
# Simple Trace Tool [EXPERIMENTAL]

## Introduction

Simple Trace is a built-in debugging tool that lets you control printing out the call stack for a piece of code. You can enable this tool and have it automatically print out verbose messages of called operators in a stack format with indenting to distinguish the context. You can enable and disable this tool using a simple method.

## Use Case

To use the simple trace tool, you need to build Intel® Extension for PyTorch\* from source and add explicit calls to enable and disable tracing in your model script. When enabled, the trace messages will be printed to the console screen by default, along with verbose log messages.

### Build Tool

Add the Simple Trace tool by turning on the build option `BUILD_SIMPLE_TRACE=ON` and then build Intel® Extension for PyTorch\* from source.

```unix
BUILD_SIMPLE_TRACE=ON python setup.py install
```

### Enable and Disable Tool

In your model script, bracket code in your model script with calls to `torch.xpu.enable_simple_trace()` and `torch.xpu.disable_simple_trace()`, as shown in the following example:

```python
# import all necessary libraries
import torch
import intel_extension_for_pytorch

print(torch.xpu.using_simple_trace())   # False
a = torch.randn(100).xpu()              # this line won't be traced

torch.xpu.enable_simple_trace()         # to enable simple trace tool

# test code (with tracing enabled) begins here
b = torch.randn(100).xpu()
c = torch.unique(b)
# test code ends here

torch.xpu.disable_simple_trace()        # to disable simple trace tool
```

The simple trace output will start after being enabled, and will continue until
the call to disable it, so be careful with your model script logic so the disable call is
not unintentionally skipped.

### Results

Using the script shown above as the exmaple, you'll see these messages printed out to the console:

```text
[262618.262618]  Call  into  OP: wrapper__empty_strided -> at::AtenIpexTypeXPU::empty_strided (#0)
[262618.262618]  Step out of OP: wrapper__empty_strided -> at::AtenIpexTypeXPU::empty_strided (#0)
[262618.262618]  Call  into  OP: wrapper__copy_ -> at::AtenIpexTypeXPU::copy_ (#1)
[262618.262618]  Step out of OP: wrapper__copy_ -> at::AtenIpexTypeXPU::copy_ (#1)
[262618.262618]  Call  into  OP: wrapper___unique2 -> at::AtenIpexTypeXPU::_unique2 (#2)
[262618.262618]    Call  into  OP: wrapper__clone -> at::AtenIpexTypeXPU::clone (#3)
[262618.262618]      Call  into  OP: wrapper__empty_strided -> at::AtenIpexTypeXPU::empty_strided (#4)
[262618.262618]      Step out of OP: wrapper__empty_strided -> at::AtenIpexTypeXPU::empty_strided (#4)
[262618.262618]      Call  into  OP: wrapper__copy_ -> at::AtenIpexTypeXPU::copy_ (#5)
[262618.262618]      Step out of OP: wrapper__copy_ -> at::AtenIpexTypeXPU::copy_ (#5)
[262618.262618]    Step out of OP: wrapper__clone -> at::AtenIpexTypeXPU::clone (#3)
[262618.262618]    Call  into  OP: wrapper___reshape_alias -> at::AtenIpexTypeXPU::_reshape_alias (#6)
[262618.262618]    Step out of OP: wrapper___reshape_alias -> at::AtenIpexTypeXPU::_reshape_alias (#6)
[262618.262618]    Call  into  OP: wrapper_memory_format_empty -> at::AtenIpexTypeXPU::empty (#7)
[262618.262618]    Step out of OP: wrapper_memory_format_empty -> at::AtenIpexTypeXPU::empty (#7)
[262618.262618]    Call  into  OP: wrapper_memory_format_empty -> at::AtenIpexTypeXPU::empty (#8)
[262618.262618]    Step out of OP: wrapper_memory_format_empty -> at::AtenIpexTypeXPU::empty (#8)
[262618.262618]    Call  into  OP: wrapper_memory_format_empty -> at::AtenIpexTypeXPU::empty (#9)
[262618.262618]    Step out of OP: wrapper_memory_format_empty -> at::AtenIpexTypeXPU::empty (#9)
[262618.262618]    Call  into  OP: wrapper_memory_format_empty -> at::AtenIpexTypeXPU::empty (#10)
[262618.262618]    Step out of OP: wrapper_memory_format_empty -> at::AtenIpexTypeXPU::empty (#10)
[262618.262618]    Call  into  OP: wrapper_memory_format_empty -> at::AtenIpexTypeXPU::empty (#11)
[262618.262618]    Step out of OP: wrapper_memory_format_empty -> at::AtenIpexTypeXPU::empty (#11)
[262618.262618]    Call  into  OP: wrapper_memory_format_empty -> at::AtenIpexTypeXPU::empty (#12)
[262618.262618]    Step out of OP: wrapper_memory_format_empty -> at::AtenIpexTypeXPU::empty (#12)
[262618.262618]    Call  into  OP: wrapper_memory_format_empty -> at::AtenIpexTypeXPU::empty (#13)
[262618.262618]    Step out of OP: wrapper_memory_format_empty -> at::AtenIpexTypeXPU::empty (#13)
[262618.262618]    Call  into  OP: wrapper_memory_format_empty -> at::AtenIpexTypeXPU::empty (#14)
[262618.262618]    Step out of OP: wrapper_memory_format_empty -> at::AtenIpexTypeXPU::empty (#14)
[262618.262618]    Call  into  OP: wrapper__as_strided -> at::AtenIpexTypeXPU::as_strided (#15)
[262618.262618]    Step out of OP: wrapper__as_strided -> at::AtenIpexTypeXPU::as_strided (#15)
[262618.262618]    Call  into  OP: wrapper___local_scalar_dense -> at::AtenIpexTypeXPU::_local_scalar_dense (#16)
[262618.262618]    Step out of OP: wrapper___local_scalar_dense -> at::AtenIpexTypeXPU::_local_scalar_dense (#16)
[262618.262618]    Call  into  OP: wrapper__as_strided -> at::AtenIpexTypeXPU::as_strided (#17)
[262618.262618]    Step out of OP: wrapper__as_strided -> at::AtenIpexTypeXPU::as_strided (#17)
[262618.262618]    Call  into  OP: wrapper___local_scalar_dense -> at::AtenIpexTypeXPU::_local_scalar_dense (#18)
[262618.262618]    Step out of OP: wrapper___local_scalar_dense -> at::AtenIpexTypeXPU::_local_scalar_dense (#18)
[262618.262618]    Call  into  OP: wrapper__resize_ -> at::AtenIpexTypeXPU::resize_ (#19)
[262618.262618]    Step out of OP: wrapper__resize_ -> at::AtenIpexTypeXPU::resize_ (#19)
[262618.262618]  Step out of OP: wrapper___unique2 -> at::AtenIpexTypeXPU::_unique2 (#2)
[262618.262618]  Call  into  OP: wrapper__copy_ -> at::AtenIpexTypeXPU::copy_ (#20)
[262618.262618]  Step out of OP: wrapper__copy_ -> at::AtenIpexTypeXPU::copy_ (#20)
```

The meanings of each field are shown as below:

- `pid.tid`, `[262618.262618]`: the process id and the thread id responsible to the printed-out line.
- `behavior`, `Call into OP`, `Step out of OP`: call-in or step-out behavior of the operators in a run.
- `name1 -> name2`, `wrapper__empty_strided -> at::AtenIpexTypeXPU::empty_strided`: the calling operator for the current step. The name1 before the arrow shows the wrapper from PyTorch. The name2 after the arrow shows the function name of which was called in or stepped out in Intel® Extension for PyTorch\* at the current step.
- `(#No.)`, `(#0)`: index of the called operators. This index was numbered from 0 in the order of each operator to be called.
- `indent`: the indent ahead of every behavior shows the nested relationship between operators. The operator call-in line with more indent should be a child of what was called above it.

With this output, you can see the calling stack of the traced script without using complicated debug tools such as gdb.
