Codeless Optimization (Experimental)
====================================

This feature aims to get inference performance benefits from Intel® Extension for PyTorch\* without changing code in your python scripts, which can raise Out-of-Box (OOB) experience to get started with Intel® Extension for PyTorch\* easily. Users who already known how to apply optimizations with Intel® Extension for PyTorch\* APIs are not targeted for this feature, due to the inevitable overhead and limitations we mentioned below.

## Motivation

A typical use case of inference as in [transformer](https://github.com/huggingface/transformers/blob/v4.21.1/src/transformers/trainer.py#L3187) can be simplified as the code snippet below:

```
import torch
model = Model().eval()
with torch.no_grad():
    for input in dataloader():
        model(**input)
```

To utilize optimizations of Intel® Extension for PyTorch\* for optimum performance, several lines code changes are required/recommended.

```
import torch
impot intel_extension_for_pytorch as ipex # clause added
model = Model().eval()
model = ipex.optimization(model)          # clause added
with torch.no_grad():
  with torch.cpu.amp.autocast():          # clause added for running with BFloat16 (Optional)
    input = ...                           # clause added for TorchScript (Optional, but recommended) 
    model = torch.jit.trace(input)        # clause added for TorchScript (Optional, but recommended) 
    model = torch.jit.freeze()            # clause added for TorchScript (Optional, but recommended) 
    for input in dataloader():
      model(**input)
```

With this feature, code changes above done manually are not required any more. Intel® Extension for PyTorch\* optimizations will be applied automatically during execution in a monkey patch way. 
* Automatically import `intel_extension_for_pytorch` package: It applies Intel® Extension for PyTorch\* optimizations, such as: `torch.embedding_bag`, `torch.cpu.amp.autocast`. It also registers Intel® Extension for PyTorch\* JIT fusion pass and thus benefits the graph mode inference performance.
* Automatically apply `ipex.optimize()` function. Only features enabled by default parameter values are supported, such as:
    * Auto generate FX or Jit Graph.
    * Auto Channel Last convert.
    * Conv-Bn folding.
    * Weight prepack.
    * Replace dropout with identity.
    * Optimize LSTM.
* Automatically apply `torch.cpu.amp.autocast` with BFloat16 data type for inference.

## Example Usage with HuggingFace
Let's take the [QA case](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) in HuggingFace as an example.

### The origin command with ipex launch
Here is the command to run with [`ipexrun`](../performance_tuning/launch_script.md).
```
clear && ipexrun --use_default_allocator --ninstance 2 --ncore_per_instance 28 run_qa.py --model_name_or_path bert-base-uncased --dataset_name squad --do_eval --per_device_train_batch_size 12 --learning_rate 3e-5 --num_train_epochs 2 --max_seq_length 384 --doc_stride 128 --output_dir /tmp/debug_squad/
```

### Command to apply ipex optimization for FP32
Added `--auto_ipex`
```
clear && ipexrun --use_default_allocator --ninstance 2 --ncore_per_instance 28 --auto_ipex run_qa.py --model_name_or_path bert-base-uncased --dataset_name squad --do_eval --per_device_train_batch_size 12 --learning_rate 3e-5 --num_train_epochs 2 --max_seq_length 384 --doc_stride 128 --output_dir /tmp/debug_squad/
```

### Command to apply ipex optimization for BF16
Added `--auto_ipex --dtype bfloat16`
```
clear && ipexrun --use_default_allocator --ninstance 2 --ncore_per_instance 28 --auto_ipex --dtype bfloat16 run_qa.py --model_name_or_path bert-base-uncased --dataset_name squad --do_eval --per_device_train_batch_size 12 --learning_rate 3e-5 --num_train_epochs 2 --max_seq_length 384 --doc_stride 128 --output_dir /tmp/debug_squad/
```

## Use Case not supported
### Module uses forward method explicitly instead of the `__call__` attr 
```
import torch
class DummyModule(torch.nn.Module):
    def __init__(self,):
        super(DummyModule, self).__init__()
        self.input1 = torch.randn(1, 3, 224, 224)
        self.conv = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.bn = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        return self.bn(self.conv(x))

    def customized_forward(self, x):
        return self.bn(self.conv(x))

# Method1 will success
DummyModule()(input)
# Method2 will fail to apply ipex.optimize in the top-level model
DummyModule().customized_forward(input)
```
If a model uses forward method explicitly instead of the `__call__` attr, we are unable to hook the execution of this model. As result, we are unable to auto apply the optimizations to this `DummyModule()`.

### Already using `ipex.optimize`
User already invokes `ipex.optimize` in script is not targeted for this feature. The behaviour as repeated invoking of `ipex.optimize` is not defined. The second invoking of `ipex.optimize` for the same module will fail with error message to avoid this behaviour.

### Already using Jit Trace
For Jit trace case (as below example code) is not planned to support at first stage:
```
import torch
model = Model().eval()
traced_model = torch.jit.trace(model, x).eval()
traced_model = torch.jit.freeze(traced_model)
with torch.no_grad():
    for input in dataloader():
        traced_model(input)
```
For 2 reasons:
* The auto graph mode support has already been included in `ipex.optimize` with graph first API in 1.13.
* Extra launch parameters and Monkey patches are needed to support above case. We will focus on the feasibility of first use case in TorchVision and HuggingFace workloads. 
