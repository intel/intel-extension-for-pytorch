# Quick Start

The following instructions assume you have installed the Intel® Extension for PyTorch\*. For installation instructions, refer to [Installation](../../../index.html#installation?platform=cpu&version=main).

To start using the Intel® Extension for PyTorch\* in your code, you need to make the following changes:

1. Import the extension with `import intel_extension_for_pytorch as ipex`.
2. Invoke the `optimize()` function to apply optimizations.
3. Convert the eager mode model to a graph mode model.
    - For TorchScript, invoke `torch.jit.trace()` and `torch.jit.freeze()`
    - For TorchDynamo, invoke `torch.compile(model, backend="ipex")`(*Beta feature*)

**Important:** It is highly recommended to `import intel_extension_for_pytorch` right after `import torch`, prior to importing other packages.

The example below demostrates how to use the Intel® Extension for PyTorch\* with TorchScript:

```python
import torch
############## import ipex ###############
import intel_extension_for_pytorch as ipex
##########################################

model = Model()
model.eval()
data = ...

############## TorchScript ###############
model = ipex.optimize(model, dtype=torch.bfloat16)

with torch.no_grad(), torch.cpu.amp.autocast():
  model = torch.jit.trace(model, data)
  model = torch.jit.freeze(model)
  model(data)
##########################################
```

The example below demostrates how to use the Intel® Extension for PyTorch\* with TorchDynamo:

```python
import torch
############## import ipex ###############
import intel_extension_for_pytorch as ipex
##########################################

model = Model()
model.eval()
data = ...

############## TorchDynamo ###############
model = ipex.optimize(model, weights_prepack=False)

model = torch.compile(model, backend="ipex")
with torch.no_grad():
  model(data)
##########################################
```

More examples, including training and usage of low precision data types are available in the [Examples](./examples.md) section.

In [Cheat Sheet](./cheat_sheet.md), you can find more commands that can help you start using the Intel® Extension for PyTorch\*.


## LLM Quick Start

`ipex.llm.optimize` is used for Large Language Models (LLM).


```python
import torch
#################### code changes ####################  
import intel_extension_for_pytorch as ipex
######################################################  
import argparse
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

# args
parser = argparse.ArgumentParser("Generation script (fp32/bf16 path)", add_help=False)
parser.add_argument(
    "--dtype",
    type=str,
    choices=["float32", "bfloat16"],
    default="float32",
    help="choose the weight dtype and whether to enable auto mixed precision or not",
)
parser.add_argument(
    "--max-new-tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument(
    "--prompt", default="What are we having for dinner?", type=str, help="input prompt"
)
parser.add_argument("--greedy", action="store_true")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
args = parser.parse_args()
print(args)

# dtype
amp_enabled = True if args.dtype != "float32" else False
amp_dtype = getattr(torch, args.dtype)

# load model
model_id = MODEL_ID
config = AutoConfig.from_pretrained(
    model_id, torchscript=True, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=amp_dtype,
    config=config,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)
model = model.eval()
model = model.to(memory_format=torch.channels_last)

# Intel(R) Extension for PyTorch*
#################### code changes ####################  # noqa F401
model = ipex.llm.optimize(
    model,
    dtype=amp_dtype,
    inplace=True,
    deployment_mode=True,
)
######################################################  # noqa F401

# generate args
num_beams = 1 if args.greedy else 4
generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=num_beams)

# input prompt
prompt = args.prompt
input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
print("---- Prompt size:", input_size)
prompt = [prompt] * args.batch_size

# inference
with torch.inference_mode(), torch.cpu.amp.autocast(enabled=amp_enabled):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    gen_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        **generate_kwargs
    )
    gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    input_tokens_lengths = [x.shape[0] for x in input_ids]
    output_tokens_lengths = [x.shape[0] for x in gen_ids]
    total_new_tokens = [
        o - i for i, o in zip(input_tokens_lengths, output_tokens_lengths)
    ]
    print(gen_text, total_new_tokens, flush=True)
```

More LLM examples, including usage of low precision data types are available in the [LLM Examples](https://github.com/intel/intel-extension-for-pytorch/tree/main/examples/cpu/inference/python/llm) section.
