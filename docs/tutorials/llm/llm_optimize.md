LLM Optimizations Frontend API
======================================

The new API function, `ipex.llm.optimize`, is designed to optimize transformer-based models within frontend Python modules, with a particular focus on Large Language Models (LLMs).
It provides optimizations for both model-wise and content-generation-wise.
You just need to invoke the `ipex.llm.optimize` function instead of the `ipex.optimize` function to apply all optimizations transparently.

This API currently supports for inference workloads of certain models.
API documentation is available at [API Docs page](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/api_doc.html#ipex.llm.optimize),
and supported model list can be found at [this page](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/llm.html#ipexllm-optimized-model-list-for-inference).

## Pseudocode of Common Usage Scenarios

The following sections show pseudocode snippets to invoke Intel® Extension for PyTorch\* APIs to work with LLM models.
Complete examples can be found at [the Example directory](https://github.com/intel/intel-extension-for-pytorch/tree/release/2.8/examples/cpu/llm/inference).

### FP32/BF16

``` python
import torch
import intel_extension_for_pytorch as ipex
import transformers

model= transformers.AutoModelForCausalLM(model_name_or_path).eval()

dtype = torch.float # or torch.bfloat16
model = ipex.llm.optimize(model, dtype=dtype)

# inference with model.generate()
...
```

### Weight Only Quantization (WOQ)

Supports INT8 and INT4.

``` python
import torch
import intel_extension_for_pytorch as ipex
import transformers

model= transformers.AutoModelForCausalLM(model_name_or_path).eval()

qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
  weight_dtype=ipex.quantization.WoqWeightDtype.INT8, # or INT4/NF4
  lowp_mode=ipex.quantization.WoqLowpMode.NONE, # or FP16, BF16, INT8
)

checkpoint = None # optionally load int4 or int8 checkpoint
model = ipex.llm.optimize(model, quantization_config=qconfig, low_precision_checkpoint=checkpoint)

# inference with model.generate()
...
```

### Distributed Inference with DeepSpeed

Distributed inference can be performed with `DeepSpeed`. Based on original Intel® Extension for PyTorch\* scripts, the following code changes are required.

Check [LLM distributed inference examples](https://github.com/intel/intel-extension-for-pytorch/tree/release/2.8/examples/cpu/llm/inference/distributed) for complete codes.

``` python
import torch
import intel_extension_for_pytorch as ipex
import deepspeed
import transformers

dtype = torch.float # or torch.bfloat16
deepspeed.init_distributed(deepspeed.accelerator.get_accelerator().communication_backend_name())

world_size = ... # get int from env var "WORLD_SIZE" or "PMI_SIZE"
with deepspeed.OnDevice(dtype=dtype, device="meta"):
  model= transformers.AutoModelForCausalLM(model_name_or_path).eval()
model = deepspeed.init_inference(
  model,
  mp_size=world_size,
  base_dir=repo_root,
  dtype=dtype,
  checkpoint=checkpoints_json,
  **kwargs,
)
model = model.module

model = ipex.llm.optimize(model, dtype=dtype)

# inference
...
```
