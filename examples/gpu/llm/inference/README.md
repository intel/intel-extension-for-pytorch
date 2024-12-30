# LLM Inference Overview

Here you can find the inference benchmarking scripts for large language models (LLM) text generation. These scripts:

- Support Llama, GPT-J, Qwen, OPT, Bloom model families and some other Chinese models such as GLM4-9B, Baichuan2-13B and Phi3-mini. 
- Include both single instance and distributed (DeepSpeed) use cases for FP16 optimization.
- Cover model generation inference with low precision cases for different models with best performance and accuracy (fp16 AMP and weight only quantization)


## Validated Models

Currently, only support Transformers 4.44.2. Support for newer versions of Transformers and more models will be available in the future.

| MODEL FAMILY | Verified < MODEL ID > (Huggingface hub)| FP16 | Weight only quantization INT4 | Optimized on Intel® Data Center GPU Max Series (1550/1100) | Optimized on Intel® Core™ Ultra Processors with Intel® Arc™ Graphics | Optimized on Intel® Arc™ B-Series Graphics (B580) | 
|---|:---:|:---:|:---:|:---:|:---:|:---:|
|Llama 2| "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-70b-hf" |🟩| 🟩|🟩|🟩|$🟩^1$|
|Llama 3| "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-70B" |🟩| 🟩|🟩|🟩|$🟩^2$|
|Phi-3 mini| "microsoft/Phi-3-mini-128k-instruct", "microsoft/Phi-3-mini-4k-instruct" |🟩| 🟩|🟩|🟩|$🟩^3$|
|GPT-J| "EleutherAI/gpt-j-6b" | 🟩 | 🟩 |🟩 | 🟩| |
|Qwen|"Qwen/Qwen2-7B"|🟩 | 🟩 |🟩 | 🟩| |
|Qwen|"Qwen/Qwen2-7B-Instruct"| | | | | 🟩 |
|OPT|"facebook/opt-6.7b", "facebook/opt-30b"| 🟩 | 🟥 |🟩 | 🟥 |
|Bloom|"bigscience/bloom-7b1", "bigscience/bloom"| 🟩 | 🟥 |🟩 | 🟥 |
|GLM4-9B|"THUDM/glm-4-9b"| 🟩 | 🟥 |🟩 | 🟥 |
|Baichuan2-13B|"baichuan-inc/Baichuan2-13B-Chat"| 🟩 | 🟥 |🟩 | 🟥 |

- 🟩 signifies that it is supported.

- 🟥 signifies that it is not supported yet.
  
-  1: signifies that Llama-2-7b-hf is verified.
  
-  2: signifies that Meta-Llama-3-8B is verified.
  
-  3: signifies that Phi-3-mini-4k-instruct is verified.



**Note**: The verified models mentioned above (including other models in the same model family, like "codellama/CodeLlama-7b-hf" from LLAMA family) are well-supported with all optimizations like indirect access KV cache and fused ROPE. For other LLM families, we are actively working to implement these optimizations, which will be reflected in the expanded model list above. 

## Supported Platforms

\* Intel® Data Center GPU Max Series (1550/1100) : support all the models in the model list above.<br />
\* Intel® Core™ Ultra Processors with Intel® Arc™ Graphics : support Llama2-7B, Llama3-8B, Qwen2-7B, Phi3-mini-128k, Phi3-mini-4k.<br />

## Run Models

| Benchmark mode | FP16 | Weight only quantization INT4 |
|---|:---:|:---:|
|Single instance | 🟩 | 🟩 |
| Distributed (autotp) |  🟩 | 🟥 |

- 🟩 signifies that it is supported.

- 🟥 signifies that it is not supported yet.

**Note**: During the execution, you may need to log in your Hugging Face account to access model files. Refer to [HuggingFace Login](https://huggingface.co/docs/huggingface_hub/quick-start#login)

```
huggingface-cli login --token <your_token_here>
```

### Environment Set Up
Set up environment by following [LLM Environment Set Up](../README.md).

**Note**: Before running script with weight only quantization, make sure Install [intel-neural-compressor](https://github.com/intel/neural-compressor)


### Learn to Run LLM inference on Intel GPU in FP16

Follow the following several steps to run on Intel GPU with Optimizations.


1. Load the tokenizer and model, move the model to Intel GPU (xpu).
2. Change the memory format of the model to channels_last for optimization.
3. Optimize the model using Intel Extension for PyTorch (IPEX).
4. Tokenize the input prompt, convert it to tensor format, and move the input tensor to the XPU.
5. Generate text based on the input prompt with a maximum of 512 new tokens.
6. Decode the generated token IDs to a string, skipping special tokens.
7. Print the generated text.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import intel_extension_for_pytorch as ipex

# Define the model ID and prompt, here we take llama2 as an example
model_id = "meta-llama/Llama-2-7b-hf"
prompt = 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n'

# Load the tokenizer and model, move model to Intel GPU(xpu)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)
model = model.eval().to("xpu")

# Change the memory format of the model to channels_last for optimization
model = model.to(memory_format=torch.channels_last)

# Optimize the model using Intel Extension for PyTorch (IPEX)
model = ipex.llm.optimize(model.eval(), dtype=torch.float16, device="xpu")

# Tokenize the input prompt and convert it to tensor format and Move the input tensor to the XPU
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
input_ids = input_ids.to("xpu")

# Generate text based on the input prompt with a maximum of 512 new tokens
# Set cache_implementation to static cache will improve the performance, the default is dynamic cache
generated_ids = model.generate(input_ids, max_new_tokens=512, cache_implementation="static")[0]

# Decode the generated token IDs to a string, skipping special tokens
generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

# Print the generated text
print(generated_text)
```


### Learn to Quantize LLM and run inference

1. Install intel-neural-compressor for weight only quantization.
2. Define the model ID and the prompt.
3. Load the tokenizer using the pre-trained model ID.
4. Define the quantization configuration and apply quantization.
5. Set the model to evaluation mode and move it to the XPU.
6. Optimize the model with Intel Extension for PyTorch (IPEX).
7.  Tokenize the input prompt, convert it to tensor format, and move the input tensor to the XPU.
8.  Generate text based on the input prompt with a maximum of 512 new tokens.
9.  Decode the generated token IDs to a string, skipping special tokens.
10. Print the generated text.


```python
from transformers import AutoTokenizer
import torch
import intel_extension_for_pytorch as ipex
from neural_compressor.transformers import AutoModelForCausalLM, RtnConfig
import os

# Define the model ID and the prompt, here we take llama2 as an example
model_id = "meta-llama/Llama-2-7b-hf"
prompt = 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n'
use_hf_code = True
woq_checkpoint_path = "./llama2_woq_int4"

# Load the tokenizer using the pre-trained model ID
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Define the quantization configuration
woq_quantization_config = RtnConfig(
    compute_dtype="fp16", 
    weight_dtype="int4_fullrange", 
    scale_dtype="fp16", 
    group_size=64
)
# Load the model and apply quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="xpu",
    quantization_config=woq_quantization_config,
    trust_remote_code=use_hf_code,
)

# Set the model to evaluation mode and move it to the XPU
model = model.eval().to("xpu")
model = model.to(memory_format=torch.channels_last)

# Optimize the model with Intel Extension for PyTorch (IPEX)
model = ipex.llm.optimize(
    model.eval(), 
    device="xpu", 
    inplace=True, 
    quantization_config=woq_quantization_config
)

# Tokenize the input prompt and convert it to tensor format
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
input_ids = input_ids.to("xpu")

# Generate text based on the input prompt with a maximum of 512 new tokens
# Set cache_implementation to static cache will improve the performance, the default is dynamic cache
generated_ids = model.generate(input_ids, max_new_tokens=512, cache_implementation="static")[0]

# Decode the generated token IDs to a string, skipping special tokens
generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

# Print the generated text
print(generated_text)
```



### Learn to Quantize LLM and save quantized model then run inference with quantized model

First of all, set optimization path to align with intel-neural-comporessor when save quantized model

```
export IPEX_COMPUTE_ENG=4
```

1. Install intel-neural-compressor for weight only quantization.
2. Define the model ID and the prompt.
3. Load the tokenizer using the pre-trained model ID.
4. Define the quantization configuration and apply quantization.
5. Save the quantized model and tokenizer.


```python
from transformers import AutoTokenizer
import torch
import intel_extension_for_pytorch as ipex
from neural_compressor.transformers import AutoModelForCausalLM, RtnConfig
import os

# Define the model ID and the prompt, here we take llama2 as an example
model_id = "meta-llama/Llama-2-7b-hf"
prompt = 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n'
use_hf_code = True
woq_checkpoint_path = "./llama2_woq_int4"

# Load the tokenizer using the pre-trained model ID
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Check if the quantized model checkpoint path exists
if os.path.exists(woq_checkpoint_path):
    print("Directly loading already quantized model")
    # Load the already quantized model
    model = AutoModelForCausalLM.from_pretrained(
        woq_checkpoint_path, 
        trust_remote_code=use_hf_code, 
        device_map="xpu", 
        torch_dtype=torch.float16
    )
    model = model.to(memory_format=torch.channels_last)
    woq_quantization_config = getattr(model, "quantization_config", None)
else:
    # Define the quantization configuration
    woq_quantization_config = RtnConfig(
        compute_dtype="fp16", 
        weight_dtype="int4_fullrange", 
        scale_dtype="fp16", 
        group_size=64
    )
    # Load the model and apply quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="xpu",
        quantization_config=woq_quantization_config,
        trust_remote_code=use_hf_code,
    )
    # Save the quantized model and tokenizer
    model.save_pretrained(woq_checkpoint_path)
    tokenizer.save_pretrained(woq_checkpoint_path)

# Set the model to evaluation mode and move it to the XPU
model = model.eval().to("xpu")
model = model.to(memory_format=torch.channels_last)

# Optimize the model with Intel Extension for PyTorch (IPEX)
model = ipex.llm.optimize(
    model.eval(), 
    device="xpu", 
    inplace=True, 
    quantization_config=woq_quantization_config
)

# Tokenize the input prompt and convert it to tensor format
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
input_ids = input_ids.to("xpu")

# Generate text based on the input prompt with a maximum of 512 new tokens
# Set cache_implementation to static cache will improve the performance, the default is dynamic cache
generated_ids = model.generate(input_ids, max_new_tokens=512, cache_implementation="static")[0]

# Decode the generated token IDs to a string, skipping special tokens
generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

# Print the generated text
print(generated_text)
```

Then run inference with quantized model in oneDNN path.

```
unset IPEX_COMPUTE_ENG=4
```

1. Install intel-neural-compressor for weight only quantization.
2. Define the model ID and the prompt.
3. Load the quantized model and tokenizer.
4. Set the model to evaluation mode and move it to the XPU.
5. Optimize the model with Intel Extension for PyTorch (IPEX).
6.  Tokenize the input prompt, convert it to tensor format, and move the input tensor to the XPU.
7.  Generate text based on the input prompt with a maximum of 512 new tokens.
8.  Decode the generated token IDs to a string, skipping special tokens.
9.  Print the generated text.

```python
from transformers import AutoTokenizer
import torch
import intel_extension_for_pytorch as ipex
from neural_compressor.transformers import AutoModelForCausalLM, RtnConfig
import os

# Define the model ID and the prompt, here we take llama2 as an example
model_id = "meta-llama/Llama-2-7b-hf"
prompt = 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n'
use_hf_code = True
woq_checkpoint_path = "./llama2_woq_int4"

# Load the tokenizer using the pre-trained model ID
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Check if the quantized model checkpoint path exists
if os.path.exists(woq_checkpoint_path):
    print("Directly loading already quantized model")
    # Load the quantized model
    model = AutoModelForCausalLM.from_pretrained(
        woq_checkpoint_path, 
        trust_remote_code=use_hf_code, 
        device_map="xpu", 
        torch_dtype=torch.float16
    )
    model = model.to(memory_format=torch.channels_last)
    woq_quantization_config = getattr(model, "quantization_config", None)

# Tokenize the input prompt and convert it to tensor format
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
input_ids = input_ids.to("xpu")

# Generate text based on the input prompt with a maximum of 512 new tokens
# Set cache_implementation to static cache will improve the performance, the default is dynamic cache
generated_ids = model.generate(input_ids, max_new_tokens=512, cache_implementation="static")[0]

# Decode the generated token IDs to a string, skipping special tokens
generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

# Print the generated text
print(generated_text)
```

### Benchmark validated models

#### Run all validated models with Bash Script

The related code and run script are prepared in the folder. Run all inference cases with the one-click bash script `run_benchmark.sh`:

```
bash run_benchmark.sh
```

##### Single Instance Performance

**Note**: Only support LLM optimizations with datatype float16, so please don't change datatype to float32 or bfloat16.

```bash
# fp16 benchmark
python -u run_generation.py --benchmark -m ${model} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${output} --device xpu --ipex --dtype float16 --token-latency
```

**Note**: By default, generations are based on `bs = 1`, input token size = 1024, output toke size = 128, iteration num = 10 and `beam search`, and beam size = 4. For beam size = 1 and other settings, please export env settings, such as: `beam=1`, `input=32`, `output=32`, `iter=5`.

##### Distributed Performance with DeepSpeed

For all distributed inference cases, run LLM with the one-click bash script `run_benchmark_ds.sh`:
```
bash run_benchmark_ds.sh
```

```bash
# fp16 benchmark
mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py --benchmark -m ${model} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${output} --device xpu --ipex --dtype float16 --token-latency
```

**Note**: By default, generations are based on `bs = 1`, input token size = 1024, output toke size = 128, iteration num = 10 and `beam search`, and beam size = 4. For beam size = 1 and other settings, please export env settings, such as: `beam=1`, `input=32`, `output=32`, `iter=5`.

### Advanced Usage

**Note**: Unset the variable before running.
```bash
# Adding this variable to run multi-tile cases might cause an Out Of Memory (OOM) issue. 
unset TORCH_LLM_ALLREDUCE
```

#### Single Instance Accuracy

```bash
Accuracy test {TASK_NAME}, choice in this [link](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/docs/task_table.md), by default we use "lambada_standard"

# one-click bash script
bash run_accuracy.sh

# float16
LLM_ACC_TEST=1 python -u run_generation.py -m ${model} --ipex --dtype float16 --accuracy-only --acc-tasks ${task}
```

#### Distributed Accuracy with DeepSpeed

```bash
# one-click bash script
bash run_accuracy_ds.sh

# float16
LLM_ACC_TEST=1 mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py -m ${model} --ipex --dtype float16 --accuracy-only --acc-tasks ${task} 2>&1
```


### Benchmark Weight Only Quantization with low precision checkpoint (Prototype)

Using INT4 weights can further improve performance by reducing memory bandwidth. However, direct per-channel quantization of weights to INT4 may result in poor accuracy. Some algorithms can modify weights through calibration before quantizing weights to minimize accuracy drop. You may generate modified weights and quantization info (scales, zero points) for a Llama 2/GPT-J/Qwen2 models with a dataset for specified tasks by such algorithms. We recommend intel extension for transformer to quantize the LLM model.

Check [WOQ INT4](../../../../docs/tutorials/llm/int4_weight_only_quantization.md) for more details.


#### Run the weight only quantization and inference

```
bash run_benchmark_woq.sh
```

>Note:
> * Saving quantized model should be executed before the optimize_transformers function is called.
> * The optimize_transformers function is designed to optimize transformer-based models within frontend Python modules, with a particular focus on Large Language Models (LLMs). It provides optimizations for both model-wise and content-generation-wise. The detail of `optimize_transformers`, please refer to [Transformers Optimization Frontend API](../../../../docs/tutorials/llm/llm_optimize_transformers.md).

