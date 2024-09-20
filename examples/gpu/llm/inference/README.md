# LLM Inference Overview

Here you can find the inference benchmarking scripts for large language models (LLM) text generation. These scripts:

- Support Llama, GPT-J, Qwen, OPT, Bloom model families and some other Chinese models such as ChatGLMv3-6B, Baichuan2-13B and Phi3-mini. 
- Include both single instance and distributed (DeepSpeed) use cases for FP16 optimization.
- Cover model generation inference with low precision cases for different models with best performance and accuracy (fp16 AMP and weight only quantization)


## Validated Models

Currently, only support Transformers 4.38.1. Support for newer versions of Transformers and more models will be available in the future.

| MODEL FAMILY | Verified < MODEL ID > (Huggingface hub)| FP16 | Weight only quantization INT4 | Optimized on Intel® Data Center GPU Max Series (1550/1100) | Optimized on Intel® Core™ Ultra Processors with Intel® Arc™ Graphics |
|---|:---:|:---:|:---:|:---:|:---:|
|Llama 2| "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-70b-hf" |🟩| 🟩|🟩|🟩|
|Llama 3| "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-70B" |🟩| 🟩|🟩|🟩|
|Phi-3 mini| "microsoft/Phi-3-mini-128k-instruct", "microsoft/Phi-3-mini-4k-instruct" |🟩| 🟩|🟩|🟩|
|GPT-J| "EleutherAI/gpt-j-6b" | 🟩 | 🟩 |🟩 | 🟩|
|Qwen|"Qwen/Qwen-7B"|🟩 | 🟩 |🟩 | 🟩|
|OPT|"facebook/opt-6.7b", "facebook/opt-30b"| 🟩 | 🟥 |🟩 | 🟥 |
|Bloom|"bigscience/bloom-7b1", "bigscience/bloom"| 🟩 | 🟥 |🟩 | 🟥 |
|ChatGLM3-6B|"THUDM/chatglm3-6b"| 🟩 | 🟥 |🟩 | 🟥 |
|Baichuan2-13B|"baichuan-inc/Baichuan2-13B-Chat"| 🟩 | 🟥 |🟩 | 🟥 |

- 🟩 signifies that it is supported.

- 🟥 signifies that it is not supported yet.


**Note**: The verified models mentioned above (including other models in the same model family, like "codellama/CodeLlama-7b-hf" from LLAMA family) are well-supported with all optimizations like indirect access KV cache and fused ROPE. For other LLM families, we are actively working to implement these optimizations, which will be reflected in the expanded model list above. 

## Supported Platforms

\* Intel® Data Center GPU Max Series (1550/1100) : support all the models in the model list above.<br />
\* Intel® Core™ Ultra Processors with Intel® Arc™ Graphics : support Llama2-7B, Llama3-8B, Qwen-7B, Phi3-mini-128k, Phi3-mini-4k.<br />

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

**Note**: Before running script with weight only quantization, make sure [Install intel-extension-for-transformers and intel-neural-compressor](#install-intel-extension-for-transformers-and-intel-neural-compressor)


### Run with Bash Script

Run all inference cases with the one-click bash script `run_benchmark.sh`:
```
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
bash run_benchmark.sh
```

#### Single Instance Performance

**Note**: Only support LLM optimizations with datatype float16, so please don't change datatype to float32 or bfloat16.

```bash
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
# fp16 benchmark
python -u run_generation.py --benchmark -m ${model} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${output} --device xpu --ipex --dtype float16 --token-latency
```

**Note**: By default, generations are based on `bs = 1`, input token size = 1024, output toke size = 128, iteration num = 10 and `beam search`, and beam size = 4. For beam size = 1 and other settings, please export env settings, such as: `beam=1`, `input=32`, `output=32`, `iter=5`.

**Note**: Don't support beam=4 for model Phi3 mini. 

#### Distributed Performance with DeepSpeed

For all distributed inference cases, run LLM with the one-click bash script `run_benchmark_ds.sh`:
```
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
bash run_benchmark_ds.sh
```

```bash
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
# fp16 benchmark
mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py --benchmark -m ${model} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${output} --device xpu --ipex --dtype float16 --token-latency
```

**Note**: By default, generations are based on `bs = 1`, input token size = 1024, output toke size = 128, iteration num = 10 and `beam search`, and beam size = 4. For beam size = 1 and other settings, please export env settings, such as: `beam=1`, `input=32`, `output=32`, `iter=5`.

## Advanced Usage

**Note**: Unset the variable before running.
```bash
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
# Adding this variable to run multi-tile cases might cause an Out Of Memory (OOM) issue. 
unset TORCH_LLM_ALLREDUCE
```

### Single Instance Accuracy

```bash
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
Accuracy test {TASK_NAME}, choice in this [link](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/docs/task_table.md), by default we use "lambada_standard"

# one-click bash script
bash run_accuracy.sh

# float16
LLM_ACC_TEST=1 python -u run_generation.py -m ${model} --ipex --dtype float16 --accuracy-only --acc-tasks ${task}
```

### Distributed Accuracy with DeepSpeed

```bash
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
# one-click bash script
bash run_accuracy_ds.sh

# float16
LLM_ACC_TEST=1 mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py -m ${model} --ipex --dtype float16 --accuracy-only --acc-tasks ${task} 2>&1
```


### Weight Only Quantization with low precision checkpoint (Prototype)

Using INT4 weights can further improve performance by reducing memory bandwidth. However, direct per-channel quantization of weights to INT4 may result in poor accuracy. Some algorithms can modify weights through calibration before quantizing weights to minimize accuracy drop. You may generate modified weights and quantization info (scales, zero points) for a Llama 2/GPT-J/Qwen models with a dataset for specified tasks by such algorithms. We recommend intel extension for transformer to quantize the LLM model.

Check [WOQ INT4](../../../../docs/tutorials/llm/int4_weight_only_quantization.md) for more details.

#### Install intel-extension-for-transformers and intel-neural-compressor 

```
pip install numpy<2
git clone https://github.com/intel/neural-compressor.git -b v3.0
cd neural-compressor
python setup.py install
cd ..
 
git clone https://github.com/intel/intel-extension-for-transformers.git
cd intel-extension-for-transformers
git checkout c263d09f0899741142b35ebed3159e9425b1ac8f
pip install -r requirements.txt
python setup.py install
```


#### Run the weight only quantization and inference

```python
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
bash run_benchmark_woq.sh
```

>Note:
> * Saving quantized model should be executed before the optimize_transformers function is called.
> * The optimize_transformers function is designed to optimize transformer-based models within frontend Python modules, with a particular focus on Large Language Models (LLMs). It provides optimizations for both model-wise and content-generation-wise. The detail of `optimize_transformers`, please refer to [Transformers Optimization Frontend API](../../../../docs/tutorials/llm/llm_optimize_transformers.md).
