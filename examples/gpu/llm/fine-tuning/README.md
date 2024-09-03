## LLM fine-tuning

Here we provide the large language models (LLM) fine-tuning examples. These scripts:

- Support Llama 2 (7B and 70B), Llama 3 8B, Phi-3-Mini 3.8B model families and Chinese model Qwen-7B.
- Include both single GPU and Multi-GPU (distributed fine-tuning based on PyTorch FSDP) use cases for mixed precision with BF16 and FP32.
- Support popular recipes with both Full fine-tuning and LoRA.


Our examples integrate with the popular tools and libraries from the ecosystem:
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html) for distributed training
- [Hugging Face Hub](https://huggingface.co/docs/hub/en/index) for [accessing model weights](https://huggingface.co/models)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/index) for training and evaluation datasets
- [Transformers](https://github.com/huggingface/transformers) for training (Trainer) and modeling script
- [PEFT](https://github.com/huggingface/peft) for providing method such as LoRA
- [Accelerate](https://github.com/huggingface/accelerate) for Multi-GPUs launch


### LLM fine-tuning optimized with Intel® Data Center Max 1550 GPU on Linux

**Note**:
Here we mainly focus on the memory-constrained fine-tuning on single GPU, and provide examples for LoRA fine-tuning. If you want to take a try for full fine-tuning, you could set the number of GPU in distributed cases as 1, and make sure your GPU memory is large enough for model states (parameters, gradients, optimizer states) and residual states (activation, temporary buffers and unusable fragmented memory).

| Benchmark mode | Full fine-tuning | LoRA |
|---|:---:|:---:|
|Single-GPU | 🟥 | 🟩 |
|Multi-GPU (FSDP) |  🟩 | 🟩 |

| MODEL FAMILY | Verified < MODEL ID > (Hugging Face hub)| Mixed Precision (BF16+FP32) | Full fine-tuning  | LoRA |  
|---|:---:|:---:|:---:|:---:|
|[Llama 2 7B](./Llama2/)| "meta-llama/Llama-2-7b-hf" | 🟩 | 🟩 | 🟩 | 
|[Llama 2 70B](./Llama2/)| "meta-llama/Llama-2-70b-hf" | 🟩 | 🟥 |🟩 | 
|[Llama 3 8B](./Llama3/)| "meta-llama/Meta-Llama-3-8B" | 🟩 | 🟩 |🟩 | 
|[Llama 3 70B](./Llama3/)| "meta-llama/Meta-Llama-3-70B" | 🟩 | 🟥 |🟩 | 
|[Qwen 7B](./Qwen/)|"Qwen/Qwen-7B"| 🟩 | 🟩 |🟩 | 
|[Phi-3-mini 3.8B](./Phi3/README.md#fine-tuning-on-intel-data-center-max-1550-gpu-on-linux)|"Phi-3-mini-4k-instruct"| 🟩 | 🟩 |🟩 | 


\* Intel® Data Center Max 1550 GPU: support all the models in the model list above.

### LLM fine-tuning optimized with Intel® Core™ Ultra Processors with Intel® Arc™ Graphics 

| MODEL FAMILY | Verified < MODEL ID > (Hugging Face hub)| Mixed Precision (BF16+FP32) | Full fine-tuning  | LoRA |  
|---|:---:|:---:|:---:|:---:|
|[Phi-3-mini 3.8B](./Phi3/README.md#fine-tuning-on-intel-core-ultra-processors-with-intel-arc-graphics)|"Phi-3-mini-4k-instruct"| 🟩 | 🟩 |🟩 | 


- 🟩 signifies that it is supported.

- 🟥 signifies that it is not supported yet.

\* Intel® Core™ Ultra Processors with Intel® Arc™ Graphics: support Phi-3-Mini 3.8B.


### Profile the finetuning

For profiling the process of finetuning, Apply the `patches/transformers.patch` to transformers v4.41.2 and set the following VARIABLE before finetuning.

```bash
export PROFILE=1
```

