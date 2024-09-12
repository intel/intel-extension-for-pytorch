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



## Models

| MODEL FAMILY | Verified < MODEL ID > (Hugging Face hub)| Mixed Precision (BF16+FP32) | Full fine-tuning | LoRA | Intel® Data Center Max 1550 GPU | Intel® Core™ Ultra Processors with Intel® Arc™ Graphics |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
|Llama 2 7B| "meta-llama/Llama-2-7b-hf" | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |
|Llama 2 70B| "meta-llama/Llama-2-70b-hf" | 🟩 | 🟥 |🟩 | 🟩 | 🟥 |
|Llama 3 8B| "meta-llama/Meta-Llama-3-8B" | 🟩 | 🟩 |🟩 | 🟩 | 🟩 |
|Qwen 7B|"Qwen/Qwen-7B"| 🟩 | 🟩 |🟩 | 🟩| 🟥 |
|Phi-3-mini 3.8B|"Phi-3-mini-4k-instruct"| 🟩 | 🟩 |🟩 | 🟥 | 🟩 |

- 🟩 signifies that it is supported.

- 🟥 signifies that it is not supported yet.


## Supported Platforms

\* Intel® Data Center Max 1550 GPU: support all the models in the model list above.

\* Intel® Core™ Ultra Processors with Intel® Arc™ Graphics: support Llama 2 7B, Llama 3 8B and Phi-3-Mini 3.8B.


## Run Models

| Benchmark mode | Full fine-tuning | LoRA |
|---|:---:|:---:|
|Single-GPU | 🟥 | 🟩 |
|Multi-GPU (FSDP) |  🟩 | 🟩 |

- 🟩 signifies that it is supported.

- 🟥 signifies that it is not supported yet.

**Note**:
Here we mainly focus on the memory-constrained fine-tuning on single GPU, and provide examples for LoRA fine-tuning. If you want to take a try for full fine-tuning, you could set the number of GPU in distributed cases as 1, and make sure your GPU memory is large enough for model states (parameters, gradients, optimizer states) and residual states (activation, temporary buffers and unusable fragmented memory).


### Download a Model
During the execution, you may need to log in your Hugging Face account to access model files. Refer to [Hugging Face Login](https://huggingface.co/docs/huggingface_hub/quick-start#login).

### Download a Dataset
For Alpaca dataset, you can get here: [Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json).

For more Hugging Face dataset, you can download from [Hugging Face Datasets](https://huggingface.co/docs/datasets/index), or you can load the dataset in Hugging Face online mode.


### Single GPU

**Note**:
- For memory-constrained setup on single GPU, we support mixed precision fine-tuning with LoRA (set by --custom_mp), which keeps the base weight as low precision (e.g. BF16) and LoRA as high precision (e.g. FP32).

Example: Llama 3 8B + LoRA on single GPU

```bash
# in Llama3/run_llama3-8b_ft.sh
  python llama3_ft.py \
      --model_name_or_path ${model} \
      --use_flashattn True \
      --custom_mp True \
      --use_peft True \
      --max_seq_length 128 \
      --output_dir="output" \
      --evaluation_strategy="epoch" \
      --learning_rate=1e-3 \
      --auto_find_batch_size=True \
      --num_train_epochs=1 \
      --save_steps=500 \
      --logging_steps=1 \
      --save_total_limit=8
```


### Multi-GPU

#### Full fine-tuning

Example: Llama 2 7B Full fine-tuning

```bash
  # in Llama2/run_llama2_7b_fsdp.sh
  accelerate launch --config_file "fsdp_config.yaml" train.py \
      --model_name_or_path ${model} \
      --data_path ./alpaca_data.json \
      --bf16 True \
      --use_flashattn True \
      --output_dir ./result \
      --num_train_epochs 1 \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 1 \
      --gradient_accumulation_steps 1 \
      --evaluation_strategy "no" \
      --save_strategy "steps" \
      --save_steps 2000 \
      --save_total_limit 1 \
      --learning_rate 2e-5 \
      --weight_decay 0. \
      --warmup_ratio 0.03 \
      --lr_scheduler_type "cosine" \
      --logging_steps 1 \
      --optim "adamw_torch_fused"
```

#### LoRA

Example: Llama 2 7B LoRA fine-tuning

```bash
  # in Llama2/run_llama2_7b_fsdp.sh
  accelerate launch --config_file "fsdp_config.yaml" train.py \
      --model_name_or_path ${model} \
      --data_path ./alpaca_data.json \
      --bf16 True \
      --use_flashattn True \
      --use_peft True \
      --output_dir ./result \
      --num_train_epochs 1 \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 1 \
      --gradient_accumulation_steps 1 \
      --evaluation_strategy "no" \
      --save_strategy "steps" \
      --save_steps 2000 \
      --save_total_limit 1 \
      --learning_rate 2e-5 \
      --weight_decay 0. \
      --warmup_ratio 0.03 \
      --lr_scheduler_type "cosine" \
      --logging_steps 1 \
      --optim "adamw_torch_fused"
```

**Note**:
- We provide 2 examples for Alpaca dataset with 52k data and guanaco-llama2-1k dataset from Hugging Face. We recommend Alpaca dataset, which has been recognized by some popular projects, e.g. [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca).
- PyTorch FSDP related configurations could be set in fsdp_config.yaml.


