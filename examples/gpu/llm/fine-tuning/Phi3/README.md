## Phi3 fine-tuning

### Environment Set Up
Set up environment by following [LLM Environment Set Up](../../README.md).

### Download a Model
During the execution, you may need to log in your Hugging Face account to download model files from online mode. Refer to [HuggingFace Login](https://huggingface.co/docs/huggingface_hub/quick-start#login)

```
huggingface-cli login --token <your_token_here>
```

**Note**: During the execution, you need to log in your wandb account. Refer to [Wandb Login](https://docs.wandb.ai/ref/cli/wandb-login)
```
wandb login
```


### Fine-tuning on Intel® Core™ Ultra Processors with Intel® Arc™ Graphics 

**Note**: Not support full finetuning and flash attention on this platform.

```bash
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
export model="microsoft/Phi-3-mini-4k-instruct"

python phi3_ft.py \
    --model_name_or_path ${model} \
    --custom_mp True \
    --use_peft True \
    --max_seq_length 128 \
    --output_dir="output" \
    --evaluation_strategy="epoch" \
    --learning_rate=1e-3 \
    --auto_find_batch_size=True \
    --num_train_epochs=1 \
    --save_steps=100 \
    --logging_steps=1 \
    --save_total_limit=8 
```


### Fine-tuning on Intel® Data Center Max 1550 GPU on Linux

#### Fine-tuning on single card

Example: Phi-3 Mini 4k full fine-tuning on single card. The default dataset `financial_phrasebank` is loaded in `phi3_ft.py`.

```bash
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
export TORCH_LLM_ALLREDUCE=1

export model="microsoft/Phi-3-mini-4k-instruct"

python phi3_ft.py \
    --model_name_or_path ${model} \
    --use_flashattn False \
    --custom_mp True \
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


Example: Phi-3 Mini 4k LoRA fine-tuning on single card. The default dataset `financial_phrasebank` is loaded in `phi3_ft.py`.

```bash
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
export TORCH_LLM_ALLREDUCE=1

export model="microsoft/Phi-3-mini-4k-instruct"

python phi3_ft.py \
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

#### Fine-tuning on multi-GPU

**Note**:
The default `fsdp_config.yml` is set with 1 machine with 4 cards 8 tiles, If you are using different setting, please change the `num_processes: 8` accordingly. For example, to use 8 cards 16 tiles, the line in `fsdp_config.yml` should be changed to `num_processes: 16`.


Example: Phi-3 Mini 4k full fine-tuning.


```bash
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
export CCL_PROCESS_LAUNCHER=none
export TORCH_LLM_ALLREDUCE=1

export model="microsoft/Phi-3-mini-4k-instruct"

accelerate launch --config_file "fsdp_config.yaml"  phi3_ft.py \
    --model_name_or_path ${model} \
    --use_flashattn False \
    --bf16 True \
    --max_seq_length 128 \
    --output_dir="output" \
    --evaluation_strategy="epoch" \
    --learning_rate=1e-3 \
    --gradient_accumulation_steps=1 \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --num_train_epochs=1 \
    --save_steps=500 \
    --logging_steps=1 \
    --save_total_limit=8 2>&1 | tee phi3-mini_ft_fsdp_converge.log
```


Example: Phi-3 Mini 4k LoRA fine-tuning.


```bash
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
export CCL_PROCESS_LAUNCHER=none
export TORCH_LLM_ALLREDUCE=1

export model="microsoft/Phi-3-mini-4k-instruct"

accelerate launch --config_file "fsdp_config.yaml"  phi3_ft.py \
    --model_name_or_path ${model} \
    --use_flashattn False \
    --bf16 True \
    --use_peft True \
    --max_seq_length 128 \
    --output_dir="output" \
    --evaluation_strategy="epoch" \
    --learning_rate=1e-3 \
    --gradient_accumulation_steps=1 \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --num_train_epochs=1 \
    --save_steps=500 \
    --logging_steps=1 \
    --save_total_limit=8 2>&1 | tee phi3-mini_ft_fsdp_converge.log
```


Example: Phi3-Mini 4k LoRA fine-tuning.


```bash
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
export CCL_PROCESS_LAUNCHER=none
export TORCH_LLM_ALLREDUCE=1

export model="microsoft/Phi-3-mini-4k-instruct"

accelerate launch --config_file "fsdp_config.yaml"  phi3_ft.py \
    --model_name_or_path ${model} \
    --use_flashattn False \
    --bf16 True \
    --use_peft True \
    --max_seq_length 128 \
    --output_dir="output" \
    --evaluation_strategy="epoch" \
    --learning_rate=1e-3 \
    --gradient_accumulation_steps=1 \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --num_train_epochs=1 \
    --save_steps=500 \
    --logging_steps=1 \
    --save_total_limit=8 
```



