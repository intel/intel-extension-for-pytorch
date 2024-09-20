## Llama3 fine-tuning

### Environment Set Up
Set up environment by following [LLM Environment Set Up](../../README.md).

### Download a Model
During the execution, you may need to log in your Hugging Face account to download model files from online mode. Refer to [HuggingFace Login](https://huggingface.co/docs/huggingface_hub/quick-start#login)

```
huggingface-cli login --token <your_token_here>
```

**Note**: If you have download a Llama3 model from Meta official Github, you can also convert it to huggingface format by following the [guide](https://huggingface.co/docs/transformers/main/en/model_doc/llama3#usage-tips).


**Note**: During the execution, you need to log in your wandb account. Refer to [Wandb Login](https://docs.wandb.ai/ref/cli/wandb-login)
```
wandb login
```

### Fine-tuning on single card

**Note**:
Full-finetuning on single card will cause OOM.

Example: Llama 3 8B LoRA fine-tuning on single card. The default dataset `financial_phrasebank` is loaded in `llama3_ft.py`.

```bash
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
export TORCH_LLM_ALLREDUCE=1

export model="meta-llama/Meta-Llama-3-8B"

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

### Fine-tuning on multi-GPU

**Note**:
The default `fsdp_config.yml` is set with 1 machine with 4 cards 8 tiles, If you are using different setting, please change the `num_processes: 8` accordingly. For example, to use 8 cards 16 tiles, the line in `fsdp_config.yml` should be changed to `num_processes: 16`.


Example: Llama 3 8B full fine-tuning, you can change the model name/path for another Llama3 Model.


```bash
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
export CCL_PROCESS_LAUNCHER=none
export TORCH_LLM_ALLREDUCE=1

export model="meta-llama/Meta-Llama-3-8B"

accelerate launch --config_file "fsdp_config.yaml" llama3_ft.py \
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
    --save_total_limit=8 
```


Example: Llama 3 8B LoRA fine-tuning, you can change the model name/path for another Llama3 Model.


```bash
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
export CCL_PROCESS_LAUNCHER=none
export TORCH_LLM_ALLREDUCE=1

export model="meta-llama/Meta-Llama-3-8B"

accelerate launch --config_file "fsdp_config.yaml" llama3_ft.py \
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
