## Qwen fine-tuning

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


### Fine-tuning on multi-GPU

**Note**:
The default `fsdp_config.yml` is set with 1 machine with 4 cards 8 tiles, If you are using different setting, please change the `num_processes: 8` accordingly. For example, to use 8 cards 16 tiles, the line in `fsdp_config.yml` should be changed to `num_processes: 16`.

**Note**: The default dataset is `dataset.json` which locates in current Qwen folder. 



Example: Qwen 7B full fine-tuning.


```bash
export CCL_PROCESS_LAUNCHER=none
export TORCH_LLM_ALLREDUCE=1

export model="Qwen/Qwen1.5-7B"

accelerate launch --config_file "fsdp_config.yaml" qwen2_ft.py \
    --model_name_or_path $model \
    --data_path "./dataset.json" \
    --bf16 True \
    --output_dir output_qwen \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 10 \
    --learning_rate 3e-4 \
    --weight_decay 0.01 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 256
```

Example: Qwen 7B LoRA fine-tuning.

```bash
export CCL_PROCESS_LAUNCHER=none
export TORCH_LLM_ALLREDUCE=1

export model="Qwen/Qwen1.5-7B"

accelerate launch --config_file "fsdp_config.yaml" qwen2_ft.py \
    --model_name_or_path $model \
    --data_path "./dataset.json" \
    --bf16 True \
    --output_dir output_qwen \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 10 \
    --learning_rate 3e-4 \
    --weight_decay 0.01 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 256 \
    --use_lora
```
