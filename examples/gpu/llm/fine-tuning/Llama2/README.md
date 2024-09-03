## Llama2 fine-tuning



### Download a Model
During the execution, you may need to log in your Hugging Face account to download model files from online mode. Refer to [HuggingFace Login](https://huggingface.co/docs/huggingface_hub/quick-start#login)

```
huggingface-cli login --token <your_token_here>
```

**Note**: If you have download a Llama2 model from Meta official Github, you can also convert it to huggingface format by following the [guide](https://huggingface.co/docs/transformers/main/en/model_doc/llama2#usage-tips).

### Download a Dataset

For Alpaca dataset, you can get here: [Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json).
```
git clone https://github.com/tatsu-lab/stanford_alpaca
cd standford_alpaca
mv alpaca_data.json <Llama2_folder>
```


**Note**: During the execution, you need to log in your wandb account. Refer to [Wandb Login](https://docs.wandb.ai/ref/cli/wandb-login)
```
wandb login
```

### Fine-tuning on multi-GPU

**Note**:
The default `fsdp_config.yml` is set with 1 machine with 4 cards 8 tiles, If you are using different setting, please change the `num_process` accordingly.

#### Full-finetuning 


Example: Llama 2 7B full fine-tuning with Alpaca dataset, you can change the model name/path for another Llama2 Model.


**Note**:
We provide examples for Alpaca dataset with 52k data and guanaco-llama2-1k dataset from Hugging Face. We recommend [Alpaca dataset](#download-a-dataset), which has been recognized by some popular projects.
Remove the flags `--data_path` in fine-tuning command will load the guanaco-llama2-1k dataset from Hugging Face by default in `llama2_ft.py`.



```bash
export CCL_PROCESS_LAUNCHER=none
export TORCH_LLM_ALLREDUCE=1

export model='meta-llama/Llama-2-7b-hf'

accelerate launch --config_file "fsdp_config.yaml" llama2_ft.py \
        --model_name_or_path ${model} \
        --data_path ./alpaca_data.json \
        --bf16 True \
        --use_flashattn True \
        --output_dir ./result \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 2000 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --optim "adamw_torch_fused" \
```


#### LoRA finetuning

Example: Llama 2 7B LoRA fine-tuning with Alpaca dataset, you can change the model name/path for another Llama2 Model.

**Note**:
We provide examples for Alpaca dataset with 52k data and guanaco-llama2-1k dataset from Hugging Face. We recommend [Alpaca dataset](#download-a-dataset), which has been recognized by some popular projects.
Remove the flags `--data_path` in fine-tuning command will load the guanaco-llama2-1k dataset from Hugging Face by default in `llama2_ft.py`.


```bash
export CCL_PROCESS_LAUNCHER=none
export TORCH_LLM_ALLREDUCE=1

export model='meta-llama/Llama-2-7b-hf'

accelerate launch --config_file "fsdp_config.yaml" llama2_ft.py \
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
