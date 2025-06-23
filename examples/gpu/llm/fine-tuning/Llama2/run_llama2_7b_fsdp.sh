
export CCL_PROCESS_LAUNCHER=none

# settings for torch-ccl
export TORCH_LLM_ALLREDUCE=1

## alpaca dataset full-ft
Run_llama2-7b_fsdp_alpaca_dataset() {

    accelerate launch --config_file "fsdp_config.yaml" llama2_ft.py \
        --model_name_or_path ${model} \
        --data_path ./alpaca_data.json \
        --bf16 True \
        --use_flashattn True \
        --output_dir ./result \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --eval_strategy "no" \
        --save_strategy "steps" \
        --save_steps 2000 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --optim "adamw_torch_fused" 2>&1 | tee llama2_fsdp_alpaca_adamfuse_bs1.log

}

## alpaca dataset peft lora
Run_llama2-7b_fsdp_alpaca_dataset_peft() {

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
        --eval_strategy "no" \
        --save_strategy "steps" \
        --save_steps 2000 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --optim "adamw_torch_fused" 2>&1 | tee llama2_fsdp_alpaca_peft_adamfuse_bs1.log
}

## huggingface dataset full-ft
Run_llama2-7b_fsdp_huggingface_dataset() {

    accelerate launch --config_file "fsdp_config.yaml" llama2_ft.py \
        --model_name_or_path ${model} \
        --bf16 True \
        --use_flashattn True \
        --output_dir ./result \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --eval_strategy "no" \
        --save_strategy "steps" \
        --save_steps 2000 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --max_seq_length 2048 \
        --optim "adamw_torch_fused" 2>&1 | tee llama2_fsdp_huggingface_adamfuse_seq256_bs1_ww14.log
                 
}

## huggingface dataset peft lora
Run_llama2-7b_fsdp_huggingface_dataset_peft() {
 
    accelerate launch --config_file "fsdp_config.yaml" llama2_ft.py \
        --model_name_or_path ${model} \
        --bf16 True \
        --use_flashattn True \
        --use_peft True \
        --output_dir ./result \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --eval_strategy "no" \
        --save_strategy "steps" \
        --save_steps 2000 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --max_seq_length 2048 \
        --optim "adamw_torch_fused" 2>&1 | tee llama2_fsdp_huggingface_peft_adamwfuse_seq200_bs1.log

}


main() {

    model=meta-llama/Llama-2-7b-hf 

    Run_llama2-7b_fsdp_alpaca_dataset
    # Run_llama2-7b_fsdp_alpaca_dataset_peft
    # Run_llama2-7b_fsdp_huggingface_dataset
    # Run_llama2-7b_fsdp_huggingface_dataset_peft

}

main
