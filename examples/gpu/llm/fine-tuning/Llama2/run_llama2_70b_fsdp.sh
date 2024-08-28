export CCL_PROCESS_LAUNCHER=none

export TORCH_LLM_ALLREDUCE=1

# llama2-70b alpaca dataset peft lora
Run_llama2-70b_fsdp_alpaca_dataset_peft() {

    accelerate launch --config_file "fsdp_config.yaml"  train.py \
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
        --logging_steps 1 2>&1 | tee llama2_70b_fsdp_alpaca_peft_bs1.log
 
}

# llama2-70b huggingface dataset peft lora
Run_llama2-70b_fsdp_huggingface_dataset_peft() {

    accelerate launch --config_file "fsdp_config.yaml"  train.py \
        --model_name_or_path ${model} \
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
        --max_seq_length 256 2>&1 | tee llama3_70b_fsdp_huggingface_peft_seq256_bs1.log
 
}

main() {
    
    model=meta-llama/Llama-2-70b-hf
    
    Run_llama2-70b_fsdp_alpaca_dataset_peft
    # Run_llama2-70b_fsdp_huggingface_dataset_peft
}

main
