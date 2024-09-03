export CCL_PROCESS_LAUNCHER=none

# settings for torch-ccl
export TORCH_LLM_ALLREDUCE=1

## alpaca dataset full-ft
Run_llama2-7b_fsdp_alpaca_converge() {

    model='meta-llama/Llama-2-7b-hf'
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
        --fsdp "full_shard auto_wrap" \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' 2>&1 | tee llama2_fsdp_alpaca_adamfuse_bs4_3epoch_converge.log

}

## alpaca dataset peft lora
Run_llama2-7b_fsdp_alpaca_peft_converge() {
    
    model='meta-llama/Llama-2-7b-hf'

    accelerate launch --config_file "fsdp_config.yaml" llama2_ft.py \
        --model_name_or_path ${model} \
        --data_path ./alpaca_data.json \
        --bf16 True \
        --use_flashattn True \
        --use_peft True \
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
        --fsdp "full_shard auto_wrap" \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' 2>&1 | tee llama2_fsdp_alpaca_peft_adamfuse_bs4_3epoch_converge.log
}


# llama2-70b alpaca dataset peft lora
Run_llama2-70b_fsdp_alpaca_peft_converge() {

    model='meta-llama/Llama-2-70b-hf'

    accelerate launch --config_file "fsdp_config.yaml"  llama2_ft.py \
        --model_name_or_path ${model} \
        --data_path ./alpaca_data.json \
        --bf16 True \
        --use_flashattn True \
        --use_peft True \
        --output_dir ./result \
        --num_train_epochs 3 \
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
        --fsdp "full_shard auto_wrap" \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' 2>&1 | tee llama2_70b_fsdp_alpaca_peft_bs1_3epoch_converge.log
 
}


main() {

    Run_llama2-7b_fsdp_alpaca_converge
    # Run_llama2-7b_fsdp_alpaca_peft_converge
    # Run_llama2-70b_fsdp_alpaca_peft_converge
}

main
