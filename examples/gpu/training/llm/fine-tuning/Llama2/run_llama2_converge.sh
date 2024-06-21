
export CCL_PROCESS_LAUNCHER=none

# profiling set
# export PROFILE=1
# export KINETO=1

# settings for torch-ccl
export TORCH_LLM_ALLREDUCE=1

# torch-ccl verbose
# export ONECCL_BINDINGS_FOR_PYTORCH_ENV_VERBOSE=1

# oneccl runtime
# source $(python -c "import oneccl_bindings_for_pytorch as torch_ccl;print(torch_ccl.cwd)")/env/setvars.sh
# source /home2/zhuhong/LLM/ccl-inference-dev-3/build/_install/env/setvars.sh

## alpaca dataset full-ft
Run_llama2-7b_fsdp_alpaca_converge() {

    model='/media/newdrive2/huggingface/llama2-7b'

    torchrun --nproc_per_node=8 --master_port='29900' train.py \
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
    
    model='/media/newdrive2/huggingface/llama2-7b'

    torchrun --nproc_per_node=8 --master_port='29900' train.py \
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

    model='/media/newdrive2/huggingface/llama2-70b'

    accelerate launch --config_file "fsdp_config.yaml"  train.py \
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