
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
    # Option 1: huggingface offline mode
    # export HF_HOME=/media/newdrive2/huggingface/
    # export HF_DATASETS_OFFLINE=1
    # export TRANSFORMERS_OFFLINE=1
    # export HF_EVALUATE_OFFLINE=1

    # model=meta-llama/Llama-2-70b-hf
    
    # Option 2: specify the model path
    model='/media/newdrive2/huggingface/llama2-70b'

    Run_llama2-70b_fsdp_alpaca_dataset_peft
    # Run_llama2-70b_fsdp_huggingface_dataset_peft
}

main