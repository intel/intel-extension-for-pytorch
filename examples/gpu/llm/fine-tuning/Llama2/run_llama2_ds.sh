
export CCL_PROCESS_LAUNCHER=none
# export PROFILE=1

# setting for torchccl
export TORCH_LLM_ALLREDUCE=1

# profiling set
export PROFILE=1
export KINETO=1

## alpaca dataset
Run_deepspeed_alpaca_dataset() {
    export HF_HOME=/media/newdrive2/huggingface/
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_EVALUATE_OFFLINE=1

    model=meta-llama/Llama-2-7b-hf

    torchrun --nproc_per_node=8 --master_port='29900' train.py \
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
        --deepspeed "default_offload_opt_param.json" 2>&1 | tee llama2_ds_alpaca.log
        # --tf32 True
}

## huggingface dataset
Run_deepspeed_huggingface_dataset() {
    model='/media/newdrive2/llama2-7b'

    torchrun --nproc_per_node=8 --master_port='29900' train.py \
        --model_name_or_path ${model} \
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
        --deepspeed "default_offload_opt_param.json" 2>&1 | tee llama_ds_huggingface.log
        # --tf32 True
}

main() {

    # Run_deepspeed_alpaca_dataset
    Run_deepspeed_huggingface_dataset
}

main