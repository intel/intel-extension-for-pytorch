#!/bin/bash


model="Qwen/Qwen1.5-7B"
data="./dataset.json"

Run_fsdp_dummy_dataset_sequence_length_256() {
    accelerate launch --config_file "fsdp_config.yaml" qwen2_ft.py \
    --model_name_or_path $model \
    --data_path $data \
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
    #--optim "adamw_torch_fused"
}

Run_fsdp_dummy_dataset_lora_sequence_length_256() {
    accelerate launch --config_file "fsdp_config.yaml" qwen2_ft.py \
    --model_name_or_path $model \
    --data_path $data \
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
    #--optim "adamw_torch_fused"
}

Run_fsdp_dummy_dataset_sequence_length_2048() {
    accelerate launch --config_file "fsdp_config.yaml" qwen2_ft.py \
    --model_name_or_path $model \
    --data_path $data \
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
    --model_max_length 2048
    #--optim "adamw_torch_fused"
}

Run_fsdp_dummy_dataset_lora_sequence_length_2048() {
    accelerate launch --config_file "fsdp_config.yaml" qwen2_ft.py \
    --model_name_or_path $model \
    --data_path $data \
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
    --model_max_length 2048 \
    --use_lora
    #--optim "adamw_torch_fused"
}

Run_fsdp_dummy_dataset_sequence_length_256
#Run_fsdp_dummy_dataset_lora_sequence_length_256
#Run_fsdp_dummy_dataset_sequence_length_2048
#Run_fsdp_dummy_dataset_lora_sequence_length_2048
