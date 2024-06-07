## alpaca dataset
Run_llama2-7b_fsdp_alpaca_dataset() {

    model='/raid/huggingface/hub/llama2-7b'

    torchrun --nproc_per_node=8 --master_port='29900' train_cuda.py \
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
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --optim "adamw_torch_fused" \
        --fsdp "full_shard auto_wrap" \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
        --tf32 True 2>&1 | tee llama2_fsdp_alpaca_adamfuse_tf32_bs1.log

}


Run_llama2-7b_fsdp_alpaca_dataset_peft() {

    model='/raid/huggingface/hub/llama2-7b'

    torchrun --nproc_per_node=8 --master_port='29900' train_cuda.py \
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
        --optim "adamw_torch_fused" \
        --fsdp "full_shard auto_wrap" \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
        --tf32 True 2>&1 | tee llama2_fsdp_alpaca_peft_adamfuse_tf32_bs1.log
 
}

## huggingface dataset
Run_llama2-7b_fsdp_huggingface_dataset() {

    model='/raid/huggingface/hub/llama2-7b'

    torchrun --nproc_per_node=8 --master_port='29900' train_cuda.py \
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
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --max_seq_length 2048 \
        --optim "adamw_torch_fused" \
        --fsdp "full_shard auto_wrap" \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
        --tf32 True 2>&1 | tee llama2_fsdp_huggingface_adamfuse_seq2048_tf32_bs1.log
}

Run_llama2-7b_fsdp_huggingface_dataset_peft() {

    model='/raid/huggingface/hub/llama2-7b'

    torchrun --nproc_per_node=8 --master_port='29900' train_cuda.py \
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
        --max_seq_length 2048 \
        --optim "adamw_torch_fused" \
        --fsdp "full_shard auto_wrap" \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
        --tf32 True 2>&1 | tee llama2_fsdp_huggingface_peft_adamwfuse_seq2048_tf32_bs1.log

}

Run_llama2-70b_fsdp_alpaca_dataset_peft() {

    model='/media/newdrive2/huggingface/llama2-70b'

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
        --logging_steps 1 \
        --fsdp "full_shard auto_wrap" \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' 2>&1 | tee llama2_70b_fsdp_alpaca_peft_bs1.log
 
}

Run_llama2-70b_fsdp_huggingface_dataset_peft() {

    model='/media/newdrive2/huggingface/llama2-70b'

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
        --max_seq_length 256 \
        --fsdp "full_shard auto_wrap" \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' 2>&1 | tee llama2_70b_fsdp_huggingface_peft_seq256_bs1.log
 
}

main() {

    # llama2-7b
    Run_llama2-7b_fsdp_alpaca_dataset
    Run_llama2-7b_fsdp_alpaca_dataset_peft
    Run_llama2-7b_fsdp_huggingface_dataset
    Run_llama2-7b_fsdp_huggingface_dataset_peft
    
    # llama2-70b
    Run_llama2-70b_fsdp_alpaca_dataset_peft
    Run_llama2-70b_fsdp_huggingface_dataset_peft

}

main

