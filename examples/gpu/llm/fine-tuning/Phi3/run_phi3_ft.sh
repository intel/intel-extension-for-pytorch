# export ZE_AFFINITY_MASK=4

# export PROFILE=1
# export KINETO=1
# export DNNL_VERBOSE=1

# settings for torchccl
export TORCH_LLM_ALLREDUCE=1

# NOTE: SDPA (use_flashattn) support is disabled in transformers https://github.com/huggingface/transformers/pull/30423 due to some issues related to https://github.com/huggingface/transformers/pull/30127. We will enable it after transformers add it back.

Run_phi3-mini_peft_singlecard() {

    python phi3_ft.py \
        --model_name_or_path ${model} \
        --use_flashattn False \
        --custom_mp True \
        --use_peft True \
        --max_seq_length 128 \
        --output_dir="output" \
        --evaluation_strategy="epoch" \
        --learning_rate=1e-3 \
        --auto_find_batch_size=True \
        --num_train_epochs=1 \
        --save_steps=500 \
        --logging_steps=1 \
        --save_total_limit=8 2>&1 | tee phi3-mini_ft_singlecard_converge.log
}


Run_phi3-mini_peft_fsdp() {

    accelerate launch --config_file "fsdp_config.yaml"  phi3_ft.py \
        --model_name_or_path ${model} \
        --use_flashattn False \
        --bf16 True \
        --use_peft True \
        --max_seq_length 128 \
        --output_dir="output" \
        --evaluation_strategy="epoch" \
        --learning_rate=1e-3 \
        --gradient_accumulation_steps=1 \
        --per_device_train_batch_size=8 \
        --per_device_eval_batch_size=8 \
        --num_train_epochs=1 \
        --save_steps=500 \
        --logging_steps=1 \
        --save_total_limit=8 2>&1 | tee phi3-mini_ft_fsdp_converge.log

}



main() {

    model=microsoft/Phi-3-mini-4k-instruct

    Run_phi3-mini_peft_singlecard
    Run_phi3-mini_peft_fsdp

}

main
