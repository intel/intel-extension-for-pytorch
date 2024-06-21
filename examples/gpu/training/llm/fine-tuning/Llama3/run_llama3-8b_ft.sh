# proxy set for downloading dataset error
# export http_proxy="http://child-jf.intel.com:912"
# export https_proxy="http://child-jf.intel.com:912"

# export ZE_AFFINITY_MASK=4

# export PROFILE=1
# export KINETO=1
# export DNNL_VERBOSE=1

# settings for torchccl
export TORCH_LLM_ALLREDUCE=1

Run_llama3-8b_peft_singlecard() {

    python llama3_ft.py \
        --model_name_or_path ${model} \
        --use_flashattn True \
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
        --save_total_limit=8 2>&1 | tee llama3-8b_ft_singlecard_converge.log
}

Run_llama3-8b_peft_fsdp() {

    accelerate launch --main_process_port "29800" --config_file "fsdp_config.yaml"  llama3_ft.py \
        --model_name_or_path ${model} \
        --use_flashattn True \
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
        --save_total_limit=8 2>&1 | tee llama3-8b_ft_fsdp_converge.log

}


main() {

    model='/path_to_llama3/llama3-8b'

    Run_llama3-8b_peft_singlecard
    Run_llama3-8b_peft_fsdp

}

main
