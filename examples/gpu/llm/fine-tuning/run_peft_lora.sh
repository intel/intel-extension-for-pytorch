max_seq_length=32
lora_r=8
lora_alpha=8
Run_peft_lora_phi3-mini() {
    model=microsoft/Phi-3-mini-4k-instruct
    lora_target_modules="qkv_proj"

    python peft_lora.py \
	--model_name_or_path ${model} \
	--custom_mp True \
	--use_peft True \
	--max_seq_length ${max_seq_length} \
	--lora_r ${lora_r} \
	--lora_alpha ${lora_alpha} \
    --lora_target_modules ${lora_target_modules} \
	--output_dir="output" \
	--evaluation_strategy="epoch" \
	--learning_rate=1e-3 \
	--per_device_train_batch_size=1 \
	--num_train_epochs=1 \
	--save_steps=100 \
	--logging_steps=1 \
	--save_total_limit=8 | tee peft_lora_bs8_seq32_converage.log
}

Run_peft_lora_qwen2-1.5b() {
    model=Qwen/Qwen2-1.5B
    lora_target_modules="q_proj"

    python peft_lora.py \
    --model_name_or_path ${model} \
    --custom_mp True \
    --use_peft True \
    --max_seq_length ${max_seq_length} \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --lora_target_modules ${lora_target_modules} \
    --output_dir="output" \
    --evaluation_strategy="epoch" \
    --learning_rate=1e-3 \
    --per_device_train_batch_size=1 \
    --num_train_epochs=1 \
    --save_steps=100 \
    --logging_steps=1 \
    --save_total_limit=8 | tee peft_lora_bs8_seq32_converage.log
}


main() {

    Run_peft_lora_phi3-mini
    Run_peft_lora_qwen2-1.5b

}

main
