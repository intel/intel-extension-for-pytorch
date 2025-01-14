Run_QLoRA_llama3-8b_pvc() {
    model=meta-llama/Meta-Llama-3-8B
    python bnb_lora_xpu.py --model_name ${model} --quant_type nf4 --device xpu --lora_r 8 --lora_alpha=16 --max_seq_length 512 --per_device_train_batch_size 2 --gradient_accumulation_steps 4 | tee llama3-8b_qlora.log
}

main() {
    Run_QLoRA_llama3-8b_pvc
}

main