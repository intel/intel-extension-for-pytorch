Run_bitsandbtyes_llama3-8b_ptl() {
    model=meta-llama/Meta-Llama-3-8B
    python bnb_lora_xpu.py --model_name ${model} --quant_type nf4 --device xpu --lora_r 8 --lora_alpha=8 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 | tee llama-8b.log
}

main() {
    Run_bitsandbtyes_llama3-8b_ptl
}

main
