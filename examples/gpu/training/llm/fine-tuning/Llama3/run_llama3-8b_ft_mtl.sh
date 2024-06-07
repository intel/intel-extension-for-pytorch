# proxy set for downloading dataset error
# export http_proxy="http://child-jf.intel.com:912"
# export https_proxy="http://child-jf.intel.com:912"

# For MTL platform not support flash attention
Run_llama3-8b_peft() {

    python llama3_ft.py \
        --model_name_or_path ${model} \
        --custom_mp True \
        --use_peft True \
        --max_seq_length 128 \
        --output_dir="output" \
        --evaluation_strategy="epoch" \
        --learning_rate=1e-3 \
        --auto_find_batch_size=True \
        --num_train_epochs=1 \
        --save_steps=100 \
        --logging_steps=1 \
        --save_total_limit=8 2>&1 | tee llama3-8b_ft_bs8_seq128_converage.log
}


main() {

    model='/path_to_llama3-8b' 

    Run_llama3-8b_peft

}

main
