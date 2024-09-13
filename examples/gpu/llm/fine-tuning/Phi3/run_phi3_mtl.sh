# For MTL platform not support flash attention
Run_phi3-mini_peft() {

    python phi3_ft.py \
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
        --save_total_limit=8 2>&1 | tee phi-3-mini_ft_bs8_seq128_converage.log

}


main() {

    model='microsoft/Phi-3-mini-4k-instruct'

    Run_phi3-mini_peft

}

main
