## Mixtral 7B training

This guide demonstrates how to integrate the Mistral model into the Mixtral model using Mixture of Experts (MoE) techniques and train the model efficiently with DeepSpeed.

### Environment Set Up
Set up environment by following [LLM Environment Set Up](../../README.md).

### Download a Model
During the execution, you may need to log in your Hugging Face account to download model files from online mode. Refer to [HuggingFace Login](https://huggingface.co/docs/huggingface_hub/quick-start#login)

```
huggingface-cli login --token <your_token_here>
```

### Training on multi-GPU

**Note**:
bf16/fp16 is validated on 4 cards 8 tiles on Intel® Data Center Max 1550 GPU on Linux
, If you are using different setting, please change the `--num_gpus=8` accordingly. For example, to use 8 cards 16 tiles, the parameter should be changed to `--num_gpus=16`. 

**Note**:
We provide examples for `wikitext-2-raw-v1` dataset from Hugging Face. 


```bash
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
export TORCH_LLM_ALLREDUCE=1

export model='mistralai/Mistral-7B-v0.1'

deepspeed --num_gpus=8 run_mixtral_training.py \
        --deepspeed ./ds_config.json \
        --model_name_or_path ${model} \
        --dataset_name wikitext \
        --dataset_config_name wikitext-2-raw-v1 \
        --dataloader_num_workers 1 \
        --per_device_train_batch_size 1 \
        --warmup_steps 10 \
        --max_steps 50 \
        --bf16 \
        --do_train \
        --output_dir ./output_dir \
        --overwrite_output_dir 2>&1 | tee training.log
```


