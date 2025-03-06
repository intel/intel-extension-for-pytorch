# LLM Quantized Model Lora-Finetuning Overview

Here you can find the quantized model lora-finetuning scripts for Llama3.



## Supported Platforms

\* Intel® Data Center GPU Max Series (1550/1100) : support Llama3.1-8B.<br />
\* Intel® Core™ Ultra Processors with Intel® Arc™ B Series Graphics : support Llama3.2-3B.<br />

## Run Models

**Note**: During the execution, you may need to log in your Hugging Face account to access model files. Refer to [HuggingFace Login](https://huggingface.co/docs/huggingface_hub/quick-start#login)

```
huggingface-cli login --token <your_token_here>
```

### Environment Set Up
Set up environment by following [LLM Environment Set Up](../README.md).


### Run Qlora finetuning with quantized model using Bash Script

The related code and run script are prepared in the folder. Run all with the one-click bash script `run_qlora_pvc.sh` or `run_qlora_client.sh`:


If you are running on a Data Center Max Series GPU：

```
bash run_qlora_pvc.sh
```

If you are running on a Intel Client GPU:

```
bash run_qlora_client.sh
```


### Run inference with quantized model

```
# set quant_type and max_new_tokens according to your needs
python bnb_inf_xpu.py --model_name ${model} --quant_type nf4 --max_new_tokens 64 --device xpu 
```
