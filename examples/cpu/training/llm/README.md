# IPEX LLAMA2 7B lora apalca finetuning training on CPUs (distributed)

## Description

This document has instructions for running [LLaMA2 7B](https://huggingface.co/meta-llama/Llama-2-7b-hf)  lora apalca finetuning using Intel-optimized PyTorch (enable the recipes from [apalca-lora](https://github.com/tloen/alpaca-lora/tree/main) on CPUs ).

## Bare Metal
### General setup

Follow [link](https://github.com/intel/intel-extension-for-pytorch/tree/release/2.3/examples/cpu/inference/python/llm#3-environment-setup) to setup PyTorch/IPEX and some other dependency.

### Prepare dependency
```
  pip install -r requirements.txt
 ```
### Specific Setup

* Set ENV to use multi-nodes distributed training (no need for single-node multi-sockets)

In this case, we use data-parallel distributed training and every rank will hold same model replica. The NNODES is the number of ip in the HOSTFILE. To use multi-nodes distributed training you should firstly setup the passwordless login (you can refer to [link](https://linuxize.com/post/how-to-setup-passwordless-ssh-login/)) between these nodes. 
```
export NNODES=#your_node_number (default using 1 node)
# create your_ip_list_file, one ip per line, like (or self edit):
scontrol show hostname > ./hostfile

export HOSTFILE=hostfile 

```
# Quick Start Scripts  
## Run the model
```
# Get the dataset here https://github.com/tloen/alpaca-lora/blob/main/alpaca_data.json
export DATASET="./alpaca_data.json"
# Env vars
export LOCAL_BATCH_SIZE=32 #32 is default one, you can choose per need
export MODEL_NAME_OR_PATH="YOUR LOCAL PATH or MODEL_ID (HF)"
export MAXSTEP=-1 #default is -1, which means running all steps

#[optional] you may need to get access to llama2 weights from HF
Apply the access in this page [LLaMA2 7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) with your huggingface account
huggingface-cli login
{your huggingface token}
```
## Launch command
|  DataType   | Throughput  |
| ----------- | ----------- |
| BF16        | bash run_lora_finetune_ddp.sh bf16  |
