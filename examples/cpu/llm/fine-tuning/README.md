# IPEX LLAMA2 7B lora alpaca finetuning training on CPUs (distributed)

## Description

This document has instructions for running [LLaMA2 7B](https://huggingface.co/meta-llama/Llama-2-7b-hf)  lora alpaca finetuning using Intel-optimized PyTorch (enable the recipes from [alpaca-lora](https://github.com/tloen/alpaca-lora/tree/main) on CPUs).

## Distributed Computation Environment Setup

In this case, we use data-parallel distributed training and every rank will hold same model replica. The NNODES is the number of ip in the HOSTFILE.

```
export NNODES=#number_of_nodes (default using 1 node)
# Create your ip_list_file (hostfile), one IP per line. If you are using Slurm the step can be implemented by
# scontrol show hostname > ./hostfile
# You can also check the IPs of the hosts and create the hostfile manually.
export HOSTFILE=hostfile
```

*Note:* To use multi-nodes distributed training you should firstly setup the passwordless login (you can refer to [link](https://linuxize.com/post/how-to-setup-passwordless-ssh-login/)) among computation nodes. If you are using the Dockerfile, you can skip this step.

# Quick Start Scripts  

## Run the model

```
# Get the dataset here: https://github.com/tloen/alpaca-lora/blob/main/alpaca_data.json
export DATASET="./alpaca_data.json"

# Get the dataset template here: https://github.com/tloen/alpaca-lora/blob/main/templates/alpaca.json
mkdir ./templates
mv alpaca.json ./templates

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
