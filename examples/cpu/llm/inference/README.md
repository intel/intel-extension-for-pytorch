# 1. ipex.llm Optimized Model List for Inference

## 1.1 Verified for single instance mode

| MODEL FAMILY | MODEL NAME (Huggingface hub) | FP32 | BF16 | Static quantization INT8 | Weight only quantization INT8 | Weight only quantization INT4 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|LLAMA| meta-llama/Llama-2-7b-hf | 🟩 | 🟩 | 🟨 | 🟩 | 🟨 |
|LLAMA| meta-llama/Llama-2-13b-hf | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |
|LLAMA| meta-llama/Llama-2-70b-hf | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |
|LLAMA| meta-llama/Meta-Llama-3-8B | 🟩 | 🟩 | 🟨 | 🟩 | 🟨 |
|LLAMA| meta-llama/Meta-Llama-3-70B | 🟩 | 🟩 | 🟨 | 🟩 | 🟩 |
|LLAMA| meta-llama/Meta-Llama-3.1-8B-Instruct | 🟩 | 🟩 | 🟨 | 🟩 | 🟩 |
|GPT-J| EleutherAI/gpt-j-6b | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |
|GPT-NEOX| EleutherAI/gpt-neox-20b | 🟩 | 🟨 | 🟨 | 🟩 | 🟨 |
|DOLLY| databricks/dolly-v2-12b | 🟩 | 🟨 | 🟨 | 🟩 | 🟨 |
|FALCON| tiiuae/falcon-7b  | 🟩 | 🟩 | 🟩 | 🟩 |   |
|FALCON| tiiuae/falcon-11b | 🟩 | 🟩 | 🟩 | 🟩 | 🟨 |
|FALCON| tiiuae/falcon-40b | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |
|OPT| facebook/opt-30b | 🟩 | 🟩 | 🟩 | 🟩 | 🟨 |
|OPT| facebook/opt-1.3b | 🟩 | 🟩 | 🟩 | 🟩 | 🟨 |
|Bloom| bigscience/bloom-1b7 | 🟩 | 🟨 | 🟩 | 🟩  | 🟨 |
|CodeGen| Salesforce/codegen-2B-multi | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |
|Baichuan| baichuan-inc/Baichuan2-7B-Chat | 🟩 | 🟩 | 🟩 | 🟩 | 🟨 |
|Baichuan| baichuan-inc/Baichuan2-13B-Chat | 🟩 | 🟩 | 🟨 | 🟩 | 🟨 |
|Baichuan| baichuan-inc/Baichuan-13B-Chat | 🟩 | 🟨 | 🟩 | 🟩 | 🟨 |
|ChatGLM| THUDM/chatglm3-6b | 🟩 | 🟩 | 🟨 | 🟩 | 🟨 |
|ChatGLM| THUDM/chatglm2-6b | 🟩 | 🟩 | 🟩 | 🟩 | 🟨 |
|GPTBigCode| bigcode/starcoder | 🟩 | 🟩 | 🟨 | 🟩 | 🟨 |
|T5| google/flan-t5-xl | 🟩 | 🟩 | 🟨 | 🟩 |   |
|MPT| mosaicml/mpt-7b | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |
|Mistral| mistralai/Mistral-7B-v0.1 | 🟩 | 🟩 | 🟨 | 🟩 | 🟨 |
|Mixtral| mistralai/Mixtral-8x7B-v0.1 | 🟩 | 🟩 |  | 🟩 | 🟨 |
|Stablelm| stabilityai/stablelm-2-1_6b | 🟩 | 🟩 | 🟨 | 🟩 | 🟨 |
|Qwen| Qwen/Qwen-7B-Chat | 🟩 | 🟩 | 🟨 | 🟩 | 🟨 |
|Qwen| Qwen/Qwen2-7B | 🟩 | 🟩 | 🟨 | 🟩 | 🟨 |
|LLaVA| liuhaotian/llava-v1.5-7b | 🟩 | 🟩 |   | 🟩 | 🟩 |
|GIT| microsoft/git-base | 🟩 | 🟩 |   | 🟩 |   |
|Yuan| IEITYuan/Yuan2-102B-hf | 🟩 | 🟩 |   | 🟨 |   |
|Phi| microsoft/phi-2 | 🟩 | 🟩 | 🟩 | 🟩 | 🟨 |
|Phi| microsoft/Phi-3-mini-4k-instruct | 🟩 | 🟩 | 🟨 | 🟩 | 🟨 |
|Phi| microsoft/Phi-3-mini-128k-instruct | 🟩 | 🟩 | 🟨 | 🟩 | 🟨 |
|Phi| microsoft/Phi-3-medium-4k-instruct | 🟩 | 🟩 | 🟨 | 🟩 | 🟨 |
|Phi| microsoft/Phi-3-medium-128k-instruct | 🟩 | 🟩 | 🟨 | 🟩 | 🟨 |
|Whisper| openai/whisper-large-v2 | 🟩 | 🟩 | 🟩 | 🟩 |   |

## 1.2 Verified for distributed inference mode via DeepSpeed

| MODEL FAMILY | MODEL NAME (Huggingface hub) | BF16 | Weight only quantization INT8 |
|:---:|:---:|:---:|:---:|
|LLAMA| meta-llama/Llama-2-7b-hf | 🟩 | 🟩 |
|LLAMA| meta-llama/Llama-2-13b-hf | 🟩 | 🟩 |
|LLAMA| meta-llama/Llama-2-70b-hf | 🟩 | 🟩 |
|LLAMA| meta-llama/Meta-Llama-3-8B | 🟩 | 🟩 |
|LLAMA| meta-llama/Meta-Llama-3-70B | 🟩 | 🟩 |
|LLAMA| meta-llama/Meta-Llama-3.1-8B-Instruct | 🟩 | 🟩 |
|GPT-J| EleutherAI/gpt-j-6b | 🟩 | 🟩 |
|GPT-NEOX| EleutherAI/gpt-neox-20b | 🟨 | 🟩 |
|DOLLY| databricks/dolly-v2-12b | 🟨 | 🟩 |
|FALCON| tiiuae/falcon-11b | 🟩 | 🟩 |
|FALCON| tiiuae/falcon-40b | 🟩 | 🟩 |
|OPT| facebook/opt-30b | 🟨 | 🟩 |
|OPT| facebook/opt-1.3b | 🟩 | 🟩 |
|Bloom| bigscience/bloom-1b7 | 🟨 | 🟩 |
|CodeGen| Salesforce/codegen-2B-multi |  🟩 | 🟩 |
|Baichuan| baichuan-inc/Baichuan2-7B-Chat | 🟩 | 🟩 |
|Baichuan| baichuan-inc/Baichuan2-13B-Chat | 🟩 | 🟩 |
|Baichuan| baichuan-inc/Baichuan-13B-Chat | 🟨 | 🟩 |
|GPTBigCode| bigcode/starcoder | 🟩 | 🟩 |
|T5| google/flan-t5-xl | 🟩 | 🟩 |
|Mistral| mistralai/Mistral-7B-v0.1 | 🟩 | 🟩 |
|Mistral| mistralai/Mixtral-8x7B-v0.1 | 🟩 | 🟩 |
|MPT| mosaicml/mpt-7b | 🟩 | 🟩 |
|Stablelm| stabilityai/stablelm-2-1_6b | 🟩 | 🟩 |
|Qwen| Qwen/Qwen-7B-Chat | 🟩 | 🟩 |
|Qwen| Qwen/Qwen2-7B | 🟩 | 🟩 |
|GIT| microsoft/git-base | 🟩 | 🟩 |
|Phi| microsoft/phi-2 | 🟩 | 🟩 |
|Phi| microsoft/Phi-3-mini-4k-instruct | 🟩 | 🟩 |
|Phi| microsoft/Phi-3-mini-128k-instruct | 🟩 | 🟩 |
|Phi| microsoft/Phi-3-medium-4k-instruct | 🟩 | 🟩 |
|Phi| microsoft/Phi-3-medium-128k-instruct | 🟩 | 🟩 |
|Whisper| openai/whisper-large-v2 | 🟩 | 🟩 |

- 🟩 signifies that the model can perform well and with good accuracy (<1% difference as compared with FP32).

- 🟨 signifies that the model can perform well while accuracy may not been in a perfect state (>1% difference as compared with FP32).

*Note*: As the modeling file of `meta-llama/Llama-2-7b-hf` is updated,
the previous static quantization INT8 recipe is invalid and retuning is working in progress.
If needed, you can try to retune the model with the [autotune API](../../../../docs/tutorials/features/sq_recipe_tuning_api.md).

*Note*: The above verified models (including other models in the same model family, like "codellama/CodeLlama-7b-hf" from LLAMA family) are well supported with all optimizations like indirect access KV cache, fused ROPE, and prepacked TPP Linear (fp32/bf16).
We are working in progress to better support the models in the tables with various data types. In addition, more models will be optimized in the future.

<br>

# 2. How To Run LLM with ipex.llm

**ipex.llm provides a single script to facilitate running generation tasks as below:**

```
python run.py --help # for more detailed usages
```

| Key args of run.py | Notes |
|---|---|
| generation | default: beam search (beam size = 4), "--greedy" for greedy search |
| input tokens | default: 32, provide fixed sizes for input prompt size, use "--input-tokens" for [32, 64, 128, 256, 512, 1024, 2016, 2017, 2048, 4096, 8192]; if "--input-tokens" is not used, use "--prompt" to choose other strings as inputs|
| output tokens | default: 32, use "--max-new-tokens" to choose any other size |
| batch size |  default: 1, use "--batch-size" to choose any other size |
| token latency |  enable "--token-latency" to print out the first or next token latency |
| generation iterations |  use "--num-iter" and "--num-warmup" to control the repeated iterations of generation, default: 100-iter/10-warmup |
| streaming mode output | greedy search only (work with "--greedy"), use "--streaming" to enable the streaming generation output |

*Note:* You may need to log in your HuggingFace account to access the model files. Please refer to [HuggingFace login](https://huggingface.co/docs/huggingface_hub/quick-start#login).

## 2.1 Quick example for running Llama2-7b

### 2.1.1 To run generation task and benchmark performance

*Note:* The following "OMP_NUM_THREADS" and "numactl" settings are based on the assumption that the target server has 56 physical cores per numa socket, and we benchmark with 1 socket. Please adjust the settings per your hardware.

#### 2.1.1.1 Run in FP32 with stock PyTorch

```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype float32
```

#### 2.1.1.2 Run in FP32 with ipex.llm

```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype float32 --ipex 
```

#### 2.1.1.3 Run in BF16 with ipex.llm

```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex 
```

#### 2.1.1.4 Run in static quantization INT8 with ipex.llm

```bash
wget https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/cpu/2/llama2-7b_qconfig.json
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  --benchmark -m meta-llama/Llama-2-7b-hf --ipex-smooth-quant --qconfig-summary-file llama2-7b_qconfig.json --output-dir "saved_results"
```

#### 2.1.1.5 Run in weight-only quantization INT8 with ipex.llm

```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  --benchmark -m meta-llama/Llama-2-7b-hf --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp --output-dir "saved_results" 
```

#### 2.1.1.6 Run in weight-only quantization INT4 with ipex.llm

```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  --benchmark -m meta-llama/Llama-2-7b-hf --ipex-weight-only-quantization --weight-dtype INT4  --gptq --quant-with-amp --output-dir "saved_results" 
```

#### 2.1.1.7 Run in BF16 with ipex.llm in distributed way

```bash
deepspeed --bind_cores_to_rank  run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex  --autotp --shard-model
```

#### 2.1.1.8 Run in weight-only quantization INT8 with ipex.llm in distributed way

```bash
deepspeed --bind_cores_to_rank  run.py --benchmark -m meta-llama/Llama-2-7b-hf --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp  --autotp --shard-model --output-dir "saved_results"
```

### 2.1.2 To run generation task and test accuracy

For the quantized models used in accuracy tests below, we can reuse the model files that are named "best_model.pt" in the "--output-dir" path ([generated during inference performance tests above](#generation_sq)).

Check [Advanced Usage](#32-accuracy-test) for details.

#### 2.1.2.1 Single instance

```bash
# The following "OMP_NUM_THREADS" and "numactl" settings are based on the assumption that
# the target server has 56 physical cores per numa socket, and we benchmark with 1 socket.
# Please adjust the settings per your hardware.

# run_accuracy.py script is inside single_instance directory.
cd single_instance

# Running FP32 model
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run_accuracy.py  -m meta-llama/Llama-2-7b-hf --dtype float32 --ipex --tasks lambada_openai

# Running BF16 model
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run_accuracy.py  -m meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex --tasks lambada_openai

# Quantization
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run_accuracy.py -m meta-llama/Llama-2-7b-hf --quantized-model-path "../saved_results/best_model.pt" --dtype int8  --tasks lambada_openai
```

#### 2.1.2.2 Distributed inference

```bash
# run_accuracy_with_deepspeed.py script is inside distributed directory.
cd distributed
unset KMP_AFFINITY

# Distributed inference in FP32
deepspeed  --num_accelerators 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py  --model  meta-llama/Llama-2-7b-hf --dtype float32 --ipex --tasks lambada_openai 

# Distributed inference in BF16
deepspeed  --num_accelerators 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py  --model  meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex --tasks lambada_openai 

# Distributed inference with Weight-Only Quantization
deepspeed  --num_accelerators 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py  --model  meta-llama/Llama-2-7b-hf --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp --tasks lambada_openai  
```

#### 2.1.2.3 Distributed inference among multiple nodes with TCP 

A [bash script](./tools/run_scaling.sh) is provided to simplify environment configuration and the command launch.

Steps:

1. Enter the `llm` directory
2. Create a `hostfile.txt` following [instructions of deepspeed](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node)
3. Find out the network interface name used for node communication via `ifconfig` or `ibv_devices` ex : eth0
4. Open `tools/run_scaling.sh` script to update required information in line 3 to line 11 according to your environment and needs
5. run the command below to run distributed inference among nodes

```bash
bash tools/run_scaling.sh
```

The docker image built in [the environment setup tutorial](../README.md#2-environment-setup) functions ssh connection for distributed executions across multiple machines via Ethernet. However, it is supposed to be running with 1 single container on each machine. Inside each docker container, multiple inference instances can be launched by the `deepspeed` command.

Use the command below on all machines to launch the docker containers. This command uses the host network interfaces inside the docker container. Thus, you need to put the host ip addresses into the `hostfile.txt`. Do NOT launch multiple docker containers on one single machine from the same docker image. These docker containers listen on the same machine on the same port, will result in unpredicable ssh connections.

```bash
docker run --rm -it --privileged --net host ipex-llm:main bash
```

**Note:** For models on HuggingFace require access privileges, you need to run the `huggingface-cli login` command in each docker container to config a HuggingFace access token.

## 2.2 Detail usage of running LLM models

### 2.2.1 Run generation with one instance

#### 2.2.1.1 FP32:

- Command:
```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run.py --benchmark -m <MODEL_ID> --dtype float32 --ipex
```

- An example of llama2 7b model:
```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype float32 --ipex
```

#### 2.2.1.2 BF16:

- Command:
```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run.py --benchmark -m <MODEL_ID> --dtype bfloat16 --ipex
```

- An example of llama2 7b model:
```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex
```

#### 2.2.1.3 Static quantization (INT8):

We use the SmoothQuant algorithm to get good accuracy of static quantization, which is a popular method for LLM models. Besides, by default, we enable quantization mixed fp32 inference (non-quantized OPs run with fp32 dtype). To get better performance, you may add "--quant-with-amp" to enable quantization with [Automatic Mixed Precision](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html) inference (non-quantized OPs run with bf16 dtype). Please note that static quantization with AMP is still experimental and it may lead to accuracy drop and other issues.

- Command:
```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run.py  --benchmark -m <MODEL_ID> --ipex-smooth-quant --qconfig-summary-file <path to the qconfig of the model_id> --output-dir "saved_results" 
```

- An example of llama2 7b model:
```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  --benchmark -m meta-llama/Llama-2-7b-hf --ipex-smooth-quant --qconfig-summary-file <path to "llama-2-7b_qconfig.json"> --output-dir "saved_results"
```

We provide the following qconfig summary files with good quality (calibration on "NeelNanda/pile-10k" dataset and evaluate accuracy on "lambada_openai" dataset):

| Model ID | Download links |
|---|:---:|
| meta-llama/Llama-2-13b-hf | [link](https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/cpu/2/llama2-13b_qconfig.json) |
| meta-llama/Llama-2-70b-hf | [link](https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/cpu/2/llama2-70b_qconfig.json) |
| EleutherAI/gpt-j-6b | [link](https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/cpu/2/gpt-j-6b_qconfig.json) |
| tiiuae/falcon-7b | [link](https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/cpu/2/falcon-7b_qconfig.json) |
| tiiuae/falcon-11b | [link](https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/cpu/2/falcon-11b_qconfig.json) |
| tiiuae/falcon-40b | [link](https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/cpu/2/falcon-40b_qconfig.json) |
| facebook/opt-30b | [link](https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/cpu/2/opt-30b_qconfig.json) |
| facebook/opt-1.3b | [link](https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/cpu/2/opt-1b3_qconfig.json) |
| baichuan-inc/Baichuan2-7B-Chat | [link](https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/cpu/2/baichuan2-7b_qconfig.json) |
| baichuan-inc/Baichuan-13B-Chat | [link](https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/cpu/2/baichuan-13b_qconfig.json) |
| THUDM/chatglm2-6b | [link](https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/cpu/2/chatglm2_qconfig.json) |
| bigscience/bloom-1b7 | [link](https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/cpu/2/bloom-1b7_qconfig.json) |
| Salesforce/codegen-2B-multi | [link](https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/cpu/2/codegen-2b_qconfig.json) |
| mosaicml/mpt-7b | [link](https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/cpu/2/mpt-7b_qconfig.json) |
| microsoft/phi-2 | [link](https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/cpu/2/phi-2_qconfig.json) |
| openai/whisper-large-v2 | [link](https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/cpu/2/whisper_qconfig.json) |

If you would like to generate qconfig summary files (due to changes on model variants or calibration dataset), we provide the [autotune API](../../../../docs/tutorials/features/sq_recipe_tuning_api.md) and its [tuning examples](llm_sq_recipes.md), which allows an automatic global smoothquant tuning, and automatic layer-by-layer tuning provided by Intel® Neural Compressor for the best accuracy.

#### 2.2.1.4 Weight-only quantization:

By default, for weight-only quantization, we use quantization with [Automatic Mixed Precision](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html) inference ("--quant-with-amp") to get peak performance and fair accuracy.

- Command (INT8):
```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list>  python run.py  --benchmark -m <MODEL_ID> --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp --output-dir "saved_results" 
```

The command above works for most models we listed. However, to get better accuracy for the following models, some changes to the command are needed.

| Model ID | Changes to command |
| - | - |
| bigcode/starcoder | Add "`--group-size 128`" |
| baichuan-inc/Baichuan-13B-Chat | Remove "`--quant-with-amp`" |
| baichuan-inc/Baichuan2-13B-Chat | Add "`--group-size 64`" |
| bigscience/bloom-1b7 | Remove "`--quant-with-amp`"; add "`--group-size 128`" |
| EleutherAI/gpt-neox-20b | Remove "`--quant-with-amp`"; add "`--group-size 256`" |
| facebook/opt-30b | Remove "`--quant-with-amp`" |
| databricks/dolly-v2-12b | Remove "`--quant-with-amp`"; add "`--lowp-mode FP32`" |
| stabilityai/stablelm-2-1_6b | Add "`--group-size 128`" |
| meta-llama/Meta-Llama-3-70B | Add "`--group-size 128`" |

- Command (INT4):
```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run.py  --benchmark -m <MODEL_ID> --ipex-weight-only-quantization --weight-dtype INT4 --gptq --quant-with-amp --output-dir "saved_results" 
```

- An INT8 example of llama2 7b model:
```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  --benchmark -m meta-llama/Llama-2-7b-hf --ipex-weight-only-quantization  --weight-dtype INT8 --quant-with-amp --output-dir "saved_results" 
```

- An INT4 example of llama2 7b model:
```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  --benchmark -m meta-llama/Llama-2-7b-hf --ipex-weight-only-quantization --weight-dtype INT4 --gptq --quant-with-amp --output-dir "saved_results" 
```

#### 2.2.1.5 Notes:

(1) [_numactl_](https://linux.die.net/man/8/numactl) is used to specify memory and cores of your hardware to get better performance. _\<node N\>_ specifies the [numa](https://en.wikipedia.org/wiki/Non-uniform_memory_access) node id (e.g., 0 to use the memory from the first numa node). _\<physical cores list\>_ specifies phsysical cores which you are using from the _\<node N\>_ numa node (e.g., 0-56 from the first numa node). You can use [_lscpu_](https://man7.org/linux/man-pages/man1/lscpu.1.html) command in Linux to check the numa node information.

(2) The _\<MODEL_ID\>_ (e.g., "meta-llama/Llama-2-13b-hf") specifies the model you will run. we provide some verified _\<MODEL ID\>_ in the [Optimized Model List](#1-ipexllm-optimized-model-list-for-inference). You can also try other models from [HuggingFace Models](https://huggingface.co/models).

(3) <a name="generation_sq">for all quantization benchmarks</a>, both quantization and inference stages will be triggered by default. For quantization stage, it will auto-generate the quantized model named "best_model.pt" in the "--output-dir" path, and for inference stage, it will launch the inference with the quantized model "best_model.pt".  For inference-only benchmarks (avoid the repeating quantization stage), you can also reuse these quantized models for by adding "--quantized-model-path <output_dir + "best_model.pt">" .

### 2.2.2 Run generation in distributed way

#### 2.2.2.1 Prepare:

```bash
unset KMP_AFFINITY
```

In the DeepSpeed cases below, we recommend "--shard-model" to shard model weight sizes more even for better memory usage when running with DeepSpeed.

If using "--shard-model", it will save a copy of the shard model weights file in the path of "--output-dir" (default path is "./saved_results" if not provided).
If you have used "--shard-model" and generated such a shard model path (or your model weights files are already well sharded), in further repeated benchmarks, please remove "--shard-model", and replace "-m <MODEL_ID>" with "-m \<shard model path\>" to skip the repeated shard steps.

Besides, the standalone shard model function/scripts are also provided in the [Advanced Usage](#33-how-to-shard-model-for-distributed-tests-with-deepspeed-autotp) section, in case you would like to generate the shard model weights files in advance before running distributed inference.

#### 2.2.2.2 FP32:

- Command:
```bash
deepspeed --bind_cores_to_rank  run.py --benchmark -m <MODEL_ID> --dtype float32 --ipex --autotp --shard-model
```

- An example of llama2 7b model:
```bash
deepspeed --bind_cores_to_rank  run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype float32 --ipex --autotp --shard-model
```

#### 2.2.2.3 BF16:

- Command:
```bash
deepspeed --bind_cores_to_rank  run.py --benchmark -m <MODEL_ID> --dtype bfloat16 --ipex  --autotp --shard-model
```

- An example of llama2 7b model:
```bash
deepspeed --bind_cores_to_rank  run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex  --autotp --shard-model
```

#### 2.2.2.4 Weight-only quantization:

By default, for weight-only quantization, we use quantization with [Automatic Mixed Precision](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html) inference ("--quant-with-amp") to get peak performance and fair accuracy.
For weight-only quantization with deepspeed, we quantize the model then run the benchmark. The quantized model won't be saved.

- Command:
```bash
deepspeed --bind_cores_to_rank run.py  --benchmark -m <MODEL_ID> --ipex --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp  --autotp --shard-model --output-dir "saved_results"
```

Similar to single instance usage, we need to update some arguments of the running command specifically for some models to achieve better accuracy.

| Model ID | Changes to command |
| - | - |
| EleutherAI/gpt-j-6b | Remove "`--quant-with-amp`"; add "`--dtype float32`" |
| EleutherAI/gpt-neox-20b | Remove "`--quant-with-amp`"; add "`--lowp-mode FP32 --dtype float32 --group-size 256`" |
| bigcode/starcoder | Add "`--group-size 128`" |
| baichuan-inc/Baichuan-13B-Chat | Remove "`--quant-with-amp`"; add "`--dtype float32`" |
| baichuan-inc/Baichuan2-13B-Chat | Add "`--group-size 64`" |
| bigscience/bloom-1b7 | Remove "`--quant-with-amp`"; add "`--group-size 128`" |
| facebook/opt-30b | Remove "`--quant-with-amp`"; add "`--dtype float32`" |
| databricks/dolly-v2-12b | Remove "`--quant-with-amp`"; add "`--lowp-mode FP32 --dtype float32`" |
| stabilityai/stablelm-2-1_6b | Add "`--group-size 64`" |
| meta-llama/Meta-Llama-3-70B | Add "`--group-size 128`" |

- An example of llama2 7b model:
```bash
deepspeed --bind_cores_to_rank  run.py --benchmark -m meta-llama/Llama-2-7b-hf --ipex --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp --autotp --shard-model --output-dir "saved_results"
```

### 2.2.3 Additional configuration for specific models

There are some model-specific requirements to be aware of, as follows:

- For MPT models from the remote hub, we need to modify the config.json to use the modeling_mpt.py in transformers. Therefore, in the following scripts, we need to pass an extra configuration file like "--config-file=model_config/mosaicml_mpt-7b_config.json".

- For Falcon models from remote hub, we need to modify the config.json to use the modeling_falcon.py in transformers. Therefore, in the following scripts, we need to pass an extra configuration file like "--config-file=model_config/tiiuae_falcon-40b_config.json". This is optional for FP32/BF16 but needed for quantizations.

- For Llava models from remote hub, additional setup is required, i.e., `bash ./tools/prepare_llava.sh`.

## 2.3 Instructions for Running LLM with Intel® Xeon® CPU Max Series

Intel® Xeon® CPU Max Series are equipped with high bandwidth memory (HBM), which further accelerates LLM inference. For the common case that HBM and DDR are both installed in a Xeon® CPU Max Series server, the memory mode can be configured to Flat Mode or Cache Mode. Details about memory modes can be found at Section 3.1 in [the Xeon® CPU Max Series Configuration Guide](https://cdrdv2-public.intel.com/769060/354227-intel-xeon-cpu-max-series-configuration-and-tuning-guide.pdf).

### 2.3.1 Single Instance Inference with Xeon® CPU Max Series

#### 2.3.1.1 Cache Mode HBM

In cache mode, only DDR address space is visible to software and HBM functions as a transparent memory-side cache for DDR. Therefore the usage is the same with [the common usage](#221-run-generation-with-one-instance).

#### 2.3.1.2 Flat Mode HBM

In flat mode, HBM and DDR are exposed to software as separate address spaces in this mode. Therefore we need to check the `HBM_NODE_INDEX` of interest with commands like `lscpu`, then the LLM inference invoking command would be like:

- Command:
```bash
OMP_NUM_THREADS=<HBM node cores num> numactl -m <HBM_NODE_INDEX> -C <HBM cores list> python run.py --benchmark -m <MODEL_ID> --dtype bfloat16 --ipex
```

- An example of llama2 7b model with HBM numa node index being 2:
```bash
OMP_NUM_THREADS=56 numactl -m 2 -C 0-55 python run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex
```

*Note:* For some very large models we may get an "OOM Error" due to HBM capacity limitations. In this case we can change `-m` argument for `numactl` to `-p` in the above command to enable the model inference with the larger DDR memory.

- Command:
```bash
OMP_NUM_THREADS=<HBM node cores num> numactl -p <HBM_NODE_INDEX> -C <HBM cores list> python run.py --benchmark -m <MODEL_ID> --dtype bfloat16 --ipex
```
- An example of llama2 7b model with HBM numa node index being 2:
```bash
OMP_NUM_THREADS=56 numactl -p 2 -C 0-55 python run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex
```

### 2.3.2 Distributed Inference with Xeon® CPU Max Series

As HBM has memory capacity limitations, we need to shard the model in advance with DDR memory. Please follow [the example](#33-how-to-shard-model-for-distributed-tests-with-deepspeed-autotp).

Then we can invoke distributed inference with `deepspeed` command:

- Command:
```bash
deepspeed --bind_cores_to_rank run.py --benchmark -m <SHARDED_MODEL_PATH> --dtype bfloat16 --ipex --autotp
```

As the model has been sharded, we specify `SHARDED_MODEL_PATH` for `-m` argument instead of original model name or path, and `--shard-model` argument is not needed.

- An example of llama2 7b model:
```bash
python utils/create_shard_model.py -m meta-llama/Llama-2-7b-hf --save-path ./local_llama2_7b
deepspeed --bind_cores_to_rank run.py --benchmark -m ./local_llama2_7b --dtype bfloat16 --ipex --autotp
```

<br>

# 3. Advanced Usage

## 3.1 Weight-only quantization with low precision checkpoint (Prototype)

Using INT4 weights can further improve performance by reducing memory bandwidth. However, direct per-channel quantization of weights to INT4 probably results in poor accuracy. Some algorithms can modify weights through calibration before quantizing weights to minimize accuracy drop. GPTQ is one of such algorithms. You may generate modified weights and quantization info (scales, zero points) for a certain model with a dataset by such algorithms. The low precision checkpoint is saved as a `state_dict` in a `.pt` file and can be loaded later for weight only quantization. We provide an example here to run GPTQ.

*Note:* The GPTQ API is verified on the following models: gpt-j, opt, llama, Llama-2, Llama-3, chatglm2, chatglm3, bloom, bloomz, dolly-v1, dolly-v2, gpt-neo, gpt-neox, mpt, falcon, starcoder, condegen, mistral, mixtral, Qwen, stablelm, phi-2, phi-3. Pleaes note that it's still experimental, please use with care.

Here is how to use it:

```bash
# Step 1: Generate modified weights and quantization info and save as checkpoint
python utils/run_gptq.py --model <MODEL_ID> --output-dir ./saved_results
```

The dataset for calibration is `NeelNanda/pile-10k` by default. To use other dataset, such as lambada, you may use `--dataset <dataset id>` to specify. Group size is specified by `--group-size <group_size>` (default is 128). You can specify calibration sample size by modifying `--nsamples <int>` (default is 128); you can also choose whether or not to align calibration data to a fixed length by modifying `--use_max_length <bool>` and `--pad_max_length <int>`. For details please refer to [GPTQ](../../../../intel_extension_for_pytorch/quantization/_GPTQ/README.md)

It may take a few hours to finish. Modified weights and their quantization info are stored in `gptq_checkpoint_g128.pt`, where g128 means group size for input channel is 128 by default. Group size controls the granularity of quantization of weight along input channel. 
Then generate model for weight only quantization with INT4 weights and run tasks.

```bash
# Step 2: Generate quantized model with INT4 weights
# Provide checkpoint file name by --low-precision-checkpoint <file name>
python single_instance/run_quantization.py --ipex-weight-only-quantization --quant-with-amp -m <MODEL_ID> --low-precision-checkpoint "saved_results/gptq_checkpoint_g128.pt" --output-dir "saved_results" 

# Step 3: Run quantized model for latency benchmark
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python single_instance/run_quantization.py -m <MODEL_ID> --benchmark --quant-with-amp --quantized-model-path "./saved_results/best_model.pt" 
```

To run accuracy tests, please follow the instructions in the [Accuracy Test](#32-accuracy-test) part

If the checkpoint is generated by some other methods and has different keys in the state_dict, you will need to specify the keys for weight, scales, zero points and bias. Bias is optional in the state_dict while others are required. Default keys are:

```python
{
    "weight_key": "packed_weight",
    "scale_key": "scale",
    "zero_point_key": "packed_zp",
    "bias_key": "bias",
}
```

You need to make a config dict like above and pass it to `ipex.llm.optimize` together with the state_dict from the checkpoint as a tuple `(state_dict, config_dict)`. You will need to modify the example script.

```python
low_precision_checkpoint = torch.load(args.low_precision_checkpoint)
config_dict = {
    "weight_key": "...",
    "scale_key": "...",
    "zero_point_key": "...",
    "bias_key": "...",
}
state_dict_and_config = (low_precision_checkpoint, config_dict)
user_model = ipex.llm.optimize(
    user_model.eval(),
    dtype=amp_dtype,
    quantization_config=qconfig,
    inplace=True,
    low_precision_checkpoint=state_dict_and_config,
    deployment_mode=False,
)
```

**Example**

Intel® Extension for PyTorch\* with INT4 weight only quantization has been used in latest MLPerf submission (August 2023) to fully maximize the power of Intel® Xeon®, and also shows good accuracy as comparing with FP32. This example is a simplified version of the MLPerf task. It will download a finetuned FP32 GPT-J model used for MLPerf submission, quantize the model to INT4 and run a text summarization task on the `cnn_dailymail` dataset. The example runs for 1000 samples, which is a good approximation of the results for the entire dataset and saves time.

```bash
pip install evaluate nltk absl-py rouge_score
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> bash single_instance/run_int4_gpt-j_on_cnndailymail.sh
```

Please note that 100 GB disk space, 100 GB memory and Internet access are needed to run this example. The example will run for a few hours depending on your hardware and network condition. The example is verified on the 4th Generation Intel® Xeon® Scalable (Sapphire Rapids) platform. You may get different results on older platforms as some new hardware features are unavailable.

**Checkpoint Requirements**

IPEX now only supports some certain cases. Weights must be N by K and asymmetrically quantized to UINT4 and then compressed along K axis to `torch.int32`. Data type of scales can be any floating point types. Shape of scales should be [N, number_of_groups] or with additional dimensions whose length is 1. Zero points should have the same shape as scales and stored as `torch.int32` but the true data type is UINT4. Bias is optional in the `state_dict` (checkpoint). If it is present, we read bias in the `state_dict`. Otherwise we read bias from the original model. Bias is `None` if it cannot be found in both cases.

## 3.2 Accuracy test

We leverage [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for the accuracy test.

We verify and recommend to test accuracy of most models with "lambada_openai" task.
For some models, like `Salesforce/codegen-2B-multi` and `mosaicml/mpt-7b`, we verify and recommend to test their accuracy with "hellaswag" task.
For more candidate tasks for accuracy validation, please check [lm-evaluation-harness task table](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/docs/task_table.md).

### 3.2.1 Run with one instance

```bash
cd ./single_instance
```

#### 3.2.1.1 FP32:

- Command:
```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run_accuracy.py  -m <MODEL_ID> --dtype float32 --ipex --tasks {TASK_NAME}
```

- An example of llama2 7b model:
```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run_accuracy.py  -m meta-llama/Llama-2-7b-hf --dtype float32 --ipex --tasks lambada_openai
```

#### 3.2.1.2 BF16:

- Command:
```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run_accuracy.py  -m <MODEL_ID> --dtype bfloat16 --ipex --tasks {TASK_NAME}
```

- An example of llama2 7b model:
```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run_accuracy.py  -m meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex --tasks lambada_openai
```

#### 3.2.1.3 Quantizations:

For the quantized models to be used in accuracy tests, we can reuse the model files that are named "best_model.pt" in the "--output-dir" path ([generated during inference performance tests](#generation_sq)).

- Command:
```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_accuracy.py --model <MODEL ID> --quantized-model-path "../saved_results/best_model.pt" --dtype <int8 or int4> --tasks {TASK_NAME}
# Please add  "--quant-with-amp" if your model is quantized with this flag
```

- An example of llama2 7b model:
```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run_accuracy.py -m meta-llama/Llama-2-7b-hf --quantized-model-path "../saved_results/best_model.pt" --dtype int8  --tasks lambada_openai
```

### 3.2.2 Run in distributed way

#### 3.2.2.1 Prepare:

```bash
# Run distributed accuracy with 2 ranks of one node
cd ./distributed
unset KMP_AFFINITY
```

#### 3.2.2.2 FP32:

- Command:
```bash
deepspeed  --num_accelerators 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py  --model <MODEL_ID> --dtype float32 --ipex --tasks <TASK_NAME> 
```
- An example of llama2 7b model:
```bash
deepspeed  --num_accelerators 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py  --model  meta-llama/Llama-2-7b-hf --dtype float32 --ipex --tasks lambada_openai 
```

#### 3.2.2.3 BF16:
- Command:
```bash
deepspeed  --num_accelerators 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py  --model <MODEL_ID> --dtype  bfloat16 -ipex --tasks <TASK_NAME> 
```
- An example of llama2 7b model:
```bash
deepspeed  --num_accelerators 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py  --model  meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex --tasks lambada_openai 
```

#### 3.2.2.4 Weight-only quantization (INT8):

- Command:
```bash
deepspeed  --num_accelerators 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py  --model <MODEL_ID> --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp --ipex --tasks <TASK_NAME>  
```

Similar to script usage for performance benchmarking, we need to update some arguments of the running command specifically for some models to achieve better accuracy.

| Model ID | Changes to command |
| - | - |
| EleutherAI/gpt-j-6b | Remove "`--quant-with-amp`"; add "`--dtype float32`" |
| EleutherAI/gpt-neox-20b | Remove "`--quant-with-amp`"; add "`--lowp-mode FP32 --dtype float32 --group-size 256`" |
| bigcode/starcoder | Add "`--group-size 128`" |
| baichuan-inc/Baichuan-13B-Chat | Remove "`--quant-with-amp`"; add "`--dtype float32`" |
| baichuan-inc/Baichuan2-13B-Chat | Add "`--group-size 64`" |
| bigscience/bloom-1b7 | Remove "`--quant-with-amp`"; add "`--group-size 128`" |
| facebook/opt-30b | Remove "`--quant-with-amp`"; add "`--dtype float32`" |
| databricks/dolly-v2-12b | Remove "`--quant-with-amp`"; add "`--lowp-mode FP32 --dtype float32`" |

- An example of llama2 7b model:
```bash
deepspeed  --num_accelerators 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py  --model  meta-llama/Llama-2-7b-hf --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp --ipex --tasks <TASK_NAME>  
```

## 3.3 How to Shard model for Distributed tests with DeepSpeed (autoTP)

To save memory usage, we could shard the model weights under the local path before we launch distributed tests with DeepSpeed.

```
cd ./utils
# general command:
python create_shard_model.py -m <MODEL ID>  --save-path <SHARD MODEL PATH>
# After sharding the model, using -m <SHARD MODEL PATH> in later tests

# An example of llama2 7b:
python create_shard_model.py -m meta-llama/Llama-2-7b-hf --save-path ./local_llama2_7b
```

<br>

# 6. Performance Results

The performance results on AWS instances can be found [here](../../../../docs/tutorials/performance.md#llm-performance).

<br>

# 7. Miscellaneous Tips

- We can build up LLM services optimized by Intel® Extension for PyTorch\* with Triton Server. Please refer [here](../../serving/triton/README.md) for best practice.

- The LLM inference methods introduced in this page can be well applied for AWS. We can just follow the above instructions and enjoy the boosted performance of LLM with Intel® Extension for PyTorch\* optimizations on the AWS instances.
