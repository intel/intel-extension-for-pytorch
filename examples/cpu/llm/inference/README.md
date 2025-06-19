# 1. ipex.llm Optimized Model List for Inference

We have supported a long list of LLMs, including the most notable open-source models
like Llama series, Qwen series, Phi-3/Phi-4 series,
and the phenomenal high-quality reasoning model [DeepSeek-R1](#223-deepseek-r1-671b).

## 1.1 Verified for single instance mode

| MODEL FAMILY | MODEL NAME (Huggingface hub) | FP32 | BF16 | Weight only quantization INT8 | Weight only quantization INT4 |
|:---:|:---:|:---:|:---:|:---:|:---:|
|LLAMA| meta-llama/Llama-2-7b-hf | ✅ | ✅ | ✅ | ✅ |
|LLAMA| meta-llama/Llama-2-13b-hf | ✅ | ✅ | ✅ | ✅ |
|LLAMA| meta-llama/Llama-2-70b-hf | ✅ | ✅ | ✅ | ✅ |
|LLAMA| meta-llama/Meta-Llama-3-8B | ✅ | ✅ | ✅ | ✅ |
|LLAMA| meta-llama/Meta-Llama-3-70B | ✅ | ✅ | ✅ | ✅ |
|LLAMA| meta-llama/Meta-Llama-3.1-8B-Instruct | ✅ | ✅ | ✅ | ✅ |
|LLAMA| meta-llama/Llama-3.2-3B-Instruct | ✅ | ✅ | ✅ | ✅ |
|LLAMA| meta-llama/Llama-3.2-11B-Vision-Instruct | ✅ | ✅ | ✅ | ✅ |
|GPT-J| EleutherAI/gpt-j-6b | ✅ | ✅ | ✅ | ✅ |
|GPT-NEOX| EleutherAI/gpt-neox-20b | ✅ | ✅ | ✅ | ✅ |
|DOLLY| databricks/dolly-v2-12b | ✅ | ✅ | ✅ | ✅ |
|FALCON| tiiuae/falcon-7b  | ✅ | ✅ | ✅ | ✅ |
|FALCON| tiiuae/falcon-11b | ✅ | ✅ | ✅ | ✅ |
|FALCON| tiiuae/falcon-40b | ✅ | ✅ | ✅ | ✅ |
|FALCON| tiiuae/Falcon3-7B-Instruct | ✅ | ✅ | ✅ | ✅ |
|OPT| facebook/opt-30b | ✅ | ✅ | ✅ | ✅ |
|OPT| facebook/opt-1.3b | ✅ | ✅ | ✅ | ✅ |
|Bloom| bigscience/bloom-1b7 | ✅ | ✅ | ✅ | ✅ |
|CodeGen| Salesforce/codegen-2B-multi | ✅ | ✅ | ✅ | ✅ |
|Baichuan| baichuan-inc/Baichuan2-7B-Chat | ✅ | ✅ | ✅ | ✅ |
|Baichuan| baichuan-inc/Baichuan2-13B-Chat | ✅ | ✅ | ✅ | ✅ |
|Baichuan| baichuan-inc/Baichuan-13B-Chat | ✅ | ✅ | ✅ | ✅ |
|ChatGLM| THUDM/chatglm3-6b | ✅ | ✅ | ✅ | ✅ |
|ChatGLM| THUDM/chatglm2-6b | ✅ | ✅ | ✅ | ✅ |
|GPTBigCode| bigcode/starcoder | ✅ | ✅ | ✅ | ✅ |
|T5| google/flan-t5-xl | ✅ | ✅ | ✅ | ✅ |
|MPT| mosaicml/mpt-7b | ✅ | ✅ | ✅ | ✅ |
|Mistral| mistralai/Mistral-7B-v0.1 | ✅ | ✅ | ✅ | ✅ |
|Mixtral| mistralai/Mixtral-8x7B-v0.1 | ✅ | ✅ | ✅ | ✅ |
|Stablelm| stabilityai/stablelm-2-1_6b | ✅ | ✅ | ✅ | ✅ |
|Qwen| Qwen/Qwen-7B-Chat | ✅ | ✅ | ✅ | ✅ |
|Qwen| Qwen/Qwen2-7B | ✅ | ✅ | ✅ | ✅ |
|Qwen| Qwen/Qwen2.5-7B-Instruct | ✅ | ✅ | ✅ | ✅ |
|LLaVA| liuhaotian/llava-v1.5-7b | ✅ | ✅ | ✅ | ✅ |
|GIT| microsoft/git-base | ✅ | ✅ | ✅ | ✅ |
|Yuan| IEITYuan/Yuan2-102B-hf | ✅ | ✅ | ✅ |   |
|Phi| microsoft/phi-2 | ✅ | ✅ | ✅ | ✅ |
|Phi| microsoft/Phi-3-mini-4k-instruct | ✅ | ✅ | ✅ | ✅ |
|Phi| microsoft/Phi-3-mini-128k-instruct | ✅ | ✅ | ✅ | ✅ |
|Phi| microsoft/Phi-3-medium-4k-instruct | ✅ | ✅ | ✅ | ✅ |
|Phi| microsoft/Phi-3-medium-128k-instruct | ✅ | ✅ | ✅ | ✅ |
|Phi| microsoft/Phi-4-mini-instruct | ✅ | ✅ | ✅ |   |
|Phi| microsoft/Phi-4-multimodal-instruct | ✅ | ✅ | ✅ |   |
|Whisper| openai/whisper-large-v2 | ✅ | ✅ | ✅ | ✅ |
|Maira| microsoft/maira-2 | ✅ | ✅ | ✅ | ✅ |
|Jamba| ai21labs/Jamba-v0.1 | ✅ | ✅ | ✅ | ✅ |
|DeepSeek| deepseek-ai/DeepSeek-V2.5-1210 | ✅ | ✅ | ✅ | ✅ |
|DeepSeek| meituan/DeepSeek-R1-Channel-INT8 |   |   | ✅ |   |

## 1.2 Verified for distributed inference mode via DeepSpeed

| MODEL FAMILY | MODEL NAME (Huggingface hub) | BF16 | Weight only quantization INT8 |
|:---:|:---:|:---:|:---:|
|LLAMA| meta-llama/Llama-2-7b-hf | ✅ | ✅ |
|LLAMA| meta-llama/Llama-2-13b-hf | ✅ | ✅ |
|LLAMA| meta-llama/Llama-2-70b-hf | ✅ | ✅ |
|LLAMA| meta-llama/Meta-Llama-3-8B | ✅ | ✅ |
|LLAMA| meta-llama/Meta-Llama-3-70B | ✅ | ✅ |
|LLAMA| meta-llama/Meta-Llama-3.1-8B-Instruct | ✅ | ✅ |
|LLAMA| meta-llama/Llama-3.2-3B-Instruct | ✅ | ✅ |
|LLAMA| meta-llama/Llama-3.2-11B-Vision-Instruct | ✅ | ✅ |
|GPT-J| EleutherAI/gpt-j-6b | ✅ | ✅ |
|GPT-NEOX| EleutherAI/gpt-neox-20b | ✅ | ✅ |
|DOLLY| databricks/dolly-v2-12b | ✅ | ✅ |
|FALCON| tiiuae/falcon-11b | ✅ | ✅ |
|FALCON| tiiuae/falcon-40b | ✅ | ✅ |
|FALCON| tiiuae/Falcon3-7B-Instruct | ✅ | ✅ |
|OPT| facebook/opt-30b | ✅ | ✅ |
|OPT| facebook/opt-1.3b | ✅ | ✅ |
|Bloom| bigscience/bloom-1b7 | ✅ | ✅ |
|CodeGen| Salesforce/codegen-2B-multi | ✅ | ✅ |
|Baichuan| baichuan-inc/Baichuan2-7B-Chat | ✅ | ✅ |
|Baichuan| baichuan-inc/Baichuan2-13B-Chat | ✅ | ✅ |
|Baichuan| baichuan-inc/Baichuan-13B-Chat | ✅ | ✅ |
|GPTBigCode| bigcode/starcoder | ✅ | ✅ |
|T5| google/flan-t5-xl | ✅ | ✅ |
|Mistral| mistralai/Mistral-7B-v0.1 | ✅ | ✅ |
|Mistral| mistralai/Mixtral-8x7B-v0.1 | ✅ | ✅ |
|MPT| mosaicml/mpt-7b | ✅ | ✅ |
|Stablelm| stabilityai/stablelm-2-1_6b | ✅ | ✅ |
|Qwen| Qwen/Qwen-7B-Chat | ✅ | ✅ |
|Qwen| Qwen/Qwen2-7B | ✅ | ✅ |
|Qwen| Qwen/Qwen2.5-7B-Instruct | ✅ | ✅ |
|GIT| microsoft/git-base | ✅ | ✅ |
|Phi| microsoft/phi-2 | ✅ | ✅ |
|Phi| microsoft/Phi-3-mini-4k-instruct | ✅ | ✅ |
|Phi| microsoft/Phi-3-mini-128k-instruct | ✅ | ✅ |
|Phi| microsoft/Phi-3-medium-4k-instruct | ✅ | ✅ |
|Phi| microsoft/Phi-3-medium-128k-instruct | ✅ | ✅ |
|Whisper| openai/whisper-large-v2 | ✅ | ✅ |
|DeepSeek| deepseek-ai/DeepSeek-V2.5-1210 | ✅ | ✅ |
|DeepSeek| meituan/DeepSeek-R1-Channel-INT8 |   | ✅ |

*Note*: The above verified models (including other models in the same model family, like "codellama/CodeLlama-7b-hf" from LLAMA family)
are well supported with all optimizations like indirect access KV cache, fused ROPE, and customized linear kernels.
We are working in progress to better support the models in the tables with various data types.
In addition, more models will be optimized in the future.

<br>

# 2. How To Run LLM with ipex.llm

**ipex.llm provides a single script to facilitate running generation tasks as below:**

```
python run.py --help # for more detailed usages
```

| Key args of run.py | Notes |
|---|---|
| generation | default: beam search (beam size = 4), "--greedy" for greedy search |
| input tokens or prompt | provide fixed sizes for input prompt size, use "--input-tokens" for <INPUT_LENGTH> in [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 32768, 130944]; if "--input-tokens" is not used, use "--prompt" to choose other strings as inputs|
| output tokens | default: 32, use "--max-new-tokens" to choose any other size |
| batch size |  default: 1, use "--batch-size" to choose any other size |
| token latency |  enable "--token-latency" to print out the first or next token latency |
| generation iterations |  use "--num-iter" and "--num-warmup" to control the repeated iterations of generation, default: 100-iter/10-warmup |
| streaming mode output | greedy search only (work with "--greedy"), use "--streaming" to enable the streaming generation output |
| KV Cache dtype |   default: auto, use "--kv-cache-dtype=fp8_e5m2" to enable e5m2 KV Cache. More information refer to [vLLM FP8 E5M2 KV Cache](https://docs.vllm.ai/en/v0.6.6/quantization/fp8_e5m2_kvcache.html) |
| input mode | default: 0, use "--input-mode" to choose input mode for multimodal models. 0: language; 1: vision; 2: speech; 3: vision and speech |
| input images | default: None, use "--image-url" to choose the image file address for vision-text tasks |
| input audios | default: None, use "--audio" to choose the audio file address for speech tasks |

*Note:* You may need to log in your HuggingFace account to access the model files. Please refer to [HuggingFace login](https://huggingface.co/docs/huggingface_hub/quick-start#login).

**Alternatively, you can run the Jupyter Notebook to see ipex.llm with BF16 and various other quick start examples.**

Additional setup instructions for running the notebook can be found [here](../README.md#23-optional-setup-for-running-jupyter-notebooks).

## 2.1 Quick example for running Llama3.1-8B

### 2.1.1 To run generation task and benchmark performance

*Note:* The following "OMP_NUM_THREADS" and "numactl" settings are based on the assumption that
the target server has 56 physical cores per numa socket, and we benchmark with 1 socket.
Please adjust the settings per your hardware.

#### 2.1.1.1 Run in FP32 with stock PyTorch

```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m meta-llama/Meta-Llama-3.1-8B-Instruct --dtype float32
```

#### 2.1.1.2 Run in FP32 with ipex.llm

```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m meta-llama/Meta-Llama-3.1-8B-Instruct --dtype float32 --ipex
```

#### 2.1.1.3 Run in BF16 with ipex.llm

```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m meta-llama/Meta-Llama-3.1-8B-Instruct --dtype bfloat16 --ipex
```

#### 2.1.1.4 Run in weight-only quantization INT8 with ipex.llm

```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m meta-llama/Meta-Llama-3.1-8B-Instruct --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp --output-dir "saved_results"
```

#### 2.1.1.5 Run in weight-only quantization INT4 with ipex.llm

```bash
huggingface-cli download hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 --local-dir ./Llama-3.1-8B-GPTQ
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m ./Llama-3.1-8B-GPTQ --ipex-weight-only-quantization --weight-dtype INT4 --lowp-mode BF16 --quant-with-amp --output-dir "saved_results"
```

#### 2.1.1.6 Run in BF16 with ipex.llm in distributed way

```bash
deepspeed --bind_cores_to_rank run.py --benchmark -m meta-llama/Meta-Llama-3.1-8B-Instruct --dtype bfloat16 --ipex  --autotp --shard-model
```

#### 2.1.1.7 Run in weight-only quantization INT8 with ipex.llm in distributed way

```bash
deepspeed --bind_cores_to_rank run.py --benchmark -m meta-llama/Meta-Llama-3.1-8B-Instruct --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp --autotp --shard-model --output-dir "saved_results"
```

#### 2.1.1.8 Run in weight-only quantization INT4 with ipex.llm in distributed way

```bash
huggingface-cli download hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 --local-dir ./Llama-3.1-8B-GPTQ
deepspeed --bind_cores_to_rank run.py --benchmark -m ./Llama-3.1-8B-GPTQ --ipex-weight-only-quantization --weight-dtype INT4 --lowp-mode BF16 --quant-with-amp --autotp --output-dir "saved_results"
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
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run_accuracy.py -m meta-llama/Meta-Llama-3.1-8B-Instruct --dtype float32 --ipex --tasks lambada_openai

# Running BF16 model
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run_accuracy.py -m meta-llama/Meta-Llama-3.1-8B-Instruct --dtype bfloat16 --ipex --tasks lambada_openai

# Quantization. Assuming the quantized model is generated at "../saved_results/best_model.pt".
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run_accuracy.py -m meta-llama/Meta-Llama-3.1-8B-Instruct --quantized-model-path "../saved_results/best_model.pt" --dtype int8 --ipex --quant-with-amp --tasks lambada_openai
```

#### 2.1.2.2 Distributed inference

```bash
# Assuming the pre-sharded Llama model is generated at "saved_results/llama_local_shard/" folder.
# run_accuracy_with_deepspeed.py script is under "distributed" directory.
cd distributed

# Distributed inference in FP32
deepspeed --num_accelerators 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py --model "../saved_results/llama_local_shard/" --dtype float32 --ipex --tasks lambada_openai

# Distributed inference in BF16
deepspeed --num_accelerators 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py --model "../saved_results/llama_local_shard/" --dtype bfloat16 --ipex --tasks lambada_openai

# Distributed inference with Weight-Only Quantization
deepspeed --num_accelerators 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py --model "../saved_results/llama_local_shard/" --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp --tasks lambada_openai
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
docker run --rm -it --privileged -v /dev/shm:/dev/shm --net host ipex-llm:main bash
```

**Note:** For models on HuggingFace require access privileges, you need to run the `huggingface-cli login` command in each docker container to config a HuggingFace access token.

## 2.2 Detail instructions for running LLM models

### 2.2.1 Run generation with one instance

#### 2.2.1.1 FP32:

- Command:
```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run.py --benchmark -m <MODEL_ID> --dtype float32 --ipex
```

- An example of Llama-3.1-8B model:
```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m meta-llama/Meta-Llama-3.1-8B-Instruct --dtype float32 --ipex
```

#### 2.2.1.2 BF16:

- Command:
```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run.py --benchmark -m <MODEL_ID> --dtype bfloat16 --ipex
```

- An example of Llama-3.1-8B model:
```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m meta-llama/Meta-Llama-3.1-8B-Instruct --dtype bfloat16 --ipex
```

#### 2.2.1.3 Weight-only quantization INT8:

Weights are quantized by round-to-nearest (RTN).

- **Command for WoQ INT8:**
```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list>  python run.py  --benchmark -m <MODEL_ID> --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp --output-dir "saved_results" 
```

- **An example for Llama-3.1-8B model:**
```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m meta-llama/Meta-Llama-3.1-8B-Instruct --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp --output-dir "saved_results"
```

Notes:

- Please note that `<MODEL_ID>` should be the ID of a non-quantized model instead of any quantized version on HuggingFace.

- [Automatic Mixed Precision](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html) (AMP) is recommended to get peak performance and fair accuracy. It is turned on by `--quant-with-amp` or off by removing the option.

- By default, computation is done in bfloat16 no matter AMP is turned on or not. Computation dtype can be specified by `--lowp-mode`. Available options are `FP32`, `FP16`, `BF16`, and `INT8`.

- By default, weights are quantized per channel. Use `--group-size` for group-wise quantization.

- The command above works fine for most models we listed. However, to get better accuracy for the following models, some changes to the command are needed.

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

#### 2.2.1.4 Weight-only quantization INT4:

For Weight-only Quantization (WoQ) INT4, weights are quantized into int4 by different quantization algorithms. Among them, we support RTN, GPTQ, AWQ and [intel/auto-round](https://github.com/intel/auto-round).

To run with RTN, the command is similar as WoQ INT8 and you need to provide the ID of a non-quantized model:

```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run.py --benchmark -m <MODEL_ID> --ipex-weight-only-quantization --weight-dtype INT4 --quant-with-amp --output-dir "saved_results"
```

To run with GPTQ, AWQ, and intel/auto-round, you need to download or generate quantized weights beforehand.

If the INT4 quantized weight checkpoint files of the desired model can be found in [HuggingFace Models](https://huggingface.co/models),
you can download them and benchmark with the following commands:

```bash
huggingface-cli download <INT4_MODEL_ID> --local-dir <INT4_CKPT_SAVE_PATH>
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run.py  --benchmark -m <INT4_CKPT_SAVE_PATH> --ipex-weight-only-quantization --quant-with-amp --lowp-mode [INT8|BF16]
```

Here is an example to run Llama-3.1-8B with GPTQ:
```bash
huggingface-cli download hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 --local-dir ./Llama-3.1-8B-GPTQ
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m ./Llama-3.1-8B-GPTQ --ipex-weight-only-quantization --quant-with-amp --lowp-mode BF16
```

Note:

- You cannot use the ID of a quantized model on HuggingFace directly for benchmarking. Please download them and provide the local path.

- By default, computation is done in INT8 for WoQ INT4 if `--lowp-mode` is not specified.

- For GPTQ with `desc_act=True`, INT8 computation is not available. You have to set `--lowp-mode BF16` explicitly.

If the quantized INT4 checkpoint of the desired model is not available in HuggingFace Models,
you can quantize the model using [Intel® Neural Compressor (INC)](https://github.com/intel/neural-compressor).
INC supports WoQ INT4 quantization with GPTQ, AWQ and [intel/auto-round](https://github.com/intel/auto-round) algorithms.

Please refer to [INC's tutorial](https://github.com/intel/neural-compressor/tree/9c3d4a1243d7ea7f0843454c37851b1f03fe695b/examples/3.x_api/pytorch/nlp/huggingface_models/language-modeling/quantization/transformers/weight_only/text-generation#quantization-for-cpu-device)
to generate the INT4 weight checkpoint files in a separate python environment.
When the quantization process finishes, use the same command to run the model:

```bash
# Switch back to IPEX environment first.
conda activate llm
# "./llama_3_1_8B_INT4_GPTQ" is the example path of the output INT4 checkpoint.
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m ./llama_3_1_8B_INT4_GPTQ --ipex-weight-only-quantization --quant-with-amp --lowp-mode BF16
```

If your INT4 checkpoints are not from HuggingFace or INC, please make sure the directory has the same structure as those on HuggingFace.

#### 2.2.1.5 Notes:

(1) [_numactl_](https://linux.die.net/man/8/numactl) is used to specify memory and cores of your hardware to get better performance. _\<node N\>_ specifies the [numa](https://en.wikipedia.org/wiki/Non-uniform_memory_access) node id (e.g., 0 to use the memory from the first numa node). _\<physical cores list\>_ specifies phsysical cores which you are using from the _\<node N\>_ numa node (e.g., 0-56 from the first numa node). You can use [_lscpu_](https://man7.org/linux/man-pages/man1/lscpu.1.html) command in Linux to check the numa node information.

(2) The _\<MODEL_ID\>_ (e.g., "meta-llama/Llama-2-13b-hf") specifies the model you will run. we provide some verified _\<MODEL ID\>_ in the [Optimized Model List](#1-ipexllm-optimized-model-list-for-inference). You can also try other models from [HuggingFace Models](https://huggingface.co/models).

(3) <a name="generation_sq">For all quantization benchmarks</a>, both quantization and inference stages will be triggered by default. For quantization stage, it will auto-generate the quantized model named "best_model.pt" in the "--output-dir" path, and for inference stage, it will launch the inference with the quantized model "best_model.pt".  For inference-only benchmarks (avoid the repeating quantization stage), you can also reuse these quantized models for by adding "--quantized-model-path <output_dir + "best_model.pt">" .

### 2.2.2 Run generation in distributed way

#### 2.2.2.1 Prologue:

In the DeepSpeed cases below, we recommend "--shard-model" to shard model weight sizes more even for better memory usage when running with DeepSpeed.

If using "--shard-model", it will save a copy of the shard model weights file in the path of "--output-dir" (default path is "./saved_results" if not provided).
If you have used "--shard-model" and generated such a shard model path (or your model weights files are already well sharded), in further repeated benchmarks, please remove "--shard-model", and replace "-m <MODEL_ID>" with "-m \<shard model path\>" to skip the repeated shard steps.

Besides, the standalone shard model function/scripts are also provided in the [Advanced Usage](#31-how-to-shard-model-for-distributed-tests-with-deepspeed-autotp) section, in case you would like to generate the shard model weights files in advance before running distributed inference.

#### 2.2.2.2 FP32:

- Command:
```bash
deepspeed --bind_cores_to_rank run.py --benchmark -m <MODEL_ID> --dtype float32 --ipex --autotp --shard-model
```

- An example of Llama-3.1-8B model:
```bash
deepspeed --bind_cores_to_rank run.py --benchmark -m meta-llama/Meta-Llama-3.1-8B-Instruct --dtype float32 --ipex --autotp --shard-model
```

#### 2.2.2.3 BF16:

- Command:
```bash
deepspeed --bind_cores_to_rank run.py --benchmark -m <MODEL_ID> --dtype bfloat16 --ipex  --autotp --shard-model
```

- An example of Llama-3.1-8B model:
```bash
deepspeed --bind_cores_to_rank run.py --benchmark -m meta-llama/Meta-Llama-3.1-8B-Instruct --dtype bfloat16 --ipex --autotp --shard-model
```

#### 2.2.2.4 Weight-only quantization INT8:

More details about WoQ INT8 can be found in [the section above](#2213-weight-only-quantization-int8).

For weight-only quantization with deepspeed, we quantize the model then run the benchmark. The quantized model won't be saved.

- Command:
```bash
deepspeed --bind_cores_to_rank run.py --benchmark -m <MODEL_ID> --ipex --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp --autotp --shard-model --output-dir "saved_results"
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
| stabilityai/stablelm-2-1_6b | Add "`--group-size 128`" |
| meta-llama/Meta-Llama-3-70B | Add "`--group-size 128`" |

- An example of Llama-3.1-8B model:
```bash
deepspeed --bind_cores_to_rank run.py --benchmark -m meta-llama/Meta-Llama-3.1-8B-Instruct --ipex --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp --autotp --shard-model --output-dir "saved_results"
```

#### 2.2.2.5 Weight-only quantization INT4:

We can either download a quantized weight checkpoint from Huggingface Models,
or quantize the model using INC with GPTQ/AWQ/AutoRound algorithms,
or quantize the model with RTN algorithm within IPEX. Please refer
[the instructions](#2214-weight-only-quantization-int4) for details.

- Command:
```bash
deepspeed --bind_cores_to_rank run.py --benchmark -m <INT4_CKPT_PATH> --ipex --ipex-weight-only-quantization --weight-dtype INT4 --lowp-mode BF16 --quant-with-amp  --autotp --output-dir "saved_results"
```

- Example with GPTQ INT4 Llama-3.1-8B model:
```bash
huggingface-cli download hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 --local-dir ./Llama-3.1-8B-GPTQ
deepspeed --bind_cores_to_rank run.py --benchmark -m ./Llama-3.1-8B-GPTQ --ipex --ipex-weight-only-quantization --weight-dtype INT4 --lowp-mode BF16 --quant-with-amp --autotp --output-dir "saved_results"
```

### 2.2.3 DeepSeek-R1 671B

IPEX applies dedicated optimizations on the full version of `DeepSeek-R1` model
and it can be showcased with `run.py` script now!

- Currently, weight only quantization INT8 precision is supported.
Please download the INT8 quantized version from [HuggingFace Models](https://huggingface.co/meituan/DeepSeek-R1-Channel-INT8).

```bash
huggingface-cli download --resume meituan/DeepSeek-R1-Channel-INT8 --local-dir <DEEPSEEK_INT8_CKPT_SAVE_PATH>
```

- A change is required in the `config.json` file of the downloaded checkpoint path in order to apply the optimizations.
Please add the `quantization_config` field to the end of the file as below.

```diff
   "transformers_version": "4.46.3",
   "use_cache": true,
   "v_head_dim": 128,
-  "vocab_size": 129280
+  "vocab_size": 129280,
+  "quantization_config": {
+    "quant_method": "int8",
+    "bits": 8,
+    "group_size": -1
+  }
 }
```

- Use the following command to run the test.

```bash
# at examples/cpu/llm/inference
deepspeed --bind_cores_to_rank run.py -m <DEEPSEEK_INT8_CKPT_SAVE_PATH> --benchmark --input-tokens 1024 --max-new-tokens 1024 --ipex-weight-only-quantization --weight-dtype INT8 --ipex --batch-size 1 --autotp --greedy --quant-with-amp --token-latency
```

- Notes

(1) Since the hugeness of the model size as well as the cache based optimizations, it is recommended to use a server with 1.5TB
or larger memory amount. The memory comsumption optimizations are in progress.

(2) Please add `--num_accelerators` and `--bind_core_list` arguments for `deepspeed` command based on your SNC configurations.
For example, for a server having 2 sockets, 128 physical cores per socket with a total number of 6 sub-numa clusters,
it is recommended to set `--num_accelerators 6 --bind_core_list 0-41,43-84,86-127,128-169,171-212,214-255`.

(3) The provided script is mainly for showcasing performance with the default input prompts.
We can replace the prompts in `prompt.json` under `deepseekr1` key with your own inputs.
Also, we can change the script, applying [the chat template](https://huggingface.co/docs/transformers/chat_templating)
to get outputs with higher quality.

(4) We can enlarge `--max-new-tokens` setting for longer outputs and add `--streaming` to get streaming outputs in the console.

### 2.2.4 Additional configuration for specific models

There are some model-specific requirements to be aware of, as follows:

- For MPT models from the remote hub, we need to modify the config.json to use the modeling_mpt.py in transformers. Therefore, in the following scripts, we need to pass an extra configuration file like "--config-file=model_config/mosaicml_mpt-7b_config.json".

- For Falcon models from remote hub, we need to modify the config.json to use the modeling_falcon.py in transformers. Therefore, in the following scripts, we need to pass an extra configuration file like "--config-file=model_config/tiiuae_falcon-40b_config.json". This is optional for FP32/BF16 but needed for quantizations.

- For Llava models from remote hub, additional setup is required, i.e., `bash ./tools/prepare_llava.sh`.

## 2.3 Instructions for Running Multimodal LLMs

Multimodal LLMs are large language models capable of processing multiple types of inputs,
like images and audios, in addition to text prompts.
We have optimized the performance of some popular multimodal LLMs like `microsoft/Phi-4-multimodal-instruct`
and `meta-llama/Llama-3.2-11B-Vision-Instruct`, which can be showcased with the provided `run.py` script.
In the commands, the additional arguments need to be specified are highlighted here:

| Special args for multimodal | Notes |
|---|---|
| input mode | Use "--input-mode" to choose input mode for multimodal models. 0: language; 1: vision; 2: speech; 3: vision and speech |
| input image | Use "--image-url" to specify the image link address or local path for vision-text tasks |
| input audio | Use "--audio" to specify the audio file local path for speech tasks |

Meanwhile, for multimodal tasks we need to set the text prompt and bind it with the input image/audio.
The binding is realized with the special tokens, like the image tag `<|image|>` and the audio tag `<|audio|>`.
We provide the following example commands to showcase the argument settings in detail.

### 2.3.1 Phi-4-multimodal-instruct

You can download the sample image and audio to your local folder beforehand.
Also, `peft` package is required for running the model.

```bash
wget https://www.ilankelman.org/stopsigns/australia.jpg
wget https://voiceage.com/wbsamples/in_mono/Trailer.wav
pip install peft
```

- BF16, single instance

We provide example commands running in BF16 precision for all the input modes.
The OMP thread num and `numactl` setup parts are omitted.

Example command for pure text input

```bash
python run.py --input-mode 0 --benchmark -m microsoft/Phi-4-multimodal-instruct --ipex --token-latency --greedy --dtype bfloat16 --max-new-tokens 128 --prompt "<|system|>You are a helpful assistant.<|end|><|user|>How to explain Internet for a medieval knight?<|end|><|assistant|>"
```

Example command for image comprehension

```bash
python run.py --input-mode 1 --benchmark -m microsoft/Phi-4-multimodal-instruct --ipex --token-latency --greedy --dtype bfloat16 --max-new-tokens 128 --prompt "<|user|><|image_1|>What is shown in this image?<|end|><|assistant|>" --image-url australia.jpg
```

Example command for speech comprehension

```bash
python run.py --input-mode 2 --benchmark -m microsoft/Phi-4-multimodal-instruct --ipex --token-latency --greedy --dtype bfloat16 --max-new-tokens 128 --prompt "<|user|><|audio_1|>Transcribe the audio to text, and then translate the audio to French. Use <sep> as a separator between the original transcript and the translation.<|end|><|assistant|>" --audio Trailer.wav
```

Example command for image and speech comprehension

```bash
python run.py --input-mode 3 --benchmark -m microsoft/Phi-4-multimodal-instruct --ipex --token-latency --greedy --dtype bfloat16 --max-new-tokens 128 --prompt "<|user|><|image_1|><|audio_1|><|end|><|assistant|>" --audio Trailer.wav --image-url australia.jpg
```

- Weight-only quantization INT8, single instance

For WoQ INT8 precision, we need to replace the arguments `--ipex` and `--dtype bfloat16`
`--ipex-weight-only-quantization`, `--weight-dtype INT8` and `--quant-with-amp`.
In addition, `--group-size 128` is needed as group-wise quantization should be applied.

Example command for image and speech comprehension

```bash
python run.py --input-mode 3 --benchmark -m microsoft/Phi-4-multimodal-instruct --token-latency --greedy --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp --group-size 128 --max-new-tokens 128 --prompt "<|user|><|image_1|><|audio_1|><|end|><|assistant|>" --audio Trailer.wav --image-url australia.jpg
```

### 2.3.2 meta-llama/Llama-3.2-11B-Vision-Instruct

`Llama-3.2-11B-Vision-Instruct` model supports image comprehension tasks.
`--input-mode 1` should always be specified for this model.

- BF16, single instance

```bash
python run.py --input-mode 1 --benchmark -m meta-llama/Llama-3.2-11B-Vision-Instruct --ipex --dtype bfloat16 --prompt "<|image|>Describe the contents of this image." --image-url australia.jpg
```

- Weight-only quantization INT8, single instance

```bash
python run.py --input-mode 1 --benchmark -m meta-llama/Llama-3.2-11B-Vision-Instruct --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp --prompt "<|image|>Describe the contents of this image." --image-url australia.jpg
```

- BF16, distributed inference

```bash
deepspeed --bind_cores_to_rank run.py --input-mode 1 --benchmark -m meta-llama/Llama-3.2-11B-Vision-Instruct --ipex --dtype bfloat16 --prompt "<|image|>Describe the contents of this image." --image-url australia.jpg --autotp --shard-model
```

- Weight-only quantization INT8, distributed inference

```bash
deepspeed --bind_cores_to_rank run.py --input-mode 1 --benchmark -m meta-llama/Llama-3.2-11B-Vision-Instruct --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp --prompt "<|image|>Describe the contents of this image." --image-url australia.jpg --autotp --shard-model
```

## 2.4 Instructions for Running LLM with Intel® Xeon® CPU Max Series

Intel® Xeon® CPU Max Series are equipped with high bandwidth memory (HBM), which further accelerates LLM inference. For the common case that HBM and DDR are both installed in a Xeon® CPU Max Series server, the memory mode can be configured to Flat Mode or Cache Mode.
Details about memory modes can be found at Section 3.1 in [the Xeon® CPU Max Series Configuration Guide](https://cdrdv2-public.intel.com/769060/354227-intel-xeon-cpu-max-series-configuration-and-tuning-guide.pdf).

### 2.4.1 Single Instance Inference with Xeon® CPU Max Series

#### 2.4.1.1 Cache Mode HBM

In cache mode, only DDR address space is visible to software and HBM functions as a transparent memory-side cache for DDR.
Therefore the usage is the same with [the common usage](#221-run-generation-with-one-instance).

#### 2.4.1.2 Flat Mode HBM

In flat mode, HBM and DDR are exposed to software as separate address spaces.
Therefore we need to check the `HBM_NODE_INDEX` of interest with commands like `lscpu`, then the LLM inference invoking command would be like:

- Command:
```bash
OMP_NUM_THREADS=<HBM node cores num> numactl -m <HBM_NODE_INDEX> -C <HBM cores list> python run.py --benchmark -m <MODEL_ID> --dtype bfloat16 --ipex
```

- An example of Llama-3.1-8B model with HBM numa node index being 2:
```bash
OMP_NUM_THREADS=56 numactl -m 2 -C 0-55 python run.py --benchmark -m meta-llama/Meta-Llama-3.1-8B-Instruct --dtype bfloat16 --ipex
```

*Note:* For some very large models we may get an "OOM Error" due to HBM capacity limitations. In this case we can change `-m` argument for `numactl` to `-p` in the above command to enable the model inference with the larger DDR memory.

- Command:
```bash
OMP_NUM_THREADS=<HBM node cores num> numactl -p <HBM_NODE_INDEX> -C <HBM cores list> python run.py --benchmark -m <MODEL_ID> --dtype bfloat16 --ipex
```

- An example of Llama-3.1-8B model with HBM numa node index being 2:
```bash
OMP_NUM_THREADS=56 numactl -p 2 -C 0-55 python run.py --benchmark -m meta-llama/Meta-Llama-3.1-8B-Instruct --dtype bfloat16 --ipex
```

### 2.4.2 Distributed Inference with Xeon® CPU Max Series

As HBM has memory capacity limitations, we need to shard the model in advance with DDR memory.
Please follow [the example](#31-how-to-shard-model-for-distributed-tests-with-deepspeed-autotp).

Then we can invoke distributed inference with `deepspeed` command:

- Command:
```bash
deepspeed --bind_cores_to_rank run.py --benchmark -m <SHARDED_MODEL_PATH> --dtype bfloat16 --ipex --autotp
```

As the model has been sharded, we specify `SHARDED_MODEL_PATH` for `-m` argument instead of original model name or path, and `--shard-model` argument is not needed.

- An example of Llama-3.1-8B model:
```bash
python utils/create_shard_model.py -m meta-llama/Meta-Llama-3.1-8B-Instruct --save-path ./local_llama3_1_8b
deepspeed --bind_cores_to_rank run.py --benchmark -m ./local_llama3_1_8b --dtype bfloat16 --ipex --autotp
```

<br>

# 3. Advanced Usage

## 3.1 How to Shard model for Distributed tests with DeepSpeed (autoTP)

To save memory usage, we could shard the model weights under the local path before we launch distributed tests with DeepSpeed.

```
cd ./utils
# general command:
python create_shard_model.py -m <MODEL ID>  --save-path <SHARD MODEL PATH>
# After sharding the model, using -m <SHARD MODEL PATH> in later tests

# An example of Llama-3.1-8B:
python create_shard_model.py -m meta-llama/Meta-Llama-3.1-8B-Instruct --save-path ./local_llama3_1_8b
```

## 3.2 Accuracy test

`ipex.llm` is focusing on LLM performance optimizations,
yet we also provide example scripts for the validation of the model from accuracy perspective.
We leverage [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for the accuracy test,
and recommend to test accuracy of most models with "lambada_openai" task.
For some models, like `Salesforce/codegen-2B-multi` and `mosaicml/mpt-7b`, we recommend to test their accuracy with "hellaswag" task.
For more candidate tasks for accuracy validation, please check [lm-evaluation-harness task table](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/docs/task_table.md).

### 3.2.1 Run with one instance

```bash
cd ./single_instance
```

#### 3.2.1.1 FP32:

- Command:
```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run_accuracy.py -m <MODEL_ID> --dtype float32 --ipex --tasks {TASK_NAME}
```

- An example of Llama-3.1-8B model:
```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run_accuracy.py -m meta-llama/Meta-Llama-3.1-8B-Instruct --dtype float32 --ipex --tasks lambada_openai
```

#### 3.2.1.2 BF16:

- Command:
```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run_accuracy.py -m <MODEL_ID> --dtype bfloat16 --ipex --tasks {TASK_NAME}
```

- An example of Llama-3.1-8B model:
```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run_accuracy.py -m meta-llama/Meta-Llama-3.1-8B-Instruct --dtype bfloat16 --ipex --tasks lambada_openai
```

#### 3.2.1.3 Quantizations:

For the quantized models to be used in accuracy tests, we can reuse the model files that are named "best_model.pt" in the "--output-dir" path ([generated during inference performance tests](#generation_sq)).

- Command:
```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_accuracy.py --model <MODEL ID> --quantized-model-path "../saved_results/best_model.pt" --dtype <int8 or int4> --tasks <TASK_NAME>
# Please add "--quant-with-amp" if your model is quantized with this flag
```

- An example of Llama-3.1-8B model:
```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run_accuracy.py -m meta-llama/Meta-Llama-3.1-8B-Instruct --quantized-model-path "../saved_results/best_model.pt" --dtype int8 --ipex --quant-with-amp --tasks lambada_openai
```

### 3.2.2 Run in distributed way

#### 3.2.2.1 Prologue:

We provided a `run_accuracy_with_deepspeed.py` script for testing accuracy
for the models benchmarked in distributed way via `deepspeed`.

Prior to the accuracy testing, we need to have the sharded model. The sharded model should have been generated
following [the instruction for performance benchmarking with deepspeed](#2221-prologue) where `--shard-model`
flag is set. The generated model shards will be placed in the folder specified by `--output-dir` argument.
Alternatively, the model sharding process can also be accomplished [in a standalone way](#31-how-to-shard-model-for-distributed-tests-with-deepspeed-autotp).

Then we can test the accuracy with the following commands, in which `-m` or `--model` is specified with
the path of the folder of the sharded model instead of original model ID.

```bash
# Run distributed accuracy with 2 ranks of one node
cd ./distributed
```

#### 3.2.2.2 FP32:

- Command:
```bash
deepspeed --num_accelerators 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py --model <SHARD MODEL PATH> --dtype float32 --ipex --tasks <TASK_NAME>
```

- An example of a pre-sharded Llama model:
```bash
deepspeed --num_accelerators 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py --model ../saved_results/llama_local_shard --dtype float32 --ipex --tasks lambada_openai
```

#### 3.2.2.3 BF16:
- Command:
```bash
deepspeed --num_accelerators 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py --model <SHARD MODEL PATH> --dtype bfloat16 --ipex --tasks <TASK_NAME>
```

- An example of a pre-sharded Llama model:
```bash
deepspeed  --num_accelerators 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py --model ../saved_results/llama_local_shard --dtype bfloat16 --ipex --tasks lambada_openai
```

#### 3.2.2.4 Weight-only quantization INT8:

- Command:
```bash
deepspeed  --num_accelerators 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py --model <SHARD MODEL PATH> --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp --ipex --tasks <TASK_NAME>
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

- An example of a pre-sharded INT8 Llama model:
```bash
deepspeed --num_accelerators 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py --model ../saved_results/llama_local_shard --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp --ipex --tasks <TASK_NAME>
```

#### 3.2.2.5 Weight-only quantization INT4:

Please check [the instructions for WoQ INT4 performance benchmarking](#2215-weight-only-quantization-int4)
for the details on how to download or generate the INT4 quantized checkpoint files.
INT4 checkpoints cannot be pre-sharded, so in the command `--model` should be set as the path of the downloaded or generated checkpoint.

- Command:
```bash
deepspeed --num_accelerators 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py --model <INT4_CKPT_PATH> --ipex-weight-only-quantization --weight-dtype INT4 --lowp-mode BF16 --quant-with-amp --ipex --tasks <TASK_NAME>
```

- An example to run Llama-3.1-8B:
```bash
huggingface-cli download hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 --local-dir ./Llama-3.1-8B-GPTQ
deepspeed --num_accelerators 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py --model ./Llama-3.1-8B-GPTQ --ipex-weight-only-quantization --weight-dtype INT4 --lowp-mode BF16 --quant-with-amp --ipex --tasks lambada_openai
```

<br>

# 4. Performance Results

The performance results on AWS instances can be found [here](../../../../docs/tutorials/performance.md#llm-performance).

<br>

# 5. Miscellaneous Tips

The LLM inference methods introduced in this page can be well applied for AWS. We can just follow the above instructions and enjoy the boosted performance of LLM with Intel® Extension for PyTorch\* optimizations on the AWS instances.
