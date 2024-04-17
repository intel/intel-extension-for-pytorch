# 1. LLM Optimization Overview

ipex.llm provides dedicated optimization for running Large Language Models (LLM) faster, including technical points like paged attention, ROPE fusion, etc. 
To further provide optimized modules or functions to help build modelings, ipex supports the following module/function level APIs:

```
import intel_extension_for_pytorch as ipex
```

### linear post-op fusions
```
#using module init and forward
ipex.llm.modules.linearMul
ipex.llm.modules.linearGelu
ipex.llm.modules.linearNewGelu
ipex.llm.modules.linearAdd
ipex.llm.modules.linearAddAdd
ipex.llm.modules.linearSilu
ipex.llm.modules.linearSiluMul
ipex.llm.modules.linear2SiluMul
ipex.llm.modules.linearRelu
```

###  Attention related fusions 
```
#using module init and forward
ipex.llm.modules.RotaryEmbedding
ipex.llm.modules.RMSNorm
ipex.llm.modules.FastLayerNorm
ipex.llm.modules.VarlenAttention
ipex.llm.modules.PagedAttention
ipex.llm.modules.IndirectAccessKVCacheAttention

#using as functions
ipex.llm.functional.rotary_embedding
ipex.llm.functional.rms_norm
ipex.llm.functional.fast_layer_norm
ipex.llm.functional.indirect_access_kv_cache_attention
ipex.llm.functional.varlen_attention

```

### Generation related fusions 
```
ipex.llm.generation.hf_beam_search
ipex.llm.generation.hf_greedy_search
ipex.llm.generation.hf_sample
```

<br>

# 2. Show cases of ipex.llm optimized modules and functions based modeling
We provide LLAMA, GPTJ and OPT modeling as show cases that apply the optimized modules or functions from ipex.llm layers.

| MODEL FAMILY | MODEL NAME (Huggingface hub) |
|:---:|:---:|
|LLAMA| "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", etc. |
|GPT-J| "EleutherAI/gpt-j-6b", etc. |
|OPT| "facebook/opt-30b", "facebook/opt-1.3b", etc. |

## How To Run LLM with ipex.llm

**ipex.llm provides a single script to facilitate running generation tasks as below:**
Note that please setup ENV according to the ../llm/README.md

```
python run.py --help # for more detailed usages
```

| Key args of run.py | Notes |
|---|---|
| model name | use "-m MODEL_NAME" to choose models to run |
| generation | default: beam search (beam size = 4), "--greedy" for greedy search |
| input tokens | default: 32, provide fixed sizes for input prompt size, use "--input-tokens" for [32, 64, 128, 256, 512, 1024, 2016, 2017, 2048, 4096, 8192]; if "--input-tokens" is not used, use "--prompt" to choose other strings as inputs|
| output tokens | default: 32, use "--max-new-tokens" to choose any other size |
| batch size |  default: 1, use "--batch-size" to choose any other size |
| generation iterations |  use "--num-iter" and "--num-warmup" to control the repeated iterations of generation, default: 100-iter/10-warmup |
| ipex prepack | apply ipex weight prepack optimization by "--use-ipex-optimize"|
| profiling | enable pytorch profiling by " --profile"|

*Note:* You may need to log in your HuggingFace account to access the model files. Please refer to [HuggingFace login](https://huggingface.co/docs/huggingface_hub/quick-start#login).


## Run commands

```bash
# The following "OMP_NUM_THREADS" and "numactl" settings are based on the assumption that
# the target server has 56 physical cores per numa socket, and we benchmark with 1 socket.
# Please adjust the settings per your hardware.

# Running FP32 model
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  -m meta-llama/Llama-2-7b-hf --dtype float32  --use-ipex-optimize

# Running BF16 model
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  -m meta-llama/Llama-2-7b-hf --dtype bfloat16 --use-ipex-optimize

```
