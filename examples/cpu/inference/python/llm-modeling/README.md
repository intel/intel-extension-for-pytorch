# 1. LLM Module Level Optimizations Overview

Intel® Extension for PyTorch* provides dedicated optimization for running Large Language Models (LLMs) faster, including technical points like paged attention, ROPE fusion, etc. 
To further provide optimized modules or functions to help build modelings, `ipex.llm` supports the following module/function level APIs:

```python
import intel_extension_for_pytorch as ipex
```

## Linear post-op fusions

```python
# using module init and forward
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

##  Attention related fusions

```python
# using module init and forward
ipex.llm.modules.RotaryEmbedding
ipex.llm.modules.RMSNorm
ipex.llm.modules.FastLayerNorm
ipex.llm.modules.VarlenAttention
ipex.llm.modules.PagedAttention
ipex.llm.modules.IndirectAccessKVCacheAttention

# using as functions
ipex.llm.functional.rotary_embedding
ipex.llm.functional.rms_norm
ipex.llm.functional.fast_layer_norm
ipex.llm.functional.indirect_access_kv_cache_attention
ipex.llm.functional.varlen_attention
ipex.llm.functional.add_layer_norm
ipex.llm.functional.add_rms_norm
ipex.llm.functional.silu_mul
ipex.llm.functional.gelu_mul
```

## Generation related fusions

```python
# using for optimizing huggingface generation APIs with prompt sharing
ipex.llm.generation.hf_beam_sample
ipex.llm.generation.hf_beam_search
ipex.llm.generation.hf_greedy_search
ipex.llm.generation.hf_sample
```

<br>

# 2. Showcases of ipex.llm optimized modules and functions based modeling

We provide optimized LLAMA, GPT-J and OPT modeling files on the basis of [huggingface modeling APIs](https://huggingface.co/docs/transformers/en/main_classes/model) and a entry script `run.py` as showcases that apply the optimized modules or functions from `ipex.llm`.

## Running example script

Please install the required packages via the following commands.

```bash
python -m pip install torch intel-extension-for-pytorch intel-openmp
conda install gperftools -y
# The example LLM modelings are showcased based on transformers v4.38.1
python -m pip install transformers==4.38.1 accelerate

# Set the environment variables for performance on Xeon
export LD_PRELOAD=$(bash ../../../llm/tools/get_libstdcpp_lib.sh):${CONDA_PREFIX}/lib/libiomp5.so:${CONDA_PREFIX}/lib/libtcmalloc.so:${LD_PRELOAD}
export KMP_BLOCKTIME=1
export KMP_TPAUSE=0
export KMP_FORKJOIN_BARRIER_PATTERN=dist,dist
export KMP_PLAIN_BARRIER_PATTERN=dist,dist
export KMP_REDUCTION_BARRIER_PATTERN=dist,dist

# Download the example prompt file
wget https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/prompt.json
```

The detail usage of `run.py` can be obtained by running

```bash
python run.py --help
```

Example commands are listed below:

```bash
# The following "OMP_NUM_THREADS" and "numactl" settings are based on the assumption that
# the target server has 56 physical cores per numa socket, and we benchmark with 1 socket.
# Please adjust the settings per your hardware.

# Running FP32 model
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py -m meta-llama/Llama-2-7b-hf --dtype float32

# Running BF16 model
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py -m meta-llama/Llama-2-7b-hf --dtype bfloat16
```

*Note:* You may need to log in your HuggingFace account to access the model files. Please refer to [HuggingFace login](https://huggingface.co/docs/huggingface_hub/quick-start#login).

<br>

# 3. Optimize your own LLM with ipex.llm

## Changes required in the modeling file

The changes required for applying `ipex.llm` optimizations for the customized LLMs are highly diverse based on their respective model architectures and implementations.
Generally speaking, the key steps would be:

1. Analyze the model to find out the parts that are suitable for utilizing the optimizations.

2. Re-write these parts, applying the optimized `ipex.llm` operators.

3. Some refactor of model architecture definition may be required to connect the original and optimized modules.

## Changes required in the inference entry script

Some key updates are required in the LLM inference entry script:

1. Optimization for linear modules and their fusions: realized by weight prepacking with `ipex.optimize()`.

```python
from intel_extension_for_pytorch.cpu._auto_kernel_selection import (
    _enable_tpp,
    _disable_tpp,
)

_disable_tpp()
if args.dtype == "bfloat16":
    _enable_tpp()
    model = ipex.optimize(model.eval(), dtype=torch.bfloat16, inplace=True)
else:
    model = ipex.optimize(
        model.eval(),
        dtype=torch.float32,
        inplace=True,
        auto_kernel_selection=True,
    )
```

*Note:* The example is for FP32/BF16 optimization.
Please refer to [Advanced Usage](#4-advanced-usage) part for weight only quantization enabling.

2. Optimizations for [the huggingface text generation API](https://huggingface.co/docs/transformers/en/main_classes/text_generation):

- Using `ipex.llm.generation` functions to get prompt sharing for first token acceleration when `num_beams > 1`.

```python
# Taking beam search as example here, please check complete code updates in run.py
hf_beam_search = ipex.llm.generation.hf_beam_search.__get__(model, model.__class__)
setattr(model, "beam_search", hf_beam_search)
```

- Using PyTorch jit to further reduce dispatch overhead for first token and next tokens acceleration.

```python
# Please create a dummy `sample_inputs` in advance 
# as the example input for jit.trace()
with torch.no_grad(), torch.cpu.amp.autocast(enabled=amp_enabled):
    trace_model = torch.jit.trace(
        model,
        example_kwarg_inputs=sample_inputs,
        strict=False,
        check_trace=False,
    )
    trace_model = torch.jit.freeze(trace_model)
    model = ipex._set_optimized_model_for_generation(
        model, optimized_model=trace_model
    )
```

Please read `run.py` and the example modeling files for detail of the changes.
The key parts are highlighted with comments.

<br>

# 4. Advanced usage

## How to apply weight only quantization int8

Intel® Extension for PyTorch* also provides weight only quantization for int8 precision optimization
(replace the part using `ipex.optimize()`, which is for fp32/bf16 optimization in above showcases).

```python
from intel_extension_for_pytorch.quantization import WoqWeightDtype
from intel_extension_for_pytorch.quantization import prepare, convert
weight_dtype = WoqWeightDtype.INT8 # weight dtype is int8
lowp_mode = ipex.quantization.WoqLowpMode.BF16 # lowest precision for computation
qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
    weight_dtype=weight_dtype,
    lowp_mode=lowp_mode,
    group_size= -1, # default is -1, can be further tuned in [32, 64, 128, 256, 512] (recommend) for better accuracy if needed
)
prepared_model = prepare(model, qconfig)
with torch.no_grad(), torch.cpu.amp.autocast(enabled=True): # we recommend to use quantization with AMP for better perf
    converted_model = convert(prepared_model).to(torch.bfloat16)
```

<br>

# 5. Miscellaneous Tips

- For LLMs, usually the query, key and value linear operations in Attention layer can be fused into one linear as kind of concat linear optimization. (e.g., [modeling_gpt_neox](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L175) from transformers)
 
- LLM generation tasks are based on the [assumption](https://huggingface.co/blog/how-to-generate) that the probability distribution of a word sequence can be decomposed into the product of conditional next word distributions.
Thus the model's computation of `lm_head` layer during the first token's generation can be reduced with using last token as its inputs (instead of using the full tokens from input prompt).
The showcases we provide contain such optimization (set with `lm_head_generation` flag). This is also optimized in LLM serving [text-generation-inference](https://github.com/huggingface/text-generation-inference/blob/main/server/text_generation_server/models/custom_modeling/flash_llama_modeling.py#L419).