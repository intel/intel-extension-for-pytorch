import gc
import json
import math
import pathlib
import os
import time
from argparse import ArgumentParser
from pathlib import Path
import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
import deepspeed.comm as dist
from huggingface_hub import snapshot_download
from transformers.models.bloom.modeling_bloom import BloomBlock as BloomBlock
from transformers.utils import is_offline_mode
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
)


# supported models now
MODEL_CLASSES = {
    "gpt-j": (AutoModelForCausalLM, AutoTokenizer),
    "gpt-neox": (AutoModelForCausalLM, AutoTokenizer),
    "llama": (AutoModelForCausalLM, LlamaTokenizer),
    "opt": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}

# the Deepspeed team made these so it's super fast to load (~1 minute), rather than wait 10-20min loading time.
tp_presharded_models = [
    "microsoft/bloom-deepspeed-inference-int8",
    "microsoft/bloom-deepspeed-inference-fp16",
]

t_start = time.time()

parser = ArgumentParser()

parser.add_argument(
    "-m",
    "--model-id",
    type=str,
    default="bigscience/bloom",
    help="the huggingface mdoel id",
)
parser.add_argument(
    "--device",
    type=str,
    choices=["cpu"],
    help="cpu",
    default="cpu",
)
parser.add_argument(
    "--dtype",
    type=str,
    help="float16 or bfloat16 or int8",
    choices=["int8", "float16", "bfloat16", "float32"],
    default="float16",
)
parser.add_argument(
    "--local_rank", required=False, type=int, help="used by dist launchers"
)
parser.add_argument(
    "--batch-size", "--batch-size", default=1, type=int, help="batch size"
)
parser.add_argument("--num-iter", default=50, type=int, help="num iter")
parser.add_argument("--num-warmup", default=5, type=int, help="num warmup")
parser.add_argument(
    "--benchmark", action="store_true", help="additionally run benchmark"
)
parser.add_argument("--greedy", action="store_true")
parser.add_argument("--profile", action="store_true")
parser.add_argument("--ki", action="store_true")
parser.add_argument(
    "--max-new-tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument("--input-tokens", default="32", type=str)
parser.add_argument("--prompt", default=None, type=str)
parser.add_argument("--ipex", action="store_true", help="ipex is not enabled now")
parser.add_argument(
    "--ipex-weight-only-quantization",
    action="store_true",
    help="enable ipex WOQ optimization (int8 mix fp32)",
)
parser.add_argument("--jit", action="store_true")
parser.add_argument("--print-memory", action="store_true")
parser.add_argument("--token-latency", action="store_true")
args = parser.parse_args()


num_tokens = args.max_new_tokens
# import extension
if args.ipex:
    import intel_extension_for_pytorch as ipex

    try:
        ipex._C.disable_jit_linear_repack()
    except Exception:
        pass


def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


local_rank = get_int_from_env(["LOCAL_RANK", "MPI_LOCALRANKID"], "0")
world_size = get_int_from_env(["WORLD_SIZE", "PMI_SIZE"], "1")

deepspeed.init_distributed(get_accelerator().communication_backend_name())


def print_rank0(*msg):
    if local_rank != 0:
        return
    print(*msg)


### Model loading and instantiating on GPUs
def get_repo_root(model_name_or_path):
    local_prefix = ("/", "./", "../")
    if model_name_or_path.startswith(local_prefix):
        return model_name_or_path
    # checks if online or not
    if is_offline_mode():
        print_rank0("Offline mode: forcing local_files_only=True")
    # download only on first process
    allow_patterns = ["*.bin", "*.model", "*.json", "*.txt", "*.py", "*LICENSE"]
    if local_rank == 0:
        snapshot_download(
            model_name_or_path,
            local_files_only=is_offline_mode(),
            cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
            allow_patterns=allow_patterns,
            # ignore_patterns=["*.safetensors"],
        )

    dist.barrier()

    return snapshot_download(
        model_name_or_path,
        local_files_only=is_offline_mode(),
        cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
        allow_patterns=allow_patterns,
        # ignore_patterns=["*.safetensors"],
    )


def get_checkpoint_files(model_name_or_path):
    cached_repo_dir = get_repo_root(model_name_or_path)

    # extensions: .bin | .pt
    # creates a list of paths from all downloaded files in cache dir
    file_list = [
        str(entry)
        for entry in Path(cached_repo_dir).rglob("*.[bp][it][n]")
        if entry.is_file()
    ]
    return file_list


model_name = args.model_id
if args.dtype == "float16":
    load_dtype = torch.half
    infer_dtype = torch.half
elif args.dtype == "bfloat16":
    load_dtype = torch.bfloat16
    infer_dtype = torch.bfloat16
elif args.dtype == "int8":
    load_dtype = torch.half
    infer_dtype = torch.int8
elif args.dtype == "float32":
    load_dtype = torch.float32
    infer_dtype = torch.float32

tp_presharded_mode = True if model_name in tp_presharded_models else False

# print(get_checkpoint_files(model_name))

print_rank0(f"*** Loading the model {model_name}")
model_type = next((x for x in MODEL_CLASSES.keys() if x in model_name.lower()), "auto")
model_class = MODEL_CLASSES[model_type]
tokenizer = model_class[1].from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name, torchscript=args.jit)
if not hasattr(config, "text_max_length") and args.prompt is None:
    config.text_max_length = int(args.input_tokens) + int(args.max_new_tokens)

# XXX: can't automatically derive dtype via config's `from_pretrained`
# dtype = torch.bfloat16 if model_name in ["bigscience/bloom", "bigscience/bigscience-small-testing"] else torch.float16


# use one of these args to `init_inference`
# 1. injection_policy is the slower version, but it's plain pytorch so it'll always work
# 2. replace_with_kernel_inject is the faster one (fast fused kernels)
kernel_inject = args.ki

if args.benchmark:
    get_accelerator().empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage("pre-from-pretrained", force=True)

# Construct model with fake meta tensors, later will be replaced during ds-inference ckpt load
with deepspeed.OnDevice(dtype=load_dtype, device="meta"):
    # Even inside the meta device context, from_pretrained still loads the
    # model to cpu instead of meta device. Use from_config instead to solve the issue for big models.
    # We add the instance type check here since some of the models haven't yet supported from_config.
    if model_class[0] == AutoModelForCausalLM:
        model = model_class[0].from_config(config).to(load_dtype)
    else:
        model = model_class[0].from_pretrained(
            model_name, config=config, low_cpu_mem_usage=True, torch_dtype=load_dtype
        )

if args.benchmark:
    deepspeed.runtime.utils.see_memory_usage("post-from-pretrained", force=True)

model = model.eval()
model = model.to(memory_format=torch.channels_last)

if args.benchmark:
    get_accelerator().empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage("post-init-ds-zero-init", force=True)

### Deepspeed-Inference Loading

checkpoints_json = "checkpoints.json"


def write_checkpoints_json():
    checkpoint_files = get_checkpoint_files(model_name)
    if local_rank == 0:
        # model.config.model_type.upper()
        data = {"type": "BLOOM", "checkpoints": checkpoint_files, "version": 1.0}
        json.dump(data, open(checkpoints_json, "w"))


if args.benchmark:
    get_accelerator().empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage("pre-ds-inference-init", force=True)

if kernel_inject:
    kwargs = dict(replace_with_kernel_inject=True)
else:
    kwargs = dict(replace_with_kernel_inject=False)

repo_root = get_repo_root(model_name)
if tp_presharded_mode:
    # tp presharded repos come with their own checkpoints config file
    checkpoints_json = os.path.join(repo_root, "ds_inference_config.json")
else:
    # for normal bloom repo we need to write the checkpoints config file
    write_checkpoints_json()
    dist.barrier()

model = deepspeed.init_inference(
    model,
    mp_size=world_size,
    base_dir=repo_root,
    dtype=infer_dtype,
    checkpoint=checkpoints_json,
    **kwargs,
)

if args.benchmark:
    get_accelerator().empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage("post-ds-inference-init", force=True)


model = model.module

if args.benchmark:
    t_ready = time.time()

# to ipex
if args.ipex:
    ipex_woq_enabled = args.ipex_weight_only_quantization and args.dtype == "float32"
    model = ipex._optimize_transformers(
        model.eval(),
        dtype=infer_dtype if not ipex_woq_enabled else torch.int8,
        inplace=True,
    )
    if ipex_woq_enabled:
        from intel_extension_for_pytorch.quantization import convert, prepare

        lowp_mode = ipex.quantization.WoqLowpMode.BF16
        qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
            lowp_mode=lowp_mode
        )
        model = prepare(model.eval(), qconfig, inplace=True, bn_folding=False)
        with torch.no_grad():
            model = convert(model.eval(), inplace=True).eval()

### Generate


print_rank0(f"*** Starting to generate {num_tokens} tokens with bs={args.batch_size}")

# input tokens
input_sentences = []
current_path = pathlib.Path(__file__).parent.resolve()
with open(str(current_path) + "/prompt.json") as f:
    prompt_pool = json.load(f)
if args.prompt is not None:
    input_sentences.append(args.prompt)
elif model_type == "auto":
    raise SystemExit(
        "[ERROR] model prompt is not supported, please use --prompt for this model: "
        + args.model_id
    )
elif int(args.input_tokens) > 8192:
    input_sentences.append(prompt_pool[model_type]["8192"] * int(int(args.input_tokens) / 8192))
elif args.input_tokens in prompt_pool[model_type]:
    input_sentences.append(prompt_pool[model_type][args.input_tokens])
else:
    raise SystemExit("[ERROR] Plese use --prompt if want to use custom input.")


if args.batch_size > len(input_sentences):
    # dynamically extend to support larger bs by repetition
    input_sentences *= math.ceil(args.batch_size / len(input_sentences))
num_beams = 1 if args.greedy else 4
generate_kwargs = dict(max_new_tokens=num_tokens, do_sample=False, num_beams=num_beams)
if args.token_latency:
    if not hasattr(model.config, "token_latency"):
        model.config.token_latency = True

if args.jit:
    torch._C._jit_set_texpr_fuser_enabled(False)
    past_key_values = None
    import re

    if re.search("GPTJ", model.config.architectures[0]):
        beam_idx_tmp = torch.zeros(
            (2048, int(args.batch_size * num_beams)), dtype=torch.long
        ).contiguous()
        past_key_values = tuple(
            [
                (
                    torch.zeros([1, 1, 1, 1]).contiguous(),
                    torch.zeros([1, 1, 1, 1]).contiguous(),
                    beam_idx_tmp,
                    torch.zeros(1, dtype=torch.long).contiguous(),
                )
                for i in range(model.config.n_layer)
            ]
        )
    elif re.search("llama", model.config.architectures[0], re.IGNORECASE):
        beam_idx_tmp = torch.zeros(
            (2048, int(args.batch_size * num_beams)), dtype=torch.long
        ).contiguous()
        past_key_values = tuple(
            [
                (
                    torch.zeros([1, 32, 1, 128]).contiguous(),
                    torch.zeros([1, 32, 1, 128]).contiguous(),
                    beam_idx_tmp,
                    torch.zeros(1, dtype=torch.long).contiguous(),
                )
                for i in range(model.config.num_hidden_layers)
            ]
        )
    elif re.search("gptneox", model.config.architectures[0], re.IGNORECASE):
        beam_idx_tmp = torch.zeros(
            (2048, int(args.batch_size * num_beams)), dtype=torch.long
        ).contiguous()
        past_key_values = tuple(
            [
                (
                    torch.zeros([1, 1, 1, 1]).contiguous(),
                    torch.zeros([1, 1, 1, 1]).contiguous(),
                    beam_idx_tmp,
                    torch.zeros(1, dtype=torch.long).contiguous(),
                )
                for i in range(model.config.num_hidden_layers)
            ]
        )
    elif re.search("opt", model.config.architectures[0], re.IGNORECASE):
        beam_idx_tmp = torch.zeros(
            (2048, int(args.batch_size * num_beams)), dtype=torch.long
        ).contiguous()
        past_key_values = tuple(
            [
                (
                    torch.zeros([1, 1, 1, 1]).contiguous(),
                    torch.zeros([1, 1, 1, 1]).contiguous(),
                    beam_idx_tmp,
                    torch.zeros(1, dtype=torch.long).contiguous(),
                )
                for i in range(model.config.num_hidden_layers)
            ]
        )
    else:
        print("does not support jit yet on your model, please re-run without jit")
        exit(0)
    example_inputs = None
    input_ids = torch.ones(32).to(torch.long)
    attention_mask = torch.ones(len(input_ids))
    position_ids = torch.arange(len(input_ids))
    if re.search("opt", model.config.architectures[0], re.IGNORECASE):
        example_inputs = {
            "input_ids": input_ids.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
            "past_key_values": past_key_values,
        }
    else:
        example_inputs = {
            "input_ids": input_ids.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
            "position_ids": position_ids.unsqueeze(0),
            "past_key_values": past_key_values,
        }

    with torch.inference_mode(), torch.no_grad(), torch.autocast(
        device_type=args.device,
        enabled=infer_dtype is torch.bfloat16,
        dtype=infer_dtype if infer_dtype is torch.bfloat16 else None,
    ):
        trace_model = torch.jit.trace(
            model, example_kwarg_inputs=example_inputs, strict=False, check_trace=False
        )
        trace_model = torch.jit.freeze(trace_model)
        setattr(model, "trace_graph", trace_model)
print_rank0(f"Generate args {generate_kwargs}")


inputs = input_sentences[: args.batch_size]
input_size = tokenizer.batch_encode_plus(inputs, return_tensors="pt").input_ids.size(
    dim=1
)
print("*** Prompt size: ", input_size)


def generate():
    """returns a list of zipped inputs, outputs and number of new tokens"""

    input_tokens = tokenizer.batch_encode_plus(inputs, return_tensors="pt")
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(
                get_accelerator().current_device_name()
            )

    outputs = model.generate(**input_tokens, **generate_kwargs)
    gen_ids = outputs[0] if args.token_latency else outputs

    input_tokens_lengths = [x.shape[0] for x in input_tokens.input_ids]
    output_tokens_lengths = [x.shape[0] for x in gen_ids]

    total_new_tokens = [
        o - i if model.config.model_type != "t5" else o
        for i, o in zip(input_tokens_lengths, output_tokens_lengths)
    ]
    gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

    return zip(inputs, gen_text, total_new_tokens), outputs


def trace_handler(prof):
    print(prof.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=-1))

# warmup is a must if measuring speed as it's when all the optimizations are performed
# e.g. on 8x80 a100 the first pass of 100 tokens takes 23sec, and the next one is 4secs
if not args.benchmark:
    print_rank0("*** Running generate warmup")
    generated, _ = generate()

    print_rank0("*** Running generate")
    t_generate_start = time.time()
    generated, _ = generate()
    t_generate_span = time.time() - t_generate_start
    for i, o, _ in generated:
        print_rank0(f"{'-'*60}\nin={i}\nout={o}\n")

### Benchmark
# benchmark it!
else:
    get_accelerator().empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage("end-of-run", force=True)

    print_rank0("*** Running benchmark")
    total_time = 0.0
    cycles = args.num_iter
    warmup = args.num_warmup
    total_list = []
    if args.profile:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=3,
                active=1),
            on_trace_ready=trace_handler
        ) as prof:
            for i in range(5):
                gen_ids, outputs = generate()
                prof.step()
    # latency
    for i in range(cycles):
        t0 = time.time()
        gen_ids, outputs = generate()
        t1 = time.time()
        gen_ids = list(gen_ids)
        print_rank0(gen_ids[0][1:])
        print_rank0("Iteration: %d, Time: %.6f sec" % (i, t1 - t0))
        if i >= warmup:
            total_time += t1 - t0
            if args.token_latency:
                total_list.append(outputs[1])

    latency = total_time / (cycles - warmup)
    print_rank0("\n", "-" * 10, "Summary:", "-" * 10)
    print_rank0("Inference latency: %.3f sec." % latency)
    if args.token_latency:
        import numpy as np
        from itertools import chain

        first_latency = np.mean([x[0] for x in total_list])
        average_2n = list(chain(*[x[1:] for x in total_list]))
        average_2n.sort()
        average_2n_latency = np.mean(average_2n)
        p90_latency = average_2n[int(len(average_2n) * 0.9)]
        p99_latency = average_2n[int(len(average_2n) * 0.99)]
        print_rank0("First token average latency: %.3f sec." % first_latency)
        print_rank0("Average 2... latency: %.3f sec." % average_2n_latency)
        print_rank0("P90 2... latency: %.3f sec." % p90_latency)
        print_rank0("P99 2... latency: %.3f sec." % p99_latency)
