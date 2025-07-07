import contextlib
import sys
import gc
import json
import math
import pathlib
import os
import time
from argparse import ArgumentParser
from pathlib import Path
import torch
import re

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
    FalconForCausalLM,
    T5ForConditionalGeneration,
    AutoTokenizer,
)


# supported models now
MODEL_CLASSES = {
    "gpt-j": (AutoModelForCausalLM, AutoTokenizer),
    "gpt-neox": (AutoModelForCausalLM, AutoTokenizer),
    "opt": (AutoModelForCausalLM, AutoTokenizer),
    "bloom": (AutoModelForCausalLM, AutoTokenizer),
    "llama3": (AutoModelForCausalLM, AutoTokenizer),
    "llama": (AutoModelForCausalLM, AutoTokenizer),
    "t5": (T5ForConditionalGeneration, AutoTokenizer),
    "falcon": (AutoModelForCausalLM, AutoTokenizer),
    "mistral": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}

# Set console encoding to UTF-8
if os.name == 'nt':
    os.system('chcp 65001')
    sys.stdout.reconfigure(encoding='utf-8')

# the Deepspeed team made these so it's super fast to load (~1 minute), rather than wait 10-20min loading time.
tp_presharded_models = ["microsoft/bloom-deepspeed-inference-int8", "microsoft/bloom-deepspeed-inference-fp16"]

do_profiling = os.environ.get("PROFILE", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]

t_start = time.time()

parser = ArgumentParser()

parser.add_argument(
    '-m', '--model-id',
    type=str,
    default='bigscience/bloom',
    help="the huggingface mdoel id"
)
parser.add_argument(
    '--sub-model-name',
    type=str,
    help="the sub model name for accuracy check"
)
parser.add_argument(
    '--device',
    type=str,
    choices=["cpu", "cuda", "xpu"],
    help="cpu or cuda or xpu, same as --cuda or not",
    default='xpu',
)
parser.add_argument(
    "--dtype", type=str, help="float16 or bfloat16 or int8", choices=["int8", "float16", "bfloat16", "float32"], default="float16"
)
parser.add_argument("--local_rank", required=False, type=int, help="used by dist launchers")
parser.add_argument("--batch_size", "--batch-size", default=1, type=int, help="batch size")
parser.add_argument("--num-iter", default=10, type=int, help="num iter")
parser.add_argument("--num-warmup", default=5, type=int, help="num warmup")
parser.add_argument("--benchmark", action="store_true", help="additionally run benchmark")
parser.add_argument("--cuda", action="store_true", help="run in cuda")
parser.add_argument("--num-beams", type=int, nargs="+")
parser.add_argument("--greedy", action="store_true")
parser.add_argument("--ki", action="store_true")
parser.add_argument('--max-new-tokens', default=32, type=int, nargs="+", help="output max new tokens")
parser.add_argument('--input-tokens', default=32, type=str, nargs="+")
parser.add_argument('--prompt', default=None, type=str)
parser.add_argument('--ipex', action='store_true', help="ipex is not enabled now")
parser.add_argument('--print-memory', action='store_true')
parser.add_argument("--token-latency", action="store_true")
parser.add_argument("--accuracy-only", action="store_true")
parser.add_argument(
    "--acc-tasks",
    default="lambada_standard",
    type=str,
    help="tasks list for accuracy validation, only enabled lambada_standard and lambada_standard at present",
)
parser.add_argument("--acc-iter", default=-1, type=int)
parser.add_argument("--disable_optimize_transformers", action="store_true")
parser.add_argument("--use-static-cache", default=False, action="store_true", help="use static kv cache")
parser.add_argument("--use-hf-code", default=True, action="store_false", help="use hf transformers code")
args = parser.parse_args()


# import extension
if args.ipex:
    import intel_extension_for_pytorch as ipex

if args.cuda or args.device == "cuda":
    args.cuda = True
    args.device == "cuda"

def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


local_rank = get_int_from_env(["LOCAL_RANK", "MPI_LOCALRANKID", "PALS_LOCAL_RANKID"], "0")
world_size = get_int_from_env(["WORLD_SIZE", "PMI_SIZE", "PALS_LOCAL_SIZE"], "1")
port = get_int_from_env(["MASTER_PORT"], 29500)
print(f"*** local_rank={local_rank} world_size={world_size} port={port}")

deepspeed.init_distributed(get_accelerator().communication_backend_name(), distributed_port=port)
x = torch.ones(
    [4, 1, 14336], device=torch.device(args.device, local_rank), dtype=torch.bfloat16
)
dist.all_reduce(x)

def print_rank0(*msg):
    if local_rank != 0:
        return
    print(*msg)

### Model loading and instantiating on GPUs
def get_repo_root(model_name_or_path):
    if os.path.isdir(model_name_or_path):
        return model_name_or_path
    # checks if online or not
    if is_offline_mode():
        print_rank0("Offline mode: forcing local_files_only=True")
    # download only on first process
    if local_rank == 0:
        snapshot_download(
            model_name_or_path,
            local_files_only=is_offline_mode(),
            cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
            ignore_patterns=["*.msgpack", "*.h5"],
            resume_download=True,
        )

    dist.barrier()

    return snapshot_download(
        model_name_or_path,
        local_files_only=is_offline_mode(),
        cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
        ignore_patterns=["*.msgpack", "*.h5"],
        resume_download=True,
    )


def get_checkpoint_files(model_name_or_path):
    cached_repo_dir = get_repo_root(model_name_or_path)

    # extensions: .bin | .pt | .safetensors
    # creates a list of paths from all downloaded files in cache dir
    file_list = list()
    pattern_sample = re.compile(r'(.*).(safetensors|bin|pt)$')
    for entry in Path(cached_repo_dir).rglob("*"):
        match = re.match(pattern=pattern_sample, string=str(entry))
        if match:
            file_list.append(str(entry))

    return file_list


def print_mem_usage(msg):
    get_accelerator().empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage(msg, force=True)


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

print_rank0(f"*** Loading the model {model_name}")
model_type = next((x for x in MODEL_CLASSES.keys() if x in model_name.lower()), 'auto')
model_class = MODEL_CLASSES[model_type]
tokenizer = model_class[1].from_pretrained(model_name, trust_remote_code=args.use_hf_code)
config = AutoConfig.from_pretrained(model_name, trust_remote_code=args.use_hf_code)
# Avoid deepspeed tp>=2 lm_head weight reload. Not affect the results.
config.tie_word_embeddings = False
#if not hasattr(config, "text_max_length") and args.prompt is None:
#    config.text_max_length = int(args.input_tokens) + int(args.max_new_tokens)
print_rank0("*** model config:", config)

if args.benchmark:
    print_mem_usage("pre-from-pretrained")

is_meta_support = model_type not in ["auto"]

# Construct model with fake meta tensors, later will be replaced during ds-inference ckpt load
with deepspeed.OnDevice(dtype=load_dtype, device="meta", enabled=is_meta_support):
    # Even inside the meta device context, from_pretrained still loads the
    # model to cpu instead of meta device. Use from_config instead to solve the issue for big models.
    # We add the instance type check here since some of the models haven't yet supported from_config.
    if model_class[0] == AutoModelForCausalLM and is_meta_support:
        model = model_class[0].from_config(config, torch_dtype=load_dtype)
    else:
        model = model_class[0].from_pretrained(model_name, config=config, low_cpu_mem_usage=True, torch_dtype=load_dtype, trust_remote_code=args.use_hf_code)

print_rank0("*** model after load", model)

if args.benchmark:
    print_mem_usage("post-from-pretrained")

model = model.eval()

### Deepspeed-Inference Loading

checkpoints_json = "checkpoints.json"


def write_checkpoints_json():
    checkpoint_files = get_checkpoint_files(model_name)
    if local_rank == 0:
        data = {"type": "BLOOM", "checkpoints": checkpoint_files, "version": 1.0}
        json.dump(data, open(checkpoints_json, "w"))


if args.ki:
    kwargs = dict(replace_with_kernel_inject=True)
else:
    kwargs = dict(replace_with_kernel_inject=False)

repo_root = get_repo_root(model_name)
if tp_presharded_mode and args.ki:
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
    checkpoint=checkpoints_json if is_meta_support else None,
    **kwargs,
)

print_rank0("*** model after init_inference", model)

if args.benchmark:
    print_mem_usage("post-ds-inference-init")

if args.benchmark:
    t_ready = time.time()

# to ipex
if args.ipex:
    if args.disable_optimize_transformers:
        model = ipex.optimize(model.eval().to("xpu"), dtype=infer_dtype)
    else:
        if "low_precision_checkpoint" in ipex.optimize_transformers.__code__.co_varnames:
            model = ipex.llm.optimize(model.eval(), dtype=infer_dtype, device="xpu", inplace=True)
        else:
            model = ipex.llm.optimize(model.eval().to("xpu"), dtype=infer_dtype)
    print_rank0("*** model after optimize_transformers", model)

# bypass assertion for beam4
if isinstance(model, deepspeed.InferenceEngine):
    model = model.module

# reinitialize some buffers that is empty caused by meta device loading
if args.disable_optimize_transformers:
    if model_type == "llama" and isinstance(model, LlamaForCausalLM):
        if hasattr(model.model, "causal_mask"):
            model.model.causal_mask = torch.triu(torch.ones_like(model.model.causal_mask), diagonal=1)

if args.num_beams is None:
    args.num_beams = 1 if args.greedy else 4

######################## run lm eval accuracy check ########################

def run_accuracy():
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import make_table

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    hfmodel = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=args.device,
    )

    if args.acc_iter == -1:
        results = evaluator.simple_evaluate(
            model=hfmodel,
            tasks=args.acc_tasks,
        )
    else:
        results = evaluator.simple_evaluate(
            model=hfmodel,
            tasks=args.acc_tasks,
            limit=args.acc_iter
        )

    print(make_table(results))


if args.accuracy_only:
    run_accuracy()
    sys.exit(0)

######################## run generation benchmark ########################
### Generate
current_path = pathlib.Path(__file__).parent.resolve()
with open(str(current_path) + '/prompt.json') as f:
    prompt_pool = json.load(f)

def run_generate(num_tokens, num_input_tokens, num_beams):
    print_rank0(f"*** Starting to generate {num_tokens} tokens for {num_input_tokens} tokens with num_beams={num_beams}")
    # input tokens
    input_sentences = []
    if args.prompt is not None:
        input_sentences.append(args.prompt)
    elif int(num_input_tokens) > 8192:
        input_sentences.append(prompt_pool[model_type]["8192"] * int(int(num_input_tokens) / 8192))
    elif num_input_tokens in prompt_pool[model_type]:
        input_sentences.append(prompt_pool[model_type][num_input_tokens])
    else:
        raise SystemExit('[ERROR] Plese use --prompt if want to use custom input.')

    if args.batch_size > len(input_sentences):
        # dynamically extend to support larger bs by repetition
        input_sentences *= math.ceil(args.batch_size / len(input_sentences))

    generate_kwargs = dict(max_new_tokens=num_tokens,
                           do_sample=False, num_beams=num_beams)
    if args.use_static_cache:
        generate_kwargs["cache_implementation"] = "static"
    if args.token_latency:
        generate_kwargs["token_latency"] = True

    inputs = input_sentences[: args.batch_size]

    def generate():
        """returns a list of zipped inputs, outputs and number of new tokens"""

        input_tokens = tokenizer.batch_encode_plus(inputs, return_tensors="pt", return_token_type_ids=False)
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(get_accelerator().current_device_name())

        outputs = model.generate(**input_tokens, **generate_kwargs)
        gen_ids = outputs[0] if args.token_latency else outputs

        input_tokens_lengths = [x.shape[0] for x in input_tokens.input_ids]
        output_tokens_lengths = [x.shape[0] for x in gen_ids]

        total_new_tokens = [o - i if model.config.model_type != 't5' else o for i, o in zip(input_tokens_lengths, output_tokens_lengths)]
        gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        return zip(inputs, gen_text, total_new_tokens), outputs

    # Accuracy check, take the ref_prompt as reference for check
    f1 = open(os.path.join(os.path.dirname(__file__), "ref_prompt.json"))
    prompt_json = json.load(f1)
    f1.close()
    ref_prompt = None
    ref_prompt_cuda = None
    token_support = [(32, 32), (1024, 128)]
    if (int(num_input_tokens), num_tokens) in token_support and args.sub_model_name in prompt_json:
        ref_prompt = prompt_json[args.sub_model_name][f"{num_input_tokens}-{num_tokens}"][f"{num_beams}"]
        try:
            ref_prompt_cuda = prompt_json[args.sub_model_name][f"{num_input_tokens}-{num_tokens}"][f"cuda-result: {num_beams}"]
        except Exception:
            pass
    acc_pass = 0

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
        print_rank0("*** Running benchmark")
        total_time = 0.0
        cycles = args.num_iter
        warmup = args.num_warmup
        total_list = []
        with torch.inference_mode():
            # latency
            for i in range(cycles):
                enable_profile = do_profiling and i == cycles - 1
                with (
                    contextlib.nullcontext(None) if not enable_profile else
                    torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.XPU],
                        record_shapes=True,
                    )
                ) as prof:
                    t0 = time.time()
                    gen_ids, outputs = generate()
                    if args.cuda:
                        torch.cuda.synchronize()
                    t1 = time.time()

                if enable_profile:
                    with open("./profile_{}_{}.log".format(num_beams, local_rank), "w") as f:
                        f.write(prof.key_averages().table(sort_by="self_xpu_time_total"))
                    # Cannot sort by id when using kineto
                    #with open('./profile_{}_{}_id.log'.format(num_beams, local_rank), "w") as f:
                    #    f.write(prof.table(sort_by="id", row_limit=-1))
                    with open("./profile_{}_{}_detail.log".format(num_beams, local_rank), "w") as f:
                        f.write(prof.key_averages(group_by_input_shape=True).table())
                    #prof.export_chrome_trace(f"./trace_{local_rank}.json")

                gen_ids = list(gen_ids)
                for p, o, _ in gen_ids:
                    print_rank0(f"*** In: {p}\n*** Out: {o[len(p):]}")
                print_rank0("Iteration: %d, Time: %.6f sec" % (i, t1 - t0))
                print_mem_usage("post-iteration-%d" % i)
                if i >= warmup and i < cycles - int(do_profiling):
                    total_time += (t1 - t0)
                    if args.token_latency:
                        total_list.append(outputs[1])
                if ref_prompt is not None and ref_prompt in gen_ids[0][1:]:
                    acc_pass += 1
                elif ref_prompt_cuda is not None and ref_prompt_cuda in gen_ids[0][1:]:
                    acc_pass += 1

        latency = total_time / (cycles - warmup - int(do_profiling))
        print_rank0("\n", "-" * 10, "Summary:", "-" * 10)
        print_rank0("Inference latency: %.5f sec." % latency)
        if args.token_latency:
            import numpy as np
            from itertools import chain
            #if local_rank == 0:
            #    with open("raw_latency.json", "w") as f:
            #        json.dump(total_list, f)
            first_latency = np.mean([x[0] for x in total_list])
            average_2n = list(chain(*[x[1:] for x in total_list]))
            average_2n.sort()
            average_2n_latency = np.mean(average_2n)
            p90_latency = average_2n[int(len(average_2n) * 0.9)]
            p99_latency = average_2n[int(len(average_2n) * 0.99)]
            print_rank0("First token average latency: %.5f sec." % first_latency)
            print_rank0("Average 2... latency: %.5f sec." % average_2n_latency)
            print_rank0("P90 2... latency: %.5f sec." % p90_latency)
            print_rank0("P99 2... latency: %.5f sec." % p99_latency)
        print_rank0(f"Generate args: batch_size={args.batch_size}, max_new_tokens={num_tokens}, input_tokens={num_input_tokens}, num_beams={num_beams}")

        if ref_prompt is None:
            print_rank0("""Accuracy check skip""")
        elif acc_pass == args.num_iter:
            print_rank0("""Accuracy check pass""")
        else:
            print_rank0(f"""Accuracy check fail, the wrong iteration number is: {args.num_iter - acc_pass}""")


def to_list(obj):
    if not isinstance(obj, list):
        return [obj]
    else:
        return obj


for o, i, g in zip(to_list(args.max_new_tokens), to_list(args.input_tokens), to_list(args.num_beams)):
    run_generate(o, i, g)
