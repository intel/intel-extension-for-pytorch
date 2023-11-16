import torch
import time
import json
import pathlib
import argparse

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
)


# supported models
MODEL_CLASSES = {
    "gpt-j": (AutoModelForCausalLM, AutoTokenizer),
    "gpt-neox": (AutoModelForCausalLM, AutoTokenizer),
    "llama": (AutoModelForCausalLM, LlamaTokenizer),
    "opt": (AutoModelForCausalLM, AutoTokenizer),
    "falcon": (AutoModelForCausalLM, AutoTokenizer),
    "bloom": (AutoModelForCausalLM, AutoTokenizer),
    "codegen": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}

# args
parser = argparse.ArgumentParser("Generation script (fp32/bf16 path)", add_help=False)
parser.add_argument(
    "-m",
    "--model-id",
    type=str,
    default="EleutherAI/gpt-j-6B",
    help="the huggingface mdoel id",
)
parser.add_argument(
    "--dtype",
    type=str,
    choices=["float32", "bfloat16"],
    default="bfloat16",
    help="bfloat16, float32",
)
parser.add_argument(
    "--input-tokens",
    default="32",
    type=str,
    help="input tokens length if needed from prompt.json",
)
parser.add_argument(
    "--max-new-tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument(
    "--prompt", default=None, type=str, help="input prompt for self-defined if needed"
)
parser.add_argument(
    "--config-file", default=None, type=str, help="specific configuration file"
)
parser.add_argument("--greedy", action="store_true")
parser.add_argument("--ipex", action="store_true")
parser.add_argument("--deployment-mode", action="store_true")
parser.add_argument("--torch-compile", action="store_true")
parser.add_argument("--backend", default="ipex", type=str, help="backend of torch.compile")
parser.add_argument("--profile", action="store_true")
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--num-iter", default=100, type=int, help="num iter")
parser.add_argument("--num-warmup", default=10, type=int, help="num warmup")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
parser.add_argument(
    "--token-latency", action="store_true", help="get token latency breakdown"
)
args = parser.parse_args()
print(args)

# import ipex
if args.ipex:
    import intel_extension_for_pytorch as ipex

    torch._C._jit_set_texpr_fuser_enabled(False)
    try:
        ipex._C.disable_jit_linear_repack()
    except Exception:
        pass

# dtype
amp_enabled = True if args.dtype != "float32" else False
amp_dtype = getattr(torch, args.dtype)

# load model
model_type = next(
    (x for x in MODEL_CLASSES.keys() if x in args.model_id.lower()), "auto"
)
model_class = MODEL_CLASSES[model_type]
if args.config_file is None:
    config = AutoConfig.from_pretrained(
        args.model_id, torchscript=args.deployment_mode, trust_remote_code=True
    )
else:
    config = AutoConfig.from_pretrained(
        args.config_file, torchscript=args.deployment_mode, trust_remote_code=True
    )
if not hasattr(config, "text_max_length") and args.prompt is None:
    config.text_max_length = int(args.input_tokens) + int(args.max_new_tokens)

if not hasattr(config, "lm_head_generation"):
    config.lm_head_generation = True

model = model_class[0].from_pretrained(
    args.model_id,
    torch_dtype=amp_dtype,
    config=config,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
tokenizer = model_class[1].from_pretrained(args.model_id, trust_remote_code=True)
model = model.eval()
model = model.to(memory_format=torch.channels_last)

# to ipex
if args.ipex:
    model = ipex.optimize_transformers(
        model.eval(),
        dtype=amp_dtype,
        inplace=True,
        deployment_mode=args.deployment_mode,
    )

if args.torch_compile:
    if args.deployment_mode:
        raise SystemExit("[ERROR] deployment_mode cannot co-work with torch.compile, please set deployment_mode to False if want to use torch.compile.")
    model.generate = torch.compile(model.generate, backend=args.backend)

num_beams = 1 if args.greedy else 4
# generate args
generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=num_beams)


def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))


if args.benchmark:
    if args.token_latency:
        if not hasattr(model.config, "token_latency"):
            model.config.token_latency = True
    # input prompt
    current_path = pathlib.Path(__file__).parent.resolve()
    with open(str(current_path) + "/prompt.json") as f:
        prompt_pool = json.load(f)
    if args.prompt is not None:
        prompt = args.prompt
    elif model_type == "auto":
        raise SystemExit(
            "[ERROR] model prompt is not supported, please use --prompt for this model: "
            + args.model_id
        )
    elif int(args.input_tokens) > 8192:
        prompt = prompt_pool[model_type]["8192"] * int(int(args.input_tokens) / 8192)
    elif args.input_tokens in prompt_pool[model_type]:
        prompt = prompt_pool[model_type][args.input_tokens]
    else:
        raise SystemExit("[ERROR] Plese use --prompt if want to use custom input.")

    input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
    print("---- Prompt size:", input_size)

    # start
    total_time = 0.0
    num_iter = args.num_iter
    num_warmup = args.num_warmup
    prompt = [prompt] * args.batch_size
    total_list = []
    with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
        enabled=amp_enabled
    ):
        if args.profile:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                schedule=torch.profiler.schedule(wait=1, warmup=3, active=1),
                on_trace_ready=trace_handler,
            ) as prof:
                for i in range(5):
                    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                    output = model.generate(
                        input_ids, max_new_tokens=args.max_new_tokens, **generate_kwargs
                    )
                    prof.step()
        for i in range(num_iter):
            tic = time.time()
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            output = model.generate(
                input_ids, max_new_tokens=args.max_new_tokens, **generate_kwargs
            )
            gen_ids = output[0] if args.token_latency else output
            gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            toc = time.time()
            input_tokens_lengths = [x.shape[0] for x in input_ids]
            output_tokens_lengths = [x.shape[0] for x in gen_ids]
            total_new_tokens = [
                o - i if model.config.model_type != "t5" else o
                for i, o in zip(input_tokens_lengths, output_tokens_lengths)
            ]
            print(gen_text, total_new_tokens, flush=True)
            print("Iteration: %d, Time: %.6f sec" % (i, toc - tic), flush=True)
            if i >= num_warmup:
                total_time += toc - tic
                if args.token_latency:
                    total_list.append(output[1])

    print("\n", "-" * 10, "Summary:", "-" * 10)
    latency = total_time / (num_iter - num_warmup)
    print("Inference latency: %.3f sec." % latency)

    if args.token_latency:
        import numpy as np
        from itertools import chain

        first_latency = np.mean([x[0] for x in total_list])
        average_2n = list(chain(*[x[1:] for x in total_list]))
        average_2n.sort()
        average_2n_latency = np.mean(average_2n)
        p90_latency = average_2n[int(len(average_2n) * 0.9)]
        p99_latency = average_2n[int(len(average_2n) * 0.99)]
        print("First token average latency: %.3f sec." % first_latency)
        print("Average 2... latency: %.3f sec." % average_2n_latency)
        print("P90 2... latency: %.3f sec." % p90_latency)
        print("P99 2... latency: %.3f sec." % p99_latency)
