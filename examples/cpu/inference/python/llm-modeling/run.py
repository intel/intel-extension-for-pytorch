import torch
import time
import json
import pathlib
import argparse
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForCausalLM,
)
import transformers
import intel_extension_for_pytorch as ipex
from modeling_llama import IPEXLlamaForCausalLM

transformers.models.llama.modeling_llama.LlamaForCausalLM = IPEXLlamaForCausalLM
from modeling_gptj import IPEXGPTJForCausalLM

transformers.models.gptj.modeling_gptj.GPTJForCausalLM = IPEXGPTJForCausalLM
from modeling_opt import IPEXOPTForCausalLM

transformers.models.opt.modeling_opt.OPTForCausalLM = IPEXOPTForCausalLM

MODEL_CLASSES = {
    "gpt-j": (AutoModelForCausalLM, AutoTokenizer),
    "llama": (AutoModelForCausalLM, LlamaTokenizer),
    "opt": (AutoModelForCausalLM, AutoTokenizer),
}

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
parser.add_argument("--greedy", action="store_true")
parser.add_argument("--profile", action="store_true")
parser.add_argument("--use-ipex-optimize", action="store_true")
parser.add_argument("--num-iter", default=100, type=int, help="num iter")
parser.add_argument("--num-warmup", default=10, type=int, help="num warmup")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
args = parser.parse_args()
print(args)

model_type = next(
    (x for x in MODEL_CLASSES.keys() if x in args.model_id.lower()), "auto"
)
model_class = MODEL_CLASSES[model_type]

amp_enabled = True if args.dtype != "float32" else False
amp_dtype = getattr(torch, args.dtype)

model = model_class[0].from_pretrained(
    args.model_id,
    torch_dtype=amp_dtype,
    low_cpu_mem_usage=True,
    attn_implementation="eager",
)
tokenizer = model_class[1].from_pretrained(args.model_id, trust_remote_code=True)

num_beams = 1 if args.greedy else 4
generate_kwargs = dict(
    do_sample=False,
    temperature=0.9,
    num_beams=num_beams,
    max_new_tokens=args.max_new_tokens,
    min_new_tokens=args.max_new_tokens,
)

model = model.eval()

if args.use_ipex_optimize:
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


def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))


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
                output = model.generate(input_ids, **generate_kwargs)
                prof.step()

    for i in range(num_iter):
        tic = time.time()
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        output = model.generate(input_ids, **generate_kwargs)
        gen_ids = output
        gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        toc = time.time()
        input_tokens_lengths = [x.shape[0] for x in input_ids]
        output_tokens_lengths = [x.shape[0] for x in gen_ids]
        total_new_tokens = [
            o - i for i, o in zip(input_tokens_lengths, output_tokens_lengths)
        ]
        print(gen_text, total_new_tokens, flush=True)
        print("Iteration: %d, Time: %.6f sec" % (i, toc - tic), flush=True)
        if i >= num_warmup:
            total_time += toc - tic

print("\n", "-" * 10, "Summary:", "-" * 10)
latency = total_time / (num_iter - num_warmup)
print("Inference latency: %.3f sec." % latency)
