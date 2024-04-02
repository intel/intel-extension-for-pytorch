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


def get_dummy_input(_model, return_dict=False):
    sample_inputs = None

    if hasattr(_model.config, "n_layer"):
        model_num_layers = _model.config.n_layer
    elif hasattr(_model.config, "num_hidden_layers"):
        model_num_layers = _model.config.num_hidden_layers
    elif hasattr(_model.config, "num_layers"):
        model_num_layers = _model.config.num_layers
    elif hasattr(_model.config, "n_layers"):
        model_num_layers = _model.config.n_layers
    else:
        AssertionError(
            False,
            "Cannot support the dummy sample_inputs for your model, please use your sample_inputs as the inputs and run again",
        )
    past_key_values = tuple(
        [
            (
                (
                    torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                    torch.zeros([1, 1, 1, 1]).contiguous(),
                    torch.zeros([1, 1, 1, 1]).contiguous(),
                    torch.zeros(1, 4, dtype=torch.long),
                )
            )
            for i in range(model_num_layers)
        ]
    )

    input_ids = torch.ones(32).to(torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    model_inputs = _model.prepare_inputs_for_generation(
        input_ids, attention_mask=attention_mask
    )
    has_position_ids = model_inputs.get("position_ids", None) is not None
    position_ids = torch.arange(input_ids.shape[-1]).unsqueeze(0)
    if has_position_ids:
        sample_inputs = (
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            if return_dict
            else (input_ids, attention_mask, past_key_values, position_ids)
        )
    else:
        sample_inputs = (
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
            }
            if return_dict
            else (input_ids, attention_mask, past_key_values)
        )

    return sample_inputs


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
parser.add_argument("--token-latency", action="store_true")
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
    # torchscript=True if args.use_ipex_optimize else False,
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
    if not hasattr(model.config, "use_ipex_optimize"):
        model.config.use_ipex_optimize = True
    # 1) using ipex weight prepack to work with IPEX linear module and their fusions
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

    # 2) using ipex geneartion function to get prompt sharing and first token optimizations
    hf_beam_search = ipex.llm.generation.hf_beam_search.__get__(model, model.__class__)
    hf_greedy_search = ipex.llm.generation.hf_greedy_search.__get__(
        model, model.__class__
    )
    hf_sample = ipex.llm.generation.hf_sample.__get__(model, model.__class__)
    hf_beam_sample = ipex.llm.generation.hf_beam_sample.__get__(model, model.__class__)

    setattr(model, "beam_search", hf_beam_search)  # noqa: B010
    setattr(model, "greedy_search", hf_greedy_search)  # noqa: B010
    setattr(model, "sample", hf_sample)  # noqa: B010
    setattr(model, "beam_sample", hf_beam_sample)  # noqa: B010

    if not hasattr(model.config, "lm_head_generation"):
        model.config.lm_head_generation = True

    # 3) using PyTorch jit to further reduce dispatch overhead
    sample_inputs = get_dummy_input(model, return_dict=True)
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


if (
    args.token_latency
    and args.use_ipex_optimize
    and not hasattr(model.config, "token_latency")
):
    model.config.token_latency = True


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
        gen_ids = output[0] if args.token_latency else output
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
