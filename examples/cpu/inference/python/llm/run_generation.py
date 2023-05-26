import torch
import time
import json
import pathlib
import argparse

from torch.nn.functional import pad
from datasets import load_dataset
from torch.utils.data import DataLoader

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
)


# supported models
MODEL_CLASSES = {
    "gpt-j": (AutoModelForCausalLM, AutoTokenizer),
    "gpt-neox": (AutoModelForCausalLM, AutoTokenizer),
    "llama": (LlamaForCausalLM, LlamaTokenizer),
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
    "--device",
    type=str,
    choices=["cpu"],
    default="cpu",
    help="cpu",
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
parser.add_argument("--ipex", action="store_true")
parser.add_argument("--jit", action="store_true")
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--lambada", action="store_true")
parser.add_argument("--dataset", default="lambada", type=str)
parser.add_argument("--accuracy-only", action="store_true")
parser.add_argument("--num-iter", default=100, type=int, help="num iter")
parser.add_argument("--num-warmup", default=10, type=int, help="num warmup")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
parser.add_argument(
    "--token-latency", action="store_true", help="get token latency breakdown"
)
args = parser.parse_args()
print(args)

# device
device = torch.device(args.device)


if not args.ipex or not args.jit:
    print("Please use --ipex and --jit to re-run this script, aborting...")
    exit(0)

# import ipex
if args.ipex:
    import intel_extension_for_pytorch as ipex

    try:
        ipex._C.disable_jit_linear_repack()
    except Exception:
        pass

if args.jit:
    torch._C._jit_set_texpr_fuser_enabled(False)

# dtype
amp_enabled = True if args.dtype != "float32" else False
amp_dtype = getattr(torch, args.dtype)

# load model
model_type = next(
    (x for x in MODEL_CLASSES.keys() if x in args.model_id.lower()), "auto"
)
model_class = MODEL_CLASSES[model_type]
config = AutoConfig.from_pretrained(args.model_id, torchscript=args.jit)
if not hasattr(config, "text_max_length") and args.prompt is None:
    config.text_max_length = int(args.input_tokens) + int(args.max_new_tokens)
model = model_class[0].from_pretrained(
    args.model_id, torch_dtype=amp_dtype, config=config, low_cpu_mem_usage=True
)
tokenizer = model_class[1].from_pretrained(args.model_id)
model = model.eval().to(device)
model = model.to(memory_format=torch.channels_last)

# to ipex
if args.ipex:
    model = ipex._optimize_transformers(model.eval(), dtype=amp_dtype, inplace=True)

num_beams = 1 if args.greedy else 4
# generate args
generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=num_beams)

# dummy past key values
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
                torch.zeros([1, 1, 1, 1]).contiguous(),
                torch.zeros([1, 1, 1, 1]).contiguous(),
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
else:
    print(
        "Currently we only support jit path on GPTJ, llama, and gpt_neox models for IPEX new API ipex._optimize_transformers(), please re-run without jit "
    )
    exit(0)

if not hasattr(model, "trace_graph") and args.jit and args.benchmark and args.ipex:
    example_inputs = None
    input_ids = torch.ones(32).to(torch.long)
    attention_mask = torch.ones(len(input_ids))
    position_ids = torch.arange(len(input_ids))
    example_inputs = {
        "input_ids": input_ids.unsqueeze(0),
        "attention_mask": attention_mask.unsqueeze(0),
        "position_ids": position_ids.unsqueeze(0),
        "past_key_values": past_key_values,
    }

    with torch.inference_mode(), torch.no_grad(), torch.autocast(
        device_type=args.device,
        enabled=amp_enabled,
        dtype=amp_dtype if amp_enabled else None,
    ):
        trace_model = torch.jit.trace(
            model, example_kwarg_inputs=example_inputs, strict=False, check_trace=False
        )
        trace_model = torch.jit.freeze(trace_model)
        setattr(model, "trace_graph", trace_model)


class Evaluator:
    def __init__(self, dataset, tokenizer, batch_size=8, pad_val=1, pad_max=196):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.pad_val = pad_val
        self.pad_max = pad_max

        # tokenize the dataset
        self.dataset = self.dataset.map(self.tokenize_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids"])

    @torch.no_grad()
    def tokenize_function(self, examples):
        example = self.tokenizer(examples["text"])
        return example

    @torch.no_grad()
    def collate_batch(self, batch):
        position_ids_padded = []
        input_ids_padded = []
        last_ind = []
        attention_mask_padded = []
        for text in batch:
            # we cut the sentence if it exceeds pad_max, we are using tuned max 196 from gptj model; TODO: tune best pad_max
            input_ids = (
                text["input_ids"]
                if text["input_ids"].shape[0] <= self.pad_max
                else text["input_ids"][0 : int(self.pad_max - 1)]
            )
            pad_len = self.pad_max - input_ids.shape[0]
            last_ind.append(input_ids.shape[0] - 1)
            attention_mask = torch.ones(len(input_ids))
            position_ids = torch.arange(len(input_ids))
            input_ids = pad(input_ids, (0, pad_len), value=self.pad_val)
            input_ids_padded.append(input_ids)
            attention_mask = pad(attention_mask, (0, pad_len), value=0)
            attention_mask_padded.append(attention_mask)
            position_ids = pad(position_ids, (0, pad_len), value=self.pad_val)
            position_ids_padded.append(position_ids)
        return (
            (
                torch.vstack(input_ids_padded),
                torch.vstack(attention_mask_padded),
                torch.vstack(position_ids_padded),
                tuple(past_key_values),
            ),
            torch.tensor(last_ind),
        )

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        latency = 0
        test_dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_batch,
        )

        for i, (
            (input_ids, attention_mask, position_ids, past_key_values),
            last_ind,
        ) in enumerate(test_dataloader):
            label = input_ids[torch.arange(len(last_ind)), last_ind]
            input_ids[torch.arange(len(last_ind)), last_ind] = self.pad_val
            pad_len = self.pad_max - last_ind - 1
            start = time.time()

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
            )

            latency += time.time() - start

            last_token_logits = outputs[0][torch.arange(len(last_ind)), -2 - pad_len, :]

            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
            if i % 50 == 0:
                print(hit / total)
                print("Processed minibatch:", i)

        acc = hit / total
        print(acc)
        lantecy = latency / len(self.dataset)
        return acc, lantecy


if args.lambada:
    full_dataset = load_dataset(args.dataset)
    dataset = full_dataset["validation"]

    model.eval()
    evaluator = Evaluator(dataset, tokenizer, args.batch_size)

    test_dataloader = DataLoader(
        evaluator.dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=evaluator.collate_batch,
    )


def eval_func(traced_model):
    acc, latency = evaluator.evaluate(traced_model)
    print("Accuracy:", acc)
    print("Latency (sec):", latency)
    return acc


if args.accuracy_only:
    if args.jit and args.ipex:
        input_ids = torch.ones(32).to(torch.long)
        attention_mask = torch.ones(len(input_ids))
        position_ids = torch.arange(len(input_ids))
        example_inputs = (
            input_ids.unsqueeze(0),
            attention_mask.unsqueeze(0),
            position_ids.unsqueeze(0),
            tuple(past_key_values),
        )
        with torch.no_grad(), torch.autocast(
            device_type=args.device,
            enabled=amp_enabled,
            dtype=amp_dtype if amp_enabled else None,
        ):
            model = torch.jit.trace(model.eval(), example_inputs, strict=False)
            model = torch.jit.freeze(model.eval())

    with torch.autocast(
        device_type=args.device,
        enabled=amp_enabled,
        dtype=amp_dtype if amp_enabled else None,
    ):
        eval_func(model)


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
    with torch.inference_mode(), torch.no_grad(), torch.autocast(
        device_type=args.device,
        enabled=amp_enabled,
        dtype=amp_dtype if amp_enabled else None,
    ):
        for i in range(num_iter):
            tic = time.time()
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
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
