#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import time
import json
import pathlib
import numpy as np
from itertools import chain
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

import torch
import sys


def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))


parser = argparse.ArgumentParser(
    "LLM generation (greedy search) script for inductor torch.compile path",
    add_help=False,
)
parser.add_argument(
    "-m",
    "--model-name-or-path",
    default="meta-llama/Llama-2-7b-hf",
    type=str,
    help="path to model or model name in HF hub",
)
parser.add_argument(
    "--dtype",
    type=str,
    choices=["fp32", "bf16", "int8-bf16", "int8", "fp16"],
    help="bf16 or fp32",
    default="bf16",
)
parser.add_argument(
    "--max-new-tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument("--input-tokens", default="32", type=str)
parser.add_argument("--page-size", default=32, type=int)
parser.add_argument("--prompt", default=None, type=str)
parser.add_argument("--num-iter", default=100, type=int, help="num iter")
parser.add_argument("--num-warmup", default=10, type=int, help="num warmup")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
parser.add_argument(
    "--group-size", default=64, type=int, help="group size for woq int4"
)
parser.add_argument("--profile", action="store_true")
parser.add_argument("--accuracy_only", action="store_true")
parser.add_argument(
    "--dataset",
    type=str,
    choices=["lambada", "mmlu", "gsm8k_cot_llama"],
    default="mmlu",
)
parser.add_argument("--disable-concat-linear", action="store_true")
parser.add_argument("--disable-grouped-gemm", action="store_true")
parser.add_argument(
    "--weight-dtype",
    type=str,
    choices=["INT8", "INT4"],
    help="int8 or int4",
    default="INT8",
)
# below are args for scripts compatibility, will be refined.
parser.add_argument(
    "--weight-only-quant",
    action="store_true",
    help="use weight-only quantization",
)
parser.add_argument("--torchao", action="store_true")
parser.add_argument("--inductor", action="store_true")
parser.add_argument("--token-latency", action="store_true")
parser.add_argument("--benchmark", action="store_true")
parser.add_argument(
    "--int8_bf16_mixed",
    action="store_true",
    help="by default it is int8-fp32 mixed, to enable int8 mixed amp bf16 (work on platforms like SPR)",
)
parser.add_argument("--asym-quant-act", action="store_true")
args = parser.parse_args()

if args.dtype == "bf16":
    amp_enabled = True
    load_dtype = torch.bfloat16
elif args.dtype == "fp32":
    amp_enabled = False
    load_dtype = torch.float
elif args.dtype in ["int8-bf16", "int8"]:
    # mixed bf16
    amp_enabled = True
    load_dtype = torch.bfloat16
elif args.dtype == "fp16":
    amp_enabled = True
    load_dtype = torch.float16
else:
    raise SystemExit(
        "This script (inductor peak perf with flexAttention) only support "
        "int8, bf16, fp16, int8-bf16, int8 (da8w8) and fp32 as dtype"
    )

attn_type = "paged_attention"
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path, torch_dtype=load_dtype, attn_implementation=attn_type
)
if attn_type == "paged_attention":
    model.generation_config.cache_implementation = "paged"
    model.config.page_size = args.page_size

if not hasattr(model.config, "no_token_latency"):
    model.config.no_token_latency = False

from torch._inductor import config as inductor_config

if args.profile:
    inductor_config.profiler_mark_wrapper_call = True
    inductor_config.cpp.enable_kernel_profile = True
    inductor_config.cpp.descriptive_names = "inductor_node"
inductor_config.cpp_wrapper = True
inductor_config.max_autotune = True
inductor_config.max_autotune_gemm_backends = "CPP,ATEN"
inductor_config.cpp.use_small_dequant_buffer = True
torch._dynamo.config.allow_unspec_int_on_nn_module = True

if args.dtype in ["fp32", "bf16", "fp16"]:
    if not args.disable_grouped_gemm and hasattr(
        inductor_config.cpp, "enable_grouped_gemm_template"
    ):
        inductor_config.cpp.enable_grouped_gemm_template = True
    elif not args.disable_concat_linear:
        inductor_config.cpp.enable_concat_linear = True
        print("---- apply concat linear ----", flush=True)
    with torch.no_grad(), torch.autocast("cpu", enabled=amp_enabled, dtype=load_dtype):
        model.forward = torch.compile(model.forward)
elif args.dtype in ["int8", "int8-bf16"]:
    from torch._inductor import config as inductor_config
    from torchao.quantization import quant_api
    from torchao.utils import unwrap_tensor_subclass

    if not args.disable_concat_linear:
        inductor_config.cpp.enable_concat_linear = True
        print("---- apply concat linear ----", flush=True)

    with torch.no_grad(), torch.autocast("cpu", enabled=True, dtype=torch.bfloat16):
        if args.weight_dtype == "INT8":
            if args.dtype == "int8-bf16":
                print("---- apply torchao woq int8 api ----", flush=True)
                quant_api.quantize_(
                    model, quant_api.Int8WeightOnlyConfig(set_inductor_config=False)
                )
                unwrap_tensor_subclass(model)
            elif args.dtype == "int8":
                print(
                    "---- apply torchao int8_dynamic_activation_int8_weight api ----",
                    flush=True,
                )
                quant_api.quantize_(
                    model,
                    quant_api.Int8DynamicActivationInt8WeightConfig(
                        set_inductor_config=False
                    ),
                )
        elif args.weight_dtype == "INT4":
            if args.dtype == "int8-bf16":
                from torchao.dtypes import Int4CPULayout

                print("---- apply torchao a16w4 api ----", flush=True)
                quant_api.quantize_(
                    model,
                    quant_api.Int4WeightOnlyConfig(
                        group_size=args.group_size,
                        layout=Int4CPULayout(),
                        set_inductor_config=False,
                        version=1,
                    ),
                )
                unwrap_tensor_subclass(model)
            elif args.dtype == "int8":
                from torchao.dtypes import Int8DynamicActInt4WeightCPULayout
                from torchao.quantization.quant_primitives import MappingType

                print("---- apply torchao da8w4 api ----", flush=True)
                quant_api.quantize_(
                    model,
                    quant_api.Int8DynamicActivationInt4WeightConfig(
                        group_size=args.group_size,
                        layout=Int8DynamicActInt4WeightCPULayout(),
                        act_mapping_type=(
                            MappingType.ASYMMETRIC
                            if args.asym_quant_act
                            else MappingType.SYMMETRIC
                        ),
                    ),
                )

        model.forward = torch.compile(model.forward)


def run_accuracy_lambada(model, dataset):
    from torch.nn.functional import pad
    from datasets import load_dataset
    from torch.utils.data import DataLoader

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
            input_ids_padded = []
            last_ind = []
            for text in batch:
                # we cut the sentence if it exceeds pad_max, we are using tuned max 196 from gptj model; TODO: tune best pad_max
                input_ids = (
                    text["input_ids"]
                    if text["input_ids"].shape[0] <= self.pad_max
                    else text["input_ids"][0 : int(self.pad_max - 1)]
                )
                pad_len = self.pad_max - input_ids.shape[0]
                last_ind.append(input_ids.shape[0] - 1)
                input_ids = pad(input_ids, (0, pad_len), value=self.pad_val)
                input_ids_padded.append(input_ids)
            return ((torch.vstack(input_ids_padded)), torch.tensor(last_ind))

        @torch.no_grad()
        def evaluate(self, model):
            # The task is to predict the last word of the input.
            total, hit = 0, 0
            latency = 0
            test_dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.collate_batch,
            )

            for i, ((input_ids), last_ind) in enumerate(test_dataloader):
                label = input_ids[torch.arange(len(last_ind)), last_ind]
                input_ids[torch.arange(len(last_ind)), last_ind] = self.pad_val
                pad_len = self.pad_max - last_ind - 1
                start = time.time()
                outputs = model(
                    input_ids,
                )
                latency += time.time() - start
                if isinstance(outputs, tuple):
                    res = outputs[0]
                else:
                    res = outputs["logits"]
                last_token_logits = res[torch.arange(len(last_ind)), -2 - pad_len, :]

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

    full_dataset = load_dataset(dataset)
    dataset = full_dataset["validation"]

    evaluator = Evaluator(dataset, tokenizer, 1)
    test_dataloader = DataLoader(
        evaluator.dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=evaluator.collate_batch,
    )

    def eval_func(traced_model):
        print("Evaluating LLM")
        acc, latency = evaluator.evaluate(traced_model)
        print("Accuracy:", acc)
        print("Latency (sec):", latency)
        return acc

    model.eval()
    with torch.autocast("cpu", enabled=amp_enabled, dtype=load_dtype):
        eval_func(model)


# Run LM eval accuracy check for MMLU and GSM8K
def run_accuracy_lmeval(model, dataset):
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import make_table
    import os

    # disable the output token latency for accuracy eval
    model.config.no_token_latency = True

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model.config.tie_word_embeddings = False

    hfmodel = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=1,
        device="cpu",
    )

    print(f"Running accuracy check for dataset: {dataset}")
    if dataset == "gsm8k_cot_llama":
        results = evaluator.simple_evaluate(
            model=hfmodel,
            tasks=dataset,
            fewshot_as_multiturn=True,
            apply_chat_template=True,
            batch_size=16,
        )
        print(results.get("results")["gsm8k_cot_llama"])
        acc = float(
            results.get("results")["gsm8k_cot_llama"]["exact_match,strict-match"]
        )
        print("Accuracy:", acc)

    elif dataset == "mmlu":
        results = evaluator.simple_evaluate(
            model=hfmodel,
            tasks=dataset,
            batch_size=16,
            num_fewshot=5,
        )
        print(results.get("results")["mmlu"])
        acc = results.get("results")["mmlu"]["acc,none"]
        print("Accuracy:", acc)

    print("========================================================")
    print(make_table(results))
    print("========================================================")


if args.accuracy_only:
    if args.dataset == "lambada":
        run_accuracy_lambada(model, args.dataset)
    else:
        run_accuracy_lmeval(model, args.dataset)
    print("Acc test done, exit...")
    sys.exit(0)

# greedy search
generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=1)
current_path = pathlib.Path(__file__).parent.resolve()

model_type = "[not support model]"
if "llama" in args.model_name_or_path.lower():
    model_type = "llama"
elif "gpt-j" in args.model_name_or_path.lower():
    model_type = "gpt-j"

if args.prompt is not None:
    prompt = args.prompt
else:
    with open(str(current_path) + "/prompt.json") as f:
        prompt_pool = json.load(f)
    if model_type in prompt_pool and args.input_tokens in prompt_pool[model_type]:
        prompt = prompt_pool[model_type][args.input_tokens]
    else:
        raise SystemExit(
            "[ERROR] No such input_tokens prompt in prompt.json, Plese use --prompt if want to use custom input."
        )

input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
print("---- Prompt size:", input_size)

prompt = [prompt] * args.batch_size

# warmup
with torch.no_grad(), torch.autocast("cpu", enabled=amp_enabled, dtype=load_dtype):
    for i in range(args.num_warmup):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        model.generate(input_ids, max_new_tokens=args.max_new_tokens, **generate_kwargs)
if args.profile:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=1),
        on_trace_ready=trace_handler,
        record_shapes=True,
    ) as prof, torch.compiler.set_stance(skip_guard_eval_unsafe=True):
        with torch.no_grad(), torch.autocast(
            "cpu", enabled=amp_enabled, dtype=load_dtype
        ):
            for i in range(3):
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                model.generate(
                    input_ids, max_new_tokens=args.max_new_tokens, **generate_kwargs
                )
                prof.step()
# benchmark
num_iter = args.num_iter - args.num_warmup
total_time = 0.0
total_list = []
with (
    torch.no_grad(),
    torch.autocast("cpu", enabled=amp_enabled, dtype=load_dtype),
    torch.compiler.set_stance(skip_guard_eval_unsafe=True),
):
    # Warm-up again. We found first run slow even with warm-up above.
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    model.generate(input_ids, max_new_tokens=args.max_new_tokens, **generate_kwargs)
    for i in range(num_iter):
        tic = time.time()
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        output = model.generate(
            input_ids, max_new_tokens=args.max_new_tokens, **generate_kwargs
        )
        gen_ids = output[0]
        gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        toc = time.time()
        total_time += toc - tic
        total_list.append(output[1])

print(gen_text, flush=True)
print("\n", "-" * 10, "Summary:", "-" * 10)
latency = total_time / (num_iter)
print("inference-latency: %.6f sec." % latency)
first_latency = np.mean([x[0] for x in total_list])
next_latency_list = list(chain(*[x[1:] for x in total_list]))
next_latency_list.sort()
average_next_latency = np.mean(next_latency_list)
p90_latency = np.percentile(next_latency_list, 90)
print("first-token-latency: %.6f sec." % first_latency)
print("rest-token-latency: %.6f sec." % average_next_latency)
print("P90-rest-token-latency: %.6f sec." % p90_latency)
