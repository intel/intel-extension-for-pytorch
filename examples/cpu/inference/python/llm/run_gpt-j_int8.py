import argparse
import time
import json
import pathlib
import torch
from datasets import load_dataset
from torch.nn.functional import pad
from torch.utils.data import DataLoader
import intel_extension_for_pytorch as ipex
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

parser = argparse.ArgumentParser("GPT-J generation script (int8 path)", add_help=False)
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
    help="cpu",
    default="cpu",
)
parser.add_argument("--dtype", type=str, default="int8")
parser.add_argument(
    "--max-new-tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument("--output-dir", nargs="?", default="./saved_results")
parser.add_argument("--lambada", action="store_true")
parser.add_argument(
    "--ipex-weight-only-quantization",
    action="store_true",
    help="use ipex weight-only quantization",
)
parser.add_argument("--jit", action="store_true")
parser.add_argument("--int8", action="store_true")
parser.add_argument("--ipex-smooth-quant", action="store_true")
parser.add_argument(
    "--int8-bf16-mixed",
    action="store_true",
    help="by default it is int8-fp32 mixed, to enable int8 mixed amp bf16 (work on platforms like SPR)",
)
parser.add_argument("--quantized-model-path", default="./saved_results/best_model.pt")
parser.add_argument("--accuracy-only", action="store_true")
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--input-tokens", default="32", type=str)
parser.add_argument("--prompt", default=None, type=str)
parser.add_argument("--num-iter", default=100, type=int, help="num iter")
parser.add_argument("--num-warmup", default=10, type=int, help="num warmup")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
parser.add_argument("--token-latency", action="store_true")
parser.add_argument("--greedy", action="store_true")
parser.add_argument("--profile", action="store_true")
parser.add_argument(
    "--lowp-mode",
    choices=["BF16","FP32","INT8","FP16"], 
    default="BF16",
    type=str,
    help="low precision mode for weight only quantization"
)
args = parser.parse_args()


# disable
try:
    ipex._C.disable_jit_linear_repack()
except Exception:
    pass

# beam search = 4
num_beams = 1 if args.greedy else 4
generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=num_beams)

# load model
config = AutoConfig.from_pretrained(args.model_id, torchscript=args.jit)
if not hasattr(config, "text_max_length") and args.prompt is None:
    config.text_max_length = int(args.input_tokens) + int(args.max_new_tokens)

if args.benchmark and args.jit and not args.ipex_weight_only_quantization:
    try:
        with ipex._IPEXOnDevice(dtype=torch.float, device="meta"):
            user_model = AutoModelForCausalLM.from_config(config)
    except:
        user_model = AutoModelForCausalLM.from_config(config)
else:
    user_model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float, config=config, low_cpu_mem_usage=True
    )

tokenizer = AutoTokenizer.from_pretrained(args.model_id)
print("Data type of the model:", user_model.dtype)

# to channels last
user_model = user_model.to(memory_format=torch.channels_last)
user_model.eval()
# calling _optimize_transformers for int8 path
user_model = ipex._optimize_transformers(
    user_model.eval(), dtype=torch.int8, inplace=True
)

beam_idx_tmp = torch.zeros((2048, int(args.batch_size * num_beams)), dtype=torch.long).contiguous()
global_past_key_value = [(torch.zeros([1,user_model.config.num_attention_heads,1,int(user_model.config.hidden_size/user_model.config.num_attention_heads)]).contiguous(),
                           torch.zeros([1,user_model.config.num_attention_heads,1,int(user_model.config.hidden_size/user_model.config.num_attention_heads)]).contiguous(), beam_idx_tmp, torch.zeros(1, dtype=torch.long).contiguous()) for i in range(user_model.config.num_hidden_layers)]
                           
# amp autocast
if args.int8_bf16_mixed:
    amp_enabled = True
    amp_dtype = torch.bfloat16
else:
    amp_enabled = False
    amp_dtype = torch.float32

if args.lambada:

    class Evaluator:
        def __init__(
            self, dataset, tokenizer, args, batch_size=8, pad_val=1, pad_max=196
        ):
            self.dataset = dataset
            self.tokenizer = tokenizer
            self.batch_size = batch_size
            self.pad_val = pad_val
            self.pad_max = pad_max
            self.args = args

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
            past_key_values = []
            attention_mask_padded = []
            position_ids_padded = []

            for text in batch:
                input_ids = text["input_ids"]
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
                # dummy past key value
                beam_idx_tmp = torch.zeros(
                    (2048, int(self.args.batch_size * num_beams)), dtype=torch.long
                ).contiguous()
                past_key_value = [
                    (
                        torch.zeros([1, 16, 1, 256]).contiguous(),
                        torch.zeros([1, 16, 1, 256]).contiguous(),
                        beam_idx_tmp,
                        torch.zeros(1, dtype=torch.long).contiguous(),
                    )
                    for i in range(28)
                ]

            return (
                (
                    torch.vstack(input_ids_padded),
                    torch.vstack(attention_mask_padded),
                    torch.vstack(position_ids_padded),
                    tuple(past_key_value),
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
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                latency += time.time() - start
                last_token_logits = outputs[0][
                    torch.arange(len(last_ind)), -2 - pad_len, :
                ]
                pred = last_token_logits.argmax(dim=-1)
                total += label.size(0)
                hit += (pred == label).sum().item()
                if i % 50 == 0:
                    print(hit / total, flush=True)
                    print("Processed minibatch:", i, flush=True)

            acc = hit / total
            print(acc)
            lantecy = latency / len(self.dataset)
            return acc, lantecy

    full_dataset = load_dataset("lambada")
    dataset = full_dataset["validation"]
    calib_dataset = full_dataset["train"]
    evaluator = Evaluator(dataset, tokenizer, args, batch_size=args.batch_size)
    calib_evaluator = Evaluator(
        calib_dataset, tokenizer, args, batch_size=args.batch_size
    )

    calib_dataloader = DataLoader(
        calib_evaluator.dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=evaluator.collate_batch,
    )


if args.jit and args.benchmark:
    torch._C._jit_set_texpr_fuser_enabled(False)
    if args.benchmark and (args.int8 or args.int8_bf16_mixed):
        if not hasattr(user_model, "trace_graph"):
            print("load_int8_model")
            self_jit = torch.jit.load(args.quantized_model_path)
            self_jit = torch.jit.freeze(self_jit.eval())
            setattr(user_model, "trace_graph", self_jit)


def calib_func(prepared_model):
    for i, (
        (input_ids, attention_mask, position_ids, past_key_values),
        last_ind,
    ) in enumerate(calib_dataloader):
        if i == 100:
            break
        prepared_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )


def eval_func(traced_model):
    acc, latency = evaluator.evaluate(traced_model)
    print("Accuracy:", acc)
    print("Latency (sec):", latency)
    return acc


if args.ipex_smooth_quant:
    from neural_compressor import PostTrainingQuantConfig, quantization

    op_type_dict = {
        "add": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}},
        "linear": {
            "weight": {
                "dtype": ["int8"],
                "scheme": ["sym"],
                "granularity": ["per_channel"],
                "algorithm": ["minmax"],
            },
            "activation": {
                "dtype": ["uint8"],
                "scheme": ["asym"],
                "granularity": ["per_tensor"],
                "algorithm": ["kl"],
            },
        },
    }

    excluded_precisions = [] if args.int8_bf16_mixed else ["bf16"]
    recipes = {"smooth_quant": True, "smooth_quant_args": {"alpha": "auto"}}

    recipes["smooth_quant_args"]["folding"] = True

    print("smooth_quant_args:", recipes)
    conf = PostTrainingQuantConfig(
        backend="ipex",
        excluded_precisions=excluded_precisions,
        op_type_dict=op_type_dict,
        recipes=recipes,
    )

    q_model = quantization.fit(
        user_model,
        conf,
        calib_dataloader=calib_dataloader,
        calib_func=calib_func,
    )

    q_model.save(args.output_dir)

if args.ipex_weight_only_quantization:

    def convert_woq(m, qconfig, inplace=True):
        import copy

        def _convert(m):
            from intel_extension_for_pytorch.nn.modules import IpexWoqLinear

            if isinstance(m, torch.nn.Linear):
                m.qconfig = qconfig.global_qconfig
                m_new = IpexWoqLinear.from_float(m)
                return m_new
            m_new = m

            for name, child in m.named_children():
                setattr(m_new, name, _convert(child))
            return m_new

        if not inplace:
            m_new = copy.deepcopy(m)
        else:
            m_new = m
        return _convert(m_new)

    example_inputs = None
    input_ids = torch.ones(32).to(torch.long)
    attention_mask = torch.ones(len(input_ids))
    position_ids = torch.arange(len(input_ids))
    example_inputs = (
        input_ids.unsqueeze(0),
        attention_mask.unsqueeze(0),
        position_ids.unsqueeze(0),
        tuple(global_past_key_value),
    )

    from intel_extension_for_pytorch.quantization import prepare, convert
    
    if args.lowp_mode == "INT8":
        lowp_mode = ipex.quantization.WoqLowpMode.INT8
    elif args.lowp_mode == "FP32":
        lowp_mode = ipex.quantization.WoqLowpMode.NONE
    elif args.lowp_mode == "FP16":
        lowp_mode = ipex.quantization.WoqLowpMode.FP16
    else:
        lowp_mode = ipex.quantization.WoqLowpMode.BF16

    qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
        lowp_mode=lowp_mode
    )
    with torch.no_grad():
        convert_model = convert_woq(user_model.eval(), qconfig)
        self_jit = torch.jit.trace(convert_model.eval(), example_inputs, strict=False)
        self_jit = torch.jit.freeze(self_jit.eval())
        self_jit.save(args.output_dir + "/best_model.pt")

if args.accuracy_only:
    if args.int8 or args.int8_bf16_mixed:
        user_model = torch.jit.load(args.quantized_model_path)
        user_model = torch.jit.freeze(user_model.eval())

    with torch.autocast(
        device_type=args.device,
        enabled=amp_enabled,
        dtype=amp_dtype if amp_enabled else None,
    ):
        eval_func(user_model)

if args.benchmark:
    # input prompt
    current_path = pathlib.Path(__file__).parent.resolve()
    with open(str(current_path) + "/prompt.json") as f:
        prompt_pool = json.load(f)
    if args.prompt is not None:
        prompt = args.prompt
    elif int(args.input_tokens) > 8192:
        prompt = prompt_pool["gpt-j"]["8192"] * int(int(args.input_tokens) / 8192)
    elif args.input_tokens in prompt_pool["gpt-j"]:
        prompt = prompt_pool["gpt-j"][args.input_tokens]
    else:
        raise SystemExit("[ERROR] Plese use --prompt if want to use custom input.")

    input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
    print("---- Prompt size:", input_size)

    if args.token_latency:
        if not hasattr(user_model.config, "token_latency"):
            user_model.config.token_latency = True

    # start
    total_time = 0.0
    num_iter = args.num_iter
    num_warmup = args.num_warmup
    prompt = [prompt] * args.batch_size
    total_list = []
    with torch.inference_mode(), torch.no_grad(), torch.autocast(
        device_type=args.device,
        enabled=amp_enabled,
        dtype=torch.bfloat16 if amp_enabled else None,
    ):
        for i in range(num_iter):
            tic = time.time()
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
            output = user_model.generate(
                input_ids, max_new_tokens=args.max_new_tokens, **generate_kwargs
            )
            gen_ids = output[0] if args.token_latency else output
            gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            toc = time.time()
            input_tokens_lengths = [x.shape[0] for x in input_ids]
            output_tokens_lengths = [x.shape[0] for x in gen_ids]
            total_new_tokens = [
                o - i if user_model.config.model_type != "t5" else o
                for i, o in zip(input_tokens_lengths, output_tokens_lengths)
            ]
            print(gen_text, total_new_tokens, flush=True)
            print("Iteration: %d, Time: %.6f sec" % (i, toc - tic), flush=True)
            # if user_model.config.model_type != 't5':
            #     assert total_new_tokens[0] == args.max_new_tokens, "Generated new tokens != max new tokens"
            if i >= num_warmup:
                total_time += toc - tic
                if args.token_latency:
                    total_list.append(output[1])

    if args.profile:
        def trace_handler(prof):
            print(prof.key_averages().table(
                sort_by="self_cpu_time_total", row_limit=-1))
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=3,
                active=1),
            on_trace_ready=trace_handler
        ) as prof:
            for i in range(5):
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
                output = user_model.generate(
                    input_ids, max_new_tokens=args.max_new_tokens, **generate_kwargs
                )
                gen_ids = output[0] if args.token_latency else output
                gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                prof.step()

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
