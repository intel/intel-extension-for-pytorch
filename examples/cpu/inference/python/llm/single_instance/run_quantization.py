import argparse
import time
import json
import pathlib
import re

from datasets import load_dataset

import torch
from torch.utils.data import DataLoader
import transformers
from transformers import AutoConfig

import intel_extension_for_pytorch as ipex

import sys

sys.path.append(sys.path[0] + "/../../")

from llm.utils.utils import (
    _get_relative_imports,
    _gradient_checkpointing_disable,
    _gradient_checkpointing_enable,
)
from llm.utils.model_class.llm import EXAMPLE_INPUTS_MODE
from llm.utils.model_class.llama import LLAMAConfig
from llm.utils.model_class.gptj import GPTJConfig
from llm.utils.model_class.gptneox import GPTNEOXConfig
from llm.utils.model_class.falcon import FALCONConfig
from llm.utils.model_class.opt import OPTConfig
from llm.utils.model_class.bloom import BloomConfig
from llm.utils.model_class.codegen import CodeGenConfig
from llm.utils.model_class.baichuan import BaichuanConfig
from llm.utils.model_class.chatglm import ChatGLMConfig
from llm.utils.model_class.gptbigcode import GPTJBigCodeConfig
from llm.utils.model_class.t5 import T5Config
from llm.utils.model_class.mistral import MistralConfig

transformers.dynamic_module_utils.get_relative_imports = _get_relative_imports
transformers.modeling_utils.PreTrainedModel.gradient_checkpointing_disable = (
    _gradient_checkpointing_disable
)
transformers.modeling_utils.PreTrainedModel.gradient_checkpointing_enable = (
    _gradient_checkpointing_enable
)

parser = argparse.ArgumentParser("LLM generation script (int8 path)", add_help=False)
parser.add_argument(
    "-m", "--model-id", default=None, type=str, required=True, help="your llm model"
)
parser.add_argument(
    "--max-new-tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument("--dataset", nargs="?", default="")
parser.add_argument("--split", nargs="?", default="validation", const="validation")
parser.add_argument("--output-dir", nargs="?", default="./saved_results")
parser.add_argument("--quant-model-name", default="best_model.pt")
parser.add_argument("--ipex-smooth-quant", action="store_true")
parser.add_argument(
    "--ipex-weight-only-quantization",
    action="store_true",
    help="use ipex weight-only quantization",
)
parser.add_argument("--int8", action="store_true")
parser.add_argument(
    "--int8-bf16-mixed",
    action="store_true",
    help="by default it is int8-fp32 mixed, to enable int8 mixed amp bf16 (work on platforms like SPR)",
)
parser.add_argument("--qconfig-summary-file", default="", help="qconfig for static quantization")
parser.add_argument("--quantized-model-path", default="./saved_results/best_model.pt")
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--input-tokens", default="32", type=str)
parser.add_argument("--prompt", default=None, type=str)
parser.add_argument("--num-iter", default=100, type=int, help="num iter")
parser.add_argument("--num-warmup", default=10, type=int, help="num warmup")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
parser.add_argument(
    "--alpha", default=0.8, type=float, help="alpha value for smoothquant"
)
parser.add_argument("--token-latency", action="store_true")
parser.add_argument("--greedy", action="store_true")
parser.add_argument("--profile", action="store_true")
parser.add_argument(
    "--config-file", default=None, type=str, help="specific configuration file"
)
parser.add_argument(
    "--lowp-mode",
    choices=["AUTO", "BF16", "FP32", "INT8", "FP16"],
    default="AUTO",
    type=str,
    help="low precision mode for weight only quantization. "
    "It indicates data type for computation for speedup at the cost "
    "of accuracy. Unrelated to activation or weight data type."
    "It is not supported yet to use lowp_mode=INT8 for INT8 weight, "
    "falling back to lowp_mode=BF16 implicitly in this case."
    "If set to AUTO, lowp_mode is determined by weight data type: "
    "lowp_mode=BF16 is used for INT8 weight "
    "and lowp_mode=INT8 used for INT4 weight",
)
parser.add_argument(
    "--weight-dtype",
    choices=["INT8", "INT4"],
    default="INT8",
    type=str,
    help="weight data type for weight only quantization. Unrelated to activation"
    " data type or lowp-mode. If `--low-precision-checkpoint` is given, weight"
    " data type is always INT4 and this argument is not needed.",
)
parser.add_argument(
    "--group-size",
    default=-1,
    type=int,
    help="For weight-only quantization only. Specifies the group size along"
    " input channel for block-wise quantization of weight. It must be a"
    " positive power of 2 or -1. If it is -1, weight is quantized per"
    " output channel. Otherwise, weight is quantized per block with block size"
    " = [1, group_size]. If `--low-precision-checkpoint` is given, group"
    " size is determined automatically and this argument has no effect.",
)
parser.add_argument(
    "--low-precision-checkpoint",
    default="",
    type=str,
    help="Low precision checkpoint file generated by calibration, such as GPTQ. It contains"
    " modified weights, scales, zero points, etc. For better accuracy of weight only"
    " quantization with INT4 weight.",
)
parser.add_argument(
    "--act-quant-mode",
    choices=["PER_TENSOR", "PER_IC_BLOCK", "PER_BATCH", "PER_BATCH_IC_BLOCK"],
    default="PER_IC_BLOCK",
    type=str,
    help="Quantization mode for activation with different granularity. "
    "For lowp-mode=INT8 only. For other cases, it has no effect. "
    "Assume the activation tensor has shape batch_size x input_channel. "
    "PER_TENSOR(0): quantize per tensor; "
    "PER_IC_BLOCK(1): quantize per group along IC with group size = IC_BLOCK; "
    "PER_BATCH(2): quantize per batch; "
    "PER_BATCH_IC_BLOCK(3): quantize per block of size 1 x IC_BLOCK. "
    "IC_BLOCK is determined by IC automatically.",
)
args = parser.parse_args()


# disable
try:
    ipex._C.disable_jit_linear_repack()
except Exception:
    pass

# amp autocast
if args.int8_bf16_mixed:
    amp_enabled = True
    amp_dtype = torch.bfloat16
else:
    amp_enabled = False
    amp_dtype = torch.float32


num_beams = 1 if args.greedy else 4
generate_kwargs = dict(
    do_sample=False,
    temperature=0.9,
    num_beams=num_beams,
    max_new_tokens=args.max_new_tokens,
    min_new_tokens=args.max_new_tokens,
)

if args.config_file is None:
    config = AutoConfig.from_pretrained(
        args.model_id, torchscript=True, trust_remote_code=True
    )
else:
    config = AutoConfig.from_pretrained(
        args.config_file, torchscript=True, trust_remote_code=True
    )
if re.search("falcon", config.architectures[0], re.IGNORECASE) or re.search(
    "rw", config.architectures[0], re.IGNORECASE
):
    model = FALCONConfig(args.model_id)
elif re.search("GPTJ", config.architectures[0], re.IGNORECASE):
    model = GPTJConfig(args.model_id)
elif re.search("llama", config.architectures[0], re.IGNORECASE):
    model = LLAMAConfig(args.model_id)
elif re.search("gptneox", config.architectures[0], re.IGNORECASE):
    model = GPTNEOXConfig(args.model_id)
elif re.search("OPT", config.architectures[0], re.IGNORECASE):
    model = OPTConfig(args.model_id)
elif re.search("bloom", config.architectures[0], re.IGNORECASE):
    model = BloomConfig(args.model_id)
elif re.search("codegen", config.architectures[0], re.IGNORECASE):
    model = CodeGenConfig(args.model_id)
elif re.search("baichuan", config.architectures[0], re.IGNORECASE):
    model = BaichuanConfig(args.model_id)
elif re.search("chatglm", config.architectures[0], re.IGNORECASE):
    model = ChatGLMConfig(args.model_id)
elif re.search("gptbigcode", config.architectures[0], re.IGNORECASE):
    model = GPTJBigCodeConfig(args.model_id)
elif re.search("t5", config.architectures[0], re.IGNORECASE):
    generate_kwargs["max_length"] = generate_kwargs["max_new_tokens"]
    generate_kwargs.pop("max_new_tokens")
    model = T5Config(args.model_id)
elif re.search("mistral", config.architectures[0], re.IGNORECASE):
    model = MistralConfig(args.model_id)
else:
    raise AssertionError("Not support %s." % (args.model_id))

if not hasattr(config, "text_max_length") and args.prompt is None:
    config.text_max_length = int(args.input_tokens) + int(args.max_new_tokens)

user_model = model.get_user_model(config, args.benchmark)

tokenizer = model.get_tokenizer()
print("Data type of the model:", user_model.dtype)

if model.to_channels_last:
    user_model = user_model.to(memory_format=torch.channels_last)
user_model.eval()

# dummy past key value
beam_idx_tmp = torch.zeros(
    (2048, int(args.batch_size * num_beams)), dtype=torch.long
).contiguous()
global_past_key_value = [
    (
        torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
        torch.zeros(
            [
                1,
                user_model.config.num_attention_heads,
                1,
                int(
                    user_model.config.hidden_size
                    / user_model.config.num_attention_heads
                ),
            ]
        ).contiguous(),
        torch.zeros(
            [
                1,
                user_model.config.num_attention_heads,
                1,
                int(
                    user_model.config.hidden_size
                    / user_model.config.num_attention_heads
                ),
            ]
        ).contiguous(),
        beam_idx_tmp,
    )
    for i in range(user_model.config.num_hidden_layers)
]


if args.ipex_smooth_quant:
    if args.qconfig_summary_file is not "":
        qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(alpha=args.alpha)
        user_model = ipex.optimize_transformers(
            user_model.eval(),
            dtype=amp_dtype,
            quantization_config=qconfig,
            qconfig_summary_file=args.qconfig_summary_file,
            inplace=True,
            deployment_mode=True,
        )
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        user_model.trace_graph.save(args.output_dir + '/' + args.quant_model_name)
        quant_model = user_model.trace_graph
    else:
        class Evaluator:
            def __init__(
                self, dataset, tokenizer, args, batch_size=1, pad_val=1, pad_max=512
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
                if "prompt" in examples:
                    example = self.tokenizer(examples["prompt"])
                elif "text" in examples:
                    example = self.tokenizer(examples["text"])
                elif "code" in examples:
                    example = self.tokenizer(examples["code"])
                return example
    
            @torch.no_grad()
            def collate_batch(self, batch):
                position_ids_padded = []
                input_ids_padded = []
                last_ind = []
                attention_mask_padded = []
                for text in batch:
                    input_ids = text["input_ids"]
                    input_ids = (
                        input_ids[: int(self.pad_max)]
                        if len(input_ids) > int(self.pad_max)
                        else input_ids
                    )
                    last_ind.append(input_ids.shape[0] - 1)
                    attention_mask = torch.ones(len(input_ids))
                    position_ids = torch.arange(len(input_ids))
                    input_ids_padded.append(input_ids)
                    attention_mask_padded.append(attention_mask)
                    position_ids_padded.append(position_ids)
                    if not model.use_global_past_key_value:
                        # dummy past key value
                        beam_idx_tmp = torch.zeros(
                            (2048, int(self.args.batch_size * num_beams)), dtype=torch.long
                        ).contiguous()
                        past_key_value = [
                            (
                                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                                torch.zeros([1, 16, 1, 256]).contiguous(),
                                torch.zeros([1, 16, 1, 256]).contiguous(),
                                beam_idx_tmp,
                            )
                            for i in range(28)
                        ]
                return (
                    (
                        torch.vstack(input_ids_padded),
                        torch.vstack(attention_mask_padded),
                        torch.vstack(position_ids_padded),
                        tuple(global_past_key_value)
                        if model.use_global_past_key_value
                        else tuple(past_key_value),
                    ),
                    torch.tensor(last_ind),
                )
    
        calib_dataset = load_dataset(
            args.dataset if args.dataset else model.default_dataset, split="train"
        )
        user_model.eval()
        calib_evaluator = Evaluator(
            calib_dataset, tokenizer, args, batch_size=args.batch_size
        )
        calib_dataloader = DataLoader(
            calib_evaluator.dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=calib_evaluator.collate_batch,
        )
    
        def calib_func(prepared_model):
            for i, (
                (input_ids, attention_mask, position_ids, past_key_values),
                last_ind,
            ) in enumerate(calib_dataloader):
                if i == 512:
                    break
                prepared_model(
                    input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                )
    
        if model.use_neural_compressor:
            from neural_compressor import PostTrainingQuantConfig, quantization
    
            qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping()
            user_model = ipex.optimize_transformers(
                user_model.eval(),
                dtype=amp_dtype,
                quantization_config=qconfig,
                inplace=True,
                deployment_mode=False,
            )
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
            recipes = {"smooth_quant": True, "smooth_quant_args": {"alpha": args.alpha}}
    
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
        else:
            example_inputs = None
            for i, (
                (input_ids, attention_mask, position_ids, past_key_values),
                last_ind,
            ) in enumerate(calib_dataloader):
                example_inputs = (input_ids, attention_mask, position_ids, past_key_values)
                break
            from intel_extension_for_pytorch.quantization import prepare, convert
    
            qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(alpha=args.alpha)
            user_model = ipex.optimize_transformers(
                user_model.eval(),
                dtype=amp_dtype,
                quantization_config=qconfig,
                inplace=True,
                deployment_mode=False,
            )
            prepared_model = prepare(
                user_model.eval(), qconfig, example_inputs=example_inputs, inplace=True
            )
            with torch.no_grad():
                for i, (
                    (input_ids, attention_mask, position_ids, past_key_values),
                    last_ind,
                ) in enumerate(calib_dataloader):
                    if i == 512:
                        break
                    prepared_model(
                        input_ids,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                    )
            with torch.no_grad(), torch.cpu.amp.autocast(
                enabled=amp_enabled,
            ):
                convert_model = convert(prepared_model.eval(), inplace=True).eval()
                self_jit = torch.jit.trace(
                    convert_model.eval(), example_inputs, strict=False, check_trace=False
                )
                self_jit = torch.jit.freeze(self_jit.eval())
                pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
                self_jit.save(args.output_dir + "/" + args.quant_model_name)
                quant_model = self_jit

elif args.ipex_weight_only_quantization:
    weight_dtype = torch.quint4x2 if args.weight_dtype == "INT4" else torch.qint8

    if args.lowp_mode == "INT8":
        lowp_mode = ipex.quantization.WoqLowpMode.INT8
    elif args.lowp_mode == "FP32":
        lowp_mode = ipex.quantization.WoqLowpMode.NONE
    elif args.lowp_mode == "FP16":
        lowp_mode = ipex.quantization.WoqLowpMode.FP16
    elif args.lowp_mode == "BF16":
        lowp_mode = ipex.quantization.WoqLowpMode.BF16
    else:  # AUTO
        if args.low_precision_checkpoint != "" or weight_dtype == torch.quint4x2:
            lowp_mode = ipex.quantization.WoqLowpMode.INT8
        else:
            lowp_mode = ipex.quantization.WoqLowpMode.BF16

    act_quant_mode_dict = {
        "PER_TENSOR": ipex.quantization.WoqActQuantMode.PER_TENSOR,
        "PER_IC_BLOCK": ipex.quantization.WoqActQuantMode.PER_IC_BLOCK,
        "PER_BATCH": ipex.quantization.WoqActQuantMode.PER_BATCH,
        "PER_BATCH_IC_BLOCK": ipex.quantization.WoqActQuantMode.PER_BATCH_IC_BLOCK,
    }
    qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
        weight_dtype=weight_dtype,
        lowp_mode=lowp_mode,
        act_quant_mode=act_quant_mode_dict[args.act_quant_mode],
        group_size=args.group_size,
    )
    if args.low_precision_checkpoint != "":
        low_precision_checkpoint = torch.load(args.low_precision_checkpoint)
    else:
        low_precision_checkpoint = None
    user_model = ipex.optimize_transformers(
        user_model.eval(),
        dtype=amp_dtype,
        quantization_config=qconfig,
        inplace=True,
        low_precision_checkpoint=low_precision_checkpoint,
        deployment_mode=False,
    )
    example_inputs = None
    input_ids = torch.ones(32).to(torch.long)
    attention_mask = torch.ones(len(input_ids))
    if model.example_inputs_mode == EXAMPLE_INPUTS_MODE.MASK_POS_KV:
        position_ids = torch.arange(len(input_ids))
        example_inputs = (
            input_ids.unsqueeze(0),
            attention_mask.unsqueeze(0),
            position_ids.unsqueeze(0),
            tuple(global_past_key_value),
        )
    elif model.example_inputs_mode == EXAMPLE_INPUTS_MODE.MASK_KV_POS:
        position_ids = torch.arange(len(input_ids))
        example_inputs = (
            input_ids.unsqueeze(0),
            attention_mask.unsqueeze(0),
            tuple(global_past_key_value),
            position_ids.unsqueeze(0),
        )
    elif model.example_inputs_mode == EXAMPLE_INPUTS_MODE.KV_MASK:
        example_inputs = (
            input_ids.unsqueeze(0),
            tuple(global_past_key_value),
            attention_mask.unsqueeze(0),
        )
    elif model.example_inputs_mode == EXAMPLE_INPUTS_MODE.MASK_KV_ENC:
        last_hidden_state = torch.rand([1, 32, 2048])
        global_past_key_value = [
            (
                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                torch.zeros(
                    [
                        1,
                        user_model.config.num_attention_heads,
                        1,
                        int(
                            user_model.config.hidden_size
                            / user_model.config.num_attention_heads
                        ),
                    ]
                ).contiguous(),
                torch.zeros(
                    [
                        1,
                        user_model.config.num_attention_heads,
                        1,
                        int(
                            user_model.config.hidden_size
                            / user_model.config.num_attention_heads
                        ),
                    ]
                ).contiguous(),
                beam_idx_tmp,
                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                torch.zeros(
                    [
                        32,
                        1,
                        user_model.config.num_attention_heads,
                        int(
                            user_model.config.hidden_size
                            / user_model.config.num_attention_heads
                        ),
                    ]
                ).contiguous(),
                torch.zeros(
                    [
                        32,
                        1,
                        user_model.config.num_attention_heads,
                        int(
                            user_model.config.hidden_size
                            / user_model.config.num_attention_heads
                        ),
                    ]
                ).contiguous(),
                beam_idx_tmp,
            )
            for i in range(user_model.config.num_hidden_layers)
        ]
        example_inputs = (
            torch.ones(1).to(torch.long).unsqueeze(0),
            attention_mask.unsqueeze(0),
            tuple(global_past_key_value),
            (last_hidden_state,),
        )
    else:
        example_inputs = (
            input_ids.unsqueeze(0),
            attention_mask.unsqueeze(0),
            tuple(global_past_key_value),
        )
    if hasattr(model, "extra_inputs"):
        example_inputs = example_inputs + model.extra_inputs
    with torch.no_grad(), torch.cpu.amp.autocast(
        enabled=amp_enabled,
    ):
        self_jit = torch.jit.trace(
            user_model.eval(), example_inputs, strict=False, check_trace=False
        )
        self_jit = torch.jit.freeze(self_jit.eval())
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        self_jit.save(args.output_dir + "/" + args.quant_model_name)
        quant_model = self_jit


if args.benchmark:
    torch._C._jit_set_texpr_fuser_enabled(False)
    if not re.search(
        "baichuan", config.architectures[0], re.IGNORECASE
    ) and not re.search("chatglm", config.architectures[0], re.IGNORECASE):
        qconfig = ipex.quantization.default_static_qconfig_mapping
        user_model = ipex.optimize_transformers(
            user_model.eval(),
            dtype=torch.float,
            inplace=True,
            quantization_config=qconfig,
            deployment_mode=False,
        )
    if not hasattr(user_model, "trace_graph"):
        print("load_quantized_model")
        try:
            self_jit = torch.jit.load(args.quantized_model_path)
            self_jit = torch.jit.freeze(self_jit.eval())
        except Exception as e:
            print("warning: loading failed.", e)
            self_jit = quant_model
        ipex._set_optimized_model_for_generation(user_model, optimized_model=self_jit)

    # input prompt
    current_path = pathlib.Path(__file__).parent.resolve()
    with open(str(current_path) + "/prompt.json") as f:
        prompt_pool = json.load(f)
    if args.prompt is not None:
        prompt = args.prompt
    elif int(args.input_tokens) > 8192:
        prompt = prompt_pool[model.name]["8192"] * int(int(args.input_tokens) / 8192)
    elif args.input_tokens in prompt_pool[model.name]:
        prompt = prompt_pool[model.name][args.input_tokens]
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
    with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
        enabled=amp_enabled
    ):
        for i in range(num_iter):
            tic = time.time()
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            output = user_model.generate(input_ids, **generate_kwargs)
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
            if i >= num_warmup:
                total_time += toc - tic
                if args.token_latency:
                    total_list.append(output[1])

    if args.profile:

        def trace_handler(prof):
            print(
                prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1)
            )

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(wait=1, warmup=3, active=1),
            on_trace_ready=trace_handler,
        ) as prof:
            for i in range(5):
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                output = user_model.generate(input_ids, **generate_kwargs)
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
