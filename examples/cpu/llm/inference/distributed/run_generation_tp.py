import torch
import time
import json
import pathlib
import argparse
import re

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    T5ForConditionalGeneration,
    AutoProcessor,
)

from transformers import TextStreamer

import sys

sys.path.append(sys.path[0] + "/../../../")

import logging

logger = logging.getLogger(__name__)

# supported models
MODEL_CLASSES = {
    "gpt-j": (AutoModelForCausalLM, AutoTokenizer),
    "gpt-neox": (AutoModelForCausalLM, AutoTokenizer),
    "llama": (AutoModelForCausalLM, LlamaTokenizer),
    "opt": (AutoModelForCausalLM, AutoTokenizer),
    "falcon": (AutoModelForCausalLM, AutoTokenizer),
    "bloom": (AutoModelForCausalLM, AutoTokenizer),
    "codegen": (AutoModelForCausalLM, AutoTokenizer),
    "baichuan2": (AutoModelForCausalLM, AutoTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "chatglm": (AutoModelForCausalLM, AutoTokenizer),
    "gptbigcode": (AutoModelForCausalLM, AutoTokenizer),
    "t5": (T5ForConditionalGeneration, AutoTokenizer),
    "mistral": (AutoModelForCausalLM, AutoTokenizer),
    "mixtral": (AutoModelForCausalLM, AutoTokenizer),
    "mpt": (AutoModelForCausalLM, AutoTokenizer),
    "stablelm": (AutoModelForCausalLM, AutoTokenizer),
    "qwen": (AutoModelForCausalLM, AutoTokenizer),
    "git": (AutoModelForCausalLM, AutoProcessor),
    "yuan": (AutoModelForCausalLM, AutoTokenizer),
    "phi-3": (AutoModelForCausalLM, AutoTokenizer),
    "phi": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}

try:
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
    from llava.model.builder import load_pretrained_model
    from llava.conversation import conv_templates
    from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
    from llava.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
    )

    MODEL_CLASSES["llava"] = (LlavaLlamaForCausalLM, AutoTokenizer)
except ImportError:
    pass

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
    "--streaming",
    action="store_true",
    help="enable streaming mode for generation output (greedy search only)",
)
parser.add_argument(
    "--image-url",
    default="https://images.cocodataset.org/val2017/000000039769.jpg",
    type=str,
    help="image url for image-to-text task",
)
parser.add_argument(
    "--config-file", default=None, type=str, help="specific configuration file"
)
parser.add_argument("--greedy", action="store_true")
parser.add_argument("--ipex", action="store_true")
parser.add_argument(
    "--ipex-weight-only-quantization",
    action="store_true",
    help="use ipex weight-only quantization",
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
    "--quant-with-amp",
    action="store_true",
    help="by default it is int8-fp32 mixed, to enable int8 mixed amp bf16 (work on platforms like SPR)",
)
parser.add_argument(
    "--weight-dtype",
    choices=["INT8", "INT4", "NF4"],
    default="INT8",
    type=str,
    help="weight data type for weight only quantization. Unrelated to activation"
    " data type or lowp-mode. If `--low-precision-checkpoint` is given, weight"
    " data type is always INT4 and this argument is not needed.",
)
parser.add_argument("--deployment-mode", action="store_true")
parser.add_argument("--torch-compile", action="store_true")
parser.add_argument(
    "--backend", default="ipex", type=str, help="backend of torch.compile"
)
parser.add_argument("--profile", action="store_true")
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--num-iter", default=100, type=int, help="num iter")
parser.add_argument("--num-warmup", default=10, type=int, help="num warmup")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
parser.add_argument(
    "--token-latency", action="store_true", help="get token latency breakdown"
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
parser.add_argument(
    "--gptq-legacy-format",
    action="store_true",
    help="Indicate that the low-precision checkpoint is in the legacy format rather than the"
    " HuggingFace Optimum format for backward compatibility. It must be used with"
    " --low-precision-checkpoint. Otherwise, it has no effect.",
)
parser.add_argument(
    "--cache-weight-for-large-batch",
    action="store_true",
    help="Cache an extra linear weight for large batch inference, such as the first token (prefill phase)."
    " It brings better performance at the cost of higher memory usage. It is only valid for full bf16 path"
    " and weight-only quantization with lowp-mode=BF16. Otherwise, it has no effect.",
)
args = parser.parse_args()
print(args)

# import ipex
if args.ipex or args.ipex_weight_only_quantization:
    import intel_extension_for_pytorch as ipex

    torch._C._jit_set_texpr_fuser_enabled(False)
    try:
        ipex._C.disable_jit_linear_repack()
    except Exception:
        pass

# dtype
amp_enabled = False if args.dtype == "float32" or not args.quant_with_amp else True
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
if model_type == "mpt" and args.prompt is None:
    config.max_seq_len = int(args.input_tokens) + int(args.max_new_tokens)
if model_type == "llava":
    config.use_cache = True

if not hasattr(config, "lm_head_generation"):
    config.lm_head_generation = True

if model_type != "llava":
    model = model_class[0].from_pretrained(
        args.model_id,
        torch_dtype=amp_dtype,
        config=config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    tokenizer = model_class[1].from_pretrained(args.model_id, trust_remote_code=True)
else:
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_id
    )
model = model.eval()
model = model.to(memory_format=torch.channels_last)

num_beams = 1 if args.greedy else 4
# generate args
if args.streaming:
    streamer = TextStreamer(tokenizer)
else:
    streamer = None
generate_kwargs = dict(
    do_sample=False,
    temperature=0.9,
    num_beams=num_beams,
    max_new_tokens=args.max_new_tokens,
    min_new_tokens=args.max_new_tokens,
    streamer=streamer,
)

if re.search("gptbigcode", model.config.architectures[0], re.IGNORECASE):
    model_type = "gptbigcode"
if re.search("gptneox", model.config.architectures[0], re.IGNORECASE):
    model_type = "gpt-neox"
elif re.search("t5", model.config.architectures[0], re.IGNORECASE):
    generate_kwargs["max_length"] = generate_kwargs["max_new_tokens"]
    generate_kwargs.pop("max_new_tokens")
elif re.search("git", model.config.architectures[0], re.IGNORECASE) or re.search(
    "llava", model.config.architectures[0], re.IGNORECASE
):
    from PIL import Image
    import requests
    from io import BytesIO

    model.config.batch_size = int(args.batch_size) * num_beams

    def load_image(image_file):
        if image_file.startswith("http://") or image_file.startswith("https://"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image


if re.search("llava", model.config.architectures[0], re.IGNORECASE):
    model_name = get_model_name_from_path(args.model_id)
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    conv = conv_templates[conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ("user", "assistant")
    else:
        roles = conv.roles
if re.search("yuan", model.config.architectures[0], re.IGNORECASE):
    model.config.batch_size = int(args.batch_size) * num_beams


def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))


# to ipex
if args.ipex:
    model = ipex.llm.optimize(
        model.eval(),
        dtype=amp_dtype,
        inplace=True,
        deployment_mode=args.deployment_mode,
        cache_weight_for_large_batch=args.cache_weight_for_large_batch,
    )
elif args.ipex_weight_only_quantization:
    from intel_extension_for_pytorch.quantization import WoqWeightDtype

    if args.weight_dtype == "INT8":
        weight_dtype = WoqWeightDtype.INT8
    elif args.weight_dtype == "INT4":
        weight_dtype = WoqWeightDtype.INT4
    else:
        assert args.weight_dtype == "NF4"
        weight_dtype = WoqWeightDtype.NF4

    if args.lowp_mode == "INT8":
        lowp_mode = ipex.quantization.WoqLowpMode.INT8
    elif args.lowp_mode == "FP32":
        lowp_mode = ipex.quantization.WoqLowpMode.NONE
    elif args.lowp_mode == "FP16":
        lowp_mode = ipex.quantization.WoqLowpMode.FP16
    elif args.lowp_mode == "BF16":
        lowp_mode = ipex.quantization.WoqLowpMode.BF16
    else:  # AUTO
        if args.low_precision_checkpoint != "" or weight_dtype == WoqWeightDtype.INT4:
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
        if args.gptq_legacy_format:
            config_dict = (
                ipex.utils.weight_only_quantization._legacy_lowp_checkpoint_config()
            )
            low_precision_checkpoint = (low_precision_checkpoint, config_dict)
    else:
        low_precision_checkpoint = None

    model = ipex.llm.optimize(
        model.eval(),
        dtype=amp_dtype,
        quantization_config=qconfig,
        inplace=True,
        cache_weight_for_large_batch=args.cache_weight_for_large_batch,
    )

if args.torch_compile:
    if args.deployment_mode:
        raise SystemExit(
            "[ERROR] deployment_mode cannot co-work with torch.compile, please set deployment_mode"
            " to False if want to use torch.compile."
        )
    model.forward = torch.compile(model.forward, dynamic=True, backend=args.backend)


if args.benchmark:
    if args.token_latency:
        if not hasattr(model.config, "token_latency"):
            model.config.token_latency = True
    if model_type == "git":
        prompt = Image.open(requests.get(args.image_url, stream=True).raw)
        generate_kwargs.pop("min_new_tokens", None)
    elif model_type == "llava":
        if args.prompt is not None:
            prompt = args.prompt
        image = load_image(args.image_url)
        image = [image] * args.batch_size
        if model.config.mm_use_im_start_end:
            prompt = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + prompt
            )
        else:
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    else:
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
        # elif int(args.input_tokens) > 8192:
        #     prompt = prompt_pool[model_type]["8192"] * int(int(args.input_tokens) / 8192)
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
                    if model_type == "llava":
                        input_ids = torch.stack(
                            [
                                tokenizer_image_token(
                                    pmt,
                                    tokenizer,
                                    IMAGE_TOKEN_INDEX,
                                    return_tensors="pt",
                                )
                                for pmt in prompt
                            ]
                        )
                        image_tensor = [
                            image_processor.preprocess(img, return_tensors="pt")[
                                "pixel_values"
                            ].to(amp_dtype)
                            for img in image
                        ]
                        output = model.generate(
                            input_ids, images=image_tensor, **generate_kwargs
                        )
                    elif model_type == "git":
                        input_ids = tokenizer(
                            images=prompt, return_tensors="pt"
                        ).pixel_values
                        output = model.generate(
                            pixel_values=input_ids, **generate_kwargs
                        )
                    else:
                        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                        output = model.generate(input_ids, **generate_kwargs)
                    prof.step()
        for i in range(num_iter):
            tic = time.time()
            if model_type == "llava":
                input_ids = torch.stack(
                    [
                        tokenizer_image_token(
                            pmt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                        )
                        for pmt in prompt
                    ]
                )
                image_tensor = [
                    image_processor.preprocess(img, return_tensors="pt")[
                        "pixel_values"
                    ].to(amp_dtype)
                    for img in image
                ]
                output = model.generate(
                    input_ids, images=image_tensor, **generate_kwargs
                )
            elif model_type == "git":
                input_ids = tokenizer(images=prompt, return_tensors="pt").pixel_values
                output = model.generate(pixel_values=input_ids, **generate_kwargs)
            else:
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                output = model.generate(input_ids, **generate_kwargs)
            gen_ids = output[0] if args.token_latency else output
            gen_text = tokenizer.batch_decode(
                gen_ids[:, input_ids.shape[1] :] if model_type == "llava" else gen_ids,
                skip_special_tokens=True,
            )

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
    latency = total_time / (num_iter - num_warmup) * 1000
    print("Inference latency: %.2f ms." % latency)

    if args.token_latency:
        import numpy as np
        from itertools import chain

        first_latency = np.mean([x[0] for x in total_list]) * 1000
        average_2n = list(chain(*[x[1:] for x in total_list]))
        average_2n.sort()
        average_2n_latency = np.mean(average_2n) * 1000
        p90_latency = average_2n[int(len(average_2n) * 0.9)] * 1000
        p99_latency = average_2n[int(len(average_2n) * 0.99)] * 1000
        print("First token average latency: %.2f ms." % first_latency)
        print("Average 2... latency: %.2f ms." % average_2n_latency)
        print("P90 2... latency: %.2f ms." % p90_latency)
        print("P99 2... latency: %.2f ms." % p99_latency)
