import torch
import time
import json
import pathlib
import argparse
import re

from transformers import AutoConfig, TextStreamer

import sys

sys.path.append(sys.path[0] + "/../../../")

from llm.inference.utils.supported_models import MODEL_CLASSES
import logging

logger = logging.getLogger(__name__)

try:
    from llava.model.builder import load_pretrained_model
    from llava.conversation import conv_templates
    from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
    from llava.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
    )

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
    "--audio",
    default="example.flac",
    type=str,
    help="audio file for speech-to-text task",
)
parser.add_argument(
    "--config-file", default=None, type=str, help="specific configuration file"
)
parser.add_argument("--greedy", action="store_true")
parser.add_argument("--ipex", action="store_true")
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
    "--cache-weight-for-large-batch",
    action="store_true",
    help="Cache an extra linear weight for large batch inference, such as the first token (prefill phase)."
    " It brings better performance at the cost of higher memory usage. It is only valid for dtype=bfloat16."
    " Otherwise, it has no effect.",
)
parser.add_argument(
    "--vision-text-model",
    action="store_true",
    help="whether or not it is vision-text multi-model structure",
)
parser.add_argument(
    "--kv-cache-dtype",
    type=str,
    choices=[
        "auto",
        "fp8_e5m2",
    ],
    default="auto",
    help='Data type for kv cache storage. If "auto", will use model '
    "data type. fp8 type now supports e5m2.",
)
parser.add_argument(
    "--input-mode",
    default="0",
    choices=["0", "1", "2", "3"],
    type=str,
    help="Input mode for multimodal models. 0: language; 1: vision; 2: speech; 3: vision_speech",
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
if model_type == "llama" and args.vision_text_model:
    model_type = "mllama"
if model_type in ["maira-2", "deepseek-v2", "deepseek-v3"]:
    model_type = model_type.replace("-", "")
model_class = MODEL_CLASSES[model_type]
if args.config_file is None:
    if model_type == "chatglm":
        # chatglm modeling is from remote hub and its torch_dtype in config.json need to be overrided
        config = AutoConfig.from_pretrained(
            args.model_id,
            torchscript=args.deployment_mode,
            trust_remote_code=True,
            torch_dtype=amp_dtype,
        )
    else:
        config = AutoConfig.from_pretrained(
            args.model_id,
            torchscript=args.deployment_mode,
            trust_remote_code=True,
        )
else:
    config = AutoConfig.from_pretrained(
        args.config_file,
        torchscript=args.deployment_mode,
        trust_remote_code=True,
        torch_dtype=amp_dtype,
    )

if args.kv_cache_dtype == "auto":
    kv_cache_dtype = None
elif args.kv_cache_dtype == "fp8_e5m2":
    kv_cache_dtype = torch.float8_e5m2
config.kv_cache_dtype = kv_cache_dtype

if not hasattr(config, "text_max_length") and args.prompt is None:
    config.text_max_length = int(args.input_tokens) + int(args.max_new_tokens)
if model_type == "mpt" and args.prompt is None:
    config.max_seq_len = int(args.input_tokens) + int(args.max_new_tokens)
if model_type == "whisper":
    config.text_max_length = config.max_source_positions + config.max_target_positions
if model_type == "jamba":
    config.use_mamba_kernels = False

if not hasattr(config, "lm_head_generation"):
    config.lm_head_generation = True
if model_type == "maira2" and not hasattr(config.text_config, "lm_head_generation"):
    config.text_config.lm_head_generation = True
if re.search("phi4mm", config.architectures[0], re.IGNORECASE):
    model_type = "phi4mm"
    model_class = MODEL_CLASSES[model_type]
    prompt = args.prompt
    _COMPATIBLE_IMAGE_SPECIAL_TOKEN_PATTERN = r"<\|image_\d+\|>"
    _COMPATIBLE_AUDIO_SPECIAL_TOKEN_PATTERN = r"<\|audio_\d+\|>"
    image_in_prompt = len(re.findall(_COMPATIBLE_IMAGE_SPECIAL_TOKEN_PATTERN, prompt))
    audio_in_prompt = len(re.findall(_COMPATIBLE_AUDIO_SPECIAL_TOKEN_PATTERN, prompt))
    is_vision = image_in_prompt > 0
    is_speech = audio_in_prompt > 0
    audio_batch_size = args.batch_size
    if is_vision:
        assert (
            image_in_prompt == args.batch_size
        ), "Prompt is invalid. For multiple images, the user needs to insert multiple image placeholders in the prompt as below: \
            <|user|><|image_1|><|image_2|><|image_3|>Summarize the content of the images.<|end|><|assistant|>"
    if is_speech:
        if not is_vision:
            assert (
                audio_in_prompt == args.batch_size
            ), "Prompt is invalid. For multiple audios, the user needs to insert multiple audio placeholders in the prompt as below: \
                <|user|><|audio_1|><|audio_2|><|audio_3|>Transcribe the audio clip into text.<|end|><|assistant|>"
        else:
            audio_batch_size = audio_in_prompt
    if not is_vision and not is_speech:
        config.input_mode = 0
    elif is_vision and not is_speech:
        config.input_mode = 1
    elif not is_vision and is_speech:
        config.input_mode = 2
    else:
        config.input_mode = 3

    assert config.input_mode == int(
        args.input_mode
    ), "Input mode in prompt is not consistent with the input mode in the command line."
if model_type != "llava":
    config._attn_implementation = "eager"
    model = model_class[0].from_pretrained(
        args.model_id,
        torch_dtype=amp_dtype,
        config=config,
        low_cpu_mem_usage=True if model_type != "maira2" else False,
        trust_remote_code=True,
        attn_implementation="eager",
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

elif re.search("mllama", model.config.architectures[0], re.IGNORECASE) or re.search(
    "phi4mm", model.config.architectures[0], re.IGNORECASE
):
    from PIL import Image

    def load_image(image_file):
        if image_file.startswith("http://") or image_file.startswith("https://"):
            import requests

            raw_image = Image.open(requests.get(args.image_url, stream=True).raw)
        else:
            raw_image = Image.open(image_file)
        return raw_image

elif re.search("maira2", model.config.architectures[0], re.IGNORECASE):
    from PIL import Image
    import requests

    def download_and_open(url: str) -> Image.Image:
        response = requests.get(url, headers={"User-Agent": "MAIRA-2"}, stream=True)
        return Image.open(response.raw)


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
if re.search("yuan", model.config.architectures[0], re.IGNORECASE) or re.search(
    "jamba", model.config.architectures[0], re.IGNORECASE
):
    model.config.batch_size = int(args.batch_size) * num_beams
if re.search("phi4mm", model.config.architectures[0], re.IGNORECASE):
    model.config.batch_size = int(args.batch_size) * num_beams
    model.config.audio_batch_size = audio_batch_size * num_beams
if re.search("whisper", model.config.architectures[0], re.IGNORECASE):
    import librosa

    sample = librosa.load(args.audio, sr=16000)
if re.search("phi4mm", model.config.architectures[0], re.IGNORECASE):
    if config.input_mode in [2, 3]:
        import soundfile

        sample = soundfile.read(args.audio)
    else:
        sample = None


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
if args.torch_compile:
    if args.deployment_mode:
        raise SystemExit(
            "[ERROR] deployment_mode cannot co-work with torch.compile, please set deployment_mode"
            " to False if want to use torch.compile."
        )
    model.forward = torch.compile(model.forward, dynamic=True, backend=args.backend)


if args.benchmark:
    if args.token_latency and not args.ipex:
        args.token_latency = False
        logger.warning("--token-latency requires --ipex. Disabling --token-latency.")
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
    elif model_type == "whisper":
        prompt = sample[0]
        generate_kwargs.pop("min_new_tokens", None)
    elif model_type == "maira2":
        prompt = args.prompt
        sample = download_and_open(args.image_url)
        process_input_func = (
            tokenizer.process_reporting_input
            if hasattr(tokenizer, "process_reporting_input")
            else tokenizer.format_and_preprocess_reporting_input
        )
    elif model_type == "phi4mm":
        prompt = args.prompt
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
        if model_type == "mllama":
            raw_image = load_image(args.image_url)
            raw_image = [raw_image] * args.batch_size
            inputs = tokenizer(raw_image, prompt, return_tensors="pt")
            input_size = inputs["input_ids"].size(dim=1)
        else:
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
            elif model_type == "whisper":
                input_ids = tokenizer(
                    prompt, sampling_rate=16000, return_tensors="pt"
                ).input_features
                output = model.generate(input_ids, **generate_kwargs)
            elif model_type == "mllama":
                raw_image = load_image(args.image_url)
                raw_image = [raw_image] * args.batch_size
                inputs = tokenizer(raw_image, prompt, return_tensors="pt")
                input_ids = inputs["input_ids"]
                output = model.generate(**inputs, **generate_kwargs)
            elif model_type == "maira2":
                processed_inputs = process_input_func(
                    current_frontal=sample,
                    current_lateral=None,
                    prior_frontal=None,
                    indication=None,
                    technique=None,
                    comparison=None,
                    prior_report=None,
                    return_tensors="pt",
                    get_grounding=False,
                )
                input_ids = processed_inputs["input_ids"]
                output = model.generate(**processed_inputs, **generate_kwargs)
            elif model_type == "phi4mm":
                raw_image = load_image(args.image_url) if is_vision else None
                raw_image = [raw_image] * args.batch_size
                samples = [sample] * audio_batch_size
                inputs = tokenizer(
                    text=prompt[0],
                    images=raw_image if is_vision else None,
                    audios=samples if is_speech else None,
                    return_tensors="pt",
                )
                input_ids = inputs["input_ids"]
                output = model.generate(**inputs, **generate_kwargs)
            else:
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                output = model.generate(input_ids, **generate_kwargs)
            gen_ids = output[0] if args.token_latency else output
            gen_text = tokenizer.batch_decode(
                (
                    gen_ids[:, input_ids.shape[1] :]
                    if model_type in ["llava", "maira2", "phi4mm"]
                    else gen_ids
                ),
                skip_special_tokens=True,
            )
            toc = time.time()
            input_tokens_lengths = [x.shape[0] for x in input_ids]
            output_tokens_lengths = [x.shape[0] for x in gen_ids]
            total_new_tokens = [
                o if model.config.model_type in ["t5", "whisper"] else o - i
                for i, o in zip(input_tokens_lengths, output_tokens_lengths)
            ]
            print(gen_text, total_new_tokens, flush=True)
            print("Iteration: %d, Time: %.6f sec" % (i, toc - tic), flush=True)
            if i >= num_warmup:
                total_time += toc - tic
                if args.token_latency:
                    total_list.append(output[1])

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
                    elif model_type == "whisper":
                        input_ids = tokenizer(
                            prompt, sampling_rate=16000, return_tensors="pt"
                        ).input_features
                        output = model.generate(input_ids, **generate_kwargs)
                    elif model_type == "mllama":
                        raw_image = [load_image(args.image_url)] * args.batch_size
                        inputs = tokenizer(raw_image, prompt, return_tensors="pt")
                        output = model.generate(**inputs, **generate_kwargs)
                    elif model_type == "maira2":
                        processed_inputs = process_input_func(
                            current_frontal=sample,
                            current_lateral=None,
                            prior_frontal=None,
                            indication=None,
                            technique=None,
                            comparison=None,
                            prior_report=None,
                            return_tensors="pt",
                            get_grounding=False,
                        )
                        output = model.generate(**processed_inputs, **generate_kwargs)
                    elif model_type == "phi4mm":
                        raw_image = load_image(args.image_url) if is_vision else None
                        raw_image = [raw_image] * args.batch_size
                        samples = [sample] * audio_batch_size
                        inputs = tokenizer(
                            text=prompt[0],
                            images=raw_image if is_vision else None,
                            audios=samples if is_speech else None,
                            return_tensors="pt",
                        )
                        output = model.generate(**inputs, **generate_kwargs)
                    else:
                        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                        output = model.generate(input_ids, **generate_kwargs)
                    prof.step()

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
