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
from transformers import TextStreamer
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.llm.utils import (
    load_low_precision_checkpoint,
)
from ast import literal_eval
import sys
import os

sys.path.append(sys.path[0] + "/../../../")

import logging

logger = logging.getLogger(__name__)


from llm.inference.utils.model_class.llm import EXAMPLE_INPUTS_MODE
from llm.inference.utils.model_class.llama import LLAMAConfig
from llm.inference.utils.model_class.mllama import MLLAMAConfig
from llm.inference.utils.model_class.gptj import GPTJConfig
from llm.inference.utils.model_class.gptneox import GPTNEOXConfig
from llm.inference.utils.model_class.falcon import FALCONConfig
from llm.inference.utils.model_class.opt import OPTConfig
from llm.inference.utils.model_class.bloom import BloomConfig
from llm.inference.utils.model_class.codegen import CodeGenConfig
from llm.inference.utils.model_class.baichuan import BaichuanConfig
from llm.inference.utils.model_class.chatglm import ChatGLMConfig
from llm.inference.utils.model_class.gptbigcode import GPTJBigCodeConfig
from llm.inference.utils.model_class.t5 import T5Config
from llm.inference.utils.model_class.mistral import MistralConfig
from llm.inference.utils.model_class.mixtral import MixtralConfig
from llm.inference.utils.model_class.mpt import MPTConfig
from llm.inference.utils.model_class.stablelm import StableLMConfig
from llm.inference.utils.model_class.qwen3 import Qwen3Config
from llm.inference.utils.model_class.qwen3moe import Qwen3moeConfig
from llm.inference.utils.model_class.qwen import QwenConfig
from llm.inference.utils.model_class.qwen2 import Qwen2Config
from llm.inference.utils.model_class.git import GitConfig
from llm.inference.utils.model_class.llava import LlavaConfig
from llm.inference.utils.model_class.phi import PhiConfig
from llm.inference.utils.model_class.phi import Phi3Config
from llm.inference.utils.model_class.phi import Phi4MMConfig
from llm.inference.utils.model_class.yuan import YuanConfig
from llm.inference.utils.model_class.whisper import WhisperConfig
from llm.inference.utils.model_class.maira2 import MAIRA2Config
from llm.inference.utils.model_class.jamba import JambaConfig
from llm.inference.utils.model_class.deepseek import DeepseekV2Config, DeepseekV3Config


def str_to_kwargs(s):
    return dict(
        (k, float(v) if "." in v else int(v))
        for k, v in (item.split("=") for item in s.split(","))
    )


parser = argparse.ArgumentParser("LLM generation script (int8 path)", add_help=False)
parser.add_argument(
    "-m", "--model-id", default=None, type=str, required=True, help="your llm model"
)
parser.add_argument(
    "--vision-text-model",
    action="store_true",
    help="[deprecated] whether it is vision-text multi-model structure",
)
parser.add_argument(
    "--max-new-tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument(
    "--streaming",
    action="store_true",
    help="enable streaming mode for generation output (greedy search only)",
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
parser.add_argument(
    "--quant-with-amp",
    action="store_true",
    help="by default it is int8-fp32 mixed, to enable int8 mixed amp bf16 (work on platforms like SPR)",
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
    "--qconfig-summary-file", default="", help="qconfig for static quantization"
)
parser.add_argument("--quantized-model-path", default="./saved_results/best_model.pt")
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--input-tokens", default="32", type=str)
parser.add_argument("--prompt", default=None, type=str)
parser.add_argument("--num-iter", default=100, type=int, help="num iter")
parser.add_argument("--num-warmup", default=10, type=int, help="num warmup")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
parser.add_argument(
    "--calib-len",
    default=512,
    type=int,
    help="calibration dataset max or padding max length for SmoothQuant autotuning",
)
parser.add_argument(
    "--calib-iters",
    default=512,
    type=int,
    help="calibration iters for SmoothQuant autotuning",
)
parser.add_argument(
    "--calib-shuffle",
    action="store_true",
    help="whether to shuffle on calibration dataset for SmoothQuant autotuning",
)
parser.add_argument(
    "--calib-padding",
    action="store_true",
    help="whether to pad on calibration dataset for SmoothQuant autotuning",
)
parser.add_argument(
    "--calib-pad-val",
    default=1,
    type=int,
    help="calibration dataset padding value for SmoothQuant autotuning",
)
parser.add_argument(
    "--fallback-add",
    action="store_true",
    help="whether to fallback add ops to fp32 for SmoothQuant autotuning",
)
parser.add_argument("--alpha", default=0.5, help="alpha value for smoothquant")
parser.add_argument(
    "--folding", action="store_true", help="whether to fold mul into the previous layer"
)
parser.add_argument(
    "--init-alpha",
    default=0.5,
    type=float,
    help="a value to get baseline quantization error for auto-tuning",
)
parser.add_argument(
    "--alpha-min",
    default=0.0,
    type=float,
    help="min value of auto-tuning alpha search space",
)
parser.add_argument(
    "--alpha-max",
    default=1.0,
    type=float,
    help="max value of auto-tuning alpha search space",
)
parser.add_argument(
    "--alpha-step",
    default=0.1,
    type=float,
    help="step_size of auto-tuning alpha search space",
)
parser.add_argument(
    "--shared-criterion",
    choices=["min", "mean", "max"],
    default="max",
    type=str,
    help="criterion for input LayerNorm op of a transformer block",
)
parser.add_argument(
    "--enable-blockwise-loss",
    action="store_true",
    help="whether to enable block-wise auto-tuning",
)
parser.add_argument("--token-latency", action="store_true")
parser.add_argument("--greedy", action="store_true")
parser.add_argument("--sample", action="store_true")
parser.add_argument("--enable-thinking", action="store_true")
parser.add_argument(
    "--generation-config",
    type=str_to_kwargs,
    default="temperature=0.9",
    help="generation config",
)
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
    choices=["INT8", "INT4", "NF4", "FP8"],
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
    help="Low precision checkpoint file generated by algorithms, such as GPTQ. It contains"
    " INT4 weights, scales, zero points, etc. For better accuracy of weight only"
    " quantization with INT4 weight.",
)
parser.add_argument(
    "--act-quant-mode",
    choices=[
        "PER_TENSOR",
        "PER_IC_BLOCK",
        "PER_BATCH",
        "PER_BATCH_IC_BLOCK",
        "PER_TENSOR_SYM",
        "PER_IC_BLOCK_SYM",
        "PER_BATCH_SYM",
        "PER_BATCH_IC_BLOCK_SYM",
    ],
    default="PER_BATCH_IC_BLOCK_SYM",
    type=str,
    help="Quantization mode for activation with different granularity. "
    "For lowp-mode=INT8 only. For other cases, it has no effect. "
    "Assume the activation tensor has shape batch_size x input_channel. "
    "PER_TENSOR(0): quantize per tensor; "
    "PER_IC_BLOCK(1): quantize per group along IC with group size = IC_BLOCK; "
    "PER_BATCH(2): quantize per batch; "
    "PER_BATCH_IC_BLOCK(3): quantize per block of size 1 x IC_BLOCK. "
    "PER_TENSOR_SYM(4): symmetrically quantize per tensor; "
    "PER_IC_BLOCK_SYM(5): symmetrically quantize per group along IC with group size = IC_BLOCK; "
    "PER_BATCH_SYM(6): symmetrically quantize per batch; "
    "PER_BATCH_IC_BLOCK_SYM(7): symmetrically quantize per block of size 1 x IC_BLOCK. "
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
    " It brings better performance at the cost of higher memory usage. It is only valid for weight-only"
    " quantization with lowp-mode=BF16. Otherwise, it has no effect.",
)
parser.add_argument(
    "--lm-head-generation",
    action="store_true",
    help="Compute lm-head only for the last token in the sequence to speed up first token inference."
    " This feature is not compatible with lambada_openai accuracy test. If you want to run"
    " lambada_openai accuracy test with the quantized model afterwards, don't turn this feature on.",
)
parser.add_argument(
    "--woq-sym-quant-weight",
    action="store_true",
    help="Quantize weight symmetrically for weight only quantization. It usually brings better latency at"
    " the cost of accuracy. It has not effect if you are loading low-precision checkpoints.",
)
parser.add_argument(
    "--verbose",
    action="store_true",
    help="Print verbose information for debugging",
)
parser.add_argument(
    "--input-mode",
    default="0",
    choices=["0", "1", "2", "3"],
    type=str,
    help="Input mode for multimodal models. 0: language; 1: vision; 2: speech; 3: vision_speech",
)
args = parser.parse_args()

if args.verbose:
    logger.setLevel(logging.DEBUG)
    ipex.set_logging_level(logging.DEBUG)

if args.vision_text_model:
    logger.warning(
        "'--vision-text-model' flag is deprecated. Please set '--input-mode 1' instead."
    )
    args.input_mode = "1"

# disable
try:
    ipex._C.disable_jit_linear_repack()
    torch._C._jit_set_texpr_fuser_enabled(False)
except Exception:
    pass

# amp autocast
if args.quant_with_amp:
    amp_enabled = True
    amp_dtype = torch.bfloat16
else:
    amp_enabled = False
    amp_dtype = torch.float32


if args.config_file is None:
    if "chatglm" in args.model_id.lower():
        # chatglm modeling is from remote hub and its torch_dtype in config.json need to be overrided
        config = AutoConfig.from_pretrained(
            args.model_id,
            torchscript=True,
            trust_remote_code=True,
            torch_dtype=torch.float,
        )
    else:
        config = AutoConfig.from_pretrained(
            args.model_id,
            torchscript=True,
            trust_remote_code=True,
        )
else:
    config = AutoConfig.from_pretrained(
        args.config_file, torchscript=True, trust_remote_code=True
    )

config.use_cache = True  # For inference, it should always be True

# For DeepSeek models
if args.ipex_weight_only_quantization and args.weight_dtype == "INT8":
    config.use_fused_moe = True
    config.use_fused_moe_woq = True

if re.search("falcon", config.architectures[0], re.IGNORECASE) or re.search(
    "rw", config.architectures[0], re.IGNORECASE
):
    model = FALCONConfig(args.model_id)
elif re.search("GPTJ", config.architectures[0], re.IGNORECASE):
    model = GPTJConfig(args.model_id)
elif re.search("llama", config.architectures[0], re.IGNORECASE) and not re.search(
    "llava", config.architectures[0], re.IGNORECASE
):

    if args.input_mode == "1":
        model = MLLAMAConfig(args.model_id)
        from PIL import Image

        def load_image(image_file):
            if image_file.startswith("http://") or image_file.startswith("https://"):
                import requests

                raw_image = Image.open(requests.get(image_file, stream=True).raw)
            else:
                raw_image = Image.open(image_file)
            return raw_image

    else:
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
    model = T5Config(args.model_id)
elif re.search("mistral", config.architectures[0], re.IGNORECASE):
    model = MistralConfig(args.model_id)
elif re.search("mpt", config.architectures[0], re.IGNORECASE):
    model = MPTConfig(args.model_id)
elif re.search("mixtral", config.architectures[0], re.IGNORECASE):
    model = MixtralConfig(args.model_id)
elif re.search("stablelm", config.architectures[0], re.IGNORECASE):
    model = StableLMConfig(args.model_id)
elif re.search("qwen3moe", config.architectures[0], re.IGNORECASE):
    model = Qwen3moeConfig(args.model_id)
elif re.search("qwen3", config.architectures[0], re.IGNORECASE):
    model = Qwen3Config(args.model_id)
elif re.search("qwen", config.architectures[0], re.IGNORECASE):
    if re.search("qwen2", config.architectures[0], re.IGNORECASE):
        model = Qwen2Config(args.model_id)
    else:
        model = QwenConfig(args.model_id)
elif re.search("git", config.architectures[0], re.IGNORECASE):
    from PIL import Image
    import requests

    model = GitConfig(args.model_id)
elif re.search("llava", config.architectures[0], re.IGNORECASE):
    from PIL import Image
    import requests
    from io import BytesIO

    try:
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
    model = LlavaConfig(args.model_id)

    def load_image(image_file):
        if image_file.startswith("http://") or image_file.startswith("https://"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image

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
elif re.search("phi3", config.architectures[0], re.IGNORECASE):
    model = Phi3Config(args.model_id)
elif re.search("phi4mm", config.architectures[0], re.IGNORECASE):
    model = Phi4MMConfig(args.model_id)
    from PIL import Image
    import soundfile

    def load_image(image_file):
        if image_file.startswith("http://") or image_file.startswith("https://"):
            import requests

            raw_image = Image.open(requests.get(image_file, stream=True).raw)
        else:
            raw_image = Image.open(image_file)
        return raw_image

    audio_batch_size = args.batch_size
    if args.prompt:
        prompt = args.prompt
        _COMPATIBLE_IMAGE_SPECIAL_TOKEN_PATTERN = r"<\|image_\d+\|>"
        _COMPATIBLE_AUDIO_SPECIAL_TOKEN_PATTERN = r"<\|audio_\d+\|>"
        image_in_prompt = len(
            re.findall(_COMPATIBLE_IMAGE_SPECIAL_TOKEN_PATTERN, prompt)
        )
        audio_in_prompt = len(
            re.findall(_COMPATIBLE_AUDIO_SPECIAL_TOKEN_PATTERN, prompt)
        )
        is_vision = image_in_prompt > 0
        is_speech = audio_in_prompt > 0
        if is_vision:
            assert (
                image_in_prompt == args.batch_size
            ), "Prompt is invalid. For multiple images, the user needs to \
                insert multiple image placeholders in the prompt as below: \
                <|user|><|image_1|><|image_2|><|image_3|>Summarize the content of the images.<|end|><|assistant|>"
        if is_speech:
            if not is_vision:
                assert (
                    audio_in_prompt == args.batch_size
                ), "Prompt is invalid. For multiple audios, the user needs to \
                    insert multiple audio placeholders in the prompt as below: \
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
    else:
        config.input_mode = int(args.input_mode)
        if config.input_mode == 3:
            audio_batch_size = 1
elif re.search("phi", config.architectures[0], re.IGNORECASE):
    model = PhiConfig(args.model_id)
elif re.search("yuan", config.architectures[0], re.IGNORECASE):
    model = YuanConfig(args.model_id)
elif re.search("whisper", config.architectures[0], re.IGNORECASE):
    import librosa

    model = WhisperConfig(args.model_id)
elif re.search("maira2", config.architectures[0], re.IGNORECASE):
    from PIL import Image
    import requests

    def download_and_open(url: str) -> Image.Image:
        response = requests.get(url, headers={"User-Agent": "MAIRA-2"}, stream=True)
        return Image.open(response.raw)

    model = MAIRA2Config(args.model_id)
elif re.search("jamba", config.architectures[0], re.IGNORECASE):
    model = JambaConfig(args.model_id)
elif re.search("deepseekv2", config.architectures[0], re.IGNORECASE):
    model = DeepseekV2Config(args.model_id)
elif re.search("deepseekv3", config.architectures[0], re.IGNORECASE):
    model = DeepseekV3Config(args.model_id)
    if "deepseek-r1" in args.model_id.lower() or "deepseekr1" in args.model_id.lower():
        model.name = "deepseekr1"
else:
    raise AssertionError("Not support %s." % (args.model_id))

num_beams = 1 if args.greedy else 4
if (
    not hasattr(config, "text_max_length")
    and args.prompt is None
    and model.name not in ["t5"]
):
    if not args.benchmark:
        if hasattr(config, "max_position_embeddings"):
            config.text_max_length = config.max_position_embeddings
        else:
            config.text_max_length = 2048
    else:
        config.text_max_length = int(args.input_tokens) + int(args.max_new_tokens)
if model.name == "mpt" and args.prompt is None:
    max_seq_len = int(args.input_tokens) + int(args.max_new_tokens)
    if hasattr(config, "max_seq_len") and config.max_seq_len > max_seq_len:
        max_seq_len = config.max_seq_len
    config.max_seq_len = max_seq_len
if model.name in ["git", "llava", "jamba"]:
    config.batch_size = int(args.batch_size) * num_beams
if model.name == "phi4mm":
    config.batch_size = int(args.batch_size) * num_beams
    config.audio_batch_size = audio_batch_size * num_beams
if model.name == "whisper":
    config.text_max_length = config.max_source_positions + config.max_target_positions

if args.lm_head_generation and not hasattr(config, "lm_head_generation"):
    config.lm_head_generation = True
if model.name == "maira2" and not hasattr(config.text_config, "lm_head_generation"):
    config.text_config.lm_head_generation = True

# Users gives the model by path to int4 checkpoint directory
if (
    not args.low_precision_checkpoint
    and hasattr(config, "quantization_config")
    and os.path.isdir(args.model_id)
):
    args.low_precision_checkpoint = args.model_id

load_to_meta_device = args.benchmark or (
    args.ipex_weight_only_quantization and args.low_precision_checkpoint != ""
)
logger.debug(f"get user model with load_to_meta_device = {load_to_meta_device}")
user_model = model.get_user_model(config, load_to_meta_device)
logger.debug("get user model done")

tokenizer = model.get_tokenizer()
print("Data type of the model:", user_model.dtype)
streamer = None
if args.streaming:
    if num_beams != 1 or args.batch_size != 1:
        print(
            "--streaming only supported in greedy search mode (--greedy) with --batch-size 1. Disabling streaming output."
        )
    else:
        streamer = TextStreamer(tokenizer)

generate_kwargs = dict(
    do_sample=True if args.sample else False,
    num_beams=num_beams,
    max_new_tokens=args.max_new_tokens,
    min_new_tokens=args.max_new_tokens,
    streamer=streamer,
    **args.generation_config,
)
if re.search("t5", config.architectures[0], re.IGNORECASE):
    generate_kwargs["max_length"] = generate_kwargs["max_new_tokens"]
    generate_kwargs.pop("max_new_tokens")
elif re.search("git", config.architectures[0], re.IGNORECASE):
    generate_kwargs.pop("min_new_tokens")

if model.to_channels_last:
    user_model = user_model.to(memory_format=torch.channels_last)
user_model.eval()

# dummy past key value
beam_idx_tmp = torch.zeros(
    (2048, int(args.batch_size * num_beams)), dtype=torch.long
).contiguous()


def _get_target_nums(names):
    for n in names:
        if hasattr(user_model.config, n):
            return getattr(user_model.config, n)
    print(f"Not found target {names[0]}")
    exit(0)


if model.name not in ["mllama", "maira2"]:
    num_heads_names = ["num_attention_heads", "n_head", "num_heads", "n_heads"]
    num_layers_names = ["num_hidden_layers", "n_layer", "num_layers", "n_layers"]
    hidden_size_names = ["hidden_size", "n_embd"]
    n_heads = _get_target_nums(num_heads_names)
    n_layers = _get_target_nums(num_layers_names)
    hidden_size = _get_target_nums(hidden_size_names)
    head_dim = int(hidden_size / n_heads)
    global_past_key_value = [
        (
            torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
            torch.zeros([1, n_heads, 1, head_dim]).contiguous(),
            torch.zeros([1, n_heads, 1, head_dim]).contiguous(),
            beam_idx_tmp,
        )
        for i in range(n_layers)
    ]
if model.name == "yuan":
    global_past_key_value = tuple(
        [
            (
                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                torch.zeros([1, 1, 1, 1]).contiguous(),
                torch.zeros([1, 1, 1, 1]).contiguous(),
                torch.zeros(1, 4, dtype=torch.long),
                torch.zeros(1, 1, 2, hidden_size),
            )
            for i in range(n_layers)
        ]
    )
if model.name == "mllama":
    head_dim = user_model.config.text_config.hidden_size // (
        user_model.config.text_config.num_hidden_layers
        - len(user_model.config.text_config.cross_attention_layers)
    )
    global_past_key_value = tuple(
        [
            (
                (
                    torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                    torch.zeros([1, 1, 1, 1]).contiguous(),
                    torch.zeros([1, 1, 1, 1]).contiguous(),
                    torch.zeros(1, 4, dtype=torch.long),
                )
                if i not in user_model.config.text_config.cross_attention_layers
                else (
                    torch.zeros([1, 1, 1, head_dim]).contiguous(),
                    torch.zeros([1, 1, 1, head_dim]).contiguous(),
                )
            )
            for i in range(user_model.config.text_config.num_hidden_layers)
        ]
    )
if model.name == "maira2":
    global_past_key_value = tuple(
        [
            (
                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                torch.zeros([1, 1, 1, 1]).contiguous(),
                torch.zeros([1, 1, 1, 1]).contiguous(),
                torch.zeros(1, 4, dtype=torch.long),
            )
            for i in range(user_model.config.text_config.num_hidden_layers)
        ]
    )
if model.name == "jamba":
    intermediate_size = user_model.config.mamba_expand * user_model.config.hidden_size
    conv_kernel_size = user_model.config.mamba_d_conv
    ssm_state_size = user_model.config.mamba_d_state
    user_model.config.dtype = amp_dtype
    global_past_key_value = tuple(
        [
            (
                (
                    torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                    torch.zeros([1, 1, 1, 1]).contiguous(),
                    torch.zeros([1, 1, 1, 1]).contiguous(),
                    torch.zeros(1, 4, dtype=torch.long),
                )
                if i % user_model.config.attn_layer_period
                == user_model.config.attn_layer_offset
                else (
                    torch.zeros(
                        int(args.batch_size * num_beams),
                        intermediate_size,
                        ssm_state_size,
                        dtype=amp_dtype,
                    ).contiguous(),
                    torch.zeros(
                        int(args.batch_size * num_beams),
                        intermediate_size,
                        conv_kernel_size,
                        dtype=amp_dtype,
                    ).contiguous(),
                    torch.tensor(False).contiguous(),
                )
            )
            for i in range(user_model.config.num_hidden_layers)
        ]
    )
if model.name in ["deepseekv2", "deepseekv3", "deepseekr1"]:
    global_past_key_value = tuple(
        [
            (
                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                torch.zeros([1, 1, 1, 1]).contiguous(),
                torch.zeros(1, 4, dtype=torch.long),
            )
            for i in range(user_model.config.num_hidden_layers)
        ]
    )


def get_example_inputs(model):
    if model.use_global_past_key_value:
        global global_past_key_value
    example_inputs = None
    input_ids = torch.ones(32).to(torch.long)
    attention_mask = torch.ones(len(input_ids))
    if model.example_inputs_mode == EXAMPLE_INPUTS_MODE.MASK_POS_KV:
        position_ids = torch.arange(len(input_ids))
        if model.name == "yuan":
            example_inputs = (
                input_ids.unsqueeze(0)[:, -1:],
                attention_mask.unsqueeze(0)[:, -1:],
                position_ids.unsqueeze(0)[:, -1:],
                tuple(global_past_key_value),
            )
        elif model.name == "maira2":
            input_ids = torch.ones(1448).to(torch.long).unsqueeze(0)
            input_ids[:, 31:1400] = user_model.config.image_token_index
            attention_mask = torch.ones_like(input_ids)
            position_ids = torch.arange(input_ids.shape[-1]).unsqueeze(0)
            example_inputs = (
                input_ids,
                attention_mask,
                position_ids,
                tuple(global_past_key_value),
            )
        elif model.name == "jamba":
            example_inputs = (
                input_ids.unsqueeze(0),
                attention_mask.unsqueeze(0),
                position_ids.unsqueeze(0),
                tuple(global_past_key_value),
                torch.tensor(False),
                torch.tensor(1),
            )
        else:
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
        if model.name == "mllama":
            cross_attention_mask = torch.ones(1, 32, 1, 4)
            example_inputs = example_inputs + (cross_attention_mask,)
        if model.name == "phi4mm":
            input_mode = config.input_mode
            batch_size = config.batch_size
            audio_batch_size = config.audio_batch_size
            example_inputs = example_inputs + (
                torch.tensor([input_mode]),
                (
                    torch.rand(1, 7, 3, 448, 448).repeat(batch_size, 1, 1, 1, 1)
                    if input_mode in [1, 3]
                    else torch.tensor([])
                ),
                (
                    torch.tensor([[896, 1344]]).repeat(batch_size, 1)
                    if input_mode in [1, 3]
                    else torch.tensor([])
                ),
                (
                    torch.ones(1, 7, 32, 32).repeat(batch_size, 1, 1, 1)
                    if input_mode in [1, 3]
                    else torch.tensor([])
                ),
                (
                    torch.rand(1, 498, 80).repeat(audio_batch_size, 1, 1)
                    if input_mode in [2, 3]
                    else torch.tensor([])
                ),
                (
                    torch.tensor([63]).repeat(audio_batch_size)
                    if input_mode in [2, 3]
                    else torch.tensor([])
                ),
            )
    elif model.example_inputs_mode == EXAMPLE_INPUTS_MODE.KV_MASK:
        example_inputs = (
            input_ids.unsqueeze(0),
            tuple(global_past_key_value),
            attention_mask.unsqueeze(0),
        )
    elif model.example_inputs_mode == EXAMPLE_INPUTS_MODE.MASK_KV:
        example_inputs = (
            input_ids.unsqueeze(0),
            attention_mask.unsqueeze(0),
            tuple(global_past_key_value),
        )
    elif model.example_inputs_mode == EXAMPLE_INPUTS_MODE.MASK_KV_ENC:
        last_hidden_state = torch.rand([1, 32, 2048])
        global_past_key_value = [
            (
                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                torch.zeros([1, n_heads, 1, head_dim]).contiguous(),
                torch.zeros([1, n_heads, 1, head_dim]).contiguous(),
                beam_idx_tmp,
                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                torch.zeros([32, 1, n_heads, head_dim]).contiguous(),
                torch.zeros([32, 1, n_heads, head_dim]).contiguous(),
                beam_idx_tmp,
            )
            for i in range(n_layers)
        ]
        example_inputs = (
            torch.ones(1).to(torch.long).unsqueeze(0),
            attention_mask.unsqueeze(0),
            tuple(global_past_key_value),
            (last_hidden_state,),
        )
    elif model.example_inputs_mode == EXAMPLE_INPUTS_MODE.MASK_KV_PIXEL:
        batch_size = int(args.batch_size) * num_beams
        past_key_value = [
            (
                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                torch.zeros([batch_size, n_heads, 1, head_dim]).contiguous(),
                torch.zeros([batch_size, n_heads, 1, head_dim]).contiguous(),
                beam_idx_tmp,
            )
            for i in range(n_layers)
        ]
        pixel_inputs = torch.ones(batch_size, 3, 224, 224)
        example_inputs = (
            torch.ones(batch_size, 1).to(torch.long),
            torch.ones(batch_size, 1),
            tuple(past_key_value),
            pixel_inputs,
        )
    elif model.example_inputs_mode == EXAMPLE_INPUTS_MODE.EMBEDS_MASK_KV:
        batch_size = int(args.batch_size) * num_beams
        past_key_value = [
            (
                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                torch.zeros([batch_size, n_heads, 1, head_dim]).contiguous(),
                torch.zeros([batch_size, n_heads, 1, head_dim]).contiguous(),
                beam_idx_tmp,
            )
            for i in range(n_layers)
        ]
        input_embeds = torch.zeros(batch_size, 1, 4096).to(amp_dtype)
        example_inputs = (
            input_embeds,
            torch.ones((batch_size, 1), dtype=torch.long),
            tuple(past_key_value),
        )
    elif model.example_inputs_mode == EXAMPLE_INPUTS_MODE.KV_ENC:
        past_key_value = tuple(
            [
                (
                    torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                    torch.zeros([1, 1, 1, 1]).contiguous(),
                    torch.zeros([1, 1, 1, 1]).contiguous(),
                    torch.zeros(1, 4, dtype=torch.long),
                    torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                    torch.zeros(
                        [1, 32, n_heads, head_dim], dtype=amp_dtype
                    ).contiguous(),
                    torch.zeros(
                        [1, 32, n_heads, head_dim], dtype=amp_dtype
                    ).contiguous(),
                    torch.zeros(1, 1, dtype=torch.long),
                )
                for i in range(n_layers)
            ]
        )
        last_hidden_state = torch.rand([1, 32, 1280]).to(amp_dtype)
        example_inputs = (
            torch.ones(4).to(torch.long).unsqueeze(0),
            past_key_value,
            (last_hidden_state,),
        )
    else:
        raise RuntimeError(
            "Your model does not match existing example inputs used in ipex quantization, exiting..."
        )
    if hasattr(model, "extra_inputs"):
        example_inputs = example_inputs + model.extra_inputs
    return example_inputs


if args.ipex_smooth_quant:
    if args.qconfig_summary_file != "":
        qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(alpha=args.alpha)
        user_model = ipex.llm.optimize(
            user_model.eval(),
            dtype=amp_dtype,
            quantization_config=qconfig,
            qconfig_summary_file=args.qconfig_summary_file,
            inplace=True,
            deployment_mode=True,
        )
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        user_model.trace_graph.save(args.output_dir + "/" + args.quant_model_name)
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
                elif "audio" in examples:
                    inputs = [d["array"] for d in examples["audio"]]
                    example = self.tokenizer(
                        inputs, sampling_rate=16000, return_tensors="pt"
                    )
                    example["input_ids"] = example["input_features"]
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
                    if not args.calib_padding:
                        input_ids = (
                            input_ids[: int(args.calib_len)]
                            if len(input_ids) > int(args.calib_len)
                            else input_ids
                        )
                    else:
                        from torch.nn.functional import pad

                        pad_len = int(args.calib_len) - input_ids.shape[0]
                        input_ids = pad(
                            input_ids, (0, pad_len), value=int(args.calib_pad_val)
                        )
                    last_ind.append(input_ids.shape[0] - 1)
                    attention_mask = torch.ones(len(input_ids))
                    position_ids = torch.arange(len(input_ids))

                    input_ids_padded.append(input_ids)
                    attention_mask_padded.append(attention_mask)
                    position_ids_padded.append(position_ids)

                if model.use_global_past_key_value:
                    global global_past_key_value
                if model.example_inputs_mode == EXAMPLE_INPUTS_MODE.MASK_POS_KV:
                    model_inputs = (
                        torch.vstack(input_ids_padded),
                        torch.vstack(attention_mask_padded),
                        torch.vstack(position_ids_padded),
                        tuple(global_past_key_value),
                    )
                elif model.example_inputs_mode == EXAMPLE_INPUTS_MODE.MASK_KV_POS:
                    model_inputs = (
                        torch.vstack(input_ids_padded),
                        torch.vstack(attention_mask_padded),
                        tuple(global_past_key_value),
                        torch.vstack(position_ids_padded),
                    )
                elif model.example_inputs_mode == EXAMPLE_INPUTS_MODE.KV_MASK:
                    model_inputs = (
                        torch.vstack(input_ids_padded),
                        tuple(global_past_key_value),
                        torch.vstack(attention_mask_padded),
                    )
                elif model.example_inputs_mode == EXAMPLE_INPUTS_MODE.MASK_KV:
                    model_inputs = (
                        torch.vstack(input_ids_padded),
                        torch.vstack(attention_mask_padded),
                        tuple(global_past_key_value),
                    )
                elif model.example_inputs_mode == EXAMPLE_INPUTS_MODE.MASK_KV_ENC:
                    model_kwargs = {
                        "attention_mask": torch.vstack(attention_mask_padded),
                    }
                    model_kwargs = (
                        user_model._prepare_encoder_decoder_kwargs_for_generation(
                            torch.vstack(input_ids_padded), model_kwargs, "input_ids"
                        )
                    )
                    input_ids, example_inputs = (
                        user_model._expand_inputs_for_generation(
                            input_ids=torch.vstack(input_ids_padded),
                            expand_size=num_beams,
                            is_encoder_decoder=True,
                            **model_kwargs,
                        )
                    )
                    input_bs = int(args.batch_size * num_beams)
                    last_hidden_state = example_inputs["encoder_outputs"][
                        "last_hidden_state"
                    ]
                    global_past_key_value = tuple(
                        [
                            (
                                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                                torch.zeros([1, n_heads, 1, head_dim]).contiguous(),
                                torch.zeros([1, n_heads, 1, head_dim]).contiguous(),
                                beam_idx_tmp,
                                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                                user_model.decoder.block[i]
                                .layer[1]
                                .EncDecAttention.k(last_hidden_state)
                                .view(input_bs, -1, n_heads, head_dim)
                                .transpose(0, 1),
                                user_model.decoder.block[i]
                                .layer[1]
                                .EncDecAttention.v(last_hidden_state)
                                .view(input_bs, -1, n_heads, head_dim)
                                .transpose(0, 1),
                                beam_idx_tmp,
                            )
                            for i in range(n_layers)
                        ]
                    )
                    decoder_input_ids = (
                        torch.zeros(input_bs).to(torch.long).unsqueeze(1)
                    )
                    model_inputs = (
                        decoder_input_ids,
                        torch.vstack(attention_mask_padded),
                        tuple(global_past_key_value),
                        (last_hidden_state,),
                    )
                elif model.example_inputs_mode == EXAMPLE_INPUTS_MODE.KV_ENC:
                    input_bs = int(args.batch_size * num_beams)
                    model_kwargs = {}
                    model_kwargs = user_model._prepare_encoder_decoder_kwargs_for_generation(
                        torch.vstack(input_ids_padded).unsqueeze(0),
                        model_kwargs,
                        "input_features",
                        transformers.generation.configuration_utils.GenerationConfig(),
                    )
                    last_hidden_state = model_kwargs["encoder_outputs"][
                        "last_hidden_state"
                    ]
                    global_past_key_value = tuple(
                        [
                            (
                                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                                torch.zeros([1, 1, 1, 1]).contiguous(),
                                torch.zeros([1, 1, 1, 1]).contiguous(),
                                beam_idx_tmp,
                                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                                user_model.model.decoder.layers[i]
                                .encoder_attn.k_proj(last_hidden_state)
                                .view(
                                    int(input_bs),
                                    -1,
                                    user_model.model.decoder.layers[
                                        i
                                    ].encoder_attn.num_heads,
                                    user_model.model.decoder.layers[
                                        i
                                    ].encoder_attn.head_dim,
                                )
                                .contiguous(),
                                user_model.model.decoder.layers[i]
                                .encoder_attn.v_proj(last_hidden_state)
                                .view(
                                    int(input_bs),
                                    -1,
                                    user_model.model.decoder.layers[
                                        i
                                    ].encoder_attn.num_heads,
                                    user_model.model.decoder.layers[
                                        i
                                    ].encoder_attn.head_dim,
                                )
                                .contiguous(),
                                beam_idx_tmp,
                            )
                            for i in range(n_layers)
                        ]
                    )
                    decoder_input_ids = (
                        torch.zeros(input_bs).to(torch.long).unsqueeze(1)
                    )
                    model_inputs = (
                        decoder_input_ids,
                        tuple(global_past_key_value),
                        (last_hidden_state,),
                    )
                else:
                    raise RuntimeError(
                        "Your model does not match existing example inputs used in ipex smooth quant, exiting..."
                    )

                if hasattr(model, "extra_inputs"):
                    model_inputs = model_inputs + model.extra_inputs

                return (model_inputs, last_ind)

        if (
            hasattr(model, "default_dataset")
            and model.default_dataset == "librispeech_asr"
        ):
            calib_dataset = load_dataset(model.default_dataset, split="train.clean.100")
        else:
            calib_dataset = load_dataset(
                args.dataset if args.dataset else model.default_dataset, split="train"
            )
        if args.calib_shuffle:
            calib_dataset = calib_dataset.shuffle(seed=42)
        user_model.eval()
        calib_evaluator = Evaluator(
            calib_dataset,
            tokenizer,
            args,
            batch_size=args.batch_size,
            pad_max=512,
        )
        calib_dataloader = DataLoader(
            calib_evaluator.dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=calib_evaluator.collate_batch,
        )

        def calib_func(prepared_model):
            for i, (model_inputs, last_ind) in enumerate(calib_dataloader):
                if i >= int(args.calib_iters):
                    break
                prepared_model(*model_inputs)

        example_inputs = get_example_inputs(model)

        from intel_extension_for_pytorch.quantization import prepare, convert

        if model.use_ipex_autotune:
            qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping()
            user_model = ipex.llm.optimize(
                user_model.eval(),
                dtype=amp_dtype,
                quantization_config=qconfig,
                inplace=True,
                deployment_mode=False,
            )
            op_type_dict = {}
            if args.fallback_add:
                op_type_dict["add"] = {
                    "weight": {"dtype": ["fp32"]},
                    "activation": {"dtype": ["fp32"]},
                }

            smoothquant_args = {
                "alpha": (
                    args.alpha if args.alpha == "auto" else literal_eval(args.alpha)
                ),
                "folding": args.folding,
            }
            if args.alpha == "auto":
                smoothquant_args["auto_alpha_args"] = {
                    "init_alpha": float(args.init_alpha),
                    "alpha_min": float(args.alpha_min),
                    "alpha_max": float(args.alpha_max),
                    "alpha_step": float(args.alpha_step),
                    "shared_criterion": args.shared_criterion,
                    "enable_blockwise_loss": args.enable_blockwise_loss,
                }
                # using specified sq recipes for llama2-7b
                if re.search("llama", config.architectures[0], re.IGNORECASE):
                    smoothquant_args = {"alpha": "auto", "folding": False}
                    smoothquant_args["auto_alpha_args"] = {
                        "init_alpha": 0.8,
                        "alpha_min": 0.8,
                        "alpha_max": 0.99,
                        "alpha_step": 0.01,
                        "shared_criterion": "mean",
                        "enable_blockwise_loss": False,
                    }

            prepared_model = ipex.quantization.autotune(
                user_model,
                calib_dataloader,
                calib_func=calib_func,
                op_type_dict=op_type_dict,
                smoothquant_args=smoothquant_args,
            )
            pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            prepared_model.save_qconf_summary(args.output_dir + "/best_configure.json")

        else:
            qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(
                alpha=args.alpha
            )
            user_model = ipex.llm.optimize(
                user_model.eval(),
                dtype=amp_dtype,
                quantization_config=qconfig,
                inplace=True,
                deployment_mode=False,
            )
            prepared_model = prepare(
                user_model.eval(), qconfig, example_inputs=example_inputs, inplace=True
            )

            for i, (model_inputs, last_ind) in enumerate(calib_dataloader):
                if i == 512:
                    break
                prepared_model(*model_inputs)

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
            if model.name == "yuan":
                input_bs = int(args.batch_size * num_beams)
                example_inputs = (
                    example_inputs[0].repeat(input_bs, 1),
                    example_inputs[1].repeat(input_bs, 1),
                    example_inputs[2].repeat(input_bs, 1),
                )
                self_jit_first = torch.jit.trace(
                    convert_model.eval(),
                    example_inputs,
                    strict=False,
                    check_trace=False,
                )
                self_jit_first = torch.jit.freeze(self_jit_first.eval())
                self_jit_first.save(args.output_dir + "/" + args.quant_model_name + "2")

elif args.ipex_weight_only_quantization:
    from intel_extension_for_pytorch.quantization import (
        WoqWeightDtype,
        WoqWeightQScheme,
    )

    if args.low_precision_checkpoint != "":
        pathname = args.low_precision_checkpoint
        logger.debug(f"Loading low precision checkpoint from {pathname}")
        low_precision_checkpoint, quant_config = load_low_precision_checkpoint(pathname)
        low_precision_checkpoint = (low_precision_checkpoint, quant_config)

        if args.gptq_legacy_format:
            raise AssertionError(
                "gptq legacy format is deprecated and not supported now."
            )
    else:
        low_precision_checkpoint = None

    if args.weight_dtype == "INT8":
        weight_dtype = WoqWeightDtype.INT8
    elif args.weight_dtype == "INT4":
        weight_dtype = WoqWeightDtype.INT4
    elif args.weight_dtype == "NF4":
        weight_dtype = WoqWeightDtype.NF4
    else:
        assert args.weight_dtype == "FP8"
        weight_dtype = WoqWeightDtype.FP8

    if args.lowp_mode == "INT8":
        lowp_mode = ipex.quantization.WoqLowpMode.INT8
    elif args.lowp_mode == "FP32":
        lowp_mode = ipex.quantization.WoqLowpMode.NONE
    elif args.lowp_mode == "FP16":
        lowp_mode = ipex.quantization.WoqLowpMode.FP16
    elif args.lowp_mode == "BF16":
        lowp_mode = ipex.quantization.WoqLowpMode.BF16
    else:  # AUTO
        if weight_dtype == WoqWeightDtype.INT4 or (
            low_precision_checkpoint is not None
            and low_precision_checkpoint[1]["bits"] == 4
        ):
            lowp_mode = ipex.quantization.WoqLowpMode.INT8
        else:
            lowp_mode = ipex.quantization.WoqLowpMode.BF16

    act_quant_mode_dict = {
        "PER_TENSOR": ipex.quantization.WoqActQuantMode.PER_TENSOR,
        "PER_IC_BLOCK": ipex.quantization.WoqActQuantMode.PER_IC_BLOCK,
        "PER_BATCH": ipex.quantization.WoqActQuantMode.PER_BATCH,
        "PER_BATCH_IC_BLOCK": ipex.quantization.WoqActQuantMode.PER_BATCH_IC_BLOCK,
        "PER_TENSOR_SYM": ipex.quantization.WoqActQuantMode.PER_TENSOR_SYM,
        "PER_IC_BLOCK_SYM": ipex.quantization.WoqActQuantMode.PER_IC_BLOCK_SYM,
        "PER_BATCH_SYM": ipex.quantization.WoqActQuantMode.PER_BATCH_SYM,
        "PER_BATCH_IC_BLOCK_SYM": ipex.quantization.WoqActQuantMode.PER_BATCH_IC_BLOCK_SYM,
    }
    weight_qscheme = (
        WoqWeightQScheme.SYMMETRIC
        if args.woq_sym_quant_weight
        else WoqWeightQScheme.UNDEFINED
    )
    qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
        weight_dtype=weight_dtype,
        lowp_mode=lowp_mode,
        act_quant_mode=act_quant_mode_dict[args.act_quant_mode],
        group_size=args.group_size,
        weight_qscheme=weight_qscheme,
    )

    logger.debug("doing ipex.llm.optimize")
    user_model = ipex.llm.optimize(
        user_model.eval(),
        dtype=amp_dtype,
        quantization_config=qconfig,
        inplace=True,
        low_precision_checkpoint=low_precision_checkpoint,
        deployment_mode=False,
        cache_weight_for_large_batch=args.cache_weight_for_large_batch,
    )
    example_inputs = get_example_inputs(model)
    with torch.no_grad(), torch.cpu.amp.autocast(
        enabled=amp_enabled,
    ):
        logger.debug("doing jit trace")
        self_jit = torch.jit.trace(
            user_model.eval(), example_inputs, strict=False, check_trace=False
        )
        logger.debug("doing jit freeze")
        self_jit = torch.jit.freeze(self_jit.eval())
        logger.debug("saving jit model")
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        self_jit.save(args.output_dir + "/" + args.quant_model_name)
        quant_model = self_jit
        if model.name == "yuan":
            input_bs = int(args.batch_size * num_beams)
            example_inputs = (
                example_inputs[0].repeat(input_bs, 1),
                example_inputs[1].repeat(input_bs, 1),
                example_inputs[2].repeat(input_bs, 1),
            )
            self_jit_first = torch.jit.trace(
                user_model.eval(), example_inputs, strict=False, check_trace=False
            )
            self_jit_first = torch.jit.freeze(self_jit_first.eval())
            self_jit_first.save(args.output_dir + "/" + args.quant_model_name + "2")
        elif model.name == "mllama":
            pixel_values = torch.rand(
                1,
                1,
                4,
                3,
                user_model.config.vision_config.image_size,
                user_model.config.vision_config.image_size,
            )
            aspect_ratio_mask = torch.tensor([[[1, 1, 1, 1]]])
            aspect_ratio_ids = torch.tensor([[6]])
            example_inputs = example_inputs + (pixel_values,)
            example_inputs = example_inputs + (aspect_ratio_mask,)
            example_inputs = example_inputs + (aspect_ratio_ids,)
            self_jit_first = torch.jit.trace(
                user_model.eval(), example_inputs, strict=False, check_trace=False
            )
            self_jit_first = torch.jit.freeze(self_jit_first.eval())
            self_jit_first.save(args.output_dir + "/" + args.quant_model_name + "2")
        elif model.name == "maira2":
            pixel_values = torch.rand(1, 3, 518, 518)
            example_inputs = example_inputs + (pixel_values,)
            self_jit_first = torch.jit.trace(
                user_model.eval(), example_inputs, strict=False, check_trace=False
            )
            self_jit_first = torch.jit.freeze(self_jit_first.eval())
            self_jit_first.save(args.output_dir + "/" + args.quant_model_name + "2")
        elif model.name == "jamba":
            input_ids = torch.ones(1).to(torch.long).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            position_ids = torch.arange(input_ids.shape[-1]).unsqueeze(0)
            past_key_values = tuple(
                [
                    (
                        global_past_key_value[i]
                        if i % user_model.config.attn_layer_period
                        == user_model.config.attn_layer_offset
                        else (
                            global_past_key_value[i][0][0, ...].unsqueeze(0),
                            global_past_key_value[i][1][0, ...].unsqueeze(0),
                            torch.tensor(True).contiguous(),
                        )
                    )
                    for i in range(user_model.config.num_hidden_layers)
                ]
            )
            example_inputs = (
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                torch.tensor(False),
                torch.tensor(1),
            )
            self_jit_next = torch.jit.trace(
                user_model.eval(), example_inputs, strict=False, check_trace=False
            )
            self_jit_next = torch.jit.freeze(self_jit_next.eval())
            self_jit_next.save(args.output_dir + "/" + args.quant_model_name + "2")
        elif model.name == "phi4mm":
            batch_size = config.batch_size
            audio_batch_size = config.audio_batch_size
            if config.input_mode == 1:
                input_ids = torch.ones(1851 * batch_size).to(torch.long).unsqueeze(0)
                input_ids[:, : 1841 * batch_size] = 200010
                example_inputs[7][:, 3, :, -1] = 0
            elif config.input_mode == 2:
                input_ids = (
                    torch.ones(96 * audio_batch_size).to(torch.long).unsqueeze(0)
                )
                input_ids[:, : 63 * audio_batch_size] = 200011
            elif config.input_mode == 3:
                input_ids = torch.ones(1907 * batch_size).to(torch.long).unsqueeze(0)
                input_ids[:, : 1841 * batch_size] = 200010
                input_ids[
                    :, 1841 * batch_size : 1841 * batch_size + 63 * audio_batch_size
                ] = 200011
                example_inputs[7][:, 3, :, -1] = 0
            if config.input_mode > 0:
                attention_mask = torch.ones_like(input_ids)
                position_ids = torch.arange(input_ids.shape[-1]).unsqueeze(0)
                example_inputs = (
                    input_ids,
                    attention_mask,
                    example_inputs[2],
                    position_ids,
                ) + example_inputs[4:]
                self_jit_first = torch.jit.trace(
                    user_model.eval(), example_inputs, strict=False, check_trace=False
                )
                self_jit_first = torch.jit.freeze(self_jit_first.eval())
                self_jit_first.save(args.output_dir + "/" + args.quant_model_name + "2")


if args.benchmark:
    torch._C._jit_set_texpr_fuser_enabled(False)
    qconfig = ipex.quantization.default_static_qconfig_mapping
    user_model = ipex.llm.optimize(
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
            if model.name in ["yuan", "mllama", "maira2"] or (
                model.name == "phi4mm" and config.input_mode > 0
            ):
                self_jit_first = torch.jit.load(args.quantized_model_path + "2")
                self_jit_first = torch.jit.freeze(self_jit_first.eval())
            if model.name in ["jamba"]:
                self_jit_next = torch.jit.load(args.quantized_model_path + "2")
                self_jit_next = torch.jit.freeze(self_jit_next.eval())
        except Exception as e:
            print("warning: loading failed.", e)
            self_jit = quant_model
        if model.name in ["yuan", "mllama", "maira2"] or (
            model.name == "phi4mm" and config.input_mode > 0
        ):
            ipex._set_optimized_model_for_generation(
                user_model,
                optimized_model=self_jit,
                first_token_optimized_model=self_jit_first,
            )
        elif model.name in ["jamba"]:
            ipex._set_optimized_model_for_generation(
                user_model,
                optimized_model=self_jit_next,
                first_token_optimized_model=self_jit,
            )
        else:
            ipex._set_optimized_model_for_generation(
                user_model, optimized_model=self_jit
            )
    for test_bs in [args.batch_size]:
        if model.name == "git":
            prompt = Image.open(requests.get(args.image_url, stream=True).raw)
        elif model.name == "llava":
            if args.prompt is not None:
                prompt = args.prompt
            image = load_image(args.image_url)
            image = [image] * test_bs
            if user_model.config.mm_use_im_start_end:
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
            image_processor = model.get_image_processor()
        elif model.name == "whisper":
            sample = librosa.load(args.audio, sr=16000)
            prompt = sample[0]
            generate_kwargs.pop("min_new_tokens", None)
        elif model.name == "maira2":
            prompt = args.prompt
            sample = download_and_open(args.image_url)
            process_input_func = (
                tokenizer.process_reporting_input
                if hasattr(tokenizer, "process_reporting_input")
                else tokenizer.format_and_preprocess_reporting_input
            )
        elif model.name == "phi4mm":
            prompt = args.prompt
            sample = soundfile.read(args.audio) if config.input_mode in [2, 3] else None
        else:
            # input prompt
            current_path = pathlib.Path(__file__).parent.resolve()
            with open(str(current_path) + "/prompt.json") as f:
                prompt_pool = json.load(f)
            if args.prompt is not None:
                prompt = args.prompt
            # elif int(args.input_tokens) > 8192:
            #     prompt = prompt_pool[model.name]["8192"] * int(int(args.input_tokens) / 8192)
            elif args.input_tokens in prompt_pool[model.name]:
                if model.name == "qwen3moe":
                    prompt_list = []
                    for i in range(test_bs):
                        prompt = prompt_pool[model.name][args.input_tokens][str(i)]
                        messages = [{"role": "user", "content": prompt}]
                        prompt = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=args.enable_thinking,
                        )
                        prompt_list.append(prompt)
                    prompt = prompt_list
                else:
                    prompt = prompt_pool[model.name][args.input_tokens]
            else:
                raise SystemExit(
                    "[ERROR] Plese use --prompt if want to use custom input."
                )

            if model.name == "mllama":
                raw_image = load_image(args.image_url)
                raw_image = [raw_image] * test_bs
                inputs = tokenizer(raw_image, prompt, return_tensors="pt")
                input_size = inputs["input_ids"].size(dim=1)
            else:
                input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(
                    dim=1
                )
            print("---- Prompt size:", input_size)

    if args.token_latency:
        if not hasattr(user_model.config, "token_latency"):
            user_model.config.token_latency = True

    # start
    total_time = 0.0
    num_iter = args.num_iter
    num_warmup = args.num_warmup
    if model.name != "qwen3moe":
        prompt = [prompt] * args.batch_size
    total_list = []
    with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
        enabled=amp_enabled
    ):
        for i in range(num_iter):
            tic = time.time()
            if model.name == "llava":
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
                output = user_model.generate(
                    input_ids, images=image_tensor, **generate_kwargs
                )
            elif model.name == "git":
                input_ids = tokenizer(images=prompt, return_tensors="pt").pixel_values
                output = user_model.generate(pixel_values=input_ids, **generate_kwargs)
            elif model.name == "whisper":
                input_ids = tokenizer(
                    prompt, sampling_rate=16000, return_tensors="pt"
                ).input_features
                output = user_model.generate(input_ids, **generate_kwargs)
            elif model.name == "mllama":
                raw_image = [load_image(args.image_url)] * args.batch_size
                inputs = tokenizer(raw_image, prompt, return_tensors="pt")
                input_ids = inputs["input_ids"]
                output = user_model.generate(**inputs, **generate_kwargs)
            elif model.name == "maira2":
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
                output = user_model.generate(**processed_inputs, **generate_kwargs)
            elif model.name == "phi4mm":
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
                output = user_model.generate(**inputs, **generate_kwargs)
            else:
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                output = user_model.generate(input_ids, **generate_kwargs)
            gen_ids = output[0] if args.token_latency else output
            gen_text = tokenizer.batch_decode(
                (
                    gen_ids[:, input_ids.shape[1] :]
                    if model.name in ["llava", "maira2", "phi4mm"]
                    else gen_ids
                ),
                skip_special_tokens=True,
            )
            toc = time.time()
            input_tokens_lengths = [x.shape[0] for x in input_ids]
            output_tokens_lengths = [x.shape[0] for x in gen_ids]
            total_new_tokens = [
                o if user_model.config.model_type in ["t5", "whisper"] else o - i
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
        ) as prof, torch.no_grad(), torch.cpu.amp.autocast(enabled=amp_enabled):
            for i in range(5):
                if model.name == "llava":
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
                    output = user_model.generate(
                        input_ids, images=image_tensor, **generate_kwargs
                    )
                elif model.name == "git":
                    input_ids = tokenizer(
                        images=prompt, return_tensors="pt"
                    ).pixel_values
                    output = user_model.generate(
                        pixel_values=input_ids, **generate_kwargs
                    )
                elif model.name == "whisper":
                    input_ids = tokenizer(
                        prompt, sampling_rate=16000, return_tensors="pt"
                    ).input_features
                    output = user_model.generate(input_ids, **generate_kwargs)
                elif model.name == "mllama":
                    raw_image = [load_image(args.image_url)] * args.batch_size
                    inputs = tokenizer(raw_image, prompt, return_tensors="pt")
                    input_ids = inputs["input_ids"]
                    output = user_model.generate(**inputs, **generate_kwargs)
                elif model.name == "maira2":
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
                    output = user_model.generate(**processed_inputs, **generate_kwargs)
                elif model.name == "phi4mm":
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
                    output = user_model.generate(**inputs, **generate_kwargs)
                else:
                    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                    output = user_model.generate(input_ids, **generate_kwargs)
                gen_ids = output[0] if args.token_latency else output
                gen_text = tokenizer.batch_decode(
                    (
                        gen_ids[:, input_ids.shape[1] :]
                        if model.name in ["llava", "phi4mm"]
                        else gen_ids
                    ),
                    skip_special_tokens=True,
                )
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
