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
from ast import literal_eval
import sys

sys.path.append(sys.path[0] + "/../../")


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
from llm.utils.model_class.mixtral import MixtralConfig
from llm.utils.model_class.mpt import MPTConfig
from llm.utils.model_class.stablelm import StableLMConfig
from llm.utils.model_class.qwen import QwenConfig
from llm.utils.model_class.git import GitConfig
from llm.utils.model_class.llava import LlavaConfig

parser = argparse.ArgumentParser("LLM generation script (int8 path)", add_help=False)
parser.add_argument(
    "-m", "--model-id", default=None, type=str, required=True, help="your llm model"
)
parser.add_argument(
    "--max-new-tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument(
    "--streaming", action="store_true", help="enable streaming mode for generation output (greedy search only)"
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
    "--image-url", default="http://images.cocodataset.org/val2017/000000039769.jpg", type=str, help="image url for image-to-text task"
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
    "--calib-len", default=512, type=int, help="calibration dataset max or padding max length for SmoothQuant autotuning"
)
parser.add_argument("--calib-iters", default=512, type=int, help="calibration iters for SmoothQuant autotuning")
parser.add_argument(
    "--calib-shuffle", action="store_true", help="whether to shuffle on calibration dataset for SmoothQuant autotuning"
)
parser.add_argument(
    "--calib-padding", action="store_true", help="whether to pad on calibration dataset for SmoothQuant autotuning"
)
parser.add_argument(
    "--calib-pad-val", default=1, type=int, help="calibration dataset padding value for SmoothQuant autotuning"
)
parser.add_argument(
    "--fallback-add", action="store_true", help="whether to fallback add ops to fp32 for SmoothQuant autotuning"
)
parser.add_argument("--alpha", default=0.5, help="alpha value for smoothquant")
parser.add_argument(
    "--folding", action="store_true", help="whether to fold mul into the previous layer"
)
parser.add_argument(
    "--init-alpha", default=0.5, type=float, help="a value to get baseline quantization error for auto-tuning"
)
parser.add_argument(
    "--alpha-min", default=0.0, type=float, help="min value of auto-tuning alpha search space"
)
parser.add_argument(
    "--alpha-max", default=1.0, type=float, help="max value of auto-tuning alpha search space"
)
parser.add_argument(
    "--alpha-step", default=0.1, type=float, help="step_size of auto-tuning alpha search space"
)
parser.add_argument(
    "--shared-criterion", choices=["min", "mean", "max"], default="max", type=str
    , help="criterion for input LayerNorm op of a transformer block"
)
parser.add_argument(
    "--enable-blockwise-loss", action="store_true", help="whether to enable block-wise auto-tuning"
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
    choices=["INT8", "INT4", "NF4"],
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
parser.add_argument(
    "--gptq-legacy-format",
    action="store_true",
    help="Indicate that the low-precision checkpoint is in the legacy format rather than the"
    " HuggingFace Optimum format for backward compatibility. It must be used with"
    " --low-precision-checkpoint. Otherwise, it has no effect."
)
args = parser.parse_args()


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
elif re.search("llama", config.architectures[0], re.IGNORECASE) and not re.search("llava", config.architectures[0], re.IGNORECASE):
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
elif re.search("qwen", config.architectures[0], re.IGNORECASE):
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
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    except ImportError:
        pass
    model = LlavaConfig(args.model_id)
    def load_image(image_file):
        if image_file.startswith('http://') or image_file.startswith('https://'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        return image
    model_name = get_model_name_from_path(args.model_id)
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    conv = conv_templates[conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles
else:
    raise AssertionError("Not support %s." % (args.model_id))

num_beams = 1 if args.greedy else 4
if not hasattr(config, "text_max_length") and args.prompt is None:
    config.text_max_length = int(args.input_tokens) + int(args.max_new_tokens)
if model.name == "mpt" and not hasattr(config, "max_seq_len") and args.prompt is None:
    config.max_seq_len = int(args.input_tokens) + int(args.max_new_tokens)
if model.name in ["git", "llava"]:
    config.batch_size = int(args.batch_size) * num_beams

user_model = model.get_user_model(config, args.benchmark)

tokenizer = model.get_tokenizer()
print("Data type of the model:", user_model.dtype)
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
    streamer=streamer
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

def get_example_inputs(model):
    if model.use_global_past_key_value:
        global global_past_key_value
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
            input_ids.unsqueeze(0).repeat(batch_size,1),
            attention_mask.unsqueeze(0).repeat(batch_size,1),
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
            attention_mask.unsqueeze(0).repeat(batch_size,1),
            tuple(past_key_value),
        )
    else:
        raise RuntimeError("Your model does not match existing example inputs used in ipex quantization, exiting...")
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
                    model_kwargs = user_model._prepare_encoder_decoder_kwargs_for_generation(
                        torch.vstack(input_ids_padded), model_kwargs, "input_ids"
                    )
                    input_ids, example_inputs = user_model._expand_inputs_for_generation(
                        input_ids=torch.vstack(input_ids_padded),
                        expand_size=num_beams,
                        is_encoder_decoder=True,
                        **model_kwargs,
                    )
                    input_bs = int(args.batch_size * num_beams)
                    last_hidden_state = example_inputs["encoder_outputs"]["last_hidden_state"]
                    global_past_key_value = tuple(
                        [
                            (
                                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                                torch.zeros([1, n_heads, 1, head_dim]).contiguous(),
                                torch.zeros([1, n_heads, 1, head_dim]).contiguous(),
                                beam_idx_tmp,
                                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                                user_model.decoder.block[i].layer[1].EncDecAttention.k(last_hidden_state)
                                .view(input_bs, -1, n_heads, head_dim).transpose(0, 1),
                                user_model.decoder.block[i].layer[1].EncDecAttention.v(last_hidden_state)
                                .view(input_bs, -1, n_heads, head_dim).transpose(0, 1),
                                beam_idx_tmp,
                            )
                            for i in range(n_layers)
                        ]
                    )
                    decoder_input_ids = (torch.zeros(input_bs).to(torch.long).unsqueeze(1))
                    model_inputs = (
                        decoder_input_ids,
                        torch.vstack(attention_mask_padded),
                        tuple(global_past_key_value),
                        (last_hidden_state,),
                    )
                else:
                    raise RuntimeError("Your model does not match existing example inputs used in ipex smooth quant, exiting...")

                if hasattr(model, "extra_inputs"):
                    model_inputs = model_inputs + model.extra_inputs

                return (model_inputs, last_ind)
    
        calib_dataset = load_dataset(
            args.dataset if args.dataset else model.default_dataset, split="train"
        )
        if args.calib_shuffle:
            calib_dataset = calib_dataset.shuffle(seed=42)
        user_model.eval()
        calib_evaluator = Evaluator(
            calib_dataset, tokenizer, args, batch_size=args.batch_size, pad_max=int(args.input_tokens) if model.name=="t5" else 512
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
            
            smoothquant_args = {"alpha": args.alpha if args.alpha == "auto" \
                                else literal_eval(args.alpha), "folding": args.folding}
            if args.alpha == "auto":
                smoothquant_args["auto_alpha_args"] = {
                        "init_alpha": float(args.init_alpha),
                        "alpha_min": float(args.alpha_min),
                        "alpha_max": float(args.alpha_max),
                        "alpha_step": float(args.alpha_step),
                        "shared_criterion": args.shared_criterion,
                        "enable_blockwise_loss": args.enable_blockwise_loss
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
                        "enable_blockwise_loss": False
                }

            prepared_model = ipex.quantization.autotune(
                user_model,
                calib_dataloader,
                calib_func=calib_func,
                op_type_dict=op_type_dict,
                smoothquant_args=smoothquant_args
            )
            pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            prepared_model.save_qconf_summary(args.output_dir + "/best_configure.json")

        else:
            qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(alpha=args.alpha)
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
    user_model = ipex.llm.optimize(
        user_model.eval(),
        dtype=amp_dtype,
        quantization_config=qconfig,
        inplace=True,
        low_precision_checkpoint=low_precision_checkpoint,
        deployment_mode=False,
    )
    example_inputs = get_example_inputs(model)
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
        except Exception as e:
            print("warning: loading failed.", e)
            self_jit = quant_model
        ipex._set_optimized_model_for_generation(user_model, optimized_model=self_jit)

    if model.name == "git":
        prompt = Image.open(requests.get(args.image_url, stream=True).raw)
    elif model.name == "llava":
        if args.prompt is not None:
            prompt = args.prompt
        image = load_image(args.image_url)
        image = [image] * args.batch_size
        if user_model.config.mm_use_im_start_end:
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        image_processor = model.get_image_processor()
    else:
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
            if model.name == "llava":
                input_ids = torch.stack([tokenizer_image_token(pmt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt') for pmt in prompt])
                image_tensor = [image_processor.preprocess(img, return_tensors='pt')['pixel_values'].to(amp_dtype) for img in image]
                output = user_model.generate(input_ids, images=image_tensor, **generate_kwargs)
            elif model.name == "git":
                input_ids = tokenizer(images=prompt, return_tensors="pt").pixel_values
                output = user_model.generate(pixel_values=input_ids, **generate_kwargs)
            else:
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                output = user_model.generate(input_ids, **generate_kwargs)
            gen_ids = output[0] if args.token_latency else output
            gen_text = tokenizer.batch_decode(gen_ids[:, input_ids.shape[1]:] if model.name=="llava" else gen_ids, skip_special_tokens=True)
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
                if model.name == "llava":
                    input_ids = torch.stack([tokenizer_image_token(pmt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt') for pmt in prompt])
                    image_tensor = [image_processor.preprocess(img, return_tensors='pt')['pixel_values'].to(amp_dtype) for img in image]
                    output = user_model.generate(input_ids, images=image_tensor, **generate_kwargs)
                elif model.name == "git":
                    input_ids = tokenizer(images=prompt, return_tensors="pt").pixel_values
                    output = user_model.generate(pixel_values=input_ids, **generate_kwargs)
                else:
                    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                    output = user_model.generate(input_ids, **generate_kwargs)
                gen_ids = output[0] if args.token_latency else output
                gen_text = tokenizer.batch_decode(gen_ids[:, input_ids.shape[1]:] if model.name=="llava" else gen_ids, skip_special_tokens=True)
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
