import torch
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, T5ForConditionalGeneration, AutoProcessor
# Here import ipex for Baichuan loading compatibility, for other models we can ignore this import
import intel_extension_for_pytorch

# supported models
MODEL_CLASSES = {
    "gpt-j": (AutoModelForCausalLM, AutoTokenizer),
    "gpt-neox": (AutoModelForCausalLM, AutoTokenizer),
    "llama": (AutoModelForCausalLM, LlamaTokenizer),
    "opt": (AutoModelForCausalLM, AutoTokenizer),
    "falcon": (AutoModelForCausalLM, AutoTokenizer),
    "bloom": (AutoModelForCausalLM, AutoTokenizer),
    "codegen": (AutoModelForCausalLM, AutoTokenizer),
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
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}

try:
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
    from llava.model.builder import load_pretrained_model
    MODEL_CLASSES["llava"] = (LlavaLlamaForCausalLM, AutoTokenizer)
except ImportError:
    pass

# args
parser = argparse.ArgumentParser("shard model weight script", add_help=False)
parser.add_argument(
    "-m",
    "--model-id",
    type=str,
    default="EleutherAI/gpt-j-6B",
    help="the huggingface mdoel id",
)
parser.add_argument(
    "--save-path",
    type=str,
    default="./",
    help="saving path",
)
parser.add_argument(
    "--dtype",
    type=str,
    choices=["float32", "bfloat16", "float16"],
    default="bfloat16",
    help="bfloat16, float32, float16",
)
parser.add_argument(
    "--max-shard-size",
    type=str,
    default="500MB",
)
parser.add_argument(
    "--local_rank", required=False, type=int, default=0, help="used by dist launchers"
)
args = parser.parse_args()
print(args)
if args.local_rank == 0 :
    model_type = next(
        (x for x in MODEL_CLASSES.keys() if x in args.model_id.lower()), "auto"
    )
    model_class = MODEL_CLASSES[model_type]
    load_dtype = torch.float32
    if args.dtype == "float16":
        load_dtype = torch.half
    elif args.dtype == "bfloat16":
        load_dtype = torch.bfloat16
    if model_type != "llava":
        tokenizer = model_class[1].from_pretrained(args.model_id, trust_remote_code=True)
        model = model_class[0].from_pretrained(
            args.model_id,
            torch_dtype=load_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_id)
    model.save_pretrained(save_directory=args.save_path, max_shard_size=args.max_shard_size, safe_serialization=False)
    tokenizer.save_pretrained(save_directory=args.save_path)
    if model_type == "llava":
        image_processor.save_pretrained(save_directory=args.save_path)
