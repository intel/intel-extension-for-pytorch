import torch
import argparse

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer
)
# supported models
MODEL_CLASSES = {
    "gpt-j": (AutoModelForCausalLM, AutoTokenizer),
    "gpt-neox": (AutoModelForCausalLM, AutoTokenizer),
    "llama": (AutoModelForCausalLM, LlamaTokenizer),
    "opt": (AutoModelForCausalLM, AutoTokenizer),
    "falcon": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}

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
args = parser.parse_args()
print(args)
model_type = next(
    (x for x in MODEL_CLASSES.keys() if x in args.model_id.lower()), "auto"
)
model_class = MODEL_CLASSES[model_type]

load_dtype = torch.float32
if args.dtype == "float16":
    load_dtype = torch.half
elif args.dtype == "bfloat16":
    load_dtype = torch.bfloat16

tokenizer = model_class[1].from_pretrained(args.model_id, trust_remote_code=True)
model = model_class[0].from_pretrained(
    args.model_id, torch_dtype=load_dtype, low_cpu_mem_usage=True, trust_remote_code=True
)

model.save_pretrained(save_directory=args.save_path, max_shard_size=args.max_shard_size)
tokenizer.save_pretrained(save_directory=args.save_path)

