import torch
import argparse

# Here import ipex for Baichuan loading compatibility, for other models we can ignore this import
import intel_extension_for_pytorch  # noqa F401
from supported_models import MODEL_CLASSES

try:
    from llava.model.builder import load_pretrained_model
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
parser.add_argument(
    "--vision-text-model",
    action="store_true",
    help="whether or not it is vision-text multi-model structure",
)
args = parser.parse_args()
print(args)
if args.local_rank == 0:
    model_type = next(
        (x for x in MODEL_CLASSES.keys() if x in args.model_id.lower()), "auto"
    )
    if model_type == "llama" and args.vision_text_model:
        model_type = "mllama"
    if model_type in ["maira-2", "deepseek-v2", "deepseek-v3", "deepseek-r1"]:
        model_type = model_type.replace("-", "")
    model_class = MODEL_CLASSES[model_type]
    load_dtype = torch.float32
    if args.dtype == "float16":
        load_dtype = torch.half
    elif args.dtype == "bfloat16":
        load_dtype = torch.bfloat16
    if model_type != "llava":
        tokenizer = model_class[1].from_pretrained(
            args.model_id, trust_remote_code=True
        )
        model = model_class[0].from_pretrained(
            args.model_id,
            torch_dtype=load_dtype,
            low_cpu_mem_usage=True if model_type != "maira2" else False,
            trust_remote_code=True,
            _attn_implementation="eager",
        )
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            args.model_id
        )
    model.save_pretrained(
        save_directory=args.save_path,
        max_shard_size=args.max_shard_size,
        safe_serialization=False,
    )
    if model_type == "phi4mm":
        tokenizer.chat_template = None
    tokenizer.save_pretrained(save_directory=args.save_path)
    if model_type == "llava":
        image_processor.save_pretrained(save_directory=args.save_path)
    if model_type in ["maira2", "deepseekv2", "deepseekv3", "deepseekr1"]:
        import inspect
        import shutil

        model_file = inspect.getfile(model.__class__)
        shutil.copy(model_file, args.save_path)
