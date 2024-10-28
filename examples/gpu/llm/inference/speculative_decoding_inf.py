import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import intel_extension_for_pytorch as ipex

from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import time

parser = argparse.ArgumentParser("* Speculative Decoding Inference script *", add_help=False)
parser.add_argument("--num-beams", default=1, type=int, help="beam width")
parser.add_argument(
    "--max-new-tokens", default=10, type=int, help="output max new tokens"
)
parser.add_argument(
    "-m",
    "--model-id",
    type=str,
    default="meta-llama/Llama-2-13b-hf",
    help="the huggingface model id, the larger one for speculative decoding",
)
parser.add_argument(
    "-assis",
    "--assistant-model-id",
    type=str,
    default="meta-llama/Llama-2-7b-hf",
    help="the huggingface model id, the assist and smaller one for speculative decoding",
)
parser.add_argument("--native-transformers", action="store_true", help="using native transformers for speculative decoding")
parser.add_argument("--turn-off-speculative-decoding", action="store_true", help="using origin hf text to generation path")
args = parser.parse_args()

device = "xpu" if torch.xpu.is_available() else "cpu"

print("start memory used total:", round(torch.xpu.memory_reserved() / 1024**3, 3), "GB")

tokenizer = AutoTokenizer.from_pretrained(args.model_id)
inputs = tokenizer("Once upon a time, there existed a little girl, who liked to have adventures.", return_tensors="pt").input_ids.to(device)

model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16).to(device)
model = model.to(memory_format=torch.channels_last)

assistant_model = AutoModelForCausalLM.from_pretrained(args.assistant_model_id, torch_dtype=torch.float16).to(device)
assistant_model = assistant_model.to(memory_format=torch.channels_last)

generate_kwargs = dict(do_sample=True, temperature=0.5)
# generate_kwargs = {}

# warm up
if not args.turn_off_speculative_decoding:
    outputs = model.generate(inputs, max_new_tokens=int(args.max_new_tokens), min_new_tokens=int(args.max_new_tokens), assistant_model=assistant_model, **generate_kwargs)
    gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
else:
    outputs = model.generate(inputs, max_new_tokens=int(args.max_new_tokens), min_new_tokens=int(args.max_new_tokens), 
                                        **generate_kwargs)
    gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

woq_config = {}
if not args.native_transformers:
    model = ipex.llm.optimize(model.eval(), dtype=torch.float16, device="xpu", inplace=True, **woq_config)
    assistant_model = ipex.llm.optimize(assistant_model.eval(), dtype=torch.float16, device="xpu", inplace=True, **woq_config)
torch.xpu.empty_cache()
start = time.time()
if not args.turn_off_speculative_decoding:
    outputs = model.generate(inputs, max_new_tokens=int(args.max_new_tokens), min_new_tokens=int(args.max_new_tokens), 
                                        assistant_model=assistant_model,
                                        **generate_kwargs)
    gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
else:
    outputs = model.generate(inputs, max_new_tokens=int(args.max_new_tokens), min_new_tokens=int(args.max_new_tokens), 
                                        **generate_kwargs)
    gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
torch.xpu.synchronize()
print("*************************************************** Time taken for optimized speculative decoding inf:", time.time() - start)
print("*************************************************** optimized result:", gen_text)
