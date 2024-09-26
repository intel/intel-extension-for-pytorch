import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import intel_extension_for_pytorch as ipex

from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import time

parser = argparse.ArgumentParser("* Speculative Decoding Inference script *", add_help=False)
parser.add_argument("--num-beams", default=1, type=int, help="beam width")
parser.add_argument(
    "--max-new-tokens", default=32, type=int, help="output max new tokens"
)
args = parser.parse_args()

device = "xpu" if torch.xpu.is_available() else "cpu"

print("start memory used total:", round(torch.xpu.memory_reserved() / 1024**3, 3), "GB")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
inputs = tokenizer("Once upon a time, there existed a little girl, who liked to have adventures.", return_tensors="pt").input_ids.to(device)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", torch_dtype=torch.float16).to(device)
model = model.to(memory_format=torch.channels_last)

assistant_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16).to(device)
assistant_model = assistant_model.to(memory_format=torch.channels_last)

generate_kwargs = dict(do_sample=True, temperature=0.5)
# generate_kwargs = {}

# warm up
outputs = model.generate(inputs, max_new_tokens=int(args.max_new_tokens), min_new_tokens=int(args.max_new_tokens), assistant_model=assistant_model, **generate_kwargs)
gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)


woq_config = {}
model = ipex.llm.optimize(model.eval(), dtype=torch.float16, device="xpu", inplace=True, **woq_config)
assistant_model = ipex.llm.optimize(assistant_model.eval(), dtype=torch.float16, device="xpu", inplace=True, **woq_config)
torch.xpu.empty_cache()
start = time.time()
outputs = model.generate(inputs, max_new_tokens=int(args.max_new_tokens), min_new_tokens=int(args.max_new_tokens), 
                                    assistant_model=assistant_model,
                                    **generate_kwargs)
gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
torch.xpu.synchronize()
print("*************************************************** Time taken for optimized speculative decoding inf:", time.time() - start)
print("*************************************************** optimized result:", gen_text)
