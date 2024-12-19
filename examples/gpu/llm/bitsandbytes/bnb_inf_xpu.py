import torch
import time
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import Accelerator

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-chat-hf", required=False, type=str, help="model_name")
parser.add_argument("--quant_type", default="int8", type=str, help="quant type", choices=["int8", "nf4", "fp4"])
parser.add_argument("--max_new_tokens", default=64, type=int, help="min_gen_len")
parser.add_argument("--device", default="cpu", type=str, help="device type", choices=["cpu", "xpu"])
args = parser.parse_args()

def get_current_device():
    return Accelerator().process_index

device_map={'':get_current_device()} if args.device == 'xpu' else None

MAX_NEW_TOKENS = args.max_new_tokens
model_id = args.model_name
torch_dtype = torch.bfloat16

text = 'I am happy because'
tokenizer = AutoTokenizer.from_pretrained(model_id)
input_ids = tokenizer(text, return_tensors="pt").input_ids

print('Loading model {}...'.format(model_id))
if args.quant_type == "int8":
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
else:
    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                            bnb_4bit_quant_type=args.quant_type,
                                            bnb_4bit_use_double_quant=False,
                                            bnb_4bit_compute_dtype=torch_dtype)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, torch_dtype=torch_dtype, )

with torch.no_grad():
    # warmup
    model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS)
    model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS)
    print("warm-up complite")
    t0 = time.time()
    generated_ids = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1)
    latency = time.time() - t0
    print(input_ids.shape)
    print(generated_ids.shape)
    result = "| latency: " + str(round(latency * 1000, 3)) + " ms |"
    print('+' + '-' * (len(result) - 2) + '+')
    print(result)
    print('+' + '-' * (len(result) - 2) + '+')

output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"output: {output}")
