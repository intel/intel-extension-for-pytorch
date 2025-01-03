import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset
from accelerate import Accelerator

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", default="facebook/opt-6.7b", required=False, type=str, help="model_name")
parser.add_argument("--quant_type", default="int8", type=str, help="quant type", choices=["int8", "nf4", "fp4"])
parser.add_argument("--device", default="cpu", type=str, help="device type", choices=["cpu", "xpu"])
parser.add_argument("--lora_r", default=8, type=int, help="LoRA rank")
parser.add_argument("--lora_alpha", default=8, type=int, help="LoRA alpha")
parser.add_argument("--max_seq_length", default=1, type=int, help="Maximum sequence length. Sequences will be right padded (and possibly truncated).")
parser.add_argument("--kbit_training_prepare", action="store_true", help="prepare model for kbit training.")
parser.add_argument("--per_device_train_batch_size", default=4, type=int)
parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
args = parser.parse_args()

import intel_extension_for_pytorch
import random
import numpy as np
seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.xpu.manual_seed_all(seed)


def get_current_device():
    return Accelerator().process_index

device_map={'':get_current_device()} if args.device == 'xpu' else None

if args.quant_type == "int8":
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
else:
    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                            bnb_4bit_quant_type=args.quant_type,
                                            bnb_4bit_use_double_quant=False,
                                            bnb_4bit_compute_dtype=torch.bfloat16)

model_id = args.model_name
model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map,quantization_config=quantization_config)

tokenizer = AutoTokenizer.from_pretrained(model_id)
if args.kbit_training_prepare:
    model = prepare_model_for_kbit_training(model)
    
tokenizer.pad_token = tokenizer.eos_token

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

config = LoraConfig(
    r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"], padding="max_length", max_length=args.max_seq_length, truncation=True), batched=True)


trainer = Trainer(
    model=model,
    train_dataset=data["train"],
    args=TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=100,
        max_steps=200,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=1,
        output_dir="outputs",
        use_cpu=True if args.device=='cpu' else False,
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
