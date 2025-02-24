from datetime import datetime
import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch.llm.functional.utils import ipex_update_causal_mask
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from peft import prepare_model_for_kbit_training,LoraConfig,PeftModel,get_peft_model
from datasets import load_dataset
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B", required=False, type=str, help="model_name")
parser.add_argument("--quant_type", default="nf4", type=str, help="quant type", choices=["int8", "nf4", "fp4"])
parser.add_argument("--device", default="xpu", type=str, help="device type", choices=["cpu", "xpu"])
parser.add_argument("--lora_r", default=8, type=int, help="LoRA rank")
parser.add_argument("--lora_alpha", default=16, type=int, help="LoRA alpha")
parser.add_argument("--max_seq_length", default=512, type=int, help="Maximum sequence length. Sequences will be right padded (and possibly truncated).")
parser.add_argument("--per_device_train_batch_size", default=2, type=int)
parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
parser.add_argument("--max_steps", default=300, type=int)
parser.add_argument("--do_eval", action="store_true")

args = parser.parse_args()

seed = 123
torch.manual_seed(seed)
torch.xpu.manual_seed_all(seed)

# Quantization configuration
if args.quant_type == "int8":
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
elif args.quant_type == "nf4":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )


# Loading the model and tokenizer
device_map={'':torch.xpu.current_device()} if args.device == 'xpu' else None
model = AutoModelForCausalLM.from_pretrained(args.model_name ,quantization_config=bnb_config, device_map=device_map)
tokenizer = AutoTokenizer.from_pretrained(
    args.model_name,
    model_max_length=args.max_seq_length,
    padding_side="left",
    add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token

train_dataset = load_dataset('gem/viggo', split='train',trust_remote_code=True)
if args.do_eval:
    eval_dataset = load_dataset('gem/viggo', split='validation',trust_remote_code=True)

def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=args.max_seq_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt =f"""Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values.
This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute'].
The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']


### Target sentence:
{data_point["target"]}


### Meaning representation:
{data_point["meaning_representation"]}
"""
    return tokenize(full_prompt)

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
if args.do_eval:
    tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model) 

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

lora_targets = []
if args.model_name in ["facebook/opt-6.7b", "Qwen/Qwen2-1.5B"]:
    lora_targets = ["q_proj", "v_proj"]
elif args.model_name in ["microsoft/Phi-3-mini-4k-instruct"]:
    lora_targets = ["o_proj", "qkv_proj"]
elif args.model_name in ["meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.2-3B-Instruct"]:
    lora_targets = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ]

assert lora_targets != [], "lora_targets should not be empty. If you use absolute path to load the model, pls manually set the target as above."

config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=lora_targets,
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

project = "viggo-finetune"
base_model_name = args.model_name
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

tokenizer.pad_token = tokenizer.eos_token
ipex_update_causal_mask(model)

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset if args.do_eval else None,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=5,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,         # Add a more higher value for better performance
        learning_rate=2.5e-5, # Want about 10x smaller than the pre-trained learning rate
        logging_steps=1,
        bf16=True,
        # optim="paged_adamw_8bit",
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="no",  # Disable saving the model
        eval_strategy="steps" if args.do_eval else "no" , # Evaluate the model every logging step
        eval_steps=50,               # Evaluate and save checkpoints every 50 steps
        do_eval=args.do_eval,                # Perform evaluation at the end of training
        report_to='tensorboard',           # set to 'wandb' for weights & baises logging
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",          # Name of the W&B run (optional)
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
