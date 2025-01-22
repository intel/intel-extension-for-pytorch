
import torch
import intel_extension_for_pytorch
import time
import os

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

from datasets import load_dataset
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer

from peft import LoraConfig, get_peft_model
from packaging import version

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_flashattn: bool = field(
        default=False,
        metadata={
            "help": "Whether to use flash attention, now we support [`torch.nn.functional.scaled_dot_product_attention`]"
        })   
    use_peft: bool = field(
        default=False,
        metadata={
            "help": "Whether to use PEFT, default is LoRA`]"
        })   
    save_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to save model`]"
        })    

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    custom_mp: bool = field(
        default=False,
        metadata={
            "help": "Use custom mixed precision setting for PEFT LoRA, keep base model weight as bfloat16 and LoRA weight as float32. This setting is for fine-tuning on single card."
        })
    max_seq_length: int = field(
        default=128,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    inf_test: bool = field(
        default=False,
        metadata={
            "help": "Test an example for inference."
        })    
    
# loading dataset
dataset = load_dataset("financial_phrasebank", "sentences_allagree")
dataset = dataset["train"].train_test_split(test_size=0.1)
dataset["validation"] = dataset["test"]
del dataset["test"]

classes = dataset["train"].features["label"].names
dataset = dataset.map(
    lambda x: {"text_label": [classes[label] for label in x["label"]]},
    batched=True,
    num_proc=1,
)

# data preprocessing
text_column = "sentence"
label_column = "text_label"


def preprocess_function(examples, tokenizer, max_length):
    batch_size = len(examples[text_column])
    inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
    targets = [str(x) for x in examples[label_column]]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets, add_special_tokens=False)  # don't add bos token because we concatenate with inputs
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
        # print(i, sample_input_ids, label_input_ids)
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
    # print(model_inputs)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][-max_length:])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][-max_length:])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][-max_length:])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

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
    
    
def train():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    start = time.time()

    # distributed setup
    os.environ['CCL_PROCESS_LAUNCHER'] = 'none'

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    os.environ['CCL_LOCAL_SIZE'] = str(world_size)
    os.environ['CCL_LOCAL_RANK'] = str(local_rank)

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if version.parse(transformers.__version__) >= version.parse("4.37.0"):
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype = torch.bfloat16 if training_args.custom_mp else torch.float32, 
            attn_implementation="sdpa" if model_args.use_flashattn else "eager")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype = torch.bfloat16 if training_args.custom_mp else torch.float32)        

    # PEFT LoRA setting
    if model_args.use_peft:
        lora_config = LoraConfig(
            r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
    
    if training_args.custom_mp:
        # custom_mp is for single card
        device = 'xpu'
        model = model.to(device)
        
        # Mixed precision for LoRA: keep base weight as bf16, and lora to fp32
        for name, child in model.named_modules():
            if ("lora_A" in name) or ("lora_B" in name):   
                child = child.to(torch.float32)

    print_trainable_parameters(model)

    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
        fn_kwargs={"tokenizer": tokenizer, "max_length": training_args.max_seq_length}
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # set fsdp wrap policy, ensure the LoRA part could be warpped separately
    if model_args.use_peft and trainer.is_fsdp_enabled:
        from peft.utils.other import fsdp_auto_wrap_policy
        fsdp_plugin = trainer.accelerator.state.fsdp_plugin
        fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

    model.config.use_cache = True  # silence the warnings. Please re-enable for inference!

    trainer.train()
    
    if training_args.inf_test:
        model.eval()
        input_text = "In January-September 2009 , the Group 's net interest income increased to EUR 112.4 mn from EUR 74.3 mn in January-September 2008 ."
        inputs = tokenizer(input_text, return_tensors="pt").to("xpu")

        outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)

        print("input sentence: ", input_text)
        print(" output prediction: ", tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))

    print("total time = %.2f sec." % (time.time() - start))
    
    
if __name__ == "__main__":
    train()