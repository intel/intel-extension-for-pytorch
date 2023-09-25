'''
Ported from Intel(R) Extension for Transformers
https://github.com/intel/intel-extension-for-transformers/blob/53bed434f16cba1fff6cdb30749d3ea545e56ee5/examples/huggingface/pytorch/language-modeling/quantization/run_clm_no_trainer.py
With unused code removed.
'''

import argparse
import sys
sys.path.append('./')
import time
import re
from pathlib import Path
import torch
from datasets import load_dataset
from torch.nn.functional import pad
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", nargs="?", default="EleutherAI/gpt-j-6b"
)
parser.add_argument("--dataset", nargs="?", default="lambada", const="lambada")
parser.add_argument("--output-dir", nargs="?", default="./saved_results")
parser.add_argument("--calib-iters", default=512, type=int,
                    help="calibration iters.")
args = parser.parse_args()


class Evaluator:
    def __init__(self, dataset, tokenizer, batch_size=8, pad_val=1, pad_max=196, is_calib=False):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.pad_val = pad_val
        self.pad_max = pad_max
        self.is_calib = is_calib

        # tokenize the dataset
        self.dataset = self.dataset.map(self.tokenize_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids"])

    @torch.no_grad()
    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"])

    @torch.no_grad()
    def collate_batch(self, batch):

        input_ids_padded = []
        last_ind = []

        for text in batch:
            input_ids = text["input_ids"]
            pad_len = self.pad_max - input_ids.shape[0]
            last_ind.append(input_ids.shape[0] - 1)
            if self.is_calib:
                input_ids = input_ids[:self.pad_max] if len(input_ids) > self.pad_max else input_ids
            else:
                input_ids = pad(input_ids, (0, pad_len), value=self.pad_val)
            input_ids_padded.append(input_ids)

        return (torch.vstack(input_ids_padded), torch.tensor(last_ind))

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        latency = 0
        test_dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_batch,
        )
        for i, (input_ids, last_ind) in enumerate(test_dataloader):
            label = input_ids[torch.arange(len(last_ind)), last_ind]
            input_ids[torch.arange(len(last_ind)), last_ind] = self.pad_val
            pad_len = self.pad_max - last_ind - 1

            start = time.time()
            outputs = model(input_ids)
            latency += time.time() - start

            last_token_logits = outputs[0][torch.arange(len(last_ind)), -2 - pad_len, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
            if (i + 1) % 50 == 0:
                print(hit / total)
                print("Processed minibatch:", i)

        acc = hit / total
        print("Accuracy: ", acc)
        latency = latency / len(self.dataset)
        print("Latency: ", latency)
        return acc


def get_user_model():
    from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
    torchscript = False
    if re.search("llama", args.model.lower()):
        from transformers import LlamaForCausalLM, LlamaTokenizer
        user_model = LlamaForCausalLM.from_pretrained(
            args.model,
            torchscript=torchscript,  # torchscript will force `return_dict=False` to avoid jit errors
        )
        tokenizer = LlamaTokenizer.from_pretrained(args.model)
    elif re.search("mpt-7b-chat", args.model.lower()):
        from mpt_7b.modeling_mpt import MPTForCausalLM
        user_model = MPTForCausalLM.from_pretrained(
            args.model,
            torchscript=torchscript,  # torchscript will force `return_dict=False` to avoid jit errors
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        user_model.config.use_cache = True
    elif re.search("falcon-7b-instruct", args.model.lower()):
        from falcon_7b_instruct.modelling_RW import RWForCausalLM
        user_model = RWForCausalLM.from_pretrained(
            args.model,
            torchscript=torchscript,  # torchscript will force `return_dict=False` to avoid jit errors
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        user_model.config.use_cache = True
    elif re.search("chatglm", args.model.lower()):
        user_model = AutoModel.from_pretrained(
            args.model,
            torchscript=torchscript,  # torchscript will force `return_dict=False` to avoid jit errors
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    else:
        user_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torchscript=torchscript,  # torchscript will force `return_dict=False` to avoid jit errors
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Set model's seq_len when GPTQ calibration is enabled.
    user_model.seqlen = 2048

    # to channels last
    user_model = user_model.to(memory_format=torch.channels_last)
    user_model.eval()
    return user_model, tokenizer


# dataset
user_model, tokenizer = get_user_model()
calib_dataset = load_dataset(args.dataset, split="train")
# calib_dataset = datasets.load_from_disk('/your/local/dataset/pile-10k/') # use this if trouble with connecting to HF
calib_dataset = calib_dataset.shuffle(seed=42)
calib_evaluator = Evaluator(calib_dataset, tokenizer, batch_size=1, pad_max=512, is_calib=True)
calib_dataloader = DataLoader(
    calib_evaluator.dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=calib_evaluator.collate_batch,
)

def calib_func(prepared_model):
    for i, calib_input in enumerate(calib_dataloader):
        if i > args.calib_iters:
            break
        prepared_model(calib_input[0])

recipes = {}
eval_func = None
from neural_compressor import PostTrainingQuantConfig, quantization
# specify the op_type_dict and op_name_dict
op_type_dict = {
    '.*': {  # re.match
        "weight": {
            'bits': 4,  # only support 4-bit for now
            'group_size': -1,  # only support per-channel for now
            'scheme': 'asym',  # only support asym for now
            'algorithm': 'GPTQ',  # RTN/AWQ/TEQ
        },
    },
}
op_name_dict = {
    'lm_head': {"weight": {'dtype': 'fp32'}, },
    'embed_out': {"weight": {'dtype': 'fp32'}, },  # for dolly_v2
}
recipes["rtn_args"] = {
    "enable_mse_search": False,
    "enable_full_range": False,
}
recipes['gptq_args'] = {
    'percdamp': .01,
    'act_order': False,
    'block_size': 128,
    'nsamples': 128,
    'use_max_length': False
}

conf = PostTrainingQuantConfig(
    approach='weight_only',
    op_type_dict=op_type_dict,
    op_name_dict=op_name_dict,
    recipes=recipes,
)

q_model = quantization.fit(
    user_model,
    conf,
    calib_dataloader=calib_dataloader,
    calib_func=calib_func,
    eval_func=eval_func,
)
compressed_model = q_model.export_compressed_model(
    compression_dtype=torch.int32,
    compression_dim=1,
    scale_dtype=torch.float16,
)
Path(args.output_dir).mkdir(parents=True, exist_ok=True)
torch.save(compressed_model.state_dict(), args.output_dir + "/gptq_checkpoint.pt")
print('\n Checkpoint saved to', args.output_dir + "/gptq_checkpoint.pt \n")
