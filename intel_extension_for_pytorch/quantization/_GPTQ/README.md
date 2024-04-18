GPTQ
====

## Introduction
Compared to normal quantization like W8A8, weight only quantization is probably a better trade-off to balance the performance and the accuracy. There are many excellent works for weight only quantization to improve its accuracy performance. GPTQ is a new one-shot weight quantization method based on approximate second-order information, that is both highly-accurate and highly-efficient. The weights of each column are updated based on the fixed-scale pseudo-quantization error and the inverse of the Hessian matrix calculated from the activations. The updated columns sharing the same scale may generate a new max/min value, so the scale needs to be saved for restoration.

## Arguments
|  gptq_args  | Default Value |                               Comments                              |
|:----------:|:-------------:|:-------------------------------------------------------------------:|
| wbits | 4 | Data type for weight |
| group_size | 128 | Controls quantization granularity along input channel (IC) dimension of weight |
| sym | False | Scheme. Default to be asym per checkpoint requirement. |
|  act_order | False |  Whether to sort Hessian's diagonal values to rearrange channel-wise quantization order|
|  percdamp | 0.01 | Percentage of Hessian's diagonal values' average, which will be added to Hessian's diagonal to increase numerical stability|
|  nsamples  | 128 |  Calibration samples' size |
|  pad_max_length  | 2048 | Whether to align calibration data to a fixed length. This value should not exceed model's acceptable sequence length.|
|  use_max_length  | False | Whether to align all calibration data to fixed length, which equals to pad_max_length. |
|  layer_wise  | False | Execute GPTQ quantization per block |
|  compression_dtype  |       torch.int32       |  Data type for compressed dtype, select from [torch.int8\|16\|32\|64]. |
|  compression_dim  |       1       |   0 means output channel while 1 means input channel.  |
|  scale_dtype  |       torch.float16       |  Data type for scale and bias.  |

## Use Case

#### Dataloader
Our GPTQ API takes an iterable containing calibration datasets, `torch.utils.data.DataLoader`, as input data.
Here is an example of creating a dataloader with a calibration dataset:
```py
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

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

calib_dataset = load_dataset(dataset_name, split="train")
calib_dataset = calib_dataset.shuffle(seed=42)
calib_evaluator = Evaluator(calib_dataset, tokenizer, batch_size=1, pad_max=512, is_calib=True)
calib_dataloader = DataLoader(
    calib_evaluator.dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=calib_evaluator.collate_batch,
)
```
For details, please refer to [example](../../../examples/cpu/inference/python/llm/utils/run_gptq.py)

#### Quantization
```py
import intel_extension_for_pytorch as ipex
from transformers import AutoModelForCausalLM
dataloader = ...
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# checkpoint will be saved under saved_dir
compressed_model = ipex.quantization.gptq(  
    model=model,
    dataloader=dataloader,
    group_size=128, 
    use_max_length=True,
    pad_max_length=512,
    compression_dtype=torch.int32,
    compression_dim=1,
    scale_dtype=torch.float16,
    saved_dir="./saved_results")

low_precision_checkpoint = torch.load(args.saved_dir)
config_dict = {
    "weight_key": "qweight",
    "scale_key": "scales",
    "zero_point_key": "qzeros",
    "bias_key": "bias"
}
state_dict_and_config = (low_precision_checkpoint, config_dict)
from intel_extension_for_pytorch.quantization import WoqWeightDtype
qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
    weight_dtype=WoqWeightDtype.INT4, # or WoqWeightDtype.INT8
    lowp_mode=ipex.quantization.WoqLowpMode.NONE, # or FP16, BF16, INT8
)
model = ipex.llm.optimize(
    model.eval(),
    dtype=amp_dtype,
    quantization_config=qconfig,
    inplace=True,
    low_precision_checkpoint=state_dict_and_config,
    deployment_mode=False,
)

# inference with model.generate()
```
For LLM example, please refer to [gpt-j](../../../examples/cpu/inference/python/llm/single_instance/run_int4_gpt-j_on_cnndailymail.py).
