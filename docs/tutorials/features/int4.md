INT4 inference [GPU] (Experimentatal)
=====================================

## INT4 DataType

INT4 is 4-bit fixed point which is used to reduce memory footprint, improve the computation efficiency and save power in Deep Learning domain.

INT4 data type is being used in weight only quantization in current stage. It will be converted to Float16 data type for computation.

## INT4 Quantization

On GPU, offline Weight Only Quantization (WOQ) is used for INT4 data compression. WOQ calibration tool using Generative Pre-trained Transformer models Quantization (GPT-Q) algorithm is created for improving the accuracy for INT4 weight quantization.

## Supported running mode

DNN Inference is supported with INT4 data type.

## Supported operators

INT4 Linear operator and widely used linear fusion operators in Large Langugue Models like `mm_qkv_int4`, `mm_bias_int4`, `mm_silu_int4`, `mm_resmul_int4`, `mm_bias_gelu_int4`, `mm_bias_resadd_resadd_int4` are supported.

## INT4 usage example

You can use a well quantized INT4 model to perform INT4 inference directly, or use the WOQ tool to compress the high precision model to INT4 model firstly, then to execute INT4 inference with IPEX on GPU.

### Weight Only Quantization Tool

This tool is used for applying quantization to the given model using gptq method.

Please note that we only support HuggingFace transformers model structure at present. GPT-J-6B is a model we intensively verified.

```python
from transformers import GPTJForCausalLM

model_path = ...
dataset = ...
model = GPTJForCausalLM.from_pretrained(model_path)
model.eval()

ipex.quantization._gptq(model, dataset, 'quantized_weight.pt', wbits=4)
```
