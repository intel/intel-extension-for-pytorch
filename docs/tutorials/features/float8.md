Float8 Data Type Support (Prototype)
====================================

## Float8 Data Type

Float8 (FP8) is a 8-bit floating point data type, which is used to reduce memory footprint, improve the computation efficiency and save power in Deep Learning domain.

Two formats are used in FP8 training and inference, in order to meet the required value range and precision of activation, weight and gradient in Deep Neural Network (DNN). One is E4M3 (sign-exponent-mantissa) for activation and weight, the other is E5M2 for gradients. These two formats are defined in [FP8 FORMATS FOR DEEP LEARNING](https://arxiv.org/pdf/2209.05433.pdf).

## FP8 Quantization

On GPU, online Dynamic Quantization is used for FP8 data compression and decompression. Delayed Scaling algorithm is used for accelerating the quantizaiton process.

## Supported running mode

Both DNN Training and Inference are supported with the FP8 data type.

## Supported operators

FP8 Linear operator is supported.

## FP8 usage example

BERT model is supported as a FP8 training showcase, see the following example:

```python
from intel_extension_for_pytorch.xpu.fp8.fp8 import fp8_autocast
from intel_extension_for_pytorch.xpu.fp8.recipe import DelayedScaling
from intel_extension_for_pytorch.nn.utils._fp8_convert import convert_fp8_model

## 'fp8_autocase' is the handler of FP8
with fp8_autocast(enabled=True, fp8_recipe=DelayedScaling()):
    ## The original model will be automatically converted to a new model with FP8 operators with 'convert_fp8_model'
    convert_fp8_model(model)
    outputs = model(input_ids=input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    labels=masked_lm_labels,
                    next_sentence_label=next_sentence_labels)
```
