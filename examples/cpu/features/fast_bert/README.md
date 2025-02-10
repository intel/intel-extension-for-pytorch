# Feature Description:

`ipex.fast_bert` proposed a technique to speed up BERT workloads. Implementation leverages the idea from [Tensor Processing Primitives](https://arxiv.org/pdf/2104.05755.pdf).

Currently `ipex.fast_bert` API is only well optimized for training. For inference, it ensures functionality, while to get peak perf, please use `ipex.optimize` API + torchscript.

# Prerequisite:
Transformers 4.6.0 ~ 4.46.2

# Usage Example:
Training:
```
python fast_bert_training_bf16.py
```
Inference:
```
python fast_bert_inference_bf16.py
```
