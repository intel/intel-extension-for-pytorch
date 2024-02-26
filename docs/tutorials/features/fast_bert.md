Fast BERT (Prototype)
========================

### Feature Description

Intel proposed a technique to speed up BERT workloads. Implementation leverages the idea from [*Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads*](https://arxiv.org/pdf/2104.05755.pdf).

Currently `ipex.fast_bert` API is only well optimized for training. For inference, it ensures functionality, while to get peak perf, please use `ipex.optimize` API + torchscript.

### Prerequisite

- Transformers 4.6.0 ~ 4.31.0

### Usage Example

An API `ipex.fast_bert` is provided for a simple usage. Usage of this API follows the pattern of `ipex.optimize` function. More detailed description of API is available at [Fast BERT API doc](../api_doc)

[//]: # (marker_feature_fastbert_bf16)
[//]: # (marker_feature_fastbert_bf16)
