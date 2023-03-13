Fast BERT (Experimental)
========================

### Feature Description

Intel proposed a technique, Tensor Processing Primitives (TPP), a programming abstraction striving for efficient, portable implementation of DL workloads with high-productivity. TPPs define a compact, yet versatile set of 2D-tensor operators (or a virtual Tensor ISA), which subsequently can be utilized as building-blocks to construct complex operators on high-dimensional tensors. Detailed contents are available at [*Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads*](https://arxiv.org/pdf/2104.05755.pdf).

Implementation of TPP is integrated into Intel® Extension for PyTorch\*. BERT could benefit from this new technique, for both training and inference.

### Prerequisite

- Transformers 4.6.0 ~ 4.20.0

### Usage Example

An API `ipex.fast_bert` is provided for a simple usage. Usage of this API follows the pattern of `ipex.optimize` function. More detailed description of API is available at [Fast BERT API doc](../api_doc)

[//]: # (marker_inf_bert_fast_bf16)
[//]: # (marker_inf_bert_fast_bf16)
