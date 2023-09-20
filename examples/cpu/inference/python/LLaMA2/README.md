<!--- 0. Title -->
# PyTorch LLaMA2 inference

<!-- 10. Description -->
## Description

This directory has a Jupyter notebook for running [LLaMA2*](https://ai.meta.com/llama/) inference using Intel-optimized PyTorch.

## General setup
This Jupyter notebook requires following instructions on https://github.com/intel/intel-extension-for-pytorch/blob/llm_feature_branch/examples/cpu/inference/python/llm/README.md to
generate prequantized LLaMA2 model.
Please note that when v2.1 of Intel Extension for PyTorch would be released, then using this custom branch wouldn't be required.

You can also [set up the environment by running this script, which would build some components from source](https://github.com/intel/intel-extension-for-pytorch/blob/llm_feature_branch/scripts/compile_bundle.sh).
We preload Intel OpenMP & tcmalloc as well.

\* Other names & brands may be claimed as the property of others
