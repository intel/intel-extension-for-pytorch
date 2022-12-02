# PyTorch JIT pass for DNNL

This folder contains experimental passes that optimize PyTorch Graph to utilize full power of DNNL (or other optimizations). It chose to use PyTorch namespace for eazy migration into main repo in the future. Abstract graph manipulation part of JIT should completely independent of other modules in extension, which means no reference to any symbols in other files of the project.

>Note:
>
>csrc/jit is only used in CPU module.
>csrc/gpu/jit is used in GPU module.
>
>We will merge and restructure jit feature in the further.