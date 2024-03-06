Performance
===========

## Overview

This page shows performance boost with Intel® Extension for PyTorch\* on several popular topologies.

## Performance Data for Intel® AI Data Center Products

Find the latest performance data for Intel® Data Center Max 1550 GPU, including detailed hardware and software configurations.

## LLM Performance

We benchmarked GPT-J 6B, LLaMA2 7B, 13B, OPT 6.7B, Bloom-7B with test input token length set to 1024. The datatype is FP16 for all the models.

![Single Tile](../images/performancev2.1.10/single_tile.png)

![Single Card](../images/performancev2.1.10/single_card.png)

![Two Card](../images/performancev2.1.10/two_card.png)

![Four Card](../images/performancev2.1.10/four_card.png)



### Configuration

#### Software Version

| Software | Version |
| :-: | :-: |
| PyTorch | [v2.1](https://pytorch.org/get-started/locally/) |
| Intel® Extension for PyTorch\* | [v2.1.10+xpu](https://github.com/intel/intel-extension-for-pytorch/releases) |
| Intel® oneAPI Base Toolkit | [2024.0](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) |
| Torch-CCL | [2.1.100](https://github.com/intel/torch-ccl/tree/ccl_torch2.1.100%2Bxpu) |
| GPU Driver |  [736.25](https://dgpu-docs.intel.com/releases/stable_736_25_20231031.html) |
| Transformers | [v4.31.0](https://github.com/huggingface/transformers/tree/v4.31.0) |  
| DeepSpeed | [commit 4fc181b0](https://github.com/microsoft/DeepSpeed/tree/4fc181b0)  |  
| Intel® Extension for DeepSpeed\* | [commit ec33277](https://github.com/intel/intel-extension-for-deepspeed/tree/ec33277)  | 


#### Hardware Configuration


CPU Configuration: 

| CPU | Intel(R) Xeon(R) Platinum 8480+ CPU  | 
| :-: | :-: | 
| Number of nodes | 1 | 
| Number of sockets | 2 | 
| Cores/Socket | 56 |
| Threads/Core | 2 | 
| uCode | 0x2b0004b1 | 
| Hyper-Threading | ON | 
| TurboBoost | ON | 
| BIOS version | SE5C7411.86B.9525.D25.2304190630 |  
| Number of DDR Memory slots | 16 | 
| Capacity of DDR memory per slot | 64GB | 
| DDR frequency | 4800 | 
| Total Memory/Node (DDR+DCPMM) | 1024GB | 
| Host OS | Ubuntu 22.04.3 LTS | 
| Host Kernel | 5.17.0-1020-oem | 
| [Spectre-Meltdown Mitigation](https://github.com/speed47/spectre-meltdown-checker) | Mitigated | 

Single tile of 4X PVC OAM Configuration: 

| GPU | Intel(R) Data Center Max 1550 GPU | 
| :-: | :-: | 
| IFWI | PVC.PS.B4.P.Si.2023.WW42.3\_25MHzi\_Quad\_DAMeni\_OAM600W\_IFRv2332i\_PSCnull\_IFWI.bin | 
| ECC | ON |
| AMC SW | AMC FW 6.2 | 
| Precision | FP16 | 

