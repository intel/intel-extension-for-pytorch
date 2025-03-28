﻿# Training Models with Intel® Extension for PyTorch\* Optimizations

We provided some examples about how to use Intel® Extension for PyTorch\* to accelerate model training.

## Environment Setup

Basically we need to install PyTorch\* (along with related packages like torchvision, torchaudio) and Intel® Extension for PyTorch\* in a Python3 environment.

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install intel-extension-for-pytorch
# oneCCL Bindings for PyTorch package is required for distributed training
python -m pip install oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
```

For more details, please check [installation guide](https://intel.github.io/intel-extension-for-pytorch/index.html#installation).

## Running Example Scripts

```bash
# Clone the repository and access to the training examples folder
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch/examples/cpu/training/python-scripts
```

Running ResNet50 distributed training example:

```bash
source $(python -c "import intel_extension_for_pytorch;import oneccl_bindings_for_pytorch as torch_ccl;print(torch_ccl.cwd)")/env/setvars.sh
export FI_TCP_IFACE="$(ip -o -4 route show to default | awk '{print $5}')"
# This example command would utilize all the numa sockets of the processor, taking each socket as a rank.
ipexrun --nnodes 1 distributed_data_parallel_training.py
```

Please check [the training examples in Intel® Extension for PyTorch\* online document](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/examples.html#training) for more details.

For more information and examples about distributed training via PyTorch\* DDP,
please visit [oneAPI Collective Communications Library Bindings for Pytorch\* Github repository](https://github.com/intel/torch-ccl).