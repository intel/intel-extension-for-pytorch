# Training Models with Intel® Extension for PyTorch\* Optimizations

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

Setting environment variables:

Set the working directory as the root folder of `oneccl_bindings_for_pytorch` package.
For a conda-based Python 3.10 environment, it should be 

```bash
cd <CONDA_ENV_ROOT>/lib/python3.10/site-packages/oneccl_bindings_for_pytorch
```

in which `CONDA_ENV_ROOT` path can be checked via `conda env list`.

Then run the environment variables activation script.

```bash
source env/setvars.sh
export FI_TCP_IFACE="$(ip -o -4 route show to default | awk '{print $5}')"
```

Clone the project repo if you haven't done so, access to the training examples folder and run the ResNet50 distributed training example:

```bash
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch/examples/cpu/training/python-scripts

# This example command would utilize all the numa sockets of the processor, taking each socket as a rank.
ipexrun --nnodes 1 distributed_data_parallel_training.py
```

Please check [the training examples in Intel® Extension for PyTorch\* online document](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/examples.html#training) for more details.

For more information and examples about distributed training via PyTorch\* DDP, please visit [oneAPI Collective Communications Library Bindings for Pytorch\* Github repository](https://github.com/intel/torch-ccl).
