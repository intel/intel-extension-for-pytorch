# Training Models with Intel® Extension for PyTorch\* Optimizations

We provided some examples about how to use Intel® Extension for PyTorch\* to accelerate model training.

## Environment Setup

Basically we need to install PyTorch\* (along with related packages like torchvision, torchaudio) and Intel® Extension for PyTorch\* in a Python3 environment.

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install intel-extension-for-pytorch
```

For more details, please check [installation guide](https://intel.github.io/intel-extension-for-pytorch/index.html#installation).

## Running Example Scripts

```bash
# Clone the repository and access to the training examples folder
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch/examples/cpu/training
```

Running ResNet50 Float32 single precision training example:

```bash
python single_instance_training_fp32.py
```

Running ResNet50 BFloat16 half precision training example:

```bash
python single_instance_training_bf16.py
```

Please check [training examples in Intel® Extension for PyTorch\* online document](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/examples.html#training) for more details.

If you would like to use distributed training via PyTorch\* DDP, please check [oneAPI Collective Communications Library Bindings for Pytorch\* Github repository](https://github.com/intel/torch-ccl) for more information and examples.