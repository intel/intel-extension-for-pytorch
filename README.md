# Intel® Extension for PyTorch

Intel Extension for PyTorch is a Python package to extend official PyTorch. It is designed to make the Out-of-Box user experience of PyTorch CPU better while achieving good performance. The extension also will be the PR(Pull-Request) buffer for the Intel PyTorch framework dev team. The PR buffer will not only contain functions, but also optimization (for example, take advantage of Intel's new hardware features).

 - [Installation](#installation)
	 - [Install PyTorch from Source](#install-pytorch-from-source)
	 - [Install Intel Extension for PyTorch from Source](#install-intel-extension-for-pytorch-from-source)
 - [Getting Started](#getting-started)
     - [Automatically Mix Precison](#automatically-mix-precision)
 - [Contribution](#contribution)
 - [License](#license)

## Installation

### Install PyTorch from Source

 1. Get PyTorch v1.5.0-rc3 source(Refer to [PyTorch guide](https://github.com/pytorch/pytorch#get-the-pytorch-source) for more details)
    ```bash
    git clone --recursive https://github.com/pytorch/pytorch
    cd pytorch

    # checkout source code to the specified version
    git checkout v1.5.0-rc3

    # update submodules for the specified PyTorch version
    git submodule sync
    git submodule update --init --recursive
    ```

 2. Get Intel PyTorch Extension source
    ```bash
    git clone --recursive https://github.com/intel/intel-extension-for-pytorch
    cd intel-extension-for-pytorch

    # if you are updating an existing checkout
    git submodule sync
    git submodule update --init --recursive
    ```

 3. Add an new backend for Intel Extension for PyTorch
    ```bash
    # Apply git patch to pytorch code
    cd ${pytorch_directory}
    git apply ${intel_extension_for_pytorch_directory}/torch_patches/dpcpp-v1.5-rc3.patch
    ```

 4. Build and install PyTorch (Refer to [PyTorch guide](https://github.com/pytorch/pytorch#install-pytorch) for more details)
    ```bash
    cd ${pytorch_directory}
    python setup.py install
    ```

### Install Intel Extension for PyTorch from Source
Install dependencies
```bash
pip install lark-parser hypothesis
```

Install the extension
```bash
cd ${intel_extension_for_pytorch_directory}
python setup.py install
```

## Getting Started

If you want to explore Intel Extension for PyTorch, you just need to convert the model and input tensors to the extension device, then the extension will be enabled automatically. Take an example, the code as follows is a model without the extension.
```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(4, 5)

    def forward(self, input):
        return self.linear(input)

input = torch.randn(2, 4)
model = Model()
res = model(input)
```
You just need to transform the above python script as follows and then the extension will be enabled and accelerate the computation automatically. Besides that the not only imperative mode but also JIT mode.
```python
import torch
import torch.nn as nn

# Import Extension
import intel_pytorch_extension as ipex

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(4, 5)

    def forward(self, input):
        return self.linear(input)

# Convert the input tensor to the Extension device
input = torch.randn(2, 4).to(ipex.DEVICE)
# Convert the model to the Extension device
model = Model().to(ipex.DEVICE)

res = model(input)
```

### Automatically Mix Precision
In addition, Intel Extension for PyTorch supports the mixed precision. It means that some operators of a model may run with Float32 and some other operators may run with BFloat16 or INT8.
In traditional, if you want to run a model with a low precision type, you need to convert the parameters and the input tensors to the low precision type manually. And if the model contains some operators that do not support the low precision type, then you have to convert back to Float32. Round after round until the model can run normally.
The extension can simply the case, you just need to enable the auto-mix-precision as follows, then you can benefit from the low precision. Currently, the extension only supports BFloat16.
```python
import torch
import torch.nn as nn

import intel_pytorch_extension as ipex
# Automatically mix precision
ipex.enable_auto_optimization(mixed_dtype = torch.bfloat16)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(4, 5)

    def forward(self, input):
        return self.linear(input)

input = torch.randn(2, 4).to(ipex.DEVICE)
model = Model().to(ipex.DEVICE)

res = model(input)
```


## Contribution

Please submit PR or issue to communicate with us or contribute code.


## License

_Apache License_, Version _2.0_. As found in [LICENSE](https://github.com/intel/intel-extension-for-pytorch/blob/master/LICENSE.txt) file.
