# Intel® Extension for PyTorch

Intel Extension for PyTorch is a Python package to extend official PyTorch. It is designed to make the Out-of-Box user experience of PyTorch CPU better while achieving good performance. The extension also will be the PR(Pull-Request) buffer for the Intel PyTorch framework dev team. The PR buffer will not only contain functions, but also optimization (for example, take advantage of Intel's new hardware features).

 - [Installation](#installation)
	 - [Install PyTorch from Source](#install-pytorch-from-source)
	 - [Install Intel PyTorch Extension from Source](#install-intel-pytorch-extension-from-source)
 - [Getting Started](#getting-started)
 - [Contribution](#contribution)
 - [License](#license)

## Installation

### Install PyTorch from Source

 1. Get PyTorch v1.5.0-rc3 source(Refer to [PyTorch guide](https://github.com/pytorch/pytorch#get-the-pytorch-source) for more details)
    ```bash
    git clone --recursive https://github.com/pytorch/pytorch
    cd pytorch

    # if you are updating an existing checkout
    git submodule sync
    git submodule update --init --recursive

    # checkout source code to the specified version
    git checkout v1.5.0-rc3
    ```

 2. Get Intel PyTorch Extension source
    ```bash
    git clone --recursive https://github.com/intel/intel-extension-for-pytorch
    cd intel-extension-for-pytorch
    
    # if you are updating an existing checkout
    git submodule sync
    git submodule update --init --recursive
    ```

 3. Add an new backend for Intel PyTorch Extension
    ```bash
    # Apply git patch to pytorch code
    cd ${intel_pytorch_extension_directory}
    git apply torch_patches/dpcpp-v1.5-rc3.patch ${pytorch_directory}
    ```
 
 4. Build and install PyTorch (Refer to [PyTorch guide](https://github.com/pytorch/pytorch#install-pytorch) for more details)
    ```bash
    cd ${pytorch_directory}
    python setup.py install
    ```

### Install Intel PyTorch Extension from Source
Install dependencies
```bash
pip install lark-parser hypothesis
```

Install the extension
```bash
cd ${intel_pytorch_extension_directory}
python setup.py install
```

## Getting Started

The user just needs to convert the model and input tensors to the extension device, then the extension will be enabled automatically. Take an example, the code as follows is a model without the extension.
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
If the user want to explore the Intel PyTorch Extension, you just need to transform the above python script as follows.
```python
import torch
import torch.nn as nn

# Import Intel PyTorch Extension
import intel_pytorch_extension

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(4, 5)

    def forward(self, input):
        return self.linear(input)

# Convert the input tensor to Intel PyTorch Extension device
input = torch.randn(2, 4).to('dpcpp')
# Convert the model to Intel PyTorch Extension device
model = Model().to('dpcpp')

res = model(input)
```

## Contribution

Please submit PR or issue to communicate with us or contribute code.


## License

_Apache License_, Version _2.0_. As found in [LICENSE](https://github.com/intel/intel-extension-for-pytorch/blob/master/LICENSE.txt) file.
