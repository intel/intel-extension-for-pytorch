# Intel® Extension for PyTorch

Intel® Extension for PyTorch (IPEX) is a Python package to extend official PyTorch. It is designed to make the Out-of-Box user experience of PyTorch CPU better while achieving good performance. The extension also will be the PR(Pull-Request) buffer for the Intel PyTorch framework dev team. The PR buffer will not only contain functions, but also optimization (for example, take advantage of Intel's new hardware features).

 - [Installation](#installation)
     - [Install PyTorch](#install-pytorch)
     - [Install Intel Extension for PyTorch from Source](#install-intel-extension-for-pytorch-from-source)
 - [Getting Started](#getting-started)
     - [Automatically Mix Precison](#automatically-mix-precision)
        - [BFloat16](#BFloat16)
        - [INT8](#int8-quantization)
     - [Supported Customized Operators](#supported-customized-operators)
     - [Supported Fusion Patterns](#supported-fusion-patterns)
 - [Tutorials](#tutorials)
 - [Joint blogs](#joint-blogs)
 - [License](#license)

## Installation

### Install PyTorch (Optional)
 |IPEX Version|PyTorch Version|
 |--|--|
 |[v1.8.0](https://github.com/intel/intel-extension-for-pytorch/tree/v1.8.0)|[v1.8.0](https://github.com/pytorch/pytorch/tree/v1.8.0 "v1.8.0")|
 |[v1.2.0](https://github.com/intel/intel-extension-for-pytorch/tree/v1.2.0)|[v1.7.0](https://github.com/pytorch/pytorch/tree/v1.7.0 "v1.7.0")|
 |[v1.1.0](https://github.com/intel/intel-extension-for-pytorch/tree/v1.1.0)|[v1.5.0-rc3](https://github.com/pytorch/pytorch/tree/v1.5.0-rc3 "v1.5.0-rc3")|
 |[v1.0.2](https://github.com/intel/intel-extension-for-pytorch/tree/v1.0.2)|[v1.5.0-rc3](https://github.com/pytorch/pytorch/tree/v1.5.0-rc3 "v1.5.0-rc3")|
 |[v1.0.1](https://github.com/intel/intel-extension-for-pytorch/tree/v1.0.1)|[v1.5.0-rc3](https://github.com/pytorch/pytorch/tree/v1.5.0-rc3 "v1.5.0-rc3")|
 |[v1.0.0](https://github.com/intel/intel-extension-for-pytorch/tree/v1.0.0)|[v1.5.0-rc3](https://github.com/pytorch/pytorch/tree/v1.5.0-rc3 "v1.5.0-rc3")|

For IPEX version earlier than 1.8.0, a patch has to be manually applied to PyTorch source code. Please check previous installation guide.

From IPEX 1.8.0, compiling PyTorch from source is not required. If you still want to compile PyTorch, please follow instructions [here](https://github.com/pytorch/pytorch#installation). Please make sure to checkout the correct PyTorch version according to the table above.

**Note:** Compiling with gcc 7 on some environments, like CentOS 7, may fail. Please use GCC >= 8 to compile.

**Note:** Installing IPEX will automatically invoke installation of the corresponding version of PyTorch.

### Install IPEX via wheel file

```
python -m pip install torch_ipex==1.8.0 -f https://software.intel.com/ipex-whl-stable
```

:information_source: Wheel files availability for Python versions

| IPEX Version | Python 3.6 | Python 3.7 | Python 3.8 | Python 3.9 |
| :--: | :--: | :--: | :--: | :--: |
| 1.8.0 |  | :heavy_check_mark: | | |

**Note**: Currently we only provide wheel file for Python 3.7. For other Python versions, please follow instructions in the following section to compile from source.

### Install IPEX by compiling from source

```bash
git clone --recursive https://github.com/intel/intel-extension-for-pytorch
cd intel-extension-for-pytorch

# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive

# run setup.py to compile IPEX and install the binaries
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
You just need to transform the above python script as follows and then the extension will be enabled and accelerate the computation automatically.
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

#### BFloat16
```python
import torch
import torch.nn as nn

import intel_pytorch_extension as ipex
# Automatically mix precision
ipex.enable_auto_mixed_precision(mixed_dtype = torch.bfloat16)

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
#### INT8 Quantization
Currently, Intel Extension for PyTorch has supported static and symmetric quantization. Development of dynamic quantization is undergoing. And asymmetric quantization will be enabled once oneDNN is upgraded to v2.0 or higher versions.

How to quantize the following model:
```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(3, 64, 7, stride=2)

    def forward(self, input):
        return self.conv(input).relu()
```
Firstly we need to do calibration step against a representative dataset (set ```running_mode``` to ```calibration```):
```python
# Convert the model to the Extension device
model = Model().to(ipex.DEVICE)

# Create a configuration file to save quantization parameters.
conf = ipex.AmpConf(torch.int8)
with torch.no_grad():
    for x in cali_dataset:
        # Run the model under calibration mode to collect quantization parameters
        with ipex.AutoMixPrecision(conf, running_mode='calibration'):
            y = model(x.to(ipex.DEVICE))
# Save the configuration file
conf.save('configure.json')
```
The content of the configuration file is as follows.

```json
[
    {
        "id": 0,
        "name": "Convolution",
        "algorithm": "min_max",
        "weight_granularity": "per_channel",
        "inputs_scale": [
            25.05583953857422
        ],
        "outputs_scale": [
            43.98969650268555
        ],
        "inputs_uint8_used": [
            false
        ],
        "outputs_uint8_used": [
            false
        ],
        "quantized": true
    },
    {
        "id": 1,
        "name": "Relu",
        "algorithm": "min_max",
        "weight_granularity": "per_channel",
        "inputs_scale": [
            43.98969650268555
        ],
        "outputs_scale": [
            43.98969650268555
        ],
        "inputs_uint8_used": [
            false
        ],
        "outputs_uint8_used": [
            false
        ],
        "quantized": true
    }
]
```
- ```id``` is a sequence number of operators which were quantized statically in the calibration step.
**Manually changing this value will cause unexpected behaviors**.
- ```name``` is the name of the operator to be quantized.
- ```algorithm``` indicates how to calculate the scales of the observed tensors. Currently only ```min_max``` is supported.
- ```weight_granularity``` controls how to quantize the operator weights. The ```Convolution``` and ```Linear``` both supports  ```per_channel``` and ```per_tensor```. And the other operators only supports ```per_tensor```.
- ```inputs_scale``` and ```outputs_scale``` are the scales to quantize the input tensors and output tensors respectively.
- ```inputs_uint8_used``` and ```outputs_uint8_used``` indicate whether to use ```int8``` or ```uint8```. Default value is ```false```, indicating that ```int8``` is used.
- ```quantized``` determines whether this operator should be quantized or not during inference.

After doing calibration step, we can use the saved configuration json file to do evalution (set ```running_mode``` to ```inference```):
```python
conf = ipex.AmpConf(torch.int8, 'configure.json')
with torch.no_grad():
    for x in cali_dataset:
        with ipex.AutoMixPrecision(conf, running_mode='inference'):
            y = model(x.to(ipex.DEVICE))
```

Supported Quantization Operators:
- ```Convoluton```
- ```BatchNorm```
- ```MaxPooling```
- ```AvgPooling```
- ```AdaptivePooling```
- ```Linear```
- ```convolution + relu```
- ```convolution + sum```
- ```convolution + sum + relu```
- ```convolution + BatchNorm```



### Supported Customized Operators
* ROIAlign
* NMS
* BatchScoreNMS
* MLP
* Interaction
* FrozenBatchNorm2d

### Supported Fusion Patterns
* Conv2D + ReLU
* Conv2D + SUM
* Conv2D + SUM + ReLU
* Conv2D + Sigmoid
* Conv2D + Sigmoid + MUL
* Conv2D + HardTanh
* Conv2D + ELU
* Conv3D + ReLU
* Conv3D + SUM
* Conv3D + SUM + ReLU
* Linear + ReLU
* Linear + GELU
* View + Transpose + Contiguous + View

## Tutorials
*  [Performance Tuning](tutorials/Performance_Tuning.md)

## Joint-blogs
* [Intel and Facebook Accelerate PyTorch Performance with 3rd Gen Intel® Xeon® Processors and Intel® Deep Learning Boost’s new BFloat16 capability](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/intel-facebook-boost-bfloat16.html)
* [Accelerate PyTorch with IPEX and oneDNN using Intel BF16 Technology](https://medium.com/pytorch/accelerate-pytorch-with-ipex-and-onednn-using-intel-bf16-technology-dca5b8e6b58f)
* [Scaling up BERT-like model Inference on modern CPU - Part 1 by IPEX launcher](https://huggingface.co/blog/bert-cpu-scaling-part-1)


## Contribution

Please submit PR or issue to communicate with us or contribute code.


## License

_Apache License_, Version _2.0_. As found in [LICENSE](https://github.com/intel/intel-extension-for-pytorch/blob/master/LICENSE.txt) file.
