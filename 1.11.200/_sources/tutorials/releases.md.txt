Releases
=============

## 1.11.200

### Highlights

- Enable more fused operators to accelerate particular models.
- Fuse `Convolution` and `LeakyReLU` ([#648](https://github.com/intel/intel-extension-for-pytorch/commit/d7603133f37375b3aba7bf744f1095b923ba979e))
- Support [`torch.einsum`](https://pytorch.org/docs/stable/generated/torch.einsum.html) and fuse it with `add` ([#684](https://github.com/intel/intel-extension-for-pytorch/commit/b66d6d8d0c743db21e534d13be3ee75951a3771d))
- Fuse `Linear` and `Tanh` ([#685](https://github.com/intel/intel-extension-for-pytorch/commit/f0f2bae96162747ed2a0002b274fe7226a8eb200))
- In addition to the original installation methods, this release provides Docker installation from [DockerHub](https://hub.docker.com/).
- Provided the <a class="reference external" href="installation.html#installation_onednn_graph_compiler">evaluation wheel packages</a> that could boost performance for selective topologies on top of oneDNN graph compiler prototype feature.
***NOTE***: This is still at an early development stage and not fully mature yet, but feel free to reach out through GitHub tickets if you have any suggestions.

**[Full Changelog](https://github.com/intel/intel-extension-for-pytorch/compare/v1.11.0...v1.11.200)**


## 1.11.0

We are excited to announce Intel® Extension for PyTorch\* 1.11.0-cpu release by tightly following PyTorch 1.11 release. Along with extension 1.11, we focused on continually improving OOB user experience and performance. Highlights include:

* Support a single binary with runtime dynamic dispatch based on AVX2/AVX512 hardware ISA detection
* Support install binary from `pip` with package name only (without the need of specifying the URL)
* Provide the C++ SDK installation to facilitate ease of C++ app development and deployment
* Add more optimizations, including graph fusions for speeding up Transformer-based models and CNN, etc
* Reduce the binary size for both the PIP wheel and C++ SDK (2X to 5X reduction from the previous version)

### Highlights
- Combine the AVX2 and AVX512 binary as a single binary and automatically dispatch to different implementations based on hardware ISA detection at runtime. The typical case is to serve the data center that mixtures AVX2-only and AVX512 platforms. It does not need to deploy the different ISA binary now compared to the previous version

    ***NOTE***:  The extension uses the oneDNN library as the backend. However, the BF16 and INT8 operator sets and features are different between AVX2 and AVX512. Please refer to [oneDNN document](https://oneapi-src.github.io/oneDNN/dev_guide_int8_computations.html#processors-with-the-intel-avx2-or-intel-avx-512-support) for more details. 

    > When one input is of type u8, and the other one is of type s8, oneDNN assumes that it is the user’s responsibility to choose the quantization parameters so that no overflow/saturation occurs. For instance, a user can use u7 [0, 127] instead of u8 for the unsigned input, or s7 [-64, 63] instead of the s8 one. It is worth mentioning that this is required only when the Intel AVX2 or Intel AVX512 Instruction Set is used.

- The extension wheel packages have been uploaded to [pypi.org](https://pypi.org/project/intel-extension-for-pytorch/). The user could directly install the extension by `pip/pip3` without explicitly specifying the binary location URL.

<table align="center">
<tbody>
<tr>
<td>v1.10.100-cpu</td>
<td>v1.11.0-cpu</td>
</tr>
<tr>
<td>

```python
python -m pip install intel_extension_for_pytorch==1.10.100 -f https://software.intel.com/ipex-whl-stable
```
</td>
<td>

```python
pip install intel_extension_for_pytorch
```
</td>
</tr>
</tbody>
</table>

- Compared to the previous version, this release provides a dedicated installation file for the C++ SDK. The installation file automatically detects the PyTorch C++ SDK location and installs the extension C++ SDK files to the PyTorch C++ SDK. The user does not need to manually add the extension C++ SDK source files and CMake to the PyTorch SDK. In addition to that, the installation file reduces the C++ SDK binary size from ~220MB to ~13.5MB. 

<table align="center">
<tbody>
<tr>
<td>v1.10.100-cpu</td>
<td>v1.11.0-cpu</td>
</tr>
<tr>
<td>

```python
intel-ext-pt-cpu-libtorch-shared-with-deps-1.10.0+cpu.zip (220M)
intel-ext-pt-cpu-libtorch-cxx11-abi-shared-with-deps-1.10.0+cpu.zip (224M)
```
</td>
<td>

```python
libintel-ext-pt-1.11.0+cpu.run (13.7M)
libintel-ext-pt-cxx11-abi-1.11.0+cpu.run (13.5M)
```
</td>
</tr>
</tbody>
</table>

- Add more optimizations, including more custom operators and fusions.
    - Fuse the QKV linear operators as a single Linear to accelerate the Transformer\*(BERT-\*) encoder part  - [#278](https://github.com/intel/intel-extension-for-pytorch/commit/0f27c269cae0f902973412dc39c9a7aae940e07b).
    - Remove Multi-Head-Attention fusion limitations to support the 64bytes unaligned tensor shape. [#531](https://github.com/intel/intel-extension-for-pytorch/commit/dbb10fedb00c6ead0f5b48252146ae9d005a0fad)
    - Fold the binary operator to Convolution and Linear operator to reduce computation. [#432](https://github.com/intel/intel-extension-for-pytorch/commit/564588561fa5d45b8b63e490336d151ff1fc9cbc) [#438](https://github.com/intel/intel-extension-for-pytorch/commit/b4e7dacf08acd849cecf8d143a11dc4581a3857f) [#602](https://github.com/intel/intel-extension-for-pytorch/commit/74aa21262938b923d3ed1e6929e7d2b629b3ff27)
    - Replace the outplace operators with their corresponding in-place version to reduce memory footprint. The extension currently supports the operators including `sliu`, `sigmoid`, `tanh`, `hardsigmoid`, `hardswish`, `relu6`, `relu`, `selu`, `softmax`. [#524](https://github.com/intel/intel-extension-for-pytorch/commit/38647677e8186a235769ea519f4db65925eca33c)
    - Fuse the Concat + BN + ReLU as a single operator. [#452](https://github.com/intel/intel-extension-for-pytorch/commit/275ff503aea780a6b741f04db5323d9529ee1081)
    - Optimize Conv3D for both imperative and JIT by enabling NHWC and pre-packing the weight. [#425](https://github.com/intel/intel-extension-for-pytorch/commit/ae33faf62bb63b204b0ee63acb8e29e24f6076f3)
- Reduce the binary size. C++ SDK is reduced from ~220MB to ~13.5MB while the wheel packaged is reduced from ~100MB to ~40MB.
- Update oneDNN and oneDNN graph to [2.5.2](https://github.com/oneapi-src/oneDNN/releases/tag/v2.5.2) and [0.4.2](https://github.com/oneapi-src/oneDNN/releases/tag/graph-v0.4.2) respectively.

### What's Changed
**Full Changelog**: https://github.com/intel/intel-extension-for-pytorch/compare/v1.10.100...v1.11.0

## 1.10.100

This release is meant to fix the following issues:
- Resolve the issue that the PyTorch Tensor Expression(TE) did not work after importing the extension.
- Wraps the BactchNorm(BN) as another operator to break the TE's BN-related fusions. Because the BatchNorm performance of PyTorch Tensor Expression can not achieve the same performance as PyTorch ATen BN.
- Update the [documentation](https://intel.github.io/intel-extension-for-pytorch/)
    - Fix the INT8 quantization example issue #205
    - Polish the installation guide

## 1.10.0

The Intel® Extension for PyTorch\* 1.10 is on top of PyTorch 1.10. In this release, we polished the front end APIs. The APIs are more simplible, stable and straightforward now. According to PyTorch community recommendation, we changed the underhood device from `XPU` to `CPU`. With this change, the model and tensor does not need to be converted to the extension device to get performance improvement. It simplifies the model changes.

Besides that, we continuously optimize the Transformer\* and CNN models by fusing more operators and applying NHWC. We measured the 1.10 performance on Torchvison and HugginFace. As expected, 1.10 can speed up the two model zones.

### Highlights

- Change the package name to `intel_extension_for_pytorch` while the original package name is `intel_pytorch_extension`. This change targets to avoid any potential legal issues.

<table align="center">
<tbody>
<tr>
<td>v1.9.0-cpu</td>
<td>v1.10.0-cpu</td>
</tr>
<tr>
<td>

```
import intel_extension_for_pytorch as ipex
```
</td>
<td>

```
import intel_extension_for_pytorch as ipex
```
</td>
</tr>
</tbody>
</table>

- The underhood device is changed from the extension-specific device(`XPU`) to the standard CPU device which aligns with PyTorch CPU device design regardless of the dispatch mechanism and operator register mechanism. The interface impactions are that the model does not need to be converted to the extension device explicitly.

<table align="center">
<tbody>
<tr>
<td>v1.9.0-cpu</td>
<td>v1.10.0-cpu</td>
</tr>
<tr>
<td>

```
import torch
import torchvision.models as models

# Import the extension
import intel_extension_for_pytorch as ipex

resnet18 = models.resnet18(pretrained = True)

# Explicitly convert the model to the extension device
resnet18_xpu = resnet18.to(ipex.DEVICE)
```
</td>
<td>

```
import torch
import torchvision.models as models

# Import the extension
import intel_extension_for_pytorch as ipex

resnet18 = models.resnet18(pretrained = True)
```
</td>
</tr>
</tbody>
</table>

- Compared to v1.9.0, v1.10.0 follows PyTorch AMP API(`torch.cpu.amp`) to support auto-mixed-precision. `torch.cpu.amp` provides convenience for auto data type conversion at runtime. Currently, `torch.cpu.amp` only supports `torch.bfloat16`. It is the default lower precision floating point data type when `torch.cpu.amp` is enabled. `torch.cpu.amp` primarily benefits on Intel CPU with BFloat16 instruction set support.

```
import torch
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x):
        return self.conv(x)
```

<table align="center">
<tbody>
<tr>
<td>v1.9.0-cpu</td>
<td>v1.10.0-cpu</td>
</tr>
<tr>
<td>

```
# Import the extension
import intel_extension_for_pytorch as ipex

# Automatically mix precision
ipex.enable_auto_mixed_precision(mixed_dtype = torch.bfloat16)

model = SimpleNet().eval()
x = torch.rand(64, 64, 224, 224)
with torch.no_grad():
    model = torch.jit.trace(model, x)
    model = torch.jit.freeze(model)
    y = model(x)
```
</td>
<td>

```
# Import the extension
import intel_extension_for_pytorch as ipex

model = SimpleNet().eval()
x = torch.rand(64, 64, 224, 224)
with torch.cpu.amp.autocast(), torch.no_grad():
    model = torch.jit.trace(model, x)
    model = torch.jit.freeze(model)
    y = model(x)
```
</td>
</tr>
</tbody>
</table>

- The 1.10 release provides the INT8 calibration as an experimental feature while it only supports post-training static quantization now. Compared to 1.9.0, the fronted APIs for qutization is more straightforward and ease-of-use.

```
import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(10, 10, 3)

    def forward(self, x):
        x = self.conv(x)
        return x

model = MyModel().eval()

# user dataset for calibration.
xx_c = [torch.randn(1, 10, 28, 28) for i in range(2))
# user dataset for validation.
xx_v = [torch.randn(1, 10, 28, 28) for i in range(20))
```
  - Clibration
<table align="center">
<tbody>
<tr>
<td>v1.9.0-cpu</td>
<td>v1.10.0-cpu</td>
</tr>
<tr>
<td>

```
# Import the extension
import intel_extension_for_pytorch as ipex

# Convert the model to the Extension device
model = Model().to(ipex.DEVICE)

# Create a configuration file to save quantization parameters.
conf = ipex.AmpConf(torch.int8)
with torch.no_grad():
    for x in xx_c:
        # Run the model under calibration mode to collect quantization parameters
        with ipex.AutoMixPrecision(conf, running_mode='calibration'):
            y = model(x.to(ipex.DEVICE))
# Save the configuration file
conf.save('configure.json')
```
</td>
<td>

```
# Import the extension
import intel_extension_for_pytorch as ipex

conf = ipex.quantization.QuantConf(qscheme=torch.per_tensor_affine)
with torch.no_grad():
    for x in xx_c:
        with ipex.quantization.calibrate(conf):
            y = model(x)

conf.save('configure.json')
```
</td>
</tr>
</tbody>
</table>

 - Inference
 <table align="center">
<tbody>
<tr>
<td>v1.9.0-cpu</td>
<td>v1.10.0-cpu</td>
</tr>
<tr>
<td>

```
# Import the extension
import intel_extension_for_pytorch as ipex

# Convert the model to the Extension device
model = Model().to(ipex.DEVICE)
conf = ipex.AmpConf(torch.int8, 'configure.json')
with torch.no_grad():
    for x in cali_dataset:
        with ipex.AutoMixPrecision(conf, running_mode='inference'):
            y = model(x.to(ipex.DEVICE))
```
</td>
<td>

```
# Import the extension
import intel_extension_for_pytorch as ipex

conf = ipex.quantization.QuantConf('configure.json')

with torch.no_grad():
    trace_model = ipex.quantization.convert(model, conf, example_input)
    for x in xx_v:
        y = trace_model(x)
```
</td>
</tr>
</tbody>
</table>


- This release introduces the `optimize` API at python front end to optimize the model and optimizer for training. The new API both supports FP32 and BF16, inference and training.

- Runtime Extension (Experimental) provides a runtime CPU pool API to bind threads to cores. It also features async tasks. Please **note**: Intel® Extension for PyTorch\* Runtime extension is still in the **POC** stage. The API is subject to change. More detailed descriptions are available in the extension documentation.

### Known Issues

- `omp_set_num_threads` function failed to change OpenMP threads number of oneDNN operators if it was set before.

  `omp_set_num_threads` function is provided in Intel® Extension for PyTorch\* to change the number of threads used with OpenMP. However, it failed to change the number of OpenMP threads if it was set before.

  pseudo-code:

  ```
  omp_set_num_threads(6)
  model_execution()
  omp_set_num_threads(4)
  same_model_execution_again()
  ```

  **Reason:** oneDNN primitive descriptor stores the omp number of threads. Current oneDNN integration caches the primitive descriptor in IPEX. So if we use runtime extension with oneDNN based pytorch/ipex operation, the runtime extension fails to change the used omp number of threads.

- Low performance with INT8 support for dynamic shapes

  The support for dynamic shapes in Intel® Extension for PyTorch\* INT8 integration is still working in progress. For the use cases where the input shapes are dynamic, for example, inputs of variable image sizes in an object detection task or of variable sequence lengths in NLP tasks, the Intel® Extension for PyTorch\* INT8 path may slow down the model inference. In this case, please utilize stock PyTorch INT8 functionality.

- Low throughput with DLRM FP32 Train

  A 'Sparse Add' [PR](https://github.com/pytorch/pytorch/pull/23057) is pending review. The issue will be fixed when the PR is merged.

### What's Changed
**Full Changelog**: https://github.com/intel/intel-extension-for-pytorch/compare/v1.9.0...v1.10.0+cpu-rc3

## 1.9.0

### What's New

* Rebased the Intel Extension for Pytorch from PyTorch-1.8.0 to the official PyTorch-1.9.0 release.
* Support binary installation.

  `python -m pip install torch_ipex==1.9.0 -f https://software.intel.com/ipex-whl-stable`
* Support the C++ library. The third party App can link the Intel-Extension-for-PyTorch C++ library to enable the particular optimizations.

## 1.8.0

### What's New

* Rebased the Intel Extension for Pytorch from Pytorch -1.7.0 to the official Pytorch-1.8.0 release. The new XPU device type has been added into Pytorch-1.8.0(49786), don’t need to patch PyTorch to enable Intel Extension for Pytorch anymore
* Upgraded the oneDNN from v1.5-rc to v1.8.1
* Updated the README file to add the sections to introduce supported customized operators, supported fusion patterns, tutorials and joint blogs with stakeholders

## 1.2.0

### What's New

* We rebased the Intel Extension for pytorch from Pytorch -1.5rc3 to the official Pytorch-1.7.0 release. It will have performance improvement with the new Pytorch-1.7 support.
* Device name was changed from DPCPP to XPU.

  We changed the device name from DPCPP to XPU to align with the future Intel GPU product for heterogeneous computation.
* Enabled the launcher for end users.
* We enabled the launch script which helps users launch the program for training and inference, then automatically setup the strategy for multi-thread, multi-instance, and memory allocator. Please refer to the launch script comments for more details.

### Performance Improvement

* This upgrade provides better INT8 optimization with refined auto mixed-precision API.
* More operators are optimized for the int8 inference and bfp16 training of some key workloads, like MaskRCNN, SSD-ResNet34, DLRM, RNNT.

### Others

* Bug fixes
  * This upgrade fixes the issue that saving the model trained by Intel extension for PyTorch caused errors.
  * This upgrade fixes the issue that Intel extension for PyTorch was slower than pytorch proper for Tacotron2.
* New custom operators

  This upgrade adds several custom operators: ROIAlign, RNN, FrozenBatchNorm, nms.
* Optimized operators/fusion

  This upgrade optimizes several operators: tanh, log_softmax, upsample, embeddingbad and enables int8 linear fusion.
* Performance

  The release has daily automated testing for the supported models: ResNet50, ResNext101, Huggingface Bert, DLRM, Resnext3d, MaskRNN, SSD-ResNet34. With the extension imported, it can bring up to 2x INT8 over FP32 inference performance improvements on the 3rd Gen Intel Xeon scalable processors (formerly codename Cooper Lake).

### Known issues

* Multi-node training still encounter hang issues after several iterations. The fix will be included in the next official release.

## 1.1.0

### What's New

* Added optimization for training with FP32 data type & BF16 data type. All the optimized FP32/BF16 backward operators include:
  * Conv2d
  * Relu
  * Gelu
  * Linear
  * Pooling
  * BatchNorm
  * LayerNorm
  * Cat
  * Softmax
  * Sigmoid
  * Split
  * Embedding_bag
  * Interaction
  * MLP
* More fusion patterns are supported and validated in the release, see table:

  |Fusion Patterns|Release|
  |--|--|
  |Conv + Sum|v1.0|
  |Conv + BN|v1.0|
  |Conv + Relu|v1.0|
  |Linear + Relu|v1.0|
  |Conv + Eltwise|v1.1|
  |Linear + Gelu|v1.1|

* Add docker support
* [Alpha] Multi-node training with oneCCL support.
* [Alpha] INT8 inference optimization.

### Performance

* The release has daily automated testing for the supported models: ResNet50, ResNext101, [Huggingface Bert](https://github.com/huggingface/transformers), [DLRM](https://github.com/intel/optimized-models/tree/master/pytorch/dlrm), [Resnext3d](https://github.com/XiaobingSuper/Resnext3d-for-video-classification), [Transformer](https://github.com/pytorch/fairseq/blob/master/fairseq/models/transformer.py). With the extension imported, it can bring up to 1.2x~1.7x BF16 over FP32 training performance improvements on the 3rd Gen Intel Xeon scalable processors (formerly codename Cooper Lake).

### Known issue

* Some workloads may crash after several iterations on the extension with [jemalloc](https://github.com/jemalloc/jemalloc) enabled.

## 1.0.2

* Rebase torch CCL patch to PyTorch 1.5.0-rc3

## 1.0.1-Alpha

* Static link oneDNN library
* Check AVX512 build option
* Fix the issue that cannot normally invoke `enable_auto_optimization`

## 1.0.0-Alpha

### What's New

* Auto Operator Optimization

  Intel Extension for PyTorch will automatically optimize the operators of PyTorch when importing its python package. It will significantly improve the computation performance if the input tensor and the model is converted to the extension device.

* Auto Mixed Precision
  Currently, the extension has supported bfloat16. It streamlines the work to enable a bfloat16 model. The feature is controlled by `enable_auto_mix_precision`. If you enable it, the extension will run the operator with bfloat16 automatically to accelerate the operator computation.

### Performance Result

We collected the performance data of some models on the Intel Cooper Lake platform with 1 socket and 28 cores. Intel Cooper Lake introduced AVX512 BF16 instructions which could improve the bfloat16 computation significantly. The detail is as follows (The data is the speedup ratio and the baseline is upstream PyTorch).

||Imperative - Operator Injection|Imperative - Mixed Precision|JIT- Operator Injection|JIT - Mixed Precision|
|:--:|:--:|:--:|:--:|:--:|
|RN50|2.68|5.01|5.14|9.66|
|ResNet3D|3.00|4.67|5.19|8.39|
|BERT-LARGE|0.99|1.40|N/A|N/A|

We also measured the performance of ResNeXt101, Transformer-FB, DLRM, and YOLOv3 with the extension. We observed that the performance could be significantly improved by the extension as expected.

### Known issue

* [#10](https://github.com/intel/intel-extension-for-pytorch/issues/10) All data types have not been registered for DPCPP
* [#37](https://github.com/intel/intel-extension-for-pytorch/issues/37) MaxPool can't get nan result when input's value is nan

### NOTE

The extension supported PyTorch v1.5.0-rc3. Support for other PyTorch versions is working in progress.
