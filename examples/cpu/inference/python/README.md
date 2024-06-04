# Model Inference with Intel® Extension for PyTorch\* Optimizations

We provided examples about how to use Intel® Extension for PyTorch\* to accelerate model inference.
The `ipex.optimize` function of Intel® Extension for PyTorch* applies optimizations to the model, bringing additional performance boosts.
For both computer vision workloads and NLP workloads, we recommend applying the `ipex.optimize` function against the model object.

## Environment Setup

Basically we need to install PyTorch\* (along with related packages like torchvision, torchaudio, transformers) and Intel® Extension for PyTorch\* in a Python3 environment.

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install intel-extension-for-pytorch
# For BERT examples, transformers package is required
python -m pip install transformers
```

For more details, please check [installation guide](https://intel.github.io/intel-extension-for-pytorch/index.html#installation).

## Running Example Scripts

We provided inference examples for eager mode as well as graph mode, in which the computational graph is generated via [TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) or [TorchDynamo](https://pytorch.org/docs/stable/torch.compiler_deepdive.html).
Eager mode is the default execution mode in PyTorch, the codes are executed in a “define-by-run” paradigm, so it is flexible, interactive and easy to debug.
On the other hand, in graph mode the codes are executed in “define-and-run” paradigm, which means the building of the entire computation graph is required before running the function.
During the graph compilation process, optimizations like layer fusion and folding are applied, and the compiled graphs are more friendly for backend optimizations, leading to accelerated execution.
TorchScript and TorchDynamo are the 2 graph compiling tools that PyTorch\* provides. 

From numerical precision perspective, we provided inference examples for [BFloat16](#bfloat16) and [INT8 quantization](#int8) in addition to the default [Float32](#float32) precision.
Low-precision approaches including [Automatic Mixed Precision (AMP)](https://pytorch.org/docs/stable/amp.html) and [quantization](https://pytorch.org/docs/stable/quantization.html) are commonly used in PyTorch\* to improve performance.
In addition, BFloat16 and INT8 calculations can be further accelerated by [Intel® Advanced Matrix Extensions (AMX)](https://en.wikipedia.org/wiki/Advanced_Matrix_Extensions) instructions.
Please refer to [best practices for x86 CPU backend](https://pytorch.org/docs/stable/torch.compiler_best_practices_for_backends.html#x86-cpu) and [Leveraging Intel® AMX](https://pytorch.org/tutorials/recipes/amx.html) for detail explanations.

```bash
# Clone the repository and access to the inference examples folder
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch/examples/cpu/inference/python
```

### Float32

Running ResNet50 inference in eager mode:

```bash
python resnet50_eager_mode_inference_fp32.py
```

Running ResNet50 inference in TorchScript mode:

```bash
python resnet50_torchscript_mode_inference_fp32.py
```

Running ResNet50 inference in TorchDynamo mode:

```bash
python resnet50_torchdynamo_mode_inference_fp32.py
```

Running BERT inference in eager mode:

```bash
python bert_eager_mode_inference_fp32.py
```

Running BERT inference in TorchScript mode:

```bash
python bert_torchscript_mode_inference_fp32.py
```

Running BERT inference in TorchDynamo mode:

```bash
python bert_torchdynamo_mode_inference_fp32.py
```

### BFloat16

Running ResNet50 inference in eager mode:

```bash
python resnet50_eager_mode_inference_bf16.py
```

Running ResNet50 inference in TorchScript mode:

```bash
python resnet50_torchscript_mode_inference_bf16.py
```

Running ResNet50 inference in TorchDynamo mode:

```bash
python resnet50_torchdynamo_mode_inference_bf16.py
```

Running BERT inference in eager mode:

```bash
python bert_eager_mode_inference_bf16.py
```

Running BERT inference in TorchScript mode:

```bash
python bert_torchscript_mode_inference_bf16.py
```

Running BERT inference in TorchDynamo mode:

```bash
python bert_torchdynamo_mode_inference_bf16.py
```

*Note:* In TorchDynamo mode, since the native PyTorch\* operators like `aten::convolution` and `aten::linear` are well supported and optimized in ipex backend, 
we need to disable weights prepacking by setting `weights_prepack=False` when calling `ipex.optimize` function.

### INT8

We support both static and dynamic INT8 quantization.

* Static quantization: both weights and activations are quantized, a calibration process required.
* Dynamic quantization: weights quantized, activations read/stored in floating point and quantized for compute.

Please read [PyTorch\* quantization introduction](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/) for a more comprehensive overview.

Running static quantization for pretrained ResNet50 model:

```bash
python int8_quantization_static.py
```

Loading the static quantized model and executing inference:

```bash
python int8_deployment.py
```

Running dynamic quantization for pretrained BERT model:

```bash
python int8_quantization_dynamic.py
```

Please check [the model inference examples in Intel® Extension for PyTorch\* online document](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/examples.html#inference) for more information.
