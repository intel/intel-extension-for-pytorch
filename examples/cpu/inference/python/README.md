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
