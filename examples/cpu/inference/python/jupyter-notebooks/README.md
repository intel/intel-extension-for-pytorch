# Intel® Extension for PyTorch* CPU Jupyter Notebooks

Intel® Extension for PyTorch (IPEX) extends PyTorch* with optimizations for extra performance boost on Intel® hardware. While most of the optimizations will be upstreamed in future PyTorch* releases, the extension delivers up-to-date features and optimizations for PyTorch workloads on Intel® hardware. The optimization approaches generally include operator optimization, graph optimization and runtime optimization.

Before selecting a sample, please make sure to (1) Check [Prerequisites](#prerequisites), (2) complete [Environment Setup](#environment-setup), and (3) see instructions to [Run the Sample](#run-the-sample).

## Jupyter Notebooks Overview

| Sample name | Description | Time to Complete             | Category   | Validated for AI Tools Selector |
|---|---|---|---|---|
[Getting Started with Intel® Extension for PyTorch* (IPEX)](https://github.com/intel/intel-extension-for-pytorch/tree/main/examples/cpu/inference/python/jupyter-notebooks/IPEX_Getting_Started.ipynb ) | This code sample demonstrates how to begin using the Intel® Extension for PyTorch* (IPEX). It will guide users how to run a PyTorch inference workload on CPU by using oneAPI AI Analytics Toolkit and also analyze the CPU usage via oneDNN verbose logs.| 15 minutes| Getting Started | Y | 
[PyTorch Inference Optimizations with Intel® Advanced Matrix Extensions (Intel® AMX) Bfloat16 Integer8](https://github.com/intel/intel-extension-for-pytorch/blob/main/examples/cpu/inference/python/jupyter-notebooks/IntelPyTorch_InferenceOptimizations_AMX_BF16_INT8.ipynb) | This code sample demonstrates how to perform inference using the ResNet50 and BERT models using the Intel® Extension for PyTorch* (IPEX). IPEX allows you to speed up inference on Intel® Xeon Scalable processors with lower precision data formats and specialized computer instructions. The bfloat16 (BF16) data format uses half the bit width of floating-point-32 (FP32), which lessens the amount of memory needed and execution time to process. Likewise, the integer8 (INT8) data format uses half the bit width of BF16. | 5 minutes     | Code Optimization | Y|
[Interactive Chat Based on DialoGPT Model Using Intel® Extension for PyTorch* Quantization](https://github.com/intel/intel-extension-for-pytorch/blob/main/examples/cpu/inference/python/jupyter-notebooks/IntelPytorch_Interactive_Chat_Quantization.ipynb)| This code sample demonstrates how to create interactive chat based on pre-trained DialoGPT model and add the Intel® Extension for PyTorch* (IPEX) quantization to it. The sample shows how to create interactive chat based on the pre-trained DialoGPT model from HuggingFace and how to add INT8 dynamic quantization to it. The Intel® Extension for PyTorch* (IPEX) gives users the ability to speed up operations on processors with INT8 data format and specialized computer instructions.| 10 minutes | Concepts and Functionality| Y|
[Optimize PyTorch Models using Intel® Extension for PyTorch (IPEX) Quantization](https://github.com/intel/intel-extension-for-pytorch/blob/main/examples/cpu/inference/python/jupyter-notebooks/IntelPytorch_Quantization.ipynb)|This code sample demonstrates how to quantize a ResNet50 model that is calibrated by the CIFAR10 dataset using the Intel® Extension for PyTorch* (IPEX). IPEX gives users the ability to speed up inference on Intel® Xeon Scalable processors with INT8 data format and specialized computer instructions. The INT8 data format uses quarter the bit width of floating-point-32 (FP32), lowering the amount of memory needed and execution time to process.| 5 minutes| Concepts and Functionality| Y|
[Optimize PyTorch Models using Intel® Extension for PyTorch* (IPEX)](https://github.com/intel/intel-extension-for-pytorch/blob/main/examples/cpu/inference/python/jupyter-notebooks/optimize_pytorch_models_with_ipex.ipynb)| This sample notebook shows how to get started with Intel® Extension for PyTorch* (IPEX) for sample Computer Vision and NLP workloads. The sample starts by loading two models from the PyTorch hub: Faster-RCNN (Faster R-CNN) and distilbert (DistilBERT). After loading the models, the sample applies sequential optimizations from Intel® Extension for PyTorch* (IPEX) and examines performance gains for each incremental change.| 30 minutes | Code Optimization |Y|

>**Note**: For Key Implementation Details, please refer to the .ipynb file of a sample.

## Prerequisites
  
| Optimized for           | Description
|:---                     |:---
| OS                      | Ubuntu* 18.04 or newer
| Hardware                | 4th Gen Intel® Xeon® Scalable processors or newer
| Software                | Intel® Extension of PyTorch* (IPEX)

## Environment Setup

### 1. Get Intel® Extension for PyTorch* CPU
**Option A: Install with PIP**\
1.1 Create new Conda environment\
Python 3.8, 3.9, 3.10, 3.11 are supported. Please create a **new conda environment** for each sample.
```
conda create -n ipex_cpu python=3.10 -y
conda activate ipex_cpu
```
1.2 Install Intel® Extension for PyTorch* CPU with PIP\
Follow these instructions to install latest released Intel® Extension for PyTorch*
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install intel-extension-for-pytorch
```

*If a specific version is needed, please follow the Installation Section and Sanity Check Section in the [installation guide](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=cpu) to install Intel® Extension for Pytorch\* with specific version.*

**Option B: Install via AI Tools Selector**\
Select and install Intel® Extension for PyTorch* CPU by following the instructions in [AI Tools Selector.](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-tools-selector.html)
*It is recommended to select Offline Installer option (validated) in AI Tools Selector*.\
> *Note: Before moving forward, make sure you have (1) AI Tools bundle environment activated (2) 'pytorch-cpu' Conda environment activated.*

### 2.  Install dependencies
```
pip install notebook ipykernel matplotlib
```
*If you encounter any issue for Jupyter notebooks, please refer to [Installing Jupyter](https://jupyter.org/install) for detailed installation instructions.*

## Run the Sample

1.  Register ipykernel with Conda
```
python -m ipykernel install --user --name=PyTorch-CPU
```
2.  Launch Jupyter Notebook
```
jupyter notebook --ip=0.0.0.0 --port 8888 --allow-root
```
3. Follow the instructions to open the URL with the token in your browser.
4. Locate and select the Notebook.
5. Change your Jupyter Notebook kernel to `PyTorch-CPU`.
6. Run every cell in the Notebook in sequence.

## Example Output

If successful, the sample displays `[CODE_SAMPLE_COMPLETED_SUCCESSFULLY]`. Additionally, the sample generates performance and analysis diagrams for comparison.

\* Other names and brands may be claimed as the property of others. [Trademarks](https://www.intel.com/content/www/us/en/legal/trademarks.html)
