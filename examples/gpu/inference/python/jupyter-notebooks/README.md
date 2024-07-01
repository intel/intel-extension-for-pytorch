# Environment Setup for Jupyter Notebook with Intel Pytorch GPU

Intel® Extension for PyTorch (IPEX) extends PyTorch* with up-to-date features optimizations for an extra performance boost on Intel hardware. Optimizations take advantage of Intel Xe Matrix Extensions (XMX) AI engines on Intel discrete GPUs. Moreover, through PyTorch* XPU device, Intel® Extension for PyTorch (IPEX) provides easy GPU acceleration for Intel discrete GPUs with PyTorch*.


## Prerequisites

| Optimized for           | Description
|:---                     |:---
| OS                      | Ubuntu* 22.04 or newer
| Hardware                | Intel® Data Center GPU Flex Series, Intel® Data Center GPU Max Series, and Intel® ARC™ A-Series GPUs(Experimental Support)
| Software                | Intel® oneAPI Base Toolkit 

## For Local Development Environments


- **Install Driver & BaseToolkit**

please follow the Preparation Section in the [installation guide](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu) to install driver and toolkit with specific version.


- **Install Jupyter Notebook with Conda**

Python 3.8,3.9,3.10,3.11 are supported.
Please create a new conda environment for each sample.

```
conda create -n ipex_gpu python=3.10 -y
conda activate ipex_gpu
pip install notebook ipykernel
```

If encounter any issue for jupyter notebook, please refer to [*Installing Jupyter*](https://jupyter.org/install) for detailed installation instructions.


- **Install Intel® Extension for Pytorch with Conda**

please follow the Installation Section and Sanity Check Section in the [installation guide](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu) to install Intel® Extension for Pytroch* with specific version.

- **Register ipykernel with Conda**
  
```
python -m ipykernel install --user --name=PyTorch-GPU
```

- **Runing the jupyter notebook**

1. Change to the sample directory.
2. Launch Jupyter Notebook.
```
jupyter notebook --ip=0.0.0.0 --port 8888 --allow-root
```
3. Follow the instructions to open the URL with the token in your browser.
4. Locate and select the Notebook.
5. Change your Jupyter Notebook kernel to **PyTorch-GPU**.
6. Run every cell in the Notebook in sequence.


## Example Output

If successful, the sample displays `[CODE_SAMPLE_COMPLETED_SUCCESSFULLY]`. Additionally, the sample generates performance and analysis diagrams for comparison.

