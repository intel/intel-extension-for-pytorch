# Environment Setup for Jupyter Notebook with Intel Pytorch CPU

The Intel® Extension for PyTorch (IPEX) extends PyTorch* with optimizations for extra performance boost on Intel® hardware. While most of the optimizations will be included in future PyTorch* releases, the extension delivers up-to-date features and optimizations for PyTorch on Intel® hardware. For example, newer optimizations include AVX-512 Vector Neural Network Instructions (AVX512 VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX).

## Prerequisites

| Optimized for           | Description
|:---                     |:---
| OS                      | Ubuntu* 18.04 or newer
| Hardware                | 4th Gen Intel® Xeon® Scalable Processors or newer
| Software                | Intel® Extension for PyTorch (IPEX)

## For Local Development Environments

- **Install Jupyter Notebook with Conda**

Python 3.8,3.9,3.10,3.11 are supported.
Please create a **new conda environment** for each sample.

```
conda create -n ipex_cpu python=3.10 -y
conda activate ipex_cpu
pip install notebook ipykernel
```

If encounter any issue for jupyter notebook, please refer to [*Installing Jupyter*](https://jupyter.org/install) for detailed installation instructions.


- **Install Intel® Extension for Pytorch* with Conda**

Follow this instructions to install latest released Intel® Extension for Pytorch*

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install intel-extension-for-pytorch
```

If a specific version is needed, please follow the Installation Section and Sanity Check Section in the [installation guide](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=cpu) to install Intel® Extension for Pytroch* with specific version.

- **Register ipykernel with Conda**
  
```
python -m ipykernel install --user --name=PyTorch-CPU
```

- **Runing the jupyter notebook**

1. Change to the sample directory.
2. Launch Jupyter Notebook.
```
jupyter notebook --ip=0.0.0.0 --port 8888 --allow-root
```
3. Follow the instructions to open the URL with the token in your browser.
4. Locate and select the Notebook.
5. Change your Jupyter Notebook kernel to **PyTorch-CPU**.
6. Run every cell in the Notebook in sequence.

## Example Output

If successful, the sample displays `[CODE_SAMPLE_COMPLETED_SUCCESSFULLY]`. Additionally, the sample generates performance and analysis diagrams for comparison.

