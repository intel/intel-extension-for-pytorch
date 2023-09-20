<!--- 0. Title -->
# PyTorch Stable Diffusion inference

<!-- 10. Description -->
## Description

This directory has a Jupyter notebook for running [Stable Diffusion*, which is a text-to-image latent diffusion model created by the researchers and engineers from CompVis, Stability AI, LAION and RunwayML.](https://huggingface.co/runwayml/stable-diffusion-v1-5) inference using Intel-optimized PyTorch.

## General setup

Install Pytorch, IPEX, TorchVison.
You can also [build them from source by running this script](https://github.com/intel/intel-extension-for-pytorch/blob/master/scripts/compile_bundle.sh).
We preload Intel OpenMP & tcmalloc as well.

### Install dependencies
```
pip install torchmetrics torch-fidelity pycocotools transformers==4.28.1
```

### Build and install a custom version of Diffusers package
```
git clone https://github.com/huggingface/diffusers.git
cd diffusers
git checkout v0.16.0
git apply ../diffusers_patch
python setup.py install
```

```


\* Other names & brands may be claimed as the property of others

