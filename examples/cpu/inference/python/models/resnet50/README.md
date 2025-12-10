# ResNet50 inference

## Description

This document has instructions for running [ResNet50](https://github.com/KaimingHe/deep-residual-networks) inference using PyTorch.

## Benchmarking with TorchInductor
### Preparation
```

# Install PyTorch, Torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu/

# Install necessary environment
pip install numpy
pip install pillow

# Install Intel OpenMP and TCMalloc
pip install packaging intel-openmp accelerate
conda install -y gperftools -c conda-forge

# Download pretrained model, you will find the model under ~/.cache/torch/hub/facebookresearch_WSL-Images_main
python download_model.py
```

### Datasets
The [ImageNet](http://www.image-net.org/) validation dataset is used.

Download and extract the ImageNet2012 dataset from https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar, then move validation images to labeled subfolders, using [the valprep.sh shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

A after running the data prep script, your folder structure should look something like this:

```
imagenet
└── val
    ├── ILSVRC2012_img_val.tar
    ├── n01440764
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   ├── ILSVRC2012_val_00002138.JPEG
    │   ├── ILSVRC2012_val_00003014.JPEG
    │   ├── ILSVRC2012_val_00006697.JPEG
    │   └── ...
    └── ...
```
The folder that contains the `val` directory should be set as the `DATASET_DIR` (for example: `export DATASET_DIR=/home/<user>/imagenet`).

### Inference
1. Setup required environment paramaters

| **Parameter**                                    |                **export command**                                 |
|:------------------------------------------------:|:-----------------------------------------------------------------:|
| **TEST_MODE** (optional)                         |     `export TEST_MODE=THROUGHPUT`(THROUGHPUT, ACCURACY, REALTIME) |
| **CORES_PER_INSTANCE** (required if no test mode)|     `export CORES_PER_INSTANCE=<the number of cores per instance>`|
| **INSTANCES** (required if no test mode)         |     `export INSTANCES=<the number of total instances>`            |
| **DATASET_DIR** (required for ACCURACY)          |     `export DATASET_DIR=<path to ImageNet>`                       |
| **OUTPUT_DIR**                                   |     `export OUTPUT_DIR=$PWD`                                      |
| **PRECISION**                                    |     `export PRECISION=bf16` (bf16)                                |
| **BATCH_SIZE** (optional)                        |     `export BATCH_SIZE=64`                                        |

2. Command lines
```
bash run_model.sh
```
