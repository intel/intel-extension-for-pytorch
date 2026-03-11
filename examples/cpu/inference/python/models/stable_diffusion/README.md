# Stable Diffusion Inference

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    PyTorch    |       https://huggingface.co/stabilityai/stable-diffusion-2-1       |           -           |         -          |

## Benchmarking with TorchInductor
### Preparation

1. Create virtual environment `venv` and activate it
    ```
    python3 -m venv venv
    . ./venv/bin/activate
    ```
2. Install PyTorch
    ```
    git clone https://github.com/yanbing-j/pytorch.git
    cd pytorch
    git checkout yanbing/tf32_dev_branch_for_test_base_1113_alpha_cand
    git submodule sync
    git submodule update --init --recursive
    conda install cmake ninja
    pip install -r requirements.txt
    pip install mkl-static mkl-include
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
    python setup.py install
    cd ..
    ```
3. Install Torchvision
    ```
    git clone https://github.com/pytorch/vision.git
    cd vision
    pip install -e .
    ```
4. Install TorchAO
    ```
    # For FP8, please use GCC 15 to build torchao.
    conda install -c conda-forge gcc=15 gxx=15
    git clone https://github.com/pytorch/ao.git
    cd ao
    git submodule sync
    git submodule update --init --recursive
    USE_CPU_KERNELS=1 python setup.py install
    cd ..
    ```
5. Install Transformers
    ```
    pip install transformers==4.57.1
    ```
6. Run setup.sh
    ```
    ./setup.sh
    ```
7. Install Intel OpenMP and TCMalloc
    ```
    pip install packaging intel-openmp accelerate
    conda install -y gperftools -c conda-forge
    ```
* Set IOMP and tcmalloc Preload for better performance
    ```
    export LD_PRELOAD="<path_to>/tcmalloc/lib/libtcmalloc.so":"<path_to_iomp>/lib/libiomp5.so":$LD_PRELOAD
    ```
* Set ENV to use fp16 AMX if you are using a supported platform
    ```
    export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX_FP16
    ```

### Datasets

Download the 2017 [COCO dataset](https://cocodataset.org) using the `download_dataset.sh` script.
Export the `DATASET_DIR` environment variable to specify the directory where the dataset will be downloaded. This environment variable will be used again when running training scripts.
```
export DATASET_DIR=<directory where the dataset will be saved>
bash download_dataset.sh
```

### Inference
1. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **TEST_MODE** (THROUGHPUT, ACCURACY)              | `export TEST_MODE=THROUGHPUT`                  |
| **DISTRIBUTED** (Only for ACCURACY)              | `export DISTRIBUTED=TRUE`                  |
| **OUTPUT_DIR**               |                               `export OUTPUT_DIR=$(pwd)`                               |
| **DATASET_DIR**       |          `export DATASET_DIR=<path_to_dataset_dir>`             |
| **MODE**      | `export MODE=<choose from: eager, compile-inductor>`     |
| **PRECISION**     |                  `export PRECISION=bf16` (fp32, bf16, fp16, int8-fp32, int8-bf16, fp8-fp32, fp8-bf16) |
| **MODEL_DIR**               |                               `export MODEL_DIR=$(pwd)`                               |
| **NNODES** (required for DISTRIBUTED)              | ` export NNODES=#your_node_number`                  |
| **HOSTFILE** (required for DISTRIBUTED)              | `export HOSTFILE=#your_ip_list_file #one ip per line`                  |

2. Command lines
```
bash run_model.sh
```

* NOTE:
Please get quantized model before running `INT8-BF16` or `INT8-FP32`.
For `compile-inductor` mode, please do calibration first:
  ```
  bash do_calibration.sh
  ```

## Output

Single-tile output will typically looks like:

```
time per prompt(s): 107.73
Latency: 107.65 s
Throughput: 0.00929 samples/sec
```
Final results of the inference run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 0.00929
   unit: samples/sec
 - key: latency
   value: 107.73
   unit: s
 - key: accuracy
   value: N/A
   unit: FID
```
