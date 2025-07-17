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
2. Install PyTorch, Torchvision
    ```
    pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu/
    ```
3. Install TorchAO
    ```
    git clone https://github.com/shiyang-weng/ao.git
    cd ao
    git checkout wengshiy/qlinear
    python setup.py install
    ```
3. Run setup.sh
    ```
    ./setup.sh
    ```
4. Install Intel OpenMP and TCMalloc
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
