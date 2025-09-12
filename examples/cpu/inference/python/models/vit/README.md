# ViT Inference

Vision Transformer inference best known configurations with PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    PyTorch    |       https://huggingface.co/google/vit-base-patch16-224        |           -           |         -          |

## Bare Metal
* Install Intel OpenMP
  ```
  pip install packaging intel-openmp accelerate==0.34.1
  ```
* Set IOMP, jemalloc and tcmalloc Preload for better performance
  ```
  export LD_PRELOAD="<path to the jemalloc directory>/lib/libjemalloc.so":"<path_to>/tcmalloc/lib/libtcmalloc.so":"<path_to_iomp>/lib/libiomp5.so":$LD_PRELOAD
  ```

* Install datasets
  ```
  pip install datasets
  ```

* Set CORE_PER_INSTANCE before running realtime mode
  ```
  export CORE_PER_INSTANCE=4
  (4cores per instance setting is preferred, while you could set any other config like 1core per instance)
  ```

* About the BATCH_SIZE in scripts
  ```
  Throughput mode is using BATCH_SIZE=[4 x core number] by default in script (which could be further tuned according to the testing host);
  Realtime mode is using BATCH_SIZE=[1] by default in script;
  ```

* [Optional] Use dummy input for performance collection
  ```
  export DUMMY_INPUT=1
  ```

# Inference
1. Install PyTorch
    ```
    # install torch from source
    git clone https://github.com/yanbing-j/pytorch.git
    cd pytorch
    git checkout yanbing/tf32_dev_branch_for_test
    git submodule sync
    git submodule update --init --recursive
    conda install cmake ninja
    pip install -r requirements.txt
    pip install mkl-static mkl-include
    python setup.py install
    cd ..
    ```
2. Install TorchAO
    ```
    git clone https://github.com/Valentine233/ao.git
    cd ao
    git checkout fp8
    git submodule sync
    git submodule update --init --recursive
    USE_CPU_KERNELS=1 python setup.py install
    cd ..
    ```
3. `cd examples/cpu/inference/python/models/vit`
4. Run `setup.sh`
    ```
    ./setup.sh
    ```
5. Prepare for downloading access
    On https://huggingface.co/datasets/ILSVRC/imagenet-1k, login your account, and click the aggreement and then generating {your huggingface token}

    huggingface-cli login
    {your huggingface token}
6. Download the dataset
  ```
   export DATASET_DIR=<path to dataset imagenet>
   #Run "download_data.sh"
   ./download_data.sh
  ```
7. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **TEST_MODE** (THROUGHPUT, ACCURACY, REALTIME)              | `export TEST_MODE=THROUGHPUT`                  |
| **OUTPUT_DIR**               |                               `export OUTPUT_DIR=$(pwd)`                               |
| **DATASET_DIR**          |  `export DATASET_DIR=<path to dataset imagenet>`    |
| **PRECISION**     |                  `export PRECISION=bf16` (fp32, bf32, bf16, fp16, int8-fp32, int8-bf16, fp8-fp32, fp8-bf16) |
| **MODEL_DIR**               |                               `export MODEL_DIR=$(pwd)`                               |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=120`                                |
| **DUMMY_INPUT**(optional)     |     `export DUMMY_INPUT=1` (This is optional; for performance collection)    |
| **CORE_PER_INSTANCE** (required for REALTIME)    |                               `export CORE_PER_INSTANCE=4`                                |
8. Run `run_model.sh`
    ```
    ./run_model.sh
    ```
## Output

Single-tile output will typically looks like:

```
2023-11-15 06:22:47,398 - __main__ - INFO - Results: {'exact': 87.01040681173131, 'f1': 93.17865304772475, 'total': 10570, 'HasAns_exact': 87.01040681173131, 'HasAns_f1': 93.17865304772475, 'HasAns_total': 10570, 'best_exact': 87.01040681173131, 'best_exact_thresh': 0.0, 'best_f1': 93.17865304772475, 'best_f1_thresh': 0.0}
```
Final results of the inference run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 405.9567
   unit: example/s
 - key: latency
   value: 0.15765228112538657
   unit: s/example
 - key: accuracy
   value: 93.179
   unit: f1
```
