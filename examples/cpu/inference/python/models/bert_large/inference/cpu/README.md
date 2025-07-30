# BERT Large Inference

BERT Large training best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Training   |    PyTorch    |       https://github.com/huggingface/transformers/tree/main/src/transformers/models/bert        |           -           |         -          |

# Pre-Requisite
* Installation of PyTorch and [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/#installation)

## Bare Metal
### General setup

Follow [link](https://github.com/IntelAI/models/blob/master/docs/general/pytorch/BareMetalSetup.md) to install Miniforge and build Pytorch, IPEX, TorchVison Jemalloc and TCMalloc.

### Model Specific Setup

* Set Jemalloc and tcmalloc Preload for better performance

  The jemalloc should be built from the [General setup](#general-setup) section.
  ```
  export LD_PRELOAD="<path to the jemalloc directory>/lib/libjemalloc.so":"path_to/tcmalloc/lib/libtcmalloc.so":$LD_PRELOAD
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  ```
* Set IOMP preload for better performance
```
  pip install packaging intel-openmp
  export LD_PRELOAD=path/lib/libiomp5.so:$LD_PRELOAD
```
* Install dependencies
```
pip install protobuf==3.20.3 numpy==1.20
```

* Set ENV to use fp16 AMX if you are using a supported platform
```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX_FP16
```

* Set ENV for model and dataset path, and optionally run with no network support
```
  export FINETUNED_MODEL=#path/bert_squad_model
  export EVAL_DATA_FILE=#/path/dev-v1.1.json


  ### [optional] Pure offline mode to benchmark:
  change --tokenizer_name to #path/bert_squad_model in scripts before running
  e.g. --tokenizer_name ${FINETUNED_MODEL} in run_model.sh
```

* [optional] Do calibration to get quantization config if you want do calibration by yourself.
```
  export INT8_CONFIG=#/path/configure.json
  run_calibration.sh
```

# Prepare Dataset
## Datasets
Please following this [link](https://github.com/huggingface/transformers/tree/v3.0.2/examples/question-answering) to get `dev-v1.1.json` and set the `EVAL_DATA_FILE` environment variable to point to the file:
```
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
export EVAL_DATA_FILE=$(pwd)/dev-v1.1.json
```
## Pre-Trained Model
Download the `config.json` and fine tuned model from huggingface and set the `FINETUNED_MODEL` environment variable to point to the directory that has both files:
```
mkdir bert_squad_model
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-config.json -O bert_squad_model/config.json
wget https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin  -O bert_squad_model/pytorch_model.bin
export FINETUNED_MODEL=$(pwd)/bert_squad_model
```

# Inference
1. `git clone https://github.com/intel/intel-extension-for-pytorch`
2. `cd intel-extension-for-pytorch/examples/cpu/inference/python/models/bert_large/inference/cpu`
3. Create virtual environment `venv` and activate it:
    ```
    python3 -m venv venv
    . ./venv/bin/activate
    ```
4. Run setup.sh
    ```
    ./setup.sh
    ```
5. Install the latest CPU versions of [torch, torchvision and intel_extension_for_pytorch](https://intel.github.io/intel-extension-for-pytorch/index.html#installation)


6. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **TEST_MODE** (THROUGHPUT, ACCURACY, REALTIME)              | `export TEST_MODE=THROUGHPUT (THROUGHPUT, ACCURACY, REALTIME)`                  |
| **EVAL_DATA_FILE**              | `export EVAL_DATA_FILE=<path to dev-v1.1.json file>`                  |
| **OUTPUT_DIR**               |                               `export OUTPUT_DIR=<path to an output directory>`                               |
| **PRECISION**     |                  `export PRECISION=bf16` (bf16, bf32, fp32, fp16, int8, avx-int8, avx-fp32 for throughput and bf16, bf32, fp32, fp16, int8, avx-fp32, avx-int8, fp8 for accuracy) |
| **FINETUNED_MODEL**               |                               `export FINETUNED_MODEL=<path to the fine tuned model>`                               |
| **MODEL_DIR**               |                               `export MODEL_DIR=$(pwd)`                               |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=<set a value for batch size, else it will run with default batch size>`                                |
7. Run `run_model.sh`

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
