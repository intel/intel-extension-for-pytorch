## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    PyTorch    |       https://github.com/huggingface/transformers/tree/main/src/transformers/models/distilbert        |           -           |         -          |

# Pre-Requisite
* Installation of PyTorch
  ```
  pip install torch --index-url https://download.pytorch.org/whl/nightly/cpu/
  ```
* Set tcmalloc Preload for better performance
  ```
  conda install -y gperftools -c conda-forge
  export LD_PRELOAD="path_to/tcmalloc/lib/libtcmalloc.so":$LD_PRELOAD
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  ```
* Set IOMP preload for better performance
  ```
  pip install packaging intel-openmp
  export LD_PRELOAD=path/lib/libiomp5.so:$LD_PRELOAD
  ```
# About the Model:
* Set SEQUENCE_LENGTH before running the model
  ```
  export SEQUENCE_LENGTH=128
  (128 is preferred, while you could set any other length)
  ```

* Set CORES_PER_INSTANCE before running realtime mode, else it will run with default CORES_PER_INSTANCE
  ```
  export CORES_PER_INSTANCE=4
  Realtime mode is using CORES_PER_INSTANCE=4 by default in script.
  (4cores per instance setting is preferred, while you could set any other config like 1core per instance)
  ```

* About the BATCH_SIZE in scripts
  ```
  Throughput mode is using BATCH_SIZE=[4 x core number] by default in script (which could be further tuned according to the testing host);
  Realtime mode is using BATCH_SIZE=[1] by default in script;
  Customized BATCH_SIZE is supposed to be no larger than dataset size 872.
  ```

* [Optional for offline tests] Prepare model and dataset files locally
  ```
  (1) download model and sst2 dataset (make sure to install git-lfs first by apt-get install git-lfs)
  bash download_model_dataset.sh
  #by default they are downloaded in current path
  #note that you should do this after you prepared model (transformers repo)

  (2) make following changes in the scirpts to run:
  delete: --task_name sst2  ==>  add: --train_file {path/to/data_file}/SST-2/train.csv --validation_file {path/to/data_file}/SST-2/dev.csv

  (3) export model path
  export FINETUNED_MODEL={path/to/model_file}/distilbert-base-uncased-finetuned-sst-2-english

  (4) run scirpt with HF_DATASETS_OFFLINE=1 flag, like:
  HF_DATASETS_OFFLINE=1 bash run_multi_instance_throughput.sh fp32

  ```

# Prepare Dataset
Use the following instructions to download the SST-2 dataset.
```
export DATASET_DIR=<path to download the dataset>
wget https://dl.fbaipublicfiles.com/glue/data/SST-2.zip -O $DATASET_DIR/SST-2.zip
unzip $DATASET_DIR/SST-2.zip -d $DATASET_DIR/
python convert.py $DATASET_DIR
```

# Pre-Trained Model
Follow the instructions below to download the pre-trained model.
```
apt-get update && apt-get install -y git-lfs
git lfs install
git clone https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
export FINETUNED_MODEL=$(pwd)/distilbert-base-uncased-finetuned-sst-2-english
```

## Inference
1. `git clone https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-cpu.git`
2. `cd examples/cpu/inference/python/models/distilbert`
3. Create virtual environment `venv` and activate it:
  ```
  python3 -m venv venv
  . ./venv/bin/activate
  ```
4. Run setup scripts:
  ```
  ./setup.sh
  ```

5. Setup required environment paramaters

# Custom mode
 Run in custom mode by export TEST_MODE="" and export BATCH_SIZE to set the batch_size, export CORES_PER_INSTANCE to set the number of cores per instance and export INSTANCES to set the number of instances.

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **TEST_MODE**              |                     `export TEST_MODE=THROUGHPUT <select from: THROUGHPUT< REALTIME, ACCURACY,"">`                                  |
| **DATASET_DIR**              |                               `export DATASET_DIR=<path-to-sst-2>`                                  |
| **PRECISION**    |                               `export PRECISION=bf16`                             |
| **OUTPUT_DIR**    |                               `export OUTPUT_DIR=$PWD`                               |
| **FINETUNED_MODEL**    |                               `export FINETUNED_MODEL=<path to pre-trained model>`                               |
| **SEQUENCE_LENGTH**    |                               `export SEQUENCE_LENGTH=128`                        |
| **HF_DATASETS_OFFLINE**    |                               `export HF_DATASETS_OFFLINE=0`                               |
| **BATCH_SIZE** (optional)  |                               `export BATCH_SIZE=<set a value for batch size, else it will run with default batch size>`                                |
 | **CORES_PER_INSTANCE** |            `export CORES_PER_INSTANCE=<set a value for cores per instance>`  |
 | **INSTANCES (Only for custom mode)** |           `export INSTANCES=<set a value for instances>`
6. Run `run_model_inductor.sh`