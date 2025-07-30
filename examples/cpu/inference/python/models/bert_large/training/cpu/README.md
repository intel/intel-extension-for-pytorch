# BERT Large Training

BERT Large training best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Training   |    PyTorch    |       https://github.com/huggingface/transformers/tree/main/src/transformers/models/bert        |           -           |         -          |

# Pre-Requisite
* Installation of PyTorch and [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/#installation)

## Bare Metal
### General setup

Follow [link]((https://github.com/IntelAI/models/blob/master/docs/general/pytorch/BareMetalSetup.md)) to install Pytorch, IPEX, TorchVison, Miniforge, Jemalloc and TCMalloc.

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

* Set ENV to use multi-nodes distributed training (no need for single-node multi-sockets)

  In this case, we use data-parallel distributed training and every rank will hold same model replica. The NNODES is the number of ip in the HOSTFILE. To use multi-nodes distributed training you should firstly setup the passwordless login (you can refer to [link](https://linuxize.com/post/how-to-setup-passwordless-ssh-login/)) between these nodes.
  ```
  export NNODES=#your_node_number
  export HOSTFILE=your_ip_list_file #one ip per line
  ```

* [optional] Compile model with PyTorch Inductor backend (support fp32/bf16/fp16)
```shell
  export TORCH_INDUCTOR=1
```


## Datasets

# Location of the input files

This [MLCommons members Google Drive location](https://drive.google.com/drive/u/0/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT) contains the following.
* TensorFlow checkpoint (bert_model.ckpt) containing the pre-trained weights (which is actually 3 files).
* Vocab file (vocab.txt) to map WordPiece to word id.
* Config file (bert_config.json) which specifies the hyperparameters of the model.

# Checkpoint conversion
python convert_tf_checkpoint.py --tf_checkpoint /cks/model.ckpt-28252 --bert_config_path /cks/bert_config.json --output_checkpoint model.ckpt-28252.pt

# Download the preprocessed text dataset

From the [MLCommons BERT Processed dataset
directory](https://drive.google.com/drive/folders/1cywmDnAsrP5-2vsr8GDc6QUc7VWe-M3v?usp=sharing)
download `results_text.tar.gz`, and `bert_reference_results_text_md5.txt`.  Then perform the following steps:

```shell
tar xf results_text.tar.gz
cd results4
md5sum --check ../bert_reference_results_text_md5.txt
cd ..
```

After completing this step you should have a directory called `results4/` that
contains 502 files for a total of about 13Gbytes.

# Generate the BERT input dataset

The [create_pretraining_data.py](/models/language_modeling/pytorch/bert_large/training/input_preprocessing/create_pretraining_data.py) script duplicates the input plain text, replaces
different sets of words with masks for each duplication, and serializes the
output into the HDF5 file format.

## Training data

The following shows how create_pretraining_data.py is called by a parallelized
script that can be called as shown below.  The script reads the text data from
the `results4/` subdirectory and outputs the resulting 500 hdf5 files to a
subdirectory named `hdf5/`.

```shell
pip install tensorflow-cpu protobuf==3.20.3 numpy==1.20
```

For phase1 the seq_len=128:
```shell
export SEQ_LEN=128
cd <path to AIRM/models_v2/pytorch/bert_large/training/cpu>
./input_preprocessing/parallel_create_hdf5.sh
```
For phase2 the seq_len=512:
```shell
export SEQ_LEN=512
cd <path to AIRM/models_v2/pytorch/bert_large/training/cpu>
./input_preprocessing/parallel_create_hdf5.sh
```

The resulting `hdf5/` subdir will have 500 files named
`part-00???-of-0500.hdf5` and have a size of about 539 Gigabytes.

Next we need to shard the data into 2048 chunks.  This is done by calling the
chop_hdf5_files.py script.  This script reads the 500 hdf5 files from
subdirectory `hdf5/` and creates 2048 hdf5 files in subdirectory
`2048_shards_uncompressed`.

For phase1:

```shell
export SEQ_LEN=128
python3 ./input_preprocessing/chop_hdf5_files.py
```

For phase2:

```shell
export SEQ_LEN=512
python3 ./input_preprocessing/chop_hdf5_files.py
```

The above will produce a subdirectory named `2048_shards_uncompressed/`
containing 2048 files named `part_*_of_2048.hdf5` and have a size of about 539 Gigabytes.
you can use "SHARD_NUM" to control the shard files number. the default "SHARD_NUM" if 2048.

```
<DATASET_DIR>
├── 2048_shards_uncompressed_512
│   └── part-00000-of-00xxx
└── 2048_shards_uncompressed_128
    └── part-00000-of-00xxx
```

# Training
1. `git clone https://github.com/intel/intel-extension-for-pytorch`
2. `cd intel-extension-for-pytorch/examples/cpu/inference/python/models/bert_large/training/cpu`
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
| **DDP** (true or false)              | `export DDP=false`                  |
| **TRAINING_PHASE** (1 or 2)              | `export TRAINING_PHASE=1`                  |
| **BERT_MODEL_CONFIG**  (1st phase only)            | `export BERT_MODEL_CONFIG=$(pwd)/bert_config.json`                  |
| **CHECKPOINT_DIR** (1st phase only)            | `export CHECKPOINT_DIR=$(pwd)/checkpoint_phase1_dir`                  |
| **PRETRAINED_MODEL** (2nd phase only)            | `export PRETRAINED_MODEL=$(pwd)/checkpoint_phase1_dir`                  |
| **DATASET_DIR**              | `export DATASET_DIR=<path/to/dataset>`                  |
| **OUTPUT_DIR**               |                               `export OUTPUT_DIR=$PWD`                               |
| **TRAIN_SCRIPT**               |                                `export TRAIN_SCRIPT=$(pwd)/run_pretrain_mlperf.py`                                |
| **PRECISION**     |                  `export PRECISION=<specify the precision to run: fp32, avx-fp32, bf16 or bf32>` |
| **MODEL_DIR**               |                               `export MODEL_DIR=$(pwd)`                               |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=256`                                |

7. Run `run_model.sh`

## Output

Single-tile output will typically looks like:

```
[info] construct file from initialization
[info] input dir =  /home/gta/Cosim_test/dataset/hdf5
[info] num files =  2
epoch: 1
Loaded 193485 samples from datafile: /home/gta/Cosim_test/dataset/hdf5/pretrain-part-01.hdf5
bert_train latency:  0.24147300720214843  s
bert_train throughput:  66.25999396531161  sentences/s
perplexity = 11.020857810974121
```
Final results of the inference run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 66.259994
   unit: sent/s
 - key: latency
   value: 0.2414730072021484
   unit: s
 - key: accuracy
   value: 11.021
   unit: perplexity
```
