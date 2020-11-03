# Guide to run DLRM with FP32/BF16 data type

## Verified on

| Item | Value |
| -: | :- |
| OS | Ubuntu 18.04 LTS |
| Compiler | gcc 7.5.0 |
| Memory | DDR4 3200MHz, 96GB/socket |

## Environment setting

1. Install anaconda 3.0
```
  wget https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh -O anaconda3.sh
  chmod +x anaconda3.sh
  ./anaconda3.sh -b -p ~/anaconda3
  ./anaconda3/bin/conda create -n ipex python=3.7
```

2. Setup anaconda virtual environment for DLRM
```
  export PATH=~/anaconda3/bin:$PATH
  source ./anaconda3/bin/activate ipex
```

3. Install dependencies
```
  conda config --append channels intel
  conda install ninja pyyaml setuptools cmake cffi typing numpy
  conda install intel-openmp  mkl-include  -c intel --no-update-deps
  conda install mkl==2018 -c intel
  conda install jemalloc

```

4. Clone source code and build

```
  # PyTorch
  git clone https://github.com/pytorch/pytorch.git
  git checkout tags/v1.5.0-rc3 -b v1.5-rc3
  git submodule sync && git submodule update --init --recursive

  # extension
  git clone https://github.com/intel/intel-extension-for-pytorch.git
  git checkout 1.1.0-preview
  git submodule update --init –recursive

  # prepare patch to PyTorch
  cp {path/to/intel-pytorch-extension}/torch_patches/dlrm_fp32.patch {path/to/pytorch}/
  cp {path/to/intel-pytorch-extension}/torch_patches/dpcpp-v1.5-rc3.patch {path/to/pytorch}/

  # build PyTorch
  cd {path/to/pytorch}
  patch -p1 < dpcpp-v1.5-rc3.patch
  patch -p1 < dlrm_fp32.patch
  pip install -r requirement.txt
  python setup.py install

  # build extension
  cd {path/to/intel-pytorch-extension}
  pip install -r requirement.txt
  python setup.py install

  ```

## Prepare Models

1. Get DLRM
```
  git clone https://github.com/facebookresearch/dlrm
  pip install -r requirement.txt
  git checkout 52b77f80a24303294a02c86b574529cdc420aac5
  patch -p1 < {path/to/intel-pytorch-extension}/torch_patches/models/dlrm.patch
```

2. Get BERT-Large
```
  git clone https://github.com/huggingface/transformers
  git checkout 1a779ad7ecb9e5215b6bd1cfa0153469d37e4274
  pip install -r ./examples/requirement.txt
  pip install --editable .
```

3. Get ResNext-101-32x4d
```
TODO
```
## Run Models
0. Enviroment setting
```
export LD_PRELOAD="${CONDA_PREFIX}/lib/libjemalloc.so:${CONDA_PREFIX}/lib/libiomp5.so"
MALLOC_CONF="retain:true,narenas:24,percpu_arena:percpu,background_thread:true,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
```

1. DLRM
```
  cd {path/to/dlrm}
  export DATASET_PATH={patch/to/dlrm_dataset}
```
```
  # run DLRM fp32 trainining
  OMP_NUM_THREADS=24 numactl -C 0-23 -m 0 python -u \
  dlrm_s_pytorch.py --data-generation=dataset --data-set=terabyte  \
  --raw-data-file=${DATASET_PATH}/day --processed-data-file=${DATASET_PATH}/terabyte_processed.npz \
  --loss-function=bce \
  --memory-map --mlperf-bin-loader --round-targets=True --learning-rate=1.0 \
  --arch-mlp-bot=13-512-256-128 --arch-mlp-top=1024-1024-512-256-1 \
  --arch-sparse-feature-size=128 --max-ind-range=4000000 \
  --numpy-rand-seed=727 \
  --print-freq=1024 --print-time --mini-batch-size=2048 --num-batches=4096 
```
```
    # run DLRM bf16 trainining
  OMP_NUM_THREADS=24 numactl -C 0-23 -m 0 python -u \
  dlrm_s_pytorch.py --data-generation=dataset --data-set=terabyte  \
  --raw-data-file=${DATASET_PATH}/day --processed-data-file=${DATASET_PATH}/terabyte_processed.npz \
  --loss-function=bce \
  --memory-map --mlperf-bin-loader --round-targets=True --learning-rate=1.0 \
  --arch-mlp-bot=13-512-256-128 --arch-mlp-top=1024-1024-512-256-1 \
  --arch-sparse-feature-size=128 --max-ind-range=4000000 \
  --numpy-rand-seed=727 \
  --print-freq=1024 --print-time --mini-batch-size=2048 --num-batches=4096 \
  --use-ipex --mix-precision 
```
```
    # run DLRM fp32 inference
  OMP_NUM_THREADS=1 numactl -C 0-23 -m 0 python -u \
  dlrm_s_pytorch.py --data-generation=dataset --data-set=terabyte  \
  --raw-data-file=${DATASET_PATH}/day --processed-data-file=${DATASET_PATH}/terabyte_processed.npz \
  --loss-function=bce \
  --memory-map --mlperf-bin-loader --round-targets=True --learning-rate=1.0 \
  --arch-mlp-bot=13-512-256-128 --arch-mlp-top=1024-1024-512-256-1 \
  --arch-sparse-feature-size=128 --max-ind-range=4000000 \
  --numpy-rand-seed=727 \
  --print-freq=1024 --print-time --mini-batch-size=16 --num-batches=4096 \
  --inference-only --share-weight --num-instance=24
```
```
    # run DLRM bf16 inference
  OMP_NUM_THREADS=1 numactl -C 0-23 -m 0 python -u \
  dlrm_s_pytorch.py --data-generation=dataset --data-set=terabyte  \
  --raw-data-file=${DATASET_PATH}/day --processed-data-file=${DATASET_PATH}/terabyte_processed.npz \
  --loss-function=bce \
  --memory-map --mlperf-bin-loader --round-targets=True --learning-rate=1.0 \
  --arch-mlp-bot=13-512-256-128 --arch-mlp-top=1024-1024-512-256-1 \
  --arch-sparse-feature-size=128 --max-ind-range=4000000 \
  --numpy-rand-seed=727 \
  --print-freq=1024 --print-time --mini-batch-size=16 --num-batches=4096 \
  --use-ipex --mix-precision --inference-only --share-weight --num-instance=24
```

2. Get BERT-Large
```
  cd {path/to/bert}
  export DATASET_PATH={patch/to/bert}
```
```
  # run Bert fp32 trainining
OMP_NUM_THREADS=24 numactl -C 0-23 -m 0 python -u  \
examples/language-modeling/run_language_modeling.py \
--output_dir=output_1 --per_gpu_train_batch_size=24  \
--model_type=bert_large --do_train --train_data_file=$DATASET_PATH/wiki.train.raw \
--overwrite_output_dir --mlm --seed=42 --max_step=30
```
```
  # run Bert bf16 trainining
OMP_NUM_THREADS=24 numactl -C 0-23 -m 0 python -u  \
examples/language-modeling/run_language_modeling.py \
--output_dir=output_1 --per_gpu_train_batch_size=24  \
--model_type=bert_large --do_train --train_data_file=$DATASET_PATH/wiki.train.raw \
--overwrite_output_dir --mlm --seed=42 --max_step=30 --ipex --dnnl --mix_precision
```
```
  # run Bert fp32 inference
for i in $(seq 0 $LAST_INSTANCE); do
    LOG_i=cpufp32_bs1_ins${i}.txt
    echo "### running on instance $i, numa node 0, core  $i"
    numactl --physcpubind=$i --membind=0  python -u \
    examples/language-modeling/run_language_modeling.py \
    --output_dir=output_$i --per_gpu_eval_batch_size=1 \
    --model_type=bert_large  --do_eval --eval_data_file=$HOME/wikitext-2-raw/wiki.train.raw \
    --overwrite_output_dir --mlm --seed=42 --max_step=30  2>&1 | tee $LOG_i &
done
```
```
  # run Bert bf16 inference
for i in $(seq 0 $LAST_INSTANCE); do
    LOG_i=cpufp32_bs1_ins${i}.txt
    echo "### running on instance $i, numa node 0, core  $i"
    numactl --physcpubind=$i --membind=0  python -u \
    examples/language-modeling/run_language_modeling.py \
    --output_dir=output_$i --per_gpu_eval_batch_size=1 \
    --model_type=bert_large  --do_eval --eval_data_file=$HOME/wikitext-2-raw/wiki.train.raw \
    --overwrite_output_dir --mlm --seed=42 --max_step=30 --ipex --dnnl --mix_precision  2>&1 | tee $LOG_i &
done
```

3. Get ResNext-101-32x4d
```
TODO
```