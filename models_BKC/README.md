# Guide to run auto-mix precision(int8) models with intel extension for pytorch

## Verified on

| Item | Value |
| -: | :- |
| OS | Ubuntu 18.04 LTS |
| Compiler | gcc 7.5.0 |
| Memory | DDR4 3200MHz, 256GB/socket |

## Environment setting

1. Install anaconda 3.0
```
  wget https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh -O anaconda3.sh
  chmod +x anaconda3.sh
  ./anaconda3.sh -b -p ~/anaconda3
  ./anaconda3/bin/conda create -n ipex python=3.7
```

2. Setup anaconda virtual environment for IPEX
```
  export PATH=~/anaconda3/bin:$PATH
  source ./anaconda3/bin/activate ipex
```

3. Install dependencies
```
  conda config --append channels intel
  conda install ninja pyyaml setuptools cmake cffi typing numpy
  conda install intel-openmp mkl mkl-include  -c intel --no-update-deps
  
  git clone  https://github.com/jemalloc/jemalloc.git    
  cd jemalloc
  git checkout c8209150f9d219a137412b06431c9d52839c7272
  ./autogen.sh
  ./configure --prefix=your_path(eg: /home/tdoux/tdoux/jemalloc/)
  make
  make install

```

4. Clone source code and build

```
  # PyTorch
  git clone https://github.com/pytorch/pytorch.git
  git checkout tags/v1.5.0-rc3 -b v1.5-rc3
  git submodule sync && git submodule update --init --recursive

  # extension
  git clone https://github.com/intel/intel-extension-for-pytorch.git
  cd intel-extension-for-pytorch
  git checkout icx
  git submodule update --init --recursive
  cd third_party/mkl-dnn
  patch -p1 < ../../torch_patches/FIFO.diff
  cd ../..
  pip install -r requirements.txt


  # prepare patch to PyTorch
  cp {path/to/intel-pytorch-extension}/torch_patches/dpcpp-v1.5-rc3.patch {path/to/pytorch}/

  # build PyTorch
  cd {path/to/pytorch}
  patch -p1 < dpcpp-v1.5-rc3.patch
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
  cd dlrm
  pip install -r requirement.txt
  git checkout 529091f61e62aa29d29cb50faa3634a273ec2bcb
  patch -p1 < {path/to/intel-pytorch-extension}/torch_patches/models/dlrm.patch
```

2. Get ResNet50
```
  git clone https://github.com/pytorch/vision
  cd vision
  git checkout v0.6.0
  python setup.py install
  cd ..
  git clone https://github.com/XiaobingSuper/examples.git
  cd examples
  git checkout int8


```

## Run Models
0. Enviroment setting
```
export LD_PRELOAD="{path/to/jemalloc}/lib/libjemalloc.so:${CONDA_PREFIX}/lib/libiomp5.so"
MALLOC_CONF="retain:true,narenas:24,percpu_arena:percpu,background_thread:true,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
```

1. DLRM
```
  cd {path/to/dlrm}
  # prepare dataset
  cd /tmp && mkdir input
  curl -O http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_{$(seq -s , 0 23)}.gz
  # unzip all download files into one folder
  export DATASET_PATH={patch/to/dlrm_dataset} 
```
```
    # run DLRM fp32 inference
  bash run_inference_latency.sh dnnl
```
```
    # run DLRM int8 inference
  bash run_inference_latency.sh dnnl int8
```

2. ResNet50
  Download imagenet dataset reference https://github.com/facebookarchive/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset
```
  cd {path/to/examples}
  cd imagenet
  export DATASET_PATH={patch/to/bert}
```
```
  # run fp32 inference throughput
bash run_int8_multi_instance_ipex.sh resnet50 dnnl fp32 jit
```
```
  # run fp32 inference latency
bash run_int8_multi_instance_latency_ipex.sh resnet50 dnnl fp32 jit
```
```
  # run int8 inference throughput
bash run_int8_multi_instance_ipex.sh resnet50 dnnl int8 jit resnet50_configure_jit.json
```
```
  # run int8 inference latency
bash run_int8_multi_instance_latency_ipex.sh resnet50 dnnl int8 jit resnet50_configure_jit.json
```

