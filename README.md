# Intel Pytorch Extension

> This is Intel Pytorch Extension Official Gitlab Repo

## PreInstall

- Ubuntu 18.04
- intel_sdk_for_opencl_applications
  - Link: https://software.intel.com/en-us/opencl-sdk/choose-download
- Intel NEO Driver/Runtime
  - Link: https://github.com/intel/compute-runtime/releases
- ComputeCpp-CE-1.1.3-Ubuntu-16.04-x86_64
  - Link: https://developer.codeplay.com/computecppce/latest/download
- Pytorch
  - Link: https://gitlab.devtools.intel.com/intel-pytorch-extension/pytorch
- gcc-5
  - **Notice:** Ubuntu default gcc is 7.5.0, u need to hack it with 5.5.0 <sub>*[1]*</sub> if ComputeCPP was used
- gitpython
- MKL (Optional)
- OpenMP (Optional)

## Install

```bash
## conda env
conda create -n <exten> --clone <yourenv>
conda activate <exten>
## check pytorch
## !!!  DO NOT IMPORT PYTORCH IN PYTORCH SOURCE DIR !!!
python -c "import torch; print('torch version: {0}'.format(torch.__version__))"
## clone source
git  clone -b dev-gpu  --depth=1 ssh://git@gitlab.devtools.intel.com:29418/intel-pytorch-extension/intel-pytorch-extension.git
cd intel-pytorch-extension
git submodule update  --init --recursive || git submodule update  --init --recursive || git submodule update  --init --recursive
## preinsatll
python -m pip install -r ./requirements.txt
## clean old build
python ./setup.py clean
## set env vars
export COMPUTECPP_DIR=/path/to/ComputeCpp-CE-1.1.3-Ubuntu-16.04-x86_64
export CXX=${COMPUTECPP_DIR}/bin/compute++
export PATH=${COMPUTECPP_DIR}/bin:${PATH}
export LD_LIBRARY_PATH=${COMPUTECPP_DIR}/include:${LD_LIBRARY_PATH}
## build
VERBOSE=1 python ./setup.py install 2>&1 | tee ./build.log
```

## Additional

- *[1]* How to hack gcc-7 with gcc-5 if ComputeCpp was used

  ```bash
  sudo apt install gcc-5 g++-5
  cd /usr/include/c++
  sudo mv 7 7_bak
  sudo ln -s 5 7
  cd  /usr/lib/gcc/x86_64-linux-gnu
  sudo mv 7 7_bak
  sudo ln -s 5 7
  ```