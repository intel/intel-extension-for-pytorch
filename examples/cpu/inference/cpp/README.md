# Model Inference using C++ with Intel® Extension for PyTorch\* Optimizations

To work with libtorch (C++ library of PyTorch), Intel® Extension for PyTorch* provides its C++ dynamic library as well. The C++ library is supposed to handle inference workload only, such as service deployment. Compilation follows the recommended methodology with CMake. 
During compilation, Intel optimizations will be activated automatically once C++ dynamic library of Intel® Extension for PyTorch* is linked.

## Dependencies Installation

```bash
# Clone the repository and access to the c++ inference example folder
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch/examples/cpu/inference/cpp
```

We can have `libtorch` and `libintel-ext-pt` installed via the following commands.

Download zip file of `libtorch` and decompress it:

```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.2.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.2.0+cpu.zip
```

Download and execute `libintel-ext-pt` installation script:

```bash
wget https://intel-extension-for-pytorch.s3.amazonaws.com/libipex/cpu/libintel-ext-pt-cxx11-abi-2.2.0%2Bcpu.run
bash libintel-ext-pt-cxx11-abi-2.2.0+cpu.run install ./libtorch
```

*Note:* If your C++ project has pre-C\+\+11 library dependencies,
you need to download and install the pre-C\+\+11 ABI version library files.

Please view the `cppsdk` part in [the installation guide](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=cpu) 
for more details about command usage.

## Running the Example

Download a pretrained ResNet50 model and save it locally:

```bash
python model_gen.py
```

Build the executable file from source code via CMake:

```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=./libtorch ..
make
```

*Note:* In the `cmake` command, it is required to specify the path of `libtorch` folder as `CMAKE_PREFIX_PATH`.
If it is specified with relative path, it should be relative to the folder containing `CMakeList.txt` file,
normally the parent folder of the `build` folder in which we execute the command.

Run the executable file:

```bash
./example-app ../resnet50.pt
```

Please view the [c++ example in Intel® Extension for PyTorch\* online document](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/examples.html#c) for more information.