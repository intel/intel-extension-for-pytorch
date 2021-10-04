# Intel® Extension for PyTorch with LLGA

## Installation

### Install PyTorch from Source


0. Create env:
    ```bash
    conda create -n ipex python=3.7 numpy ninja pyyaml pytest mkl mkl-include setuptools cmake cffi typing
    conda activate ipex
    ```


 1. Get PyTorch master source(Refer to [PyTorch guide](https://github.com/pytorch/pytorch#get-the-pytorch-source) for more details)
    ```bash
    git clone --recursive https://github.com/pytorch/pytorch
    cd pytorch

    # checkout source code to the specified version
    # TODO: need use Mingfei's branch which has fixed the NHWC accuracy issue in PyTorch master temporarily
    # Once Mingfei's PR has been merged into PyTorch master, remove this line
    git checkout db9b236

    # update submodules for the specified PyTorch version
    git submodule sync
    git submodule update --init --recursive
    ```

 2. Get the source code of Intel Extension for PyTorch
    ```bash
    git clone --recursive ssh://git@gitlab.devtools.intel.com:29418/intel-pytorch-extension/ipex-cpu-dev.git
    cd ipex-cpu-dev

    git checkout cpu-device

    # if you are updating an existing checkout
    git submodule sync
    git submodule update --init --recursive
    ```

    Tips to access intel-innersource:
    1. Make sure you have "1Source Github User" applied on [ags.intel.com](ags.intel.com)
    2. Use personal access token of Github: [link](https://docs.github.com/en/github/authenticating-to-github/creating-a-personal-access-token)
    3. If you have "local refs could not be updated" during updating the submodule,
    try:
        ```
        cd third_party/llga
        git remote prune origin
        cd ../..
        ```
        And then continue update the submodule

 3. Add Autocast Dispatch Key for PyTorch (temporary step, not needed after autocast feature landed into the stock PyTorch)
    ```bash
    # Apply git patch to pytorch code
    cd ${pytorch_directory}
    git apply ${intel_extension_for_pytorch_directory}/torch_patches/autocast.patch
    ```

 4. Build and install PyTorch (Refer to [PyTorch guide](https://github.com/pytorch/pytorch#install-pytorch) for more details)
    ```bash
    cd ${pytorch_directory}
    CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"} python setup.py develop
    ```

### Install Intel Extension for PyTorch from Source
Install dependencies
```bash
pip install lark-parser hypothesis
```

Install the extension
```bash
cd ${intel_extension_for_pytorch_directory}
python setup.py develop
```

### Run test
```bash
pip install pytest pytest-sugar==0.9.3 

python -m pytest -v tests/cpu/test_jit_llga_quantization_fuser.py
```

## Contribution

Please submit PR or issue to communicate with us or contribute code.


## License

_Apache License_, Version _2.0_. As found in [LICENSE](https://github.com/intel/intel-extension-for-pytorch/blob/master/LICENSE.txt) file.
