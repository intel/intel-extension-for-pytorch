# Apply torch patches
You need to apply these git patches for PyTorch and build PyTorch from source code.
   ```bash
   git clone https://github.com/pytorch/pytorch.git
   cd pytorch/
   git checkout b4f90aae1b375bfe06d3c4a099240e06f93c81c4
   git apply <ipex code repo>/torch_patches/00*.patch
   python setup.py install
   ```
