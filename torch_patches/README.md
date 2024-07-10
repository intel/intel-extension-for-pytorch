# Apply torch patches
You need to apply these git patches for PyTorch and build PyTorch from source code.
   ```bash
   git clone https://github.com/pytorch/pytorch.git
   cd pytorch/
   git checkout 63d5e9221bedd1546b7d364b5ce4171547db12a9
   git apply <ipex code repo>/torch_patches/00*.patch
   python setup.py install
   ```
