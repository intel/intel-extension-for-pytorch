# Apply torch patches
You need to apply these git patches for PyTorch and build PyTorch from source code.
   ```bash
   git clone https://github.com/pytorch/pytorch.git
   cd pytorch/
   git checkout 209f2fa8ff86652f67d75c2f19bf9cb9942fd018
   git apply <ipex code repo>/torch_patches/00*.patch
   python setup.py install
   ```
