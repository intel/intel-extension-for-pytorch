# Apply torch patches
You need to apply these git patches for PyTorch and build PyTorch from source code.
   ```bash
   git clone https://github.com/pytorch/pytorch.git
   cd pytorch/
   git checkout 32f585d9346e316e554c8d9bf7548af9f62141fc
   git apply <ipex code repo>/torch_patches/00*.patch
   python setup.py install
   ```
