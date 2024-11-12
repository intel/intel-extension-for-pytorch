# Apply torch patches
You need to apply these git patches for PyTorch and build PyTorch from source code.
   ```bash
   git clone https://github.com/pytorch/pytorch.git
   cd pytorch/
   git checkout a8d6afb511a69687bbb2b7e88a3cf67917e1697e
   git apply <ipex code repo>/torch_patches/00*.patch
   python setup.py install
   ```
