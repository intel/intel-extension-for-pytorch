import os
import sys
if "override.py" in os.listdir(r"./"):
    sys.exit("override.py already exist")


torch_function = ['torch.rand', 'torch.randint', 'torch.arange', 'torch.bartlett_window', 'torch.blackman_window', \
                  'torch.empty', 'torch._empty_affine_quantized', 'torch._empty_per_channel_affine_quantized', \
                  'torch.empty_strided', 'torch.eye', 'torch.full', 'torch.from_file', 'torch.from_numpy', \
                  'torch.hann_window', 'torch.hamming_window', 'torch.linspace', 'torch.logspace', 'torch.ones', \
                  'torch.scalar_tensor', 'torch.randn', 'torch.randperm', 'torch.range', 'torch.zeros', \
                  'torch.sparse_coo_tensor', 'torch.tril_indices', 'torch.triu_indices', 'torch.normal', 'torch.tensor']
code = ""
code += "import torch\n"
code += "import intel_pytorch_extension as ipex\n"
code += "import global_settings\n"
code += "ipex.core.enable_auto_dnnl()\n"
code += "\n"
code += "\n"
for func in torch_function:
    names = func.split('.')
    func_name = ""
    for name in names:
        func_name += "_"
        func_name += name
    code += func_name + " = " + func + "\n"
    code += "\n"
    code += "\n"
    code += "def dpcpp" + func_name + "(*args, **kwargs):\n"
    code += "    if 'device' in kwargs and global_settings.do_override:\n"
    code += "        return " + func_name + "(*args, **kwargs)\n"
    code += "    else:\n"
    code += "        return " + func_name + "(*args, **kwargs).to(\"dpcpp\")\n"
    code += "\n"
    code += "\n"
    code += func + " = " + "dpcpp" + func_name + "\n"
    code += "\n"



fp = open(r'./override.py', 'w')
fp.write(code)
fp.close()
