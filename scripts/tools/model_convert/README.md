
# Model Script Convert Tool

This tool can be used to help leverage effort to evaluate Pytorch model script on XPU backend for Intel® Extension for PyTorch*, but it is not a product-level tool.
It provides implicit and explicit modes.


## Implicit Mode

This mode won't change model script, but will change behavior silently during runtime of application, users won't aware the change details.

### Install
```console
$ python setup.py install
```

### Usage Example

Try following commands step by step to be familar to this tool, you will see that after import model_convert, you can run code successfully on XPU device or you will get warning if one API is not supported by XPU backend.

Run on CUDA device:
```python
import torch
torch.cuda.device_count()
torch.cuda.get_device_capability()
torch.ones(3, 4).to('cuda')
torch.ones(3, 4, device=torch.device("cuda"))
```
Run on XPU device:
```python
import torch
import intel_extension_for_pytorch
import model_convert
torch.cuda.device_count()
torch.cuda.get_device_capability()
torch.ones(3, 4).to('cuda')
torch.ones(3, 4, device=torch.device("cuda"))
```


### API Support 

check [Support List](./model_convert/yaml/api_supported_by_xpu.yaml) for supported API list.

check [Not Support List](./model_convert/yaml/api_unsupported_by_xpu.yaml) for unsupported API list.


### Run Unit Test

```console
$ cd tests
$ pytest -v test_to.py
$ pytest -v test_create_tensor.py
$ pytest -v test_data_loader.py
$ pytest -v test_device.py
# need to install mpi
$ mpirun -n 2 python test_ddp.py
```

### Enable Verbose Mode
You need to enable VERBOSE_MODEL_CONVERT flag in order to dump detailed info.

```console
$ export VERBOSE_MODEL_CONVERT=1
```

## Explicit Mode

This mode can generate modified model script directly. 


### Usage Examples

Following will explain how to use each option.

* path option: you need to create a directory and then put all files into it.

```console
$ python model_convert/export.py -p ./demo
```

* aggressive option: aggressively change behavior, might not safe sometimes. 

```console
$ python model_convert/export.py -p ./demo -a
```

* in-place option: change the file directly if add option '-i' or '--in-place', won't generate new file.

```console
$ python model_convert/export.py -p ./demo -i
```

* verbose option: turn on verbose if add option '-i' or '--in-place'

```console
$ python model_convert/export.py -p ./demo -v
```

