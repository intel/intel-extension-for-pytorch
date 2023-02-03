# Model Convert Tool

change cuda script to xpu, implicitly(during runtime) and explicitly(before runtime).

## Implicit Mode

### Install
```console
$ python setup.py install
```

### Usage Example

try following commands step by step to be familar to this tool, you will see that after import model_convert, you can run cuda code successfully on XPU device or you will get warning if one API is not supported by XPU backend.


```console
import torch
torch.cuda.device_count()
torch.cuda.get_device_capability()
torch.ones(3, 4).to('cuda')
torch.ones(3, 4, device=torch.device("cuda"))
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

```console
$ export VERBOSE_MODEL_CONVERT=1
```

## Explicit Mode

change model script explicitly


### Usage Examples

Following steps show how to use this tool

#### safety mode

Only touch .py and '.ipynb' files, won't convert the unsupported cuda api.

```console
$ python model_script_convert.py -p ./demo
```

#### aggressive mode

Modify string 'cuda' to 'xpu' in all files

```console
$ python model_script_convert.py -p ./demo -a
```
#### in-place mode

Change the file directly if add option '-i' or '--in-place'

in-place mode with safety mode

Notice: in-place mode will change the files silently, not im-place mode will dump the difference

```console
$ python model_script_convert.py -p ./demo -i
```

in-place mode with aggressive mode

```console
$ python model_script_convert.py -p ./demo -i -a
```

#### verbose mode

Turn on verbose if add option '-i' or '--in-place'

```console
$ python model_script_convert.py -p ./demo -v
```
