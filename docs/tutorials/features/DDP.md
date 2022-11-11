# DistributedDataParallel (DDP)

## Introduction
`DistributedDataParallel (DDP)` is a PyTorch\* module that implements multi-process data parallelism across multiple GPUs and machines. With DDP, the model is replicated on every process, and each model replica is fed a different set of input data samples. DDP enables overlapping between gradient communication and gradient computations to speed up training. Please refer to https://pytorch.org/tutorials/intermediate/ddp_tutorial.html for an introduction to DDP.

The PyTorch `Collective Communication (c10d)` library supports communication across processes. To run DDP on XPU, we use Intel® oneCCL Bindings for Pytorch\* (formerly known as torch-ccl) to implement the PyTorch c10d ProcessGroup API (https://github.com/intel/torch-ccl). It holds PyTorch bindings maintained by Intel for the Intel® oneAPI Collective Communications Library\* (oneCCL), a library for efficient distributed deep learning training implementing such collectives as `allreduce`, `allgather`, and `alltoall`. Refer to https://github.com/oneapi-src/oneCCL for more information about oneCCL.

## Installation of Intel® oneCCL Bindings for Pytorch\*

To use PyTorch DDP on XPU, install Intel® oneCCL Bindings for Pytorch\* as described below.
### Install PyTorch and Intel® Extension for PyTorch\*

Make sure you have installed PyTorch and Intel® Extension for PyTorch\* successfully.
For more detailed information, check [installation guide](../installation.md).  

### Install Intel® oneCCL Bindings for Pytorch\*

Install from source:
```bash
git clone https://github.com/intel/torch-ccl.git
cd torch-ccl
git submodule sync 
git submodule update --init --recursive
python setup.py install
```

Install from prebuilt wheel:

Download the prebuilt wheel from http://mlpc.intel.com/downloads/gpu-new/releases/PVC_NDA_2022ww45/IPEX/RC2/ and use the `pip install` to install it.

## DDP usage for XPU in model training 

DDP follows its usage in PyTorch. To use DDP on XPU, make the following modifications to your model script:

1. Import the necessary packages.

```python
import torch
import intel_extension_for_pytorch 
import oneccl_bindings_for_pytorch
```      

2. Initialize the process group with ccl backend.

```python
dist.init_process_group(backend='ccl')
```        

3. For DDP with each process exclusively works on a single GPU, set the device ID as `local rank`.

```python 
device = "xpu:{}".format(args.local_rank)
torch.xpu.set_device(device)
```

4. Wrap model by DDP.

```python
model = model.to(device)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
```

Note: For single-device modules, `device_ids` can contain exactly one device id, which represents the only xpu device where the input module corresponding to this process resides. Alternatively, device_ids can be `None`.

## Example Usage (MPI launch for single node):
Intel® oneCCL Bindings for Pytorch\* recommends MPI as the launcher to start multiple processes.  Here's an example to illustrate such usage.

`Example_DDP.py`

```python
"""
This example shows how to use MPI as the launcher to start DDP on single node with multiple devices.
"""
import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(4, 5)

    def forward(self, input):
        return self.linear(input)


if __name__ == "__main__":

    mpi_world_size = int(os.environ.get('PMI_SIZE', -1))
    mpi_rank = int(os.environ.get('PMI_RANK', -1))
    if mpi_world_size > 0:
        os.environ['RANK'] = str(mpi_rank)
        os.environ['WORLD_SIZE'] = str(mpi_world_size)
    else:
        # set the default rank and world size to 0 and 1
        os.environ['RANK'] = str(os.environ.get('RANK', 0))
        os.environ['WORLD_SIZE'] = str(os.environ.get('WORLD_SIZE', 1))
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # your master address
    os.environ['MASTER_PORT'] = '29500'  # your master port

    # Initialize the process group with ccl backend
    dist.init_process_group(backend='ccl')

    # For single-node distributed training, local_rank is the same as global rank
    local_rank = dist.get_rank()
    device = "xpu:{}".format(local_rank)
    model = Model().to(device)
    if dist.get_world_size() > 1:
        model = DDP(model, device_ids=[device])

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss().to(device)
    for i in range(3):
        print("Runing Iteration: {} on device {}".format(i, device))
        input = torch.randn(2, 4).to(device)
        labels = torch.randn(2, 5).to(device)
        # forward
        print("Runing forward: {} on device {}".format(i, device))
        res = model(input)
        # loss
        print("Runing loss: {} on device {}".format(i, device))
        L = loss_fn(res, labels)
        # backward
        print("Runing backward: {} on device {}".format(i, device))
        L.backward()
        # update
        print("Runing optim: {} on device {}".format(i, device))
        optimizer.step()
```

Running command:
```bash
mpirun -n 2 -l python Example_DDP.py
```

## DDP scaling API
For using one GPU card has multiple tiles, each tile could be regarded as a device for explicit scaling. We provide a DDP scaling API to enable DDP on one GPU card in the GitHub repo at `intel_extension_for_pytorch/xpu/single_card.py`.

### Usage of DDP scaling API 
Note: This API supports XPU devices on one card.

```python
Args:
model: model to be parallelized
train_dataset: dataset for training
```

If you have a model running on a single tile, you only need to make minor changes to enable the DDP training by following these steps:

1. Import the API:

```python
try:
    from intel_extension_for_pytorch.xpu.single_card import single_card_dist
except ImportError:
    raise ImportError("single_card_dist not available!")
```

2. Use multi_process_spawn launcher as a torch.multiprocessing wrapper.

```python
single_card_dist.multi_process_spawn(main_worker, (args, )) # put arguments of main_worker into a tuple
```

3. Usage of this API:

```python
dist = single_card_dist(model, train_dataset)
local_rank, model, train_sampler = dist.rank, dist.model, dist.train_sampler
```

4. Set in the model training:

```python
for epoch in range ...
    train_sampler.set_epoch(epoch)
```

5. Adjust the model to call `local_rank`, `model`, and `train_sampler` as shown here:

- device: get the xpu information used in model training

```python
xpu = "xpu:{}".format(local_rank)
print("DDP Use XPU: {} for training".format(xpu))
```

- model: use the model warpped by DDP in the following training

- train_sampler: use the train_sampler to get the train_loader

```python
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    num_workers=args.workers, pin_memory=True, sampler=train_sampler)
```
Then you can start your model training on multiple XPU devices of one card.

