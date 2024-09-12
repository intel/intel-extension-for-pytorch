Fully Sharded Data Parallel (FSDP)
=============================

## Introduction

`Fully Sharded Data Parallel (FSDP)` is a PyTorch\* module that provides industry-grade solution for large model training. FSDP is a type of data parallel training, unlike DDP, where each process/worker maintains a replica of the model, FSDP shards model parameters, optimizer states and gradients across DDP ranks to reduce the GPU memory footprint used in training. This makes the training of some large-scale models feasible. Please refer to [FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) for an introduction to FSDP.

To run FSDP on GPU, similar to DDP, we use Intel® oneCCL Bindings for Pytorch\* (formerly known as torch-ccl) to implement the PyTorch c10d ProcessGroup API (https://github.com/intel/torch-ccl). It holds PyTorch bindings maintained by Intel for the Intel® oneAPI Collective Communications Library\* (oneCCL), a library for efficient distributed deep learning training implementing collectives such as `AllGather`, `ReduceScatter`, and other needed by FSDP. Refer to [oneCCL Github page](https://github.com/oneapi-src/oneCCL) for more information about oneCCL.
To install Intel® oneCCL Bindings for Pytorch\*, follow the same installation steps as for DDP. 

## FSDP Usage (GPU only)

FSDP is designed to align with PyTorch conventions. To use FSDP with Intel® Extension for PyTorch\*, make the following modifications to your model script:

1. Import the necessary packages.
```python
import torch
import intel_extension_for_pytorch 
import oneccl_bindings_for_pytorch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
```

2. Initialize the process group with ccl backend.
```python
dist.init_process_group(backend='ccl')
``` 

3. For FSDP with each process exclusively working on a single GPU, set the device ID as `local rank`.
```python
torch.xpu.set_device("xpu:{}".format(rank))
# or
device = "xpu:{}".format(args.local_rank)
torch.xpu.set_device(device)
```

4. Wrap model by FSDP.
```python
model = model.to(device)
model = FSDP(model, device_id=device)
```

**Note**: for FSDP with XPU, you need to specify `device_ids` with XPU device; otherwise, it will trigger the CUDA path and throw an error.

## Example

Here's an example based on [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) to illustrate the usage of FSDP on XPU and the necessary changes to switch from CUDA to an XPU case.

1. Import necessary packages:

```python
"""
Import Intel® extension for Pytorch\* and Intel® oneCCL Bindings for Pytorch\*
"""
import os
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Import Intel® extension for Pytorch\* and Intel® oneCCL Bindings for Pytorch\*
import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
```

2. Set up distributed training:

```python
"""
Set the initialize the process group backend as Intel® oneCCL Bindings for Pytorch\*
"""
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group by Intel® oneCCL Bindings for Pytorch\*
    dist.init_process_group("ccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
```

3. Define the toy model for handwritten digit classification:

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
```

4. Define a training function:

```python
"""
Change the device related logic from 'rank' to '"xpu:{}".format(rank)'
"""
def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    # XPU device should be formatted as string, replace the rank with '"xpu:{}".format(rank)'
    ddp_loss = torch.zeros(2).to("xpu:{}".format(rank))
    if sampler:
        sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to("xpu:{}".format(rank)), target.to("xpu:{}".format(rank))
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction='sum')
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))
```

5. Define a validation function:

```python
"""
Change the device related logic from 'rank' to '"xpu:{}".format(rank)'
"""
def test(model, rank, world_size, test_loader):
    model.eval()
    correct = 0
    # XPU device should be formatted as string, replace the rank with '"xpu:{}".format(rank)'
    ddp_loss = torch.zeros(3).to("xpu:{}".format(rank))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to("xpu:{}".format(rank)), target.to("xpu:{}".format(rank))
            output = model(data)
            ddp_loss[0] += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        test_loss = ddp_loss[0] / ddp_loss[2]
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, int(ddp_loss[1]), int(ddp_loss[2]),
            100. * ddp_loss[1] / ddp_loss[2]))
```

6. Define a distributed training function that wraps the model in FSDP:

```python
"""
Change the device related logic from 'rank' to '"xpu:{}".format(rank)'.
Specify the argument `device_ids` as XPU device ("xpu:{}".format(rank)) in FSDP API.
"""
def fsdp_main(rank, world_size, args):
    setup(rank, world_size)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                        transform=transform)

    sampler1 = DistributedSampler(dataset1, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(dataset2, rank=rank, num_replicas=world_size)

    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}
    xpu_kwargs = {'num_workers': 2,
                    'pin_memory': True,
                    'shuffle': False}
    train_kwargs.update(xpu_kwargs)
    test_kwargs.update(xpu_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )
    torch.xpu.set_device("xpu:{}".format(rank))


    init_start_event = torch.xpu.Event(enable_timing=True)
    init_end_event = torch.xpu.Event(enable_timing=True)

    model = Net().to("xpu:{}".format(rank))
    # Specify the argument `device_ids` as XPU device ("xpu:{}".format(rank)) in FSDP API.
    model = FSDP(model, device_id="xpu:{}".format(rank))

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    init_start_event.record()
    for epoch in range(1, args.epochs + 1):
        train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
        test(model, rank, world_size, test_loader)
        scheduler.step()

    init_end_event.record()

    if rank == 0:
        print(f"XPU event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(f"{model}")

    if args.save_model:
        # use a barrier to make sure training is done on all ranks
        dist.barrier()
        states = model.state_dict()
        if rank == 0:
            torch.save(states, "mnist_cnn.pt")

    cleanup()
```

7. Finally, parse the arguments and set the main function:

```python
"""
Replace CUDA runtime API with XPU runtime API.
"""
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    WORLD_SIZE = torch.xpu.device_count()
    mp.spawn(fsdp_main,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)
```

8. Put the above code snippets to a python script `FSDP_mnist_xpu.py`, and run:

```bash
python FSDP_mnist_xpu.py
```

