# Horovod with PyTorch

Horovod is a distributed deep learning training framework for TensorFlow, Keras, PyTorch, and Apache MXNet. The goal of Horovod is to make distributed deep learning fast and easy to use. Horovod core principles are based on MPI concepts such as size, rank, local rank, allreduce, allgather, broadcast, and alltoall. To use Horovod with PyTorch, you need to install Horovod with Pytorch first, and make specific change for Horovod in your training script.

## Install Horovod with PyTorch
### Set Compiler and CCL (assume you already install oneAPI basekit)

```bash
source ${HOME}/intel/oneapi/compiler/latest/compiler/env/vars.sh
source ${HOME}/intel/oneapi/ccl/latest/env/vars.sh
```

### Build Horovod with PyTorch

```bash
git clone -b xpu --depth=1 https://github.com/intel-innersource/frameworks.ai.horovod
cd frameworks.ai.horovod
git submodule sync
git submodule update --init --recursive

I_MPI_CXX=dpcpp \
CXX=mpicxx \
LDSHARED="dpcpp -shared -fPIC" \
CC=icx \
HOROVOD_GPU=DPCPP \
HOROVOD_WITHOUT_MXNET=1 \
HOROVOD_WITHOUT_PYTORCH=0 \
HOROVOD_WITHOUT_TENSORFLOW=1 \
HOROVOD_WITHOUT_GLOO=1 \
HOROVOD_GPU_OPERATIONS=CCL \
HOROVOD_CPU_OPERATIONS=CCL \
HOROVOD_WITH_MPI=1 python setup.py install
```

## Horovod with PyTorch Usage
To use Horovod with PyTorch for XPU backend, make the following modifications to your training script:

1. Initialize Horovod.


        import torch
        import intel_extension_for_pytorch 
        import horovod.torch as hvd
        hvd.init()

2. Pin each GPU to a single process.

   With the typical setup of one GPU per process, set this to *local rank*. The first process on
   the server will be allocated the first GPU, the second process will be allocated the second GPU, and so forth.


        devid = hvd.local_rank()
        torch.xpu.set_device(devid)

3. Scale the learning rate by the number of workers.

   Effective batch size in synchronous distributed training is scaled by the number of workers.
   An increase in learning rate compensates for the increased batch size.

4. Wrap the optimizer in ``hvd.DistributedOptimizer``.

   The distributed optimizer delegates gradient computation to the original optimizer, averages gradients using *allreduce* or *allgather*, and then applies those averaged gradients.

5. Broadcast the initial variable states from rank 0 to all other processes:


       hvd.broadcast_parameters(model.state_dict(), root_rank=0)
       hvd.broadcast_optimizer_state(optimizer, root_rank=0)

   This is necessary to ensure consistent initialization of all workers when training is started with random weights or restored from a checkpoint.

6. Modify your code to save checkpoints only on worker 0 to prevent other workers from corrupting them.

   Accomplish this by guarding model checkpointing code with ``hvd.rank() != 0``.


Example:


    import torch
    import intel_extension_for_pytorch
    import horovod.torch as hvd

    # Initialize Horovod
    hvd.init()

    # Pin GPU to be used to process local rank (one GPU per process)
    devid = hvd.local_rank()
    torch.xpu.set_device(devid)
    device = "xpu:{}".format(devid)

    # Define dataset...
    train_dataset = ...

    # Partition dataset among workers using DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)

    # Build model...
    model = ...
    model.to(device)

    optimizer = optim.SGD(model.parameters())

    # Add Horovod Distributed Optimizer
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    # Broadcast parameters from rank 0 to all other processes.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    for epoch in range(100):
       for batch_idx, (data, target) in enumerate(train_loader):
           optimizer.zero_grad()
           output = model(data)
           loss = F.nll_loss(output, target)
           loss.backward()
           optimizer.step()
           if batch_idx % args.log_interval == 0:
               print('Train Epoch: {} [{}/{}]\tLoss: {}'.format(
                   epoch, batch_idx * len(data), len(train_sampler), loss.item()))

