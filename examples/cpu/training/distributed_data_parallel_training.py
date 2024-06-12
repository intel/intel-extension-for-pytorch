import os
import torch
import torch.distributed as dist
import torchvision
import oneccl_bindings_for_pytorch as torch_ccl
import intel_extension_for_pytorch as ipex

LR = 0.001
DOWNLOAD = True
DATA = 'datasets/cifar10/'

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
os.environ['RANK'] = os.environ.get('PMI_RANK', 0)
os.environ['WORLD_SIZE'] = os.environ.get('PMI_SIZE', 1)
dist.init_process_group(
backend='ccl',
init_method='env://'
)
rank = os.environ['RANK']

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = torchvision.datasets.CIFAR10(
        root=DATA,
        train=True,
        transform=transform,
        download=DOWNLOAD,
)
dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=128,
        sampler=dist_sampler
)

model = torchvision.models.resnet50()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = LR, momentum=0.9)
model.train()
model, optimizer = ipex.optimize(model, optimizer=optimizer)

model = torch.nn.parallel.DistributedDataParallel(model)

for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    print('batch_id: {}'.format(batch_idx))

if rank == 0:
    torch.save({
         'model_state_dict': model.state_dict(),
         'optimizer_state_dict': optimizer.state_dict(),
         }, 'checkpoint.pth')

dist.destroy_process_group()
print("Execution finished")