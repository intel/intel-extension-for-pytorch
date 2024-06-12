import torch
import os
import intel_extension_for_pytorch as ipex
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import argparse


class Module(torch.nn.Module):
    def __init__(
        self,
    ):
        super(Module, self).__init__()
        self.linear = torch.nn.Linear(1024, 1024, bias=False)

    def forward(self, x):
        return self.linear(x)


torch.manual_seed(10)
model = Module()
optim = torch.optim.SGD(model.parameters(), lr=1)

opt_model, opt = ipex.optimize(
    model, dtype=torch.bfloat16, optimizer=optim, inplace=False, weights_prepack=False
)


def env2int(env_list, default=-1):
    for e in env_list:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


rank = env2int(["PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK", "RANK"], 0)

os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29510"
dist.init_process_group("gloo", world_size=2, rank=rank)
my_rank = dist.get_rank()
parser = argparse.ArgumentParser()
parser.add_argument("--get-state-dict", action="store_true")
args = parser.parse_args()

opt_model = DDP(opt_model, static_graph=True)
for i in range(10):
    input = torch.randn(1024, 1024).bfloat16()
    output = opt_model(input)
    if i == 5 and my_rank == 0 and args.get_state_dict:
        state_dict = opt_model.state_dict()
    loss = output.sum()
    loss.backward()
    opt.step()
    if i == 9:
        print(f"Resume training successfully, final lose = {loss.item()}")
