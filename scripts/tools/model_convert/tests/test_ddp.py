import socket
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from mpi4py import MPI
import os
import model_convert


def main():
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ.keys():
        _local_rank = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
    else:
        _local_rank = os.environ["MPI_LOCALRANKID"]
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    proc_name = MPI.Get_processor_name()
    all_procs = MPI.COMM_WORLD.allgather(proc_name)

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(size)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_local_rank)

    _rank = rank
    _size = size

    # It will want the master address too, which we'll broadcast:
    if rank == 0:
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    else:
        master_addr = None

    master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
    print("bcasting", master_addr)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(3456)
    print(
        "Discovered MPI settings of  world_rank={}, local_rank={}, world_size={}, master_addr={}, master_port={}".format(
            os.environ["RANK"],
            os.environ["CUDA_VISIBLE_DEVICES"],
            os.environ["WORLD_SIZE"],
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
        )
    )

    cdb = torch.distributed.init_process_group(backend="nccl", init_method="env://")
    new_rank = int(torch.distributed.get_rank())
    print("new_rank:{}".format(new_rank))
    new_backend = torch.distributed.get_backend()
    print("new_backend:{}".format(new_backend))


if __name__ == "__main__":
    main()
