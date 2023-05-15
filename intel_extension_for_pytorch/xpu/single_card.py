import os
import tempfile
import torch
import torch.distributed as dist
import intel_extension_for_pytorch  # noqa F401
import oneccl_bindings_for_pytorch  # noqa F401


class single_card_dist:
    r"""DistributedDataParallel(DDP) scaling API for XPU devices on one card.

    This API wraps pytorch DDP related module, and provides a simple usage to
    enable DDP training of models based on XPU devices on one card.

    Note: This API only focus on XPU devices on one card currently. Devices on multi-cards will be supported further.

    Args:
        model: model to be parallelized
        train_dataset: dataset for training

    Example usage::
        Assuming that you have a model runs on single tile, you only need to make minor changes
        to enable the DDP training.
        Please follow these steps:
        1. Import this API:
            >>> try:
            >>>    from intel_extension_for_pytorch.xpu.single_card import single_card_dist
            >>> except ImportError:
            >>>    raise ImportError("oneccl_bindings_for_pytorch not available!")

        2. We recommend to use multi_process_spawn launcher in below, as a torch.multiprocessing wrapper.
            >>> single_card_dist.multi_process_spawn(main_worker, (args, )) # put arguments of main_worker into a tuple

        3. Usage of this API:
            >>> dist = single_card_dist(model, train_dataset)
            >>> local_rank, model, train_sampler = dist.rank, dist.model, dist.train_sampler

        4. Set in the model training:
            >>> for epoch in range ...
                train_sampler.set_epoch(epoch)

        5. Adjust the model where calls local_rank, model, train_sampler correspondingly
            e.g.,
            i). device: get the xpu information used in model training
            >>> xpu = "xpu:{}".format(local_rank)
            >>> print("DDP Use XPU: {} for training".format(xpu))
            ii). model: use the model warpped by DDP in the following training
            iii). train_sampler: use the train_sampler to get the train_loader
            >>> train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                           shuffle=(train_sampler is None),
                num_workers=args.workers, pin_memory=True, sampler=train_sampler)


    """

    def __init__(self, model=None, train_dataset=None):
        self.model = model
        self.train_dataset = train_dataset

        # Initialize the process group with ccl backend
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        dist.init_process_group(
            backend="ccl",
            init_method=os.environ["INIT_FILE"],
            rank=rank,
            world_size=world_size,
        )

        self.rank = dist.get_rank()
        self.model = self.get_ddp_model()
        self.train_sampler = self.get_train_sampler()

    @staticmethod
    def multi_process_spawn(fn=None, args=None):
        """
        Use torch.multiprocessing to spawn N (world_size) processes.
        """
        with tempfile.NamedTemporaryFile() as file:
            os.environ["INIT_FILE"] = "file://{}".format(file.name)
        proc = torch.multiprocessing.get_context("spawn").Process

        processes = []
        pid_to_pipe = {}
        world_size = len(torch.xpu.getDeviceIdListForCard())
        for rank in range(world_size):
            parent_conn, child_conn = torch.multiprocessing.Pipe()
            child_env = os.environ.copy()
            child_env["PMI_SIZE"] = str(world_size)
            child_env["PMI_RANK"] = str(rank)

            os.environ["RANK"] = child_env["PMI_RANK"]
            os.environ["WORLD_SIZE"] = child_env["PMI_SIZE"]
            process = proc(target=fn, name="process " + str(rank), args=args)

            process.start()
            print(f"Start process {rank} with pid {process.pid}")
            pid_to_pipe[process.pid] = parent_conn
            processes.append(process)
        for process in processes:
            process.join()
            return process.exitcode

    # device set (local_rank)
    def get_localrank(self):
        """
        Returns the local rank of process.
        note: for 1 card 2 tiles, local_rank is the same as global rank
        """
        return self.rank

    # model
    def get_ddp_model(self):
        """
        Returns class:`~torch.nn.parallel.DistributedDataParallel`.
        Parallelizes the application of the given module.
        """
        if not self.model:
            print("Please input the model!")
            return None
        self.xpu_device = "xpu:{}".format(self.rank)
        torch.xpu.set_device(self.xpu_device)
        self.model.xpu(self.xpu_device)
        # note we set find_unused_parameters to True defaultly, to enable models (e.g. Bert) which have
        # parameters that don't receive gradients as part of this graph are preemptively marked as
        # being ready to be reduced. Note this may bring additional overhead.
        return torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.xpu_device],
            output_device=self.xpu_device,
            find_unused_parameters=True,
        )

    # data sampler
    def get_train_sampler(self):
        """
        Returns class:`~torch.utils.data.DistributedSampler`.
        Will use no sampler if :obj:`test_dataset` is a :obj:`torch.utils.data.IterableDataset`, a sequential sampler.
        """
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            return None
        else:
            return torch.utils.data.distributed.DistributedSampler(self.train_dataset)
