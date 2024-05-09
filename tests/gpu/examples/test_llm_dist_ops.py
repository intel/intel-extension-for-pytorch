import torch
import intel_extension_for_pytorch as ipex
from torch.testing._internal.common_utils import TestCase
import torch.distributed as dist
import os


class TestDistMethod(TestCase):

    def test_all_reduce_xpu(self):

        tokens = 16
        rounds = 70 * 2 * tokens

        count = 14336

        total = 1024 * 1024 * 72
        repeat = 4
        datatype = torch.float16
        os.environ["RANK"] = str(os.environ.get("PMI_RANK", 0))
        os.environ["WORLD_SIZE"] = str(os.environ.get("PMI_SIZE", 1))
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

        if not dist.is_initialized():
            ipex.distributed.init_process_group("ccl")
        rank = dist.get_rank()
        size = dist.get_world_size()

        device = "xpu:{}".format(rank)
        # allreduce data
        data = (torch.ones(count, dtype=datatype) * 0.1).to(device)
        a = (torch.zeros((int(total / count), count), dtype=datatype)).to(device)

        for i in range(rounds):
            a[0] += data * 0.1
            for j in range(repeat):
                a += 0.01
            ipex.distributed.all_reduce(data)
            data /= size
            sync = data.cpu()
        data = data.cpu()

        tmp_a = torch.zeros(1, dtype=datatype)
        tmp_data = torch.ones(1, dtype=datatype) * 0.1

        for i in range(rounds):
            tmp_a += tmp_data * 0.1
            for j in range(repeat):
                tmp_a += 0.01
            tmp_data *= size
            tmp_data /= size

        a = a.cpu()

        for i in range(count):
            self.assertEqual(tmp_a[0], a[0][i])
        print("all_reduce passed")

    def test_all_gather_xpu(self):
        tokens = 16
        rounds = 70 * 2 * tokens

        count = 14336

        total = 1024 * 1024 * 72
        repeat = 4
        datatype = torch.float16
        os.environ["RANK"] = str(os.environ.get("PMI_RANK", 0))
        os.environ["WORLD_SIZE"] = str(os.environ.get("PMI_SIZE", 1))
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

        if not dist.is_initialized():
            ipex.distributed.init_process_group("ccl")
        rank = dist.get_rank()
        size = dist.get_world_size()

        device = "xpu:{}".format(rank)
        data = torch.arange(count, dtype=datatype, device=device) + 1 + 2 * rank

        data_list = [
            torch.randn(count, dtype=datatype, device=device) for _ in range(size)
        ]

        ipex.distributed.all_gather(data_list, data)
        for i in range(size):
            found = False
            target_data = torch.arange(count, dtype=datatype, device=device) + 1 + 2 * i
            for all_gather_data in data_list:
                found = torch.allclose(all_gather_data, target_data)
                if found:
                    break
            assert found, "Error: Allgather failed, no target tensor in tensor list."
        print("all gather test passed")

    def test_all_gather_into_tensor_xpu(self):
        tokens = 16
        rounds = 70 * 2 * tokens

        count = 14336

        total = 1024 * 1024 * 72
        repeat = 4
        datatype = torch.float16
        os.environ["RANK"] = str(os.environ.get("PMI_RANK", 0))
        os.environ["WORLD_SIZE"] = str(os.environ.get("PMI_SIZE", 1))
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        if not dist.is_initialized():
            ipex.distributed.init_process_group("ccl")
        rank = dist.get_rank()
        size = dist.get_world_size()

        device = "xpu:{}".format(rank)
        data = torch.arange(count, dtype=datatype, device=device) + 1 + 2 * rank
        dst = torch.empty(count * size, dtype=datatype, device=device)
        ipex.distributed.all_gather_into_tensor(dst, data)
        chunk_list = dst.chunk(size)
        found = False
        for i in range(size):
            target_data = torch.arange(count, dtype=datatype, device=device) + 1 + 2 * i
            for single_chunk in chunk_list:
                found = torch.allclose(single_chunk, target_data)
                if found:
                    break
            assert (
                found
            ), "Error: ipex.distributed.all_gather_into_tensor failed, no target tensor in dst tensor."
        print("all_gather_into_tensor test passed")


testcase = TestDistMethod()
testcase.test_all_reduce_xpu()
testcase.test_all_gather_into_tensor_xpu()
testcase.test_all_gather_xpu()
