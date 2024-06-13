import unittest
import os
import torch
import intel_extension_for_pytorch as ipex

has_ccl = ipex.cpu.comm.has_ccl()
world_size = 0 if not has_ccl else ipex.cpu.comm.get_world_size()


def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


ipex_llm_world_size = get_int_from_env(
    ["WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE"], 1
)


class CCLTester(unittest.TestCase):
    def init_env(self):
        _local_rank = get_int_from_env(
            [
                "LOCAL_RANK",
                "MPI_LOCALRANKID",
                "OMPI_COMM_WORLD_LOCAL_RANK",
                "MV2_COMM_WORLD_LOCAL_RANK",
            ],
            -1,
        )
        _rank = get_int_from_env(
            ["RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK"], 0
        )
        _world_size = get_int_from_env(
            ["WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE"], 1
        )
        os.environ["RANK"] = str(_rank)
        os.environ["WORLD_SIZE"] = str(_world_size)
        os.environ["LOCAL_RANK"] = str(_local_rank)
        if not os.environ.get("MASTER_PORT", None):
            os.environ["MASTER_PORT"] = "29500"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        if torch.distributed.is_available() and not torch.distributed.is_initialized():
            ipex.distributed.init_process_group()
        return _local_rank, _rank, _world_size

    @unittest.skipIf(not ipex_llm_world_size > 1, "only test with distributed")
    def test_ipex_llm_all_reduce_add(self):
        _, _rank, _world_size = self.init_env()
        ipex.enable_onednn_fusion(False)  # just to workaround the flake8
        dtypes = [torch.float32, torch.float16, torch.bfloat16]
        tensor_sizes = [4096, 4096 * 32, 8 * 1024 * 5120 * 4 * 2]
        # Less than 8 * 1024 * 5120 * 4 use SHM, otherwise use ccl allreduce
        # The above dispatch rule is transparent to users
        for dtype in dtypes:
            for tensor_size in tensor_sizes:
                input_tensor = torch.tensor([_rank + 1.0]).to(dtype).repeat(tensor_size)
                target_tensor = (
                    torch.tensor([float(_world_size * (_world_size + 1) / 2)])
                    .to(dtype)
                    .repeat(tensor_size)
                )
                ipex.distributed.all_reduce(input_tensor)
                torch.allclose(input_tensor, target_tensor)

    @unittest.skipIf(not ipex_llm_world_size > 1, "only test with distributed")
    def test_ipex_llm_allgather(self):
        _, _rank, _world_size = self.init_env()
        count = 14336
        for dtypes in [torch.float32, torch.float16, torch.bfloat16]:
            data = torch.arange(count, dtype=dtypes) + 1 + 2 * _rank
            data_list = [torch.randn(count, dtype=dtypes) for _ in range(_world_size)]
            ipex.distributed.all_gather(data_list, data)
            for i in range(_world_size):
                found = False
                target_data = (
                    torch.arange(
                        count,
                        dtype=dtypes,
                    )
                    + 1
                    + 2 * i
                )
                for all_gather_data in data_list:
                    if torch.equal(all_gather_data, target_data):
                        found = True
                        break
                assert (
                    found is True
                ), "Error: ipex.distributed.all_gather failed, no target tensor in tensor list."

    @unittest.skipIf(not ipex_llm_world_size > 1, "only test with distributed")
    def test_all_gather_into_tensor(self):
        _, _rank, _world_size = self.init_env()
        count = 14336
        for datatype in [torch.bfloat16, torch.float]:
            data = torch.arange(count, dtype=datatype) + 1 + 2 * _rank
            dst = torch.empty(count * _world_size, dtype=datatype)
            ipex.distributed.all_gather_into_tensor(dst, data)
            chunk_list = dst.chunk(_world_size)
            found = False
            for i in range(_world_size):
                target_data = torch.arange(count, dtype=datatype) + 1 + 2 * i
                for single_chunk in chunk_list:
                    found = torch.allclose(single_chunk, target_data)
                    if found:
                        break
                assert (
                    found
                ), "Error: ipex.distributed.all_gather_into_tensor failed, no target tensor in dst tensor."

    @unittest.skipIf(not (has_ccl and world_size > 1), "oneccl is not built")
    def test_all_reduce_add(self):
        mpi_world_size = int(os.environ.get("PMI_SIZE", -1))
        mpi_rank = int(os.environ.get("PMI_RANK", -1))
        ipex.enable_onednn_fusion(False)  # just to workaround the flake8
        dtypes = [torch.float32, torch.float16, torch.bfloat16]
        tensor_sizes = [4096, 4096 * 32, 8 * 1024 * 5120 * 4 * 2]
        # Less than 8 * 1024 * 5120 * 4 use SHM, otherwise use ccl allreduce
        # The above dispatch rule is transparent to users
        for dtype in dtypes:
            for tensor_size in tensor_sizes:
                input_tensor = (
                    torch.tensor([mpi_rank + 1.0]).to(dtype).repeat(tensor_size)
                )
                target_tensor = (
                    torch.tensor([float(mpi_world_size * (mpi_world_size + 1) / 2)])
                    .to(dtype)
                    .repeat(tensor_size)
                )
                ipex.cpu.comm.allreduce_add(input_tensor)
                torch.allclose(input_tensor, target_tensor)
                ipex.cpu.comm.barrier()

        self.assertEqual(mpi_world_size, ipex.cpu.comm.get_world_size())
        self.assertEqual(mpi_rank, ipex.cpu.comm.get_rank())

    @unittest.skipIf(not (has_ccl and world_size > 1), "oneccl is not built")
    def test_allgather(self):
        mpi_world_size = int(os.environ.get("PMI_SIZE", -1))
        mpi_rank = int(os.environ.get("PMI_RANK", -1))
        dtypes = [torch.float32, torch.float16, torch.bfloat16]
        for dtype in dtypes:
            for n in range(mpi_world_size + 1):
                n = n + 1
                input = (torch.tensor([n * mpi_rank])).to(dtype)
                col_per_rank = []
                for i in range(mpi_world_size + 1):
                    col_per_rank.append(i)
                expected_output = [
                    torch.tensor([i * n]).to(dtype) for i in range(mpi_world_size)
                ]
                expected_output = torch.cat(expected_output, dim=0)

                output = ipex.cpu.comm.allgather(input, col_per_rank, mpi_world_size)
                torch.allclose(expected_output, output)


if __name__ == "__main__":
    test = unittest.main()
