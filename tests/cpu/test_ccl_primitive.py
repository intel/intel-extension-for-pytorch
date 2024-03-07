import unittest
import os
import torch
import intel_extension_for_pytorch as ipex


@unittest.skip("oneccl can't works in docker")
class CCLTester(unittest.TestCase):
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
