import torch
import unittest
from torch.testing._internal.common_utils import TestCase
from bench.custom_op_bench.merged_embeddingbag import (
    EmbeddingBagList,
    MergedEmbAdaGrad,
)
import intel_extension_for_pytorch as ipex
import copy
import os

try:
    import oneccl_bindings_for_pytorch  # noqa: F401

    HAS_TORCHCCL = True
except (ImportError, RuntimeError):
    HAS_TORCHCCL = False
skipIfNoTORCHCCL = unittest.skipIf(not HAS_TORCHCCL, "torch-ccl is no installed")


class DistMergedEmbeddingTester(TestCase):
    multi_hot = [
        3,
        2,
        1,
        2,
        6,
        1,
        1,
        1,
        1,
        7,
        3,
        8,
        1,
        6,
        9,
        5,
        1,
        1,
        1,
        12,
        100,
        27,
        10,
        3,
        1,
        1,
    ]

    @unittest.skipIf(True, "TODO:Haozhe to re-enable")
    def test_training(self):
        import torch.distributed as dist

        def env2int(env_list, default=-1):
            for e in env_list:
                val = int(os.environ.get(e, -1))
                if val >= 0:
                    return val
            return default

        rank = env2int(
            ["PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK", "RANK"], 0
        )
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        NUM_TABLE = 26
        B = 1024  # B % world_size == 0
        # This is a workwaround, we use "W_SIZE" to config "WORLD_SIZE"
        # because while using IPEX launcher, the cannot find the "WORLD_SIZE"
        # even we have set them
        world_size = env2int(os.environ["W_SIZE"])
        os.environ["WORLD_SIZE"] = os.environ["W_SIZE"]
        dist.init_process_group("ccl", world_size=world_size, rank=rank)
        my_rank = dist.get_rank()
        my_size = dist.get_world_size()
        for index_type in [torch.int64, torch.int32]:
            indices = [
                torch.randint(1000, (B * self.multi_hot[i],)).to(index_type)
                for i in range(NUM_TABLE)
            ]
            for include_last_offset in [True, False]:
                n_offset = B + 1 if include_last_offset else B
                offsets = [
                    torch.arange(0, n_offset * self.multi_hot[i], self.multi_hot[i]).to(
                        index_type
                    )
                    for i in range(NUM_TABLE)
                ]
                for dtype in [
                    torch.bfloat16,
                    torch.float32,
                    torch.float64,
                ]:
                    for NUM_DIM in [64, 65, 128, 256]:
                        emb_list = EmbeddingBagList(
                            NUM_TABLE,
                            NUM_DIM,
                            dtype,
                            include_last_offset=include_last_offset,
                            mode="sum",
                        )

                        # ref result is got by run global BS on each rank with MergedEmbAdaGrad
                        ref_m = MergedEmbAdaGrad(copy.deepcopy(emb_list), lr=1)
                        ref_out = ref_m(indices, offsets)
                        ref_out = torch.cat([o.unsqueeze(1) for o in ref_out], dim=1)

                        distributed_emb = ipex.nn.modules.DistMergeEmbeddingBagWithAdaGrad.from_embeddingbag_list(
                            copy.deepcopy(emb_list.list), lr=1
                        )
                        out = distributed_emb(indices, offsets)
                        output_list = [torch.empty_like(out) for _ in range(my_size)]
                        # gather local BS for each rank and compare it with ref_out
                        dist.all_gather(output_list, out)
                        dist.barrier()
                        dist_out = torch.cat(output_list, dim=0)
                        """
                        1.Accumulate order within 1 bag may different
                        2.More fp32/bf16/fp32 cast within 1 bag since we transfer partitial acc result at bf16 level
                        [0] Mismatched elements: 4994 / 3407872 (0.1%)
                        [0] Greatest absolute difference: 0.125 at index (2, 20, 54) (up to 0.01 allowed)
                        [0] Greatest relative difference: inf at index (8, 21, 106) (up to 0.01 allowed)
                        """
                        self.assertEqual(ref_out, dist_out, atol=0.2, rtol=0.2)
                        un_updated_w = distributed_emb.weights[0].clone()
                        un_updated_hessain = distributed_emb.adagrad_args.hessian[
                            0
                        ].clone()
                        out.backward(torch.ones_like(out))
                        # check weight/hessain is updated during backward
                        self.assertNotEqual(un_updated_w, distributed_emb.weights[0])
                        self.assertNotEqual(
                            un_updated_hessain, distributed_emb.adagrad_args.hessian[0]
                        )
                        ref_out.backward(torch.ones_like(ref_out))
                        # slice out ref_weight/ref_hessian for on different ranks and compare them with
                        # the weight/hessian in DistMergeEmbeddingBagWithAdaGrad after updating
                        ref_weight = torch.cat(
                            [w.data for w in ref_m.merged_emb.weights]
                        )[my_rank::my_size, :]
                        ref_hessian = torch.cat(ref_m.merged_emb.adagrad_args.hessian)[
                            my_rank::my_size, :
                        ]
                        self.assertEqual(distributed_emb.weights[0], ref_weight)
                        self.assertEqual(
                            distributed_emb.adagrad_args.hessian[0], ref_hessian
                        )
        dist.destroy_process_group()


if __name__ == "__main__":
    test = unittest.main()
