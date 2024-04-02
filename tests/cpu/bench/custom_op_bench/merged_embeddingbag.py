import torch
import intel_extension_for_pytorch as ipex
import time
import copy

r"""
vector-size = 128
batch-size = 7168
r"""

a = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float)
b = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float)
NUM_TABLE = 26


def cache_flush():
    # We assume the cache size is <= 512MB here.
    # a = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float)
    # b = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float)
    # a, b are initialized out of this function to avoid allocate memory every time
    global a, b
    a += b


class EmbeddingBagList(torch.nn.Module):
    def __init__(
        self,
        ntables,
        num_dim,
        dtype,
        include_last_offset=False,
        sparse=False,
        mode="sum",
    ):
        super(EmbeddingBagList, self).__init__()
        self.list = torch.nn.ModuleList()
        for _ in range(ntables):
            self.list.append(
                torch.nn.EmbeddingBag(
                    1000,
                    num_dim,
                    dtype=dtype,
                    mode=mode,
                    include_last_offset=include_last_offset,
                    sparse=sparse,
                )
            )

    def forward(self, indices, offsets):
        ly = []
        for i, emb in enumerate(self.list):
            ly.append(emb(indices[i], offsets[i]))
        return ly


class EmbeddingBagListCatDense(torch.nn.Module):
    def __init__(self, emb_list):
        super(EmbeddingBagListCatDense, self).__init__()
        self.emb_list = emb_list

    def forward(self, indices, offsets, dense):
        return torch.cat([dense] + self.emb_list(indices, offsets), dim=1)


class MergedEmbCatDense(torch.nn.Module):
    def __init__(self, emblist):
        super(MergedEmbCatDense, self).__init__()
        self.merged_emb = (
            ipex.nn.modules.MergedEmbeddingBagWithCat.from_embeddingbag_list(
                emblist.list
            )
        )

    def forward(self, indices, offsets, dense):
        return self.merged_emb(indices, offsets, dense)


class MergedEmb(torch.nn.Module):
    def __init__(self, emblist):
        super(MergedEmb, self).__init__()
        self.merged_emb = ipex.nn.modules.MergedEmbeddingBag.from_embeddingbag_list(
            emblist.list
        )

    def forward(self, indices, offsets):
        return self.merged_emb(indices, offsets)


class MergedEmbSGD(torch.nn.Module):
    def __init__(self, emblist, lr=0.01, weight_decay=0):
        super(MergedEmbSGD, self).__init__()
        self.merged_emb = (
            ipex.nn.modules.MergedEmbeddingBagWithSGD.from_embeddingbag_list(
                emblist.list, lr=lr, weight_decay=weight_decay
            )
        )

    def forward(self, indices, offsets):
        return self.merged_emb(indices, offsets)


class MergedEmbAdaGrad(torch.nn.Module):
    def __init__(self, emblist, lr=0.01, eps=1e-8):
        super(MergedEmbAdaGrad, self).__init__()
        self.merged_emb = (
            ipex.nn.modules.MergedEmbeddingBagWithAdaGrad.from_embeddingbag_list(
                emblist.list, lr=lr, eps=eps
            )
        )

    def forward(self, indices, offsets):
        return self.merged_emb(indices, offsets)


def run_bench(bench_name, module, input_data, optimizer=None, training=False):
    iters = 100 if training else 1000
    for i in range(iters):
        cache_flush()
        outs = module(*input_data)
        if training:
            loss = 0
            for out in outs:
                loss += out.sum()
            loss.backward()
            if isinstance(module, EmbeddingBagList):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    start = time.time()
    exclude_time = 0
    for i in range(iters):
        flush_start = time.time()
        cache_flush()
        exclude_time += time.time() - flush_start
        outs = module(*input_data)
        if training:
            loss = 0
            sum_start = time.time()
            for out in outs:
                loss += out.sum()
            exclude_time += time.time() - sum_start
            loss.backward()
            if isinstance(module, EmbeddingBagList):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    end = time.time()
    avg_elapsed = end - start - exclude_time
    print("Took {} ms on average to run {} benchmark".format(avg_elapsed, bench_name))


def inference_bench(dataset, emb_list, merged_emb):
    emblist_input, merged_emb_input = dataset
    run_bench("EmbedddingBag List Inference", emb_list, emblist_input)
    run_bench("Merged EmbedddingBag Inference", merged_emb, merged_emb_input)


def training_bench(dataset, emb_list, merged_emb, optimizer):
    emblist_input, merged_emb_input = dataset
    run_bench(
        "EmbedddingBag List Training",
        emb_list,
        emblist_input,
        optimizer=optimizer,
        training=True,
    )
    run_bench(
        "Merged EmbedddingBag Training", merged_emb, merged_emb_input, training=True
    )


def merged_emb_cat_bench(args, input):
    assert args.inference
    indices, offsets = input
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        emblist = EmbeddingBagList(NUM_TABLE, args.vector_size, dtype)
        ref_m = EmbeddingBagListCatDense(emblist)
        m = MergedEmbCatDense(emblist)
        dense = torch.randn(args.batch_size, args.vector_size, dtype=dtype)
        with torch.no_grad():
            run_bench(
                f"MergedEmbeddingBagWithCat: value_dtype:{dtype}",
                m,
                (indices, offsets, dense),
            )
            run_bench(
                f"EmbeddingBagList+Cat: value_dtype:{dtype}",
                ref_m,
                (indices, offsets, dense),
            )


def merged_emb_with_sgd(args, input):
    for dtype in [torch.float32, torch.bfloat16]:
        if dtype == torch.bfloat16:
            # for bf16, only support split sgd
            emblist = EmbeddingBagList(NUM_TABLE, args.vector_size, torch.float32)
        else:
            emblist = EmbeddingBagList(NUM_TABLE, args.vector_size, dtype)
        m = MergedEmbAdaGrad(copy.deepcopy(emblist), lr=0.1)
        ref_m = copy.deepcopy(emblist)
        opt = torch.optim.Adagrad(ref_m.parameters(), lr=0.1)
        if dtype == torch.bfloat16:
            m.merged_emb.to_bfloat16_train()
            ref_m, opt = ipex.optimize(ref_m, dtype=torch.bfloat16, optimizer=opt)
        if args.inference:
            with torch.no_grad():
                run_bench(
                    f"MergedEmbeddingBagWithAdagrad: value_dtype:{dtype}",
                    m,
                    input,
                )
                run_bench(
                    f"EmbeddingBagList: value_dtype:{dtype}",
                    ref_m,
                    input,
                )
        else:
            run_bench(
                f"MergedEmbeddingBagWithSGD: value_dtype:{dtype}",
                m,
                input,
                training=True,
            )
            run_bench(
                f"EmbeddingBagList: value_dtype:{dtype}",
                ref_m,
                input,
                optimizer=opt,
                training=True,
            )


def merged_emb_with_adagrad(args, input):
    for dtype in [torch.float32, torch.bfloat16]:
        if dtype == torch.bfloat16:
            # for bf16, only support split sgd
            emblist = EmbeddingBagList(NUM_TABLE, args.vector_size, torch.float32)
        else:
            emblist = EmbeddingBagList(NUM_TABLE, args.vector_size, dtype)
        m = MergedEmbAdaGrad(copy.deepcopy(emblist), lr=0.1)
        ref_m = copy.deepcopy(emblist)
        opt = torch.optim.Adagrad(ref_m.parameters(), lr=0.1)
        if dtype == torch.bfloat16:
            m.merged_emb.to_bfloat16_train()
            ref_m, opt = ipex.optimize(ref_m, dtype=torch.bfloat16, optimizer=opt)
        if args.inference:
            with torch.no_grad():
                run_bench(
                    f"MergedEmbeddingBagWithAdaGrad: value_dtype:{dtype}",
                    m,
                    input,
                )
                run_bench(
                    f"EmbeddingBagList: value_dtype:{dtype}",
                    ref_m,
                    input,
                )
        else:
            run_bench(
                f"MergedEmbeddingBagWithAdaGrad: value_dtype:{dtype}",
                m,
                input,
                training=True,
            )
            run_bench(
                f"EmbeddingBagList: value_dtype:{dtype}",
                ref_m,
                input,
                optimizer=opt,
                training=True,
            )


def get_data(batch_size):
    indices = []
    offsets = []
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

    def unbalance_indices(i):
        a = torch.normal(0, 0.1, (batch_size * multi_hot[i],))
        a = a - a.min()
        a = a * 100 / a.max()
        a = a.floor().int()
        return a

    indices = [unbalance_indices(i) for i in range(26)]
    offsets = [
        torch.arange(0, batch_size * multi_hot[i], multi_hot[i]).int()
        for i in range(26)
    ]

    return (indices, offsets)


def run():
    import argparse

    parser = argparse.ArgumentParser(description="benchmark for ipex embeddingbag")
    parser.add_argument("--inference", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=7168)
    parser.add_argument("--vector-size", type=int, default=128)
    parser.add_argument("--with-cat", action="store_true", default=False)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adagrad"],
    )
    args = parser.parse_args()
    input_data = get_data(args.batch_size)
    if args.with_cat:
        assert args.inference
        merged_emb_cat_bench(args, input_data)
        exit()

    if args.optimizer == "sgd":
        merged_emb_with_sgd(args, input_data)
    else:
        merged_emb_with_adagrad(args, input_data)


if __name__ == "__main__":
    run()
