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

def cache_flush():
    # We assume the cache size is <= 512MB here.
    # a = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float)
    # b = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float)
    # a, b are initialized out of this function to avoid allocate memory every time
    global a, b
    a += b

class EmbeddingBagList(torch.nn.Module):

    def __init__(self, max_rows, vector_size):
        super(EmbeddingBagList, self).__init__()
        self.emb_list = torch.nn.ModuleList()
        for n_f in max_rows:
            self.emb_list.append(torch.nn.EmbeddingBag(n_f, vector_size, mode="sum", sparse=True))

    def forward(self, indices, offsets):
        ly = []
        for k, sparse_index_group_batch in enumerate(indices):
            sparse_offset_group_batch = offsets[k]
            E = self.emb_list[k]
            V = E(
                sparse_index_group_batch,
                sparse_offset_group_batch
            )
            ly.append(V)
        return ly

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
    avg_elapsed = (end - start - exclude_time)
    print("Took {} ms on average to run {} benchmark".format(avg_elapsed, bench_name))


def inference_bench(dataset, emb_list, merged_emb):
    emblist_input, merged_emb_input = dataset
    run_bench("EmbedddingBag List Inference", emb_list, emblist_input)
    run_bench("Merged EmbedddingBag Inference", merged_emb, merged_emb_input)

def training_bench(dataset, emb_list, merged_emb, optimizer):
    emblist_input, merged_emb_input = dataset
    run_bench("EmbedddingBag List Training", emb_list, emblist_input, optimizer=optimizer, training=True)
    run_bench("Merged EmbedddingBag Training", merged_emb, merged_emb_input, training=True)

def get_data(distribution, merged_emb, max_rows, batch_size):
    indices = []
    offsets = []
    include_last = [False for i in range(len(max_rows))]
    for i in range(len(max_rows)):
        idx = torch.empty(batch_size, dtype=torch.int64)
        if batch_size <= max_rows[i]:
            j = int(max_rows[i] / batch_size)
            for k in range(batch_size):
                value = k * j if (distribution == "balance" or k % 2 == 0) else 0
                idx[k] = value
        else:
            for k in range(batch_size):
                value = k % max_rows[i] if (distribution == "balance" or k % 2 == 0) else 0
                idx[k] = value
        indices.append(idx)
        offsets.append(torch.arange(batch_size))

    merged_input = merged_emb.linearize_indices_and_offsets(indices, offsets, include_last)
    return (indices, offsets), (merged_input, torch.BoolTensor([False]))

def run():
    import argparse
    parser = argparse.ArgumentParser(
        description="benchmark for ipex embeddingbag"
    )
    parser.add_argument("--data-distribution", type=str, choices=["balance", "unbalance"])
    parser.add_argument("--inference", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=7168)
    parser.add_argument("--vector-size", type=int, default=128)

    args = parser.parse_args()

    max_rows = [args.batch_size for i in range(26)]
    emb_list = EmbeddingBagList(max_rows, args.vector_size)
    sgd = torch.optim.SGD(emb_list.parameters(), lr=0.01)
    emb_list, sgd = ipex.optimize(model=emb_list, optimizer=sgd, dtype=torch.float)

    merged_emb = ipex.nn.modules.MergedEmbeddingBagWithSGD.from_embeddingbag_list(copy.deepcopy(emb_list.emb_list))

    input_data = get_data(args.data_distribution, merged_emb, max_rows, args.batch_size)
    if args.inference:
        inference_bench(input_data, emb_list, merged_emb)
    else:
        training_bench(input_data, emb_list, merged_emb, sgd)

if __name__ == "__main__":
    run()
