import torch
from torch import Tensor, nn
from torch.autograd import Function
from typing import List, Optional, NamedTuple
from itertools import accumulate
import enum

class PoolingMode(enum.IntEnum):
    SUM = 0
    MEAN = 1

class SGDArgs(NamedTuple):
    bf16_trail: List[Optional[torch.Tensor]]
    weight_decay: float
    lr: float

class EmbeddingSpec(NamedTuple):
    num_of_features: int
    feature_size: int
    pooling_modes: str
    dtype: torch.dtype
    weight: Optional[torch.Tensor]

def merged_embeddingbag_sgd(
    indices,
    offsets,
    indices_with_row_offsets,
    row_offsets,
    pooling_modes,
    sgd_args,
    *weights
):
    if torch.is_grad_enabled():
        return MergedEmbeddingBagSGDFunc.apply(
            indices, offsets, indices_with_row_offsets, row_offsets, pooling_modes, sgd_args, *weights
        )
    return torch.ops.torch_ipex.merged_embeddingbag_forward(indices, offsets, weights, pooling_modes)

class MergedEmbeddingBagSGDFunc(Function):
    @staticmethod
    def unpack(*args):
        return args

    @staticmethod
    def forward(ctx, indices, offsets, indices_with_row_offsets, row_offsets, pooling_modes, sgd_args, *weights):
        output = torch.ops.torch_ipex.merged_embeddingbag_forward(
            indices, offsets, weights, pooling_modes
        )
        ctx.indices = indices
        ctx.offsets = offsets
        ctx.weights = weights
        ctx.indices_with_row_offsets = indices_with_row_offsets
        ctx.row_offsets = row_offsets
        ctx.pooling_modes = pooling_modes
        ctx.sgd_args = sgd_args
        return MergedEmbeddingBagSGDFunc.unpack(*output)

    @staticmethod
    def backward(ctx, *grad_out):
        indices = ctx.indices
        offsets = ctx.offsets
        weights = ctx.weights
        indices_with_row_offsets = ctx.indices_with_row_offsets
        row_offsets = ctx.row_offsets
        pooling_modes = ctx.pooling_modes
        sgd_args = ctx.sgd_args
        bf16_trail = sgd_args.bf16_trail
        weight_decay = sgd_args.weight_decay
        lr = sgd_args.lr
        torch.ops.torch_ipex.merged_embeddingbag_backward_sgd(
            grad_out, indices, offsets, weights, indices_with_row_offsets,
            row_offsets, pooling_modes,
            bf16_trail, weight_decay, lr)
        n_tables = len(weights)
        output = [None for i in range(n_tables + 6)]
        return MergedEmbeddingBagSGDFunc.unpack(*output)

class MergedEmbeddingBag(nn.Module):
    r"""
    Merge multiple Pytorch EmbeddingBag (https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/sparse.py#L221) 
    as one torch.nn.Module. 
    Native usage for multiple EmbeddingBag is:
        >>> EmbLists = torch.nn.Modulist(emb1, emb2, emb3, ..., emb_m)
        >>> inputs = [in1, in2, in3, ..., in_m]
        >>> outputs = []
        >>> for i in range(len(EmbLists)):
        >>>     outputs.append(Emb[in_i])
    Our optimized path will be:
        >>> merged_emb = MergedEmbeddingBagWithSGD(args)
        >>> outputs = MergedEmbeddingBagWithSGD(input)
    We will have benefits for our optimized path:
      1). We will save Pytorch OP dispatch overhead, if the EmbeddingBag operations are not
      heavy, this dispatch overhead will have big impact.
    We introduce "linearize_indices_and_offsets" step to merged indices/offsets together. But consider EmbeddingBags
    are usually the first layer of a model. So "linearize_indices_and_offsets" can be considered as "data prepocess" and
    can be done offline.
    This Module can not be used alone, we suggest to use MergedEmbeddingBagWith[Optimizer] instead.
    Now we can only choose MergedEmbeddingBagWithSGD and we plan to add more optimizer support
    in the future.
    For the introduction of MergedEmbeddingBagWith[Optimizer], please find the comments at
    MergedEmbeddingBagWithSGD.
    """
    embedding_specs: List[EmbeddingSpec]

    def __init__(
        self,
        embedding_specs: List[EmbeddingSpec],
    ):
        super(MergedEmbeddingBag, self).__init__()
        self.n_tables = len(embedding_specs)
        self.weights = []
        row_offsets = []
        feature_sizes = []
        self.pooling_modes = []
        self.dtypes = []
        dtype = None
        self.weights = torch.nn.ParameterList([nn.Parameter(torch.Tensor()) for i in range(len(embedding_specs))])
        for i, emb in enumerate(embedding_specs):
            num_of_features, feature_size, mode, dtype, weight = emb
            row_offsets.append(num_of_features)
            if mode == 'sum':
                self.pooling_modes.append(PoolingMode.SUM)
            elif mode == 'mean':
                self.pooling_modes.append(PoolingMode.MEAN)
            else:
                assert False, r"MergedEmbeddingBag only support EmbeddingBag with model sum or mean"
            if weight is None:
                weight = torch.empty((num_of_features, feature_size), dtype=dtype)
            self.weights[i] = nn.Parameter(weight)
        self.register_buffer(
            "row_offsets",
            torch.tensor([0] + list(accumulate(row_offsets)), dtype=torch.int64),
        )

    def extra_repr(self) -> str:
        s = 'number of tables={}\n'.format(self.n_tables)
        for i in range(self.n_tables):
            s += "table{}: {}, {}, {}, {}".format(
                i, self.weights[i].shape[0], self.weights[i].shape[1], self.pooling_modes[i], self.weights[i].dtype)
            if i != self.n_tables - 1:
                s += '\n'
        return s

    def linearize_indices_and_offsets(
        self,
        indices: List[Tensor],
        offsets: List[Optional[Tensor]],
        include_last_offsets: List[bool]
    ):
        r"""
        To make backward/update more balance, we only have 1 logical table in MergedEmbedingBag and
        use unified indices for access the whole logical table.
        We need to re-mark the indice from different tables to distinguish them.
        For example, we have  2 tables with shape [200, 128] and [100, 128].
        The indice 50 for table1 is still 50 and the indice 50 for table2 should be set to 50 + 200 = 250.
        We assume the original indice and offset will follow the usage for Pytorch EmbeddingBag:
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/sparse.py#L355-L382
        """
        # TODO: support per_sample_weights in forward
        def get_batch_size(indice, offset, include_last_offset):
            if indice.dim() == 2:
                assert offset is None, "offset should be None if indice is 2-D tensor, https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/sparse.py#L355-L382"
                batch_size = indice.shape[0]
            else:
                batch_size = offset.numel()
                if include_last_offset:
                    batch_size -= 1
            return batch_size

        assert self.n_tables == len(indices), "expected {} but got {} indices".format(self.n_tables, len(indices))
        assert self.n_tables == len(offsets), "expected {} but got {} offsets".format(self.n_tables, len(offsets))
        assert self.n_tables == len(include_last_offsets), "expected {} but got {} include_last_offsets".format(
            self.n_tables, len(include_last_offsets))

        batch_size = get_batch_size(indices[0], offsets[0], include_last_offsets[0])
        assert all(
            batch_size == get_batch_size(idx, offset, include_last) for idx, offset, include_last in zip(indices, offsets, include_last_offsets)
        ), r"MergedEmbeddingBag only support input with same batch size"
        n_indices = sum([t.numel() for t in indices])
        n_offsets = batch_size * self.n_tables + 1  # include last offset
        merged_indices = torch.empty(n_indices, dtype=torch.int64)
        merged_indices_with_row_offsets = torch.empty(n_indices, dtype=torch.int64)  # used for sort together
        merged_offsets = torch.empty(n_offsets, dtype=torch.int64)
        idx_start = 0
        offset_start = 0
        for i in range(self.n_tables):
            n_indice = indices[i].numel()
            merged_indices[idx_start: idx_start + n_indice].copy_(indices[i].view(-1))
            merged_indices_with_row_offsets[idx_start: idx_start + n_indice].copy_(indices[i].view(-1) + self.row_offsets[i])
            if indices[i].dim() == 2:
                bag_size = indices[i].shape[1]
                offset = torch.arange(0, indices[i].numel(), bag_size)
            else:
                offset = offsets[i][:-1] if include_last_offsets[i] else offsets[i]
            assert offset.numel() == batch_size
            merged_offsets[offset_start : offset_start + batch_size].copy_(offset + idx_start)
            idx_start += n_indice
            offset_start += batch_size
        assert idx_start == n_indices
        assert offset_start == n_offsets - 1
        merged_offsets[-1] = n_indices
        return (merged_indices, merged_offsets, merged_indices_with_row_offsets)

    def forward(self, input, need_linearize_indices_and_offsets=torch.BoolTensor([True])):
        assert False, "Please use MergedEmbeddingBagWith[Optimizer]. We only support SGD now, so please create module MergedEmbeddingBagWithSGD instead"


class MergedEmbeddingBagWithSGD(MergedEmbeddingBag):
    r"""
    To support training for MergedEmbeddingBag with good performance, we fused optimizer step
    with backward function.
    Native usage for multiple EmbeddingBag is:
        >>> EmbLists = torch.nn.Modulist(emb1, emb2, emb3, ..., emb_m)
        >>> sgd = torch.optim.SGD(EmbLists.parameters(), lr=lr, weight_decay=weight_decay)
        >>> inputs = [in1, in2, in3, ..., in_m]
        >>> outputs = []
        >>> for i in range(len(EmbLists)):
        >>>     outputs.append(Emb[in_i])
        >>> sgd.zero_grad()
        >>> for i in range(len(outputs)):
        >>>     out.backward(grads[i]) 
        >>> sgd.step()
    Our optimized path will be:
        >>> # create MergedEmbeddingBagWithSGD module with optimizer args (lr and weight decay)
        >>> merged_emb = MergedEmbeddingBagWithSGD(args)
        >>> merged_input = merged_emb.linearize_indices_and_offsets(inputs)
        >>> outputs = MergedEmbeddingBagWithSGD(merged_input)
        >>> outputs.backward(grads)
    We will get further benefits in training:
      1). We will futher save Pytorch OP dispatch overhead in backward and weight update process.
      2). We will make thread loading more balance during backward/weight update. In real
      world scenario, Embedingbag are often used to represent categorical features and the 
      categorical features will often fit power law distribution. For example, if we use one
      Embeddingtable to represent the age range the users of a video game website. We might
      find most of users are from 10-19 or 20-29. So we may need update the row which represent
      10-19 or 20-29 frequently. Since update these rows need to write at the same memory address,
      we need to write it by 1 thread (or we will have write conflict or have overhead to solve the conflict).
      By merge multiple table together, we will have more friendly distribution to distribute
      backward/update tasks.
      3). We will fuse update with backward together. We can immediately update the weight after
      we get grad from backward thus the memory pattern will be more friendly. We will have 
      more chance to access data from cache. 
    """
    embedding_specs: List[EmbeddingSpec]

    def __init__(
        self,
        embedding_specs: List[EmbeddingSpec],
        lr: float = 0.01,
        weight_decay: float = 0
    ):
        super(MergedEmbeddingBagWithSGD, self).__init__(embedding_specs)
        self.sgd_args = self.init_sgd_args(lr, weight_decay)
        for i in range(self.n_tables):
            weight = self.weights[i]
            if weight.dtype == torch.bfloat16:
                self.sgd_args.bf16_trail.append(torch.zeros_like(weight, dtype=torch.bfloat16))
            else:
                self.sgd_args.bf16_trail.append(torch.empty(0, dtype=torch.bfloat16))

    def init_sgd_args(self, lr, weight_decay, bf16_trail=[]):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        return SGDArgs(
            weight_decay=weight_decay,
            lr=lr,
            bf16_trail=bf16_trail
        )

    def to_bfloat16_train(self):
        r"""
        Cast weight to bf16 and it's trail part for training
        """
        trails = []
        for i in range(len(self.weights)):
            if self.weights[i].dtype == torch.float:
                bf16_w, trail = torch.ops.torch_ipex.split_float_bfloat16(self.weights[i])
            elif self.weights[i].dtype == torch.bfloat16:
                bf16_w = self.weights[i]
                trail = torch.zeros_like(bf16_w, dtype=torch.bfloat16)
            elif self.weights[i].dtype == torch.double:
                bf16_w, trail = torch.ops.torch_ipex.split_float_bfloat16(self.weights[i].float())
            else:
                assert False, r"MergedEmbeddingBag only support dtypes with bfloat, float and double"
            trails.append(trail)
            self.weights[i] = torch.nn.Parameter(bf16_w)
        self.sgd_args = self.sgd_args._replace(bf16_trail=trails)

    def forward(self, input, need_linearize_indices_and_offsets=torch.BoolTensor([True])):
        r"""
        Args:
            input (Tuple[Tensor]): a tuple of (indices, offsets, include_last_offsets(if not merged)/indices_with_row_offsets(if merged))
            need_linearize_indices_and_offsets: indicate whether input need to be linearized
        Returns:
            List[Tensor] output shape of `(batch_size, feature_size)` which length = num of tables.
        """
        if need_linearize_indices_and_offsets.item():
            indices, offsets, include_last_offsets = input
            indices, offsets, indices_with_row_offsets = self.linearize_indices_and_offsets(indices, offsets, include_last_offsets)
        else:
            indices, offsets, indices_with_row_offsets = input
        return merged_embeddingbag_sgd(
            indices, offsets, indices_with_row_offsets, self.row_offsets,
            self.pooling_modes, self.sgd_args, *self.weights
        )

    @classmethod
    def from_embeddingbag_list(
        cls,
        tables: List[torch.nn.EmbeddingBag],
        lr: float = 0.01,
        weight_decay: float = 0
    ):
        embedding_specs = []
        for emb in tables:
            emb_shape = emb.weight.shape
            embedding_specs.append(
                EmbeddingSpec(
                    num_of_features=emb_shape[0],
                    feature_size=emb_shape[1],
                    pooling_modes=emb.mode,
                    dtype=emb.weight.dtype,
                    weight=emb.weight.detach()
                ))
        return cls(embedding_specs, lr, weight_decay)
