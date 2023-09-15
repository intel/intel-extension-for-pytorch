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
    sparse: bool


def merged_embeddingbag(
    indices, offsets, indices_with_row_offsets, row_offsets, pooling_modes, *weights
):
    if torch.is_grad_enabled():
        return MergedEmbeddingBagFunc.apply(
            indices,
            offsets,
            indices_with_row_offsets,
            row_offsets,
            pooling_modes,
            *weights
        )
    return torch.ops.torch_ipex.merged_embeddingbag_forward(
        indices, offsets, weights, pooling_modes
    )


def merged_embeddingbag_with_cat(
    weights,
    indices,
    offsets,
    dense_feature,
):
    if torch.is_grad_enabled():
        raise NotImplementedError(
            "do not support training for merged_embeddingbag_with_cat not"
        )
    return torch.ops.torch_ipex.merged_embeddingbag_cat_forward(
        weights, indices, offsets, dense_feature
    )


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
            indices,
            offsets,
            indices_with_row_offsets,
            row_offsets,
            pooling_modes,
            sgd_args,
            *weights
        )
    return torch.ops.torch_ipex.merged_embeddingbag_forward(
        indices, offsets, weights, pooling_modes
    )


class MergedEmbeddingBagFunc(Function):
    @staticmethod
    def unpack(*args):
        return args

    @staticmethod
    def forward(
        ctx,
        indices,
        offsets,
        indices_with_row_offsets,
        row_offsets,
        pooling_modes,
        *weights
    ):
        output = torch.ops.torch_ipex.merged_embeddingbag_forward(
            indices, offsets, weights, pooling_modes
        )
        ctx.offsets = offsets
        ctx.weights = weights
        ctx.indices_with_row_offsets = indices_with_row_offsets
        ctx.row_offsets = row_offsets
        ctx.pooling_modes = pooling_modes
        return MergedEmbeddingBagFunc.unpack(*output)

    @staticmethod
    def backward(ctx, *grad_out):
        offsets = ctx.offsets
        weights = ctx.weights
        indices_with_row_offsets = ctx.indices_with_row_offsets
        row_offsets = ctx.row_offsets
        pooling_modes = ctx.pooling_modes
        grad_list = torch.ops.torch_ipex.merged_embeddingbag_backward_cpu(
            grad_out,
            offsets,
            weights,
            indices_with_row_offsets,
            row_offsets,
            pooling_modes,
        )
        n_tables = len(weights)
        output = [None for i in range(5)]
        for grad in grad_list:
            output.append(grad)
        return MergedEmbeddingBagFunc.unpack(*output)


class MergedEmbeddingBagSGDFunc(Function):
    @staticmethod
    def unpack(*args):
        return args

    @staticmethod
    def forward(
        ctx,
        indices,
        offsets,
        indices_with_row_offsets,
        row_offsets,
        pooling_modes,
        sgd_args,
        *weights
    ):
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
            grad_out,
            indices,
            offsets,
            weights,
            indices_with_row_offsets,
            row_offsets,
            pooling_modes,
            bf16_trail,
            weight_decay,
            lr,
        )
        n_tables = len(weights)
        output = [None for i in range(n_tables + 6)]
        return MergedEmbeddingBagSGDFunc.unpack(*output)


class MergedEmbeddingBag(nn.Module):
    r"""
    Merge multiple Pytorch `EmbeddingBag <https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html
    #embeddingbag>`_ objects into a single `torch.nn.Module` object.

    At the current stage:

    `MergedEmbeddingBag` assumes to be constructed from `nn.EmbeddingBag` with `sparse=False`, returns dense gradients.

    `MergedEmbeddingBagWithSGD` does not return gradients, backward step and weights update step are fused.

    Native usage of multiple `EmbeddingBag` objects is:

        >>> EmbLists = torch.nn.Modulist(emb1, emb2, emb3, ..., emb_m)
        >>> inputs = [in1, in2, in3, ..., in_m]
        >>> outputs = []
        >>> for i in range(len(EmbLists)):
        >>>     outputs.append(Emb[in_i])

    The optimized path is:

        >>> EmbLists = torch.nn.Modulist(emb1, emb2, emb3, ..., emb_m)
        >>> merged_emb = MergedEmbeddingBagWithSGD.from_embeddingbag_list(EmbLists)
        >>> outputs = MergedEmbeddingBagWithSGD(inputs)

    Computation benefits from the optimized path:

        1). Pytorch OP dispatching overhead is minimized. If `EmbeddingBag` operations are not heavy, this dispatching
        overhead brings big impact.

        2). Parallelizations over embedding tables are merged into that over a single merged embedding table. This
        could benefit low parallelization efficiency scenarios when data size read out from embedding tables are not
        large enough.

    A `linearize_indices_and_offsets` step is introduced to merge indices/offsets together. Consider that `EmbeddingBag`
    objects are usually the first layer of a model, the `linearize_indices_and_offsets` step can be considered as "data
    preprocess" and can be done offline. See usage of the `linearize_indices_and_offsets` in `MergedEmbeddingBagWithSGD`.

    Now `MergedEmbeddingBagWithSGD` is the only option running with an optimizer. We plan to add more optimizer support
    in the future. Visit `MergedEmbeddingBagWithSGD` for introduction of `MergedEmbeddingBagWith[Optimizer]`.
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
        self.alldense = True
        self.weights = torch.nn.ParameterList(
            [nn.Parameter(torch.Tensor()) for i in range(len(embedding_specs))]
        )
        for i, emb in enumerate(embedding_specs):
            num_of_features, feature_size, mode, dtype, weight, sparse = emb
            row_offsets.append(num_of_features)
            if mode == "sum":
                self.pooling_modes.append(PoolingMode.SUM)
            elif mode == "mean":
                self.pooling_modes.append(PoolingMode.MEAN)
            else:
                AssertionError(
                    False
                ), r"MergedEmbeddingBag only support EmbeddingBag with model sum or mean"
            if weight is None:
                weight = torch.empty((num_of_features, feature_size), dtype=dtype)
            self.weights[i] = nn.Parameter(weight)
            if sparse:
                self.alldense = False

        self.register_buffer(
            "row_offsets",
            torch.tensor([0] + list(accumulate(row_offsets)), dtype=torch.int64),
        )

    @classmethod
    def from_embeddingbag_list(
        cls,
        tables: List[torch.nn.EmbeddingBag],
    ):
        embedding_specs = []
        for emb in tables:
            emb_shape = emb.weight.shape
            assert (
                not emb.sparse
            ), "MergedEmbeddingBag can only be used for dense gradient EmebddingBag. \
                Please use MergedEmbeddingBagWith[Optimizer] for sparse gradient."
            embedding_specs.append(
                EmbeddingSpec(
                    num_of_features=emb_shape[0],
                    feature_size=emb_shape[1],
                    pooling_modes=emb.mode,
                    dtype=emb.weight.dtype,
                    weight=emb.weight.detach(),
                    sparse=emb.sparse,
                )
            )
        return cls(embedding_specs)

    def extra_repr(self) -> str:
        s = "number of tables={}\n".format(self.n_tables)
        for i in range(self.n_tables):
            s += "table{}: {}, {}, {}, {}".format(
                i,
                self.weights[i].shape[0],
                self.weights[i].shape[1],
                self.pooling_modes[i],
                self.weights[i].dtype,
            )
            if i != self.n_tables - 1:
                s += "\n"
        return s

    def linearize_indices_and_offsets(
        self,
        indices: List[Tensor],
        offsets: List[Optional[Tensor]],
        include_last_offsets: List[bool],
    ):
        r"""
        To make backward/update more balance, we only have 1 logical table in MergedEmbedingBag and
        use unified indices for access the whole logical table.
        We need to re-mark the indice from different tables to distinguish them.
        For example, we have 2 tables with shape [200, 128] and [100, 128].
        The indice 50 for table1 is still 50 and the indice 50 for table2 should be set to 50 + 200 = 250.
        We assume the original indice and offset will follow the usage for Pytorch EmbeddingBag:
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/sparse.py#L355-L382
        """

        # TODO: support per_sample_weights in forward
        def get_batch_size(indice, offset, include_last_offset):
            if indice.dim() == 2:
                assert (
                    offset is None
                ), "offset should be None if indice is 2-D tensor, \
                    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/sparse.py#L355-L382"
                batch_size = indice.shape[0]
            else:
                batch_size = offset.numel()
                if include_last_offset:
                    batch_size -= 1
            return batch_size

        assert self.n_tables == len(indices), "expected {} but got {} indices".format(
            self.n_tables, len(indices)
        )
        assert self.n_tables == len(offsets), "expected {} but got {} offsets".format(
            self.n_tables, len(offsets)
        )
        assert self.n_tables == len(
            include_last_offsets
        ), "expected {} but got {} include_last_offsets".format(
            self.n_tables, len(include_last_offsets)
        )

        batch_size = get_batch_size(indices[0], offsets[0], include_last_offsets[0])
        assert all(
            batch_size == get_batch_size(idx, offset, include_last)
            for idx, offset, include_last in zip(indices, offsets, include_last_offsets)
        ), r"MergedEmbeddingBag only support input with same batch size"
        n_indices = sum([t.numel() for t in indices])
        n_offsets = batch_size * self.n_tables + 1  # include last offset
        merged_indices = torch.empty(n_indices, dtype=torch.int64)
        merged_indices_with_row_offsets = torch.empty(
            n_indices, dtype=torch.int64
        )  # used for sort together
        merged_offsets = torch.empty(n_offsets, dtype=torch.int64)
        idx_start = 0
        offset_start = 0
        for i in range(self.n_tables):
            n_indice = indices[i].numel()
            merged_indices[idx_start : idx_start + n_indice].copy_(indices[i].view(-1))
            merged_indices_with_row_offsets[idx_start : idx_start + n_indice].copy_(
                indices[i].view(-1) + self.row_offsets[i]
            )
            if indices[i].dim() == 2:
                bag_size = indices[i].shape[1]
                offset = torch.arange(0, indices[i].numel(), bag_size)
            else:
                offset = offsets[i][:-1] if include_last_offsets[i] else offsets[i]
            assert offset.numel() == batch_size
            merged_offsets[offset_start : offset_start + batch_size].copy_(
                offset + idx_start
            )
            idx_start += n_indice
            offset_start += batch_size
        assert idx_start == n_indices
        assert offset_start == n_offsets - 1
        merged_offsets[-1] = n_indices
        return (merged_indices, merged_offsets, merged_indices_with_row_offsets)

    def forward(
        self, input, need_linearize_indices_and_offsets=torch.BoolTensor([True])
    ):
        r"""
        Args:
            input (Tuple[Tensor]): a tuple of (indices, offsets, \
                include_last_offsets(if not merged)/indices_with_row_offsets(if merged))
            need_linearize_indices_and_offsets: indicate whether input need to be linearized
        Returns:
            List[Tensor] output shape of `(batch_size, feature_size)` which length = num of tables.
        """
        assert (
            self.alldense
        ), "MergedEmbeddingBag only support EmbeddingBag List with all dense gradient, please use \
            MergedEmbeddingBagWith[Optimizer] for sparse gridient EmbeddingBag"
        if need_linearize_indices_and_offsets.item():
            indices, offsets, include_last_offsets = input
            (
                indices,
                offsets,
                indices_with_row_offsets,
            ) = self.linearize_indices_and_offsets(
                indices, offsets, include_last_offsets
            )
        else:
            indices, offsets, indices_with_row_offsets = input
        return merged_embeddingbag(
            indices,
            offsets,
            indices_with_row_offsets,
            self.row_offsets,
            self.pooling_modes,
            *self.weights
        )


class MergedEmbeddingBagWithSGD(MergedEmbeddingBag):
    r"""
    To support training with `MergedEmbeddingBag` for good performance, optimizer step is fused with backward function.

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

    The optimized path is:

        >>> # create MergedEmbeddingBagWithSGD module with optimizer args (lr and weight decay)
        >>> EmbLists = torch.nn.Modulist(emb1, emb2, emb3, ..., emb_m)
        >>> merged_emb = MergedEmbeddingBagWithSGD.from_embeddingbag_list(EmbLists, lr=lr, weight_decay=weight_decay)
        >>> # if you need to train with BF16 dtype, we provide split sgd on it
        >>> # merged_emb.to_bfloat16_train()
        >>> merged_input = merged_emb.linearize_indices_and_offsets(inputs)
        >>> outputs = MergedEmbeddingBagWithSGD(merged_input, need_linearize_indices_and_offsets=torch.BoolTensor([False]))
        >>> outputs.backward(grads)

    Training benefits further from this optimization:

        1). Pytorch OP dispatching overhead in backward and weight update process is saved.

        2). Thread loading becomes more balanced during backward/weight update. In real world scenarios, `Embedingbag`
        are often used to represent categorical features, while the categorical features often fit power law
        distribution. For example, if we use one embedding table to represent the age range of a video game website
        users, we might find most of them are between 10-19 or 20-29. So we may need to update the row which represent
        10-19 or 20-29 frequently. Since updating these rows needs to write at the same memory address, we need to write
        it by 1 thread (otherwise we will have write conflict or overhead to solve the conflict). The potential memory
        write conflict can be simply addressed by merging multiple tables together.

        3). Weights update is fused with backward together. We can immediately update the weight right after we get
        gradients from the backward step and thus the memory access pattern becomes more friendly. Data access will
        happen on cache more than on memory.
    """
    embedding_specs: List[EmbeddingSpec]

    def __init__(
        self,
        embedding_specs: List[EmbeddingSpec],
        lr: float = 0.01,
        weight_decay: float = 0,
    ):
        super(MergedEmbeddingBagWithSGD, self).__init__(embedding_specs)
        self.sgd_args = self.init_sgd_args(lr, weight_decay)
        for i in range(self.n_tables):
            weight = self.weights[i]
            if weight.dtype == torch.bfloat16:
                self.sgd_args.bf16_trail.append(
                    torch.zeros_like(weight, dtype=torch.bfloat16)
                )
            else:
                self.sgd_args.bf16_trail.append(torch.empty(0, dtype=torch.bfloat16))

    def init_sgd_args(self, lr, weight_decay, bf16_trail=None):
        if bf16_trail is None:
            bf16_trail = []
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        return SGDArgs(weight_decay=weight_decay, lr=lr, bf16_trail=bf16_trail)

    def to_bfloat16_train(self):
        r"""
        Cast weight to bf16 and it's trail part for training
        """
        trails = []
        for i in range(len(self.weights)):
            if self.weights[i].dtype == torch.float:
                bf16_w, trail = torch.ops.torch_ipex.split_float_bfloat16(
                    self.weights[i]
                )
            elif self.weights[i].dtype == torch.bfloat16:
                bf16_w = self.weights[i]
                trail = torch.zeros_like(bf16_w, dtype=torch.bfloat16)
            elif self.weights[i].dtype == torch.double:
                bf16_w, trail = torch.ops.torch_ipex.split_float_bfloat16(
                    self.weights[i].float()
                )
            else:
                AssertionError(
                    False
                ), r"MergedEmbeddingBag only support dtypes with bfloat, float and double"
            trails.append(trail)
            self.weights[i] = torch.nn.Parameter(bf16_w)
        self.sgd_args = self.sgd_args._replace(bf16_trail=trails)

    def forward(
        self, input, need_linearize_indices_and_offsets=torch.BoolTensor([True])
    ):
        r"""
        Args:
            input (Tuple[Tensor]): a tuple of (indices, offsets, \
                include_last_offsets(if not merged)/indices_with_row_offsets(if merged))
            need_linearize_indices_and_offsets: indicate whether input need to be linearized
        Returns:
            List[Tensor] output shape of `(batch_size, feature_size)` which length = num of tables.
        """
        if need_linearize_indices_and_offsets.item():
            indices, offsets, include_last_offsets = input
            (
                indices,
                offsets,
                indices_with_row_offsets,
            ) = self.linearize_indices_and_offsets(
                indices, offsets, include_last_offsets
            )
        else:
            indices, offsets, indices_with_row_offsets = input
        return merged_embeddingbag_sgd(
            indices,
            offsets,
            indices_with_row_offsets,
            self.row_offsets,
            self.pooling_modes,
            self.sgd_args,
            *self.weights
        )

    @classmethod
    def from_embeddingbag_list(
        cls,
        tables: List[torch.nn.EmbeddingBag],
        lr: float = 0.01,
        weight_decay: float = 0,
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
                    weight=emb.weight.detach(),
                    sparse=emb.sparse,
                )
            )
        return cls(embedding_specs, lr, weight_decay)


class MergedEmbeddingBagWithCat(MergedEmbeddingBag):
    r"""
    To support `MergedEmbeddingBag` with cat all outputs with an given input.
    It is a common structure in recomendation system to cat dense output with
    sparse (embeddingbag) output together. MergedEmbeddingBagWithCat aims to
    fuse the cat together to have good memory behaviour.
    Native usage for multiple EmbeddingBag cat with dense is:

        >>> EmbLists = torch.nn.Modulist(emb1, emb2, emb3, ..., emb_m)
        >>> inputs = [in1, in2, in3, ..., in_m]
        >>> outputs = []
        >>> for i in range(len(EmbLists)):
        >>>     outputs.append(Emb[in_i])
        >>> cat_out = torch.cat([dense_feature] + outputs, dim=1)


    The optimized path is:

        >>> EmbLists = torch.nn.Modulist(emb1, emb2, emb3, ..., emb_m)
        >>> merged_emb = MergedEmbeddingBagWithCat.from_embeddingbag_list(EmbLists)
        >>> cat_out = MergedEmbeddingBagWithCat(dense_feature, inputs)
    """
    embedding_specs: List[EmbeddingSpec]

    def __init__(
        self,
        embedding_specs: List[EmbeddingSpec],
    ):
        super(MergedEmbeddingBagWithCat, self).__init__(embedding_specs)
        assert all(PoolingMode.SUM == mode for mode in self.pooling_modes)
        assert all(self.weights[0].dtype == w.dtype for w in self.weights)

    def forward(self, indices, offsets, dense_feature):
        r"""
        Args:
            indices (Tensor): a list of indices for all tables
            offsets (Tensor): a list of offsets for all tables
            dense_feature (Tensor): dense feature to be cat
        Returns:
            output shape of `(batch_size, feature_size)` which feature_size = emb_dim * (num of tables + 1).
        """
        return merged_embeddingbag_with_cat(
            self.weights,
            indices,
            offsets,
            dense_feature,
        )
