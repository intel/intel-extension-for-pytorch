import torch
from torch import nn
from torch.autograd import Function
from typing import List, Optional, NamedTuple
import enum


class PoolingMode(enum.IntEnum):
    SUM = 0
    MEAN = 1


class SGDArgs(NamedTuple):
    bf16_trail: List[Optional[torch.Tensor]]
    weight_decay: float
    lr: float


class AdaGradArgs(NamedTuple):
    hessian: List[torch.Tensor]
    bf16_trail: List[Optional[torch.Tensor]]
    eps: float
    lr: float


class EmbeddingSpec(NamedTuple):
    num_embeddings: int
    embedding_dim: int
    pooling_mode: str
    dtype: torch.dtype
    weight: Optional[torch.Tensor]
    sparse: bool
    include_last_offset: bool


def merged_embeddingbag(weights, indices, offsets, pooling_mode, include_last_offset):
    if torch.is_grad_enabled():
        return MergedEmbeddingBagFunc.apply(
            indices, offsets, pooling_mode, include_last_offset, *weights
        )
    return torch.ops.torch_ipex.merged_embeddingbag_forward(
        weights, indices, offsets, pooling_mode, include_last_offset
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
    weights, indices, offsets, pooling_mode, include_last_offset, sgd_args
):
    if torch.is_grad_enabled():
        return MergedEmbeddingBagSGDFunc.apply(
            indices,
            offsets,
            pooling_mode,
            include_last_offset,
            sgd_args,
            *weights,
        )
    return torch.ops.torch_ipex.merged_embeddingbag_forward(
        weights, indices, offsets, pooling_mode, include_last_offset
    )


def merged_embeddingbag_adagrad(
    weights, indices, offsets, pooling_mode, include_last_offset, adagrad_args
):
    if torch.is_grad_enabled():
        return MergedEmbeddingBagAdaGradFunc.apply(
            indices,
            offsets,
            pooling_mode,
            include_last_offset,
            adagrad_args,
            *weights,
        )
    return torch.ops.torch_ipex.merged_embeddingbag_forward(
        weights, indices, offsets, pooling_mode, include_last_offset
    )


class MergedEmbeddingBagFunc(Function):
    @staticmethod
    def forward(ctx, indices, offsets, pooling_mode, include_last_offset, *weights):
        output = torch.ops.torch_ipex.merged_embeddingbag_forward(
            weights, indices, offsets, pooling_mode, include_last_offset
        )
        ctx.offsets = offsets
        ctx.indices = indices
        ctx.weights = weights
        ctx.pooling_mode = pooling_mode
        ctx.include_last_offset = include_last_offset
        return tuple(output)

    @staticmethod
    def backward(ctx, *grad_out):
        offsets = ctx.offsets
        weights = ctx.weights
        indices = ctx.indices
        pooling_mode = ctx.pooling_mode
        include_last_offset = ctx.include_last_offset
        grad_list = torch.ops.torch_ipex.merged_embeddingbag_backward_cpu(
            grad_out,
            weights,
            indices,
            offsets,
            pooling_mode,
            include_last_offset,
        )
        output = [None] * 4 + grad_list
        return tuple(output)


class MergedEmbeddingBagSGDFunc(Function):
    @staticmethod
    def forward(
        ctx,
        indices,
        offsets,
        pooling_mode,
        include_last_offset,
        sgd_args,
        *weights,
    ):
        output = torch.ops.torch_ipex.merged_embeddingbag_forward(
            weights, indices, offsets, pooling_mode, include_last_offset
        )
        ctx.indices = indices
        ctx.offsets = offsets
        ctx.weights = weights
        ctx.pooling_mode = pooling_mode
        ctx.include_last_offset = include_last_offset
        ctx.sgd_args = sgd_args
        return tuple(output)

    @staticmethod
    def backward(ctx, *grad_out):
        indices = ctx.indices
        offsets = ctx.offsets
        weights = ctx.weights
        pooling_mode = ctx.pooling_mode
        include_last_offset = ctx.include_last_offset
        sgd_args = ctx.sgd_args
        bf16_trail = sgd_args.bf16_trail
        weight_decay = sgd_args.weight_decay
        lr = sgd_args.lr
        grad_list = torch.ops.torch_ipex.merged_embeddingbag_backward_sgd(
            grad_out,
            weights,
            indices,
            offsets,
            pooling_mode,
            include_last_offset,
            bf16_trail,
            weight_decay,
            lr,
        )
        output = [None] * (5 + len(weights))
        return tuple(output)


class MergedEmbeddingBagAdaGradFunc(Function):
    @staticmethod
    def forward(
        ctx,
        indices,
        offsets,
        pooling_mode,
        include_last_offset,
        adagrad_args,
        *weights,
    ):
        output = torch.ops.torch_ipex.merged_embeddingbag_forward(
            weights, indices, offsets, pooling_mode, include_last_offset
        )
        ctx.indices = indices
        ctx.offsets = offsets
        ctx.weights = weights
        ctx.pooling_mode = pooling_mode
        ctx.include_last_offset = include_last_offset
        ctx.adagrad_args = adagrad_args
        return tuple(output)

    @staticmethod
    def backward(ctx, *grad_out):
        indices = ctx.indices
        offsets = ctx.offsets
        weights = ctx.weights
        pooling_mode = ctx.pooling_mode
        include_last_offset = ctx.include_last_offset
        adagrad_args = ctx.adagrad_args
        bf16_trail = adagrad_args.bf16_trail
        hessian = adagrad_args.hessian
        eps = adagrad_args.eps
        lr = adagrad_args.lr
        grad_list = torch.ops.torch_ipex.merged_embeddingbag_backward_adagrad(
            grad_out,
            weights,
            indices,
            offsets,
            pooling_mode,
            include_last_offset,
            hessian,
            bf16_trail,
            eps,
            lr,
        )
        output = [None] * (5 + len(weights))
        return tuple(output)


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
        assert self.n_tables > 0, "MergedEmbeddingBag at least have 1 table"
        self.embedding_dim = embedding_specs[0].embedding_dim
        assert all(
            specs.embedding_dim == self.embedding_dim for specs in embedding_specs
        ), "expect all tables have same embedding_dim"
        self.dtype = embedding_specs[0].dtype
        assert all(
            specs.dtype == self.dtype for specs in embedding_specs
        ), "expect all tables have same dtype"
        self.pooling_mode = embedding_specs[0].pooling_mode
        assert self.pooling_mode in (
            "sum",
            "mean",
        ), "MergedEmbeddingBag only support EmbeddingBag with model sum or mean"
        assert all(
            specs.pooling_mode == self.pooling_mode for specs in embedding_specs
        ), "expect all tables have same pooling_mode"
        if self.pooling_mode == "sum":
            self.pooling_mode = PoolingMode.SUM
        else:
            self.pooling_mode = PoolingMode.MEAN
        self.include_last_offset = embedding_specs[0].include_last_offset
        assert all(
            specs.include_last_offset == self.include_last_offset
            for specs in embedding_specs
        ), "expect all tables have same include_last_offset"

        # Currently MergedEmbeddingBag only support all dense
        self.dense = all(not specs.sparse for specs in embedding_specs)

        self.weights = torch.nn.ParameterList(
            [nn.Parameter(torch.Tensor()) for _ in range(len(embedding_specs))]
        )

        for i, spec in enumerate(embedding_specs):
            num_embeddings, embedding_dim, _, dtype, weight, _, _ = spec
            if weight is None:
                weight = torch.empty((num_embeddings, embedding_dim), dtype=dtype)
            self.weights[i] = nn.Parameter(weight)

    @classmethod
    def from_embeddingbag_list(
        cls,
        tables: List[torch.nn.EmbeddingBag],
    ):
        embedding_specs = []
        for emb in tables:
            emb_shape = emb.weight.shape
            embedding_specs.append(
                EmbeddingSpec(
                    num_embeddings=emb_shape[0],
                    embedding_dim=emb_shape[1],
                    pooling_mode=emb.mode,
                    dtype=emb.weight.dtype,
                    weight=emb.weight.detach(),
                    sparse=emb.sparse,
                    include_last_offset=emb.include_last_offset,
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
                self.pooling_mode,
                self.weights[i].dtype,
            )
            if i != self.n_tables - 1:
                s += "\n"
        return s

    def forward(self, indices, offsets):
        r"""
        Args:
            indices (List[Tensor]):
                See https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag.forward
            offsets (List[Tensor]):
                See https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag.forward
        Returns:
            List[Tensor] output shape of `(batch_size, embedding_dim)` which length = num of tables.
        """
        assert self.dense
        return merged_embeddingbag(
            self.weights, indices, offsets, self.pooling_mode, self.include_last_offset
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

    def forward(self, indices, offsets):
        r"""
        Args:
            indices (List[Tensor]):
                See https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag.forward
            offsets (List[Tensor]):
                See https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag.forward
        Returns:
            List[Tensor] output shape of `(batch_size, embedding_dim)` which length = num of tables.
        """
        return merged_embeddingbag_sgd(
            self.weights,
            indices,
            offsets,
            self.pooling_mode,
            self.include_last_offset,
            self.sgd_args,
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
                    num_embeddings=emb_shape[0],
                    embedding_dim=emb_shape[1],
                    pooling_mode=emb.mode,
                    dtype=emb.weight.dtype,
                    weight=emb.weight.detach(),
                    sparse=emb.sparse,
                    include_last_offset=emb.include_last_offset,
                )
            )
        return cls(embedding_specs, lr, weight_decay)


class MergedEmbeddingBagWithAdaGrad(MergedEmbeddingBag):
    embedding_specs: List[EmbeddingSpec]

    def __init__(
        self,
        embedding_specs: List[EmbeddingSpec],
        lr: float = 0.01,
        eps: float = 1e-10,
    ):
        super(MergedEmbeddingBagWithAdaGrad, self).__init__(embedding_specs)
        self.adagrad_args = self.init_adagrad_args(lr, eps)
        for i in range(self.n_tables):
            weight = self.weights[i]
            if weight.dtype == torch.bfloat16:
                self.adagrad_args.bf16_trail.append(
                    torch.zeros_like(weight, dtype=torch.bfloat16)
                )
                self.adagrad_args.hessian.append(
                    torch.zeros_like(weight, dtype=torch.float)
                )
            else:
                self.adagrad_args.bf16_trail.append(
                    torch.empty(0, dtype=torch.bfloat16)
                )
                self.adagrad_args.hessian.append(torch.zeros_like(weight))

    def init_adagrad_args(self, lr, eps, bf16_trail=None, hessian=None):
        if bf16_trail is None:
            bf16_trail = []
        if hessian is None:
            hessian = []
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid eps value: {}".format(eps))
        return AdaGradArgs(eps=eps, lr=lr, bf16_trail=bf16_trail, hessian=hessian)

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
        self.adagrad_args = self.adagrad_args._replace(bf16_trail=trails)

    def forward(self, indices, offsets):
        r"""
        Args:
            indices (List[Tensor]): See
                https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag.forward
            offsets (List[Tensor]): See
                https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag.forward
        Returns:
            List[Tensor] output shape of `(batch_size, embedding_dim)` which length = num of tables.
        """
        return merged_embeddingbag_adagrad(
            self.weights,
            indices,
            offsets,
            self.pooling_mode,
            self.include_last_offset,
            self.adagrad_args,
        )

    @classmethod
    def from_embeddingbag_list(
        cls,
        tables: List[torch.nn.EmbeddingBag],
        lr: float = 0.01,
        eps: float = 1e-10,
    ):
        embedding_specs = []
        for emb in tables:
            emb_shape = emb.weight.shape
            embedding_specs.append(
                EmbeddingSpec(
                    num_embeddings=emb_shape[0],
                    embedding_dim=emb_shape[1],
                    pooling_mode=emb.mode,
                    dtype=emb.weight.dtype,
                    weight=emb.weight.detach(),
                    sparse=emb.sparse,
                    include_last_offset=emb.include_last_offset,
                )
            )
        return cls(embedding_specs, lr, eps)


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


import torch.distributed as dist


def sparse_all2all(
    world_size: int,
    send_idx: List[torch.Tensor],
    send_buf: List[torch.Tensor],
    send_ofs: List[torch.Tensor],
):
    # the first thing to know is the recv tensor sizes
    # this requires an all to all
    is_buffers = [torch.empty(1, dtype=torch.int64) for _ in range(world_size)]
    for i in range(world_size):
        is_buffers[i][0] = send_idx[i].shape[0]
    os_buffers = [torch.empty(1, dtype=torch.int64) for _ in range(world_size)]
    dist.all_to_all(os_buffers, is_buffers)
    dist.barrier()

    # init received buffers sizes
    index_type = send_idx[0].dtype
    val_type = send_buf[0].dtype
    emb_dim = send_buf[0].shape[1]
    ofs_size = send_ofs[0].shape[0]
    recv_idx = [torch.zeros(os_buffers[i], dtype=index_type) for i in range(world_size)]
    recv_buf = [
        torch.zeros((os_buffers[i], emb_dim), dtype=val_type) for i in range(world_size)
    ]
    recv_ofs = [torch.zeros((ofs_size,), dtype=torch.int64) for i in range(world_size)]
    dist.all_to_all(recv_idx, send_idx)
    dist.all_to_all(recv_buf, send_buf)
    dist.all_to_all(recv_ofs, send_ofs)
    dist.barrier()
    return recv_idx, recv_buf, recv_ofs


class DistMergeEmbeddingBagFunc(Function):
    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        row_offset: torch.Tensor,
        indices: List[torch.Tensor],
        offsets: List[torch.Tensor],
        rank: int,
        world_size: int,
        include_last_offsets: bool,
        adagrad_args: AdaGradArgs,
    ):
        global_bs = offsets[0].size(0)
        if include_last_offsets:
            global_bs -= 1
        local_bs = int(global_bs / world_size)
        ctx.indices = indices
        ctx.offsets = offsets
        ctx.weight = weight
        ctx.row_offset = row_offset
        ctx.include_last_offsets = include_last_offsets
        ctx.adagrad_args = adagrad_args
        ctx.rank = rank
        ctx.world_size = world_size
        num_emb = len(indices)
        emb_dim = weight.shape[1]
        (
            send_idx,
            send_buf,
            send_ofs,
        ) = torch.ops.torch_ipex.mergedemb_distribute_forward_local(
            weight, row_offset, indices, offsets, rank, world_size, include_last_offsets
        )
        recv_idx, recv_buf, recv_ofs = sparse_all2all(
            world_size, send_idx, send_buf, send_ofs
        )
        output = torch.empty((local_bs, num_emb, emb_dim), dtype=weight.dtype)
        torch.ops.torch_ipex.mergedemb_distribute_forward_merge(
            output, recv_idx, recv_buf, recv_ofs, num_emb
        )
        return output

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        row_offset = ctx.row_offset
        indices = ctx.indices
        offsets = ctx.offsets
        rank = ctx.rank
        world_size = ctx.world_size
        include_last_offsets = ctx.include_last_offsets
        (
            send_idx,
            send_buf,
            send_ofs,
        ) = torch.ops.torch_ipex.mergedemb_distribute_backward_local(
            grad, row_offset, indices, offsets, rank, world_size, include_last_offsets
        )
        recv_idx, recv_buf, recv_ofs = sparse_all2all(
            world_size, send_idx, send_buf, send_ofs
        )
        weight = ctx.weight
        adagrad_args = ctx.adagrad_args
        trail = adagrad_args.bf16_trail
        hessian = adagrad_args.hessian
        lr = adagrad_args.lr
        eps = adagrad_args.eps
        torch.ops.torch_ipex.mergedemb_distribute_backward_merge_adagrad_update(
            recv_idx, recv_buf, recv_ofs, weight, trail[0], hessian[0], lr, eps
        )
        return None, None, None, None, None, None, None, None


class DistMergeEmbeddingBagWithAdaGrad(MergedEmbeddingBagWithAdaGrad):
    r"""
    The distributed version or MergedEmbeddingBagWithAdaGrad
    After creating Pytorch Distributed process group, we can create DistMergeEmbeddingBagWithAdaGrad
    and the emb tables will be automatically seperated to different ranks.
    Each rank will keep particia table and will only run forward/backward/update on the rows it keeped in local.
    We will also merge the result from different ranks through all to all during forward/backward.
    The returned results for forward is shape of [local BS * num tables * emb_dim]
    Example usage:

        >>> EmbLists = torch.nn.Modulist(emb1, emb2, emb3, ..., emb_m)
        >>> dist.init_process_group("ccl", world_size=world_size, rank=rank)
        >>> distributed_emb = DistMergeEmbeddingBagWithAdaGrad.from_embeddingbag_list(EmbLists)
        >>> out = distributed_emb(indices, offsets)
    """

    def __init__(
        self,
        embedding_specs: List[EmbeddingSpec],
        lr: float = 0.01,
        eps: float = 1e-10,
    ):
        super(MergedEmbeddingBagWithAdaGrad, self).__init__(embedding_specs)
        assert (
            self.pooling_mode == PoolingMode.SUM
        ), "only support SUM for DistMergeEmbeddingBagWithAdaGrad"
        self._rank = dist.get_rank()
        self._size = dist.get_world_size()
        # create row_offset
        self._row_offset = [0 for i in range(self.n_tables + 1)]
        for i in range(self.n_tables):
            self._row_offset[i + 1] = self.weights[i].shape[0] + self._row_offset[i]
        # create allin1 weight
        # TODO: The initialization for weight here requiures 2 * total weight size PEAK memory
        # We may able to optimize here to:
        #     1. Require (1 + 1 / world_size) PEAK memory if always load all table first
        #     2. Require (1 / world_size) memory with loading optimizations like using "meta" device
        weight_allin1 = torch.cat([w.data for w in self.weights])[
            self._rank :: self._size, :
        ].clone()
        # drop the oringal weighs
        self.weights = nn.ParameterList([nn.parameter.Parameter(weight_allin1)])
        self.n_tables = 1
        self.adagrad_args = self.init_adagrad_args(lr, eps)
        if weight_allin1.dtype == torch.bfloat16:
            self.adagrad_args.bf16_trail.append(
                torch.zeros_like(weight_allin1, dtype=torch.bfloat16)
            )
            self.adagrad_args.hessian.append(
                torch.zeros_like(weight_allin1, dtype=torch.float)
            )
        else:
            self.adagrad_args.bf16_trail.append(torch.empty(0, dtype=torch.bfloat16))
            self.adagrad_args.hessian.append(torch.zeros_like(weight_allin1))

    def forward(self, indices: List[torch.Tensor], offset: List[torch.Tensor]):
        out = DistMergeEmbeddingBagFunc.apply(
            self.weights[0],
            self._row_offset,
            indices,
            offset,
            self._rank,
            self._size,
            self.include_last_offset,
            self.adagrad_args,
        )
        return out

    def extra_repr(self) -> str:
        s = ""
        s += f"world_size: {self._size}, rank_id: {self._rank}\n"
        s += super(DistMergeEmbeddingBagWithAdaGrad, self).extra_repr()
        return s
