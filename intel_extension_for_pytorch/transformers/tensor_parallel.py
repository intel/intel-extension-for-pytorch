import torch
import torch.nn as nn
from ..cpu import comm as ipex_comm
import os


class TensorParallellLinear(nn.Module):
    def __init__(
        self,
        linear,
        num_kv_heads,
        num_heads,
        head_dim,
        rank,
        world_size,
        shard_by_head,
        shard_by_col,
    ):
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rank = rank
        self.world_size = world_size
        self.shard_by_head = shard_by_head
        self.shard_by_col = shard_by_col
        self.cols_per_rank = None
        self.shard_weights(linear)

    def shard_weights_by_head(
        self,
        linear,
        num_kv_heads,
        num_heads,
        head_dim,
        rank,
        world_size,
        shard_by_col=True,
    ):
        if shard_by_col:
            total_size = linear.weight.shape[0]
        else:
            total_size = linear.weight.shape[1]
        q_bias = None
        k_bias = None
        v_bias = None
        bias_data = None
        concat_qkv = total_size > num_heads * head_dim
        kv_group_size = num_heads // num_kv_heads
        kv_head_per_rank = num_kv_heads // world_size
        if world_size == 1:
            return
        if world_size > num_kv_heads:
            RuntimeError(
                f"world_size {world_size} is larger than num_kv_heads {num_kv_heads}"
            )
        kv_head_range = [0]  # [)
        for i in range(world_size - 1, -1, -1):
            kv_head_this_rank = kv_head_per_rank
            if i < num_kv_heads % world_size:
                kv_head_this_rank += 1
            kv_head_range.append(kv_head_range[-1] + kv_head_this_rank)
        weight_data = linear.weight.data
        q_head_start = kv_head_range[rank] * kv_group_size
        q_head_end = (
            q_head_start
            + (kv_head_range[rank + 1] - kv_head_range[rank]) * kv_group_size
        )
        if shard_by_col:
            q = weight_data[q_head_start * head_dim : q_head_end * head_dim]
            if linear.bias is not None:
                q_bias = linear.bias.data[
                    q_head_start * head_dim : q_head_end * head_dim
                ]
        else:
            q = weight_data[:, q_head_start * head_dim : q_head_end * head_dim]
        if not concat_qkv:
            return torch.nn.Parameter(q), torch.nn.Parameter(q_bias)

        k_head_start = num_heads + kv_head_range[rank]
        k_head_end = k_head_start + (kv_head_range[rank + 1] - kv_head_range[rank])
        v_head_start = num_heads + num_kv_heads + kv_head_range[rank]
        v_head_end = v_head_start + (kv_head_range[rank + 1] - kv_head_range[rank])
        if shard_by_col:
            k = weight_data[k_head_start * head_dim : k_head_end * head_dim]
            v = weight_data[v_head_start * head_dim : v_head_end * head_dim]
            if linear.bias is not None:
                k_bias = linear.bias.data[
                    k_head_start * head_dim : k_head_end * head_dim
                ]
                v_bias = linear.bias.data[
                    v_head_start * head_dim : v_head_end * head_dim
                ]
                bias_data = torch.cat([q_bias, k_bias, v_bias], dim=0)
        else:
            k = weight_data[:, k_head_start * head_dim : k_head_end * head_dim]
            v = weight_data[:, v_head_start * head_dim : v_head_end * head_dim]
            if linear.bias is not None:
                bias_data = linear.bias.data
        weight_data = torch.cat([q, k, v], dim=0)
        return torch.nn.Parameter(weight_data), torch.nn.Parameter(bias_data)

    def shard_weights_by_block(
        self, linear, rank, world_size, shard_by_col=True, block_size=64
    ):
        if shard_by_col:
            total_size = linear.weight.shape[0]
        else:
            total_size = linear.weight.shape[1]
        bias_data = None
        cols_per_rank = [0]
        for i in range(world_size - 1, -1, -1):
            if total_size % block_size == 0:
                block_count = total_size // block_size
                block_per_rank = block_count // world_size
                if i < block_count % world_size:
                    block_per_rank += 1
                cols_per_rank.append(cols_per_rank[-1] + block_per_rank * block_size)
            else:
                cols = total_size // world_size
                if i < total_size % world_size:
                    cols += 1
                cols_per_rank.append(cols_per_rank[-1] + cols)
        weight_data = linear.weight.data
        if shard_by_col:
            weight_data = weight_data[cols_per_rank[rank] : cols_per_rank[rank + 1]]
            if linear.bias is not None:
                bias_data = linear.bias.data[
                    cols_per_rank[rank] : cols_per_rank[rank + 1]
                ]
        else:
            weight_data = weight_data[:, cols_per_rank[rank] : cols_per_rank[rank + 1]]
            if linear.bias is not None:
                bias_data = linear.bias.data / float(world_size)
        return (
            torch.nn.Parameter(weight_data),
            torch.nn.Parameter(bias_data),
            cols_per_rank,
        )

    def shard_weights(self, linear):
        if self.world_size == 1:
            return
        if self.shard_by_head:
            weight, bias = self.shard_weights_by_head(
                linear,
                self.num_kv_heads,
                self.num_heads,
                self.head_dim,
                self.rank,
                self.world_size,
                self.shard_by_col,
            )
        else:
            weight, bias, self.cols_per_rank = self.shard_weights_by_block(
                linear,
                self.rank,
                self.world_size,
                self.shard_by_col,
            )
        self.linear = nn.Linear(
            weight.shape[1], weight.shape[0], bias=linear.bias is not None
        )

        self.linear.weight = weight
        if linear.bias is not None:
            self.linear.bias = bias
        del linear

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.linear(input)


class TensorParallelColumnLinear(TensorParallellLinear):
    def __init__(
        self,
        linear,
        num_kv_heads,
        num_heads,
        head_dim,
        rank,
        world_size,
        shard_by_head=True,
    ):
        super().__init__(
            linear,
            num_kv_heads,
            num_heads,
            head_dim,
            rank,
            world_size,
            shard_by_head,
            shard_by_col=True,
        )


class TensorParallelRowLinear(TensorParallellLinear):
    def __init__(
        self,
        linear,
        num_kv_heads,
        num_heads,
        head_dim,
        rank,
        world_size,
        shard_by_head=True,
    ):
        super().__init__(
            linear,
            num_kv_heads,
            num_heads,
            head_dim,
            rank,
            world_size,
            shard_by_head,
            shard_by_col=False,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.linear(input)
        if self.world_size > 1:
            ipex_comm.allreduce_add(out)
        return out


class TensorParallelLMhead(TensorParallellLinear):
    def __init__(
        self,
        linear,
        num_kv_heads,
        num_heads,
        head_dim,
        rank,
        world_size,
        shard_by_col,
    ):
        super().__init__(
            linear,
            num_kv_heads,
            num_heads,
            head_dim,
            rank,
            world_size,
            shard_by_head=False,
            shard_by_col=shard_by_col,
        )
        self.gather_result = shard_by_col

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.gather_result:
            out = self.linear(input)
            out = ipex_comm.allgather(out, self.cols_per_rank, self.world_size)
        else:
            if self.world_size > 1:
                input = input[
                    ...,
                    self.cols_per_rank[self.rank] : self.cols_per_rank[self.rank + 1],
                ]
            out = self.linear(input)
            if self.world_size > 1:
                ipex_comm.allreduce_add(out)

        return out


def shard_mha_weights(
    model, target_m, num_heads, num_kv_heads, head_dim, rank, world_size
):
    if world_size == 1:
        return
    for name, sub_m in model.named_children():
        if isinstance(sub_m, target_m):
            for l_name, l_sub_m in sub_m.named_children():
                if l_name in ["q_proj"]:
                    TPLinear = TensorParallelColumnLinear(
                        l_sub_m,
                        num_kv_heads,
                        num_heads,
                        head_dim,
                        rank,
                        world_size,
                        shard_by_head=True,
                    )
                    # del sub_m.__dict__["_modules"][l_name]
                    setattr(sub_m, l_name, TPLinear.linear)
                if l_name in ["k_proj", "v_proj"]:
                    TPLinear = TensorParallelColumnLinear(
                        l_sub_m,
                        num_kv_heads,
                        num_kv_heads,
                        head_dim,
                        rank,
                        world_size,
                        shard_by_head=True,
                    )
                    # del sub_m.__dict__["_modules"][l_name]
                    setattr(sub_m, l_name, TPLinear.linear)
                if l_name in ["out_proj", "o_proj"]:
                    TPLinear = TensorParallelRowLinear(
                        l_sub_m,
                        num_kv_heads,
                        num_heads,
                        head_dim,
                        rank,
                        world_size,
                        shard_by_head=True,
                    )
                    # del sub_m.__dict__["_modules"][l_name]
                    setattr(sub_m, l_name, TPLinear)

        shard_mha_weights(
            sub_m, target_m, num_heads, num_kv_heads, head_dim, rank, world_size
        )


def shard_mlp_weights(
    model, target_m, num_heads, num_kv_heads, head_dim, rank, world_size
):
    if world_size == 1:
        return
    for _, sub_m in model.named_children():
        if isinstance(sub_m, target_m):
            for l_name, l_sub_m in sub_m.named_children():
                if l_name in ["gate_proj", "up_proj", "fc_in"]:
                    TPLinear = TensorParallelColumnLinear(
                        l_sub_m,
                        num_kv_heads,
                        num_heads,
                        head_dim,
                        rank,
                        world_size,
                        shard_by_head=False,
                    )
                    setattr(sub_m, l_name, TPLinear.linear)
                if l_name in ["down_proj", "fc_out"]:
                    TPLinear = TensorParallelRowLinear(
                        l_sub_m,
                        num_kv_heads,
                        num_kv_heads,
                        head_dim,
                        rank,
                        world_size,
                        shard_by_head=False,
                    )
                    setattr(sub_m, l_name, TPLinear)
        shard_mlp_weights(
            sub_m, target_m, num_heads, num_kv_heads, head_dim, rank, world_size
        )


def shard_lm_head_weights(
    model, supported_model_class, num_heads, num_kv_heads, head_dim, rank, world_size
):
    if world_size == 1:
        return
    if not isinstance(model, supported_model_class):
        return
    for name, sub_m in model.named_children():
        lm_head_shard_policy = os.getenv("LM_HEAD_SHARD_POLICY", "row")
        shard_by_col = lm_head_shard_policy == "col"
        if name in ["lm_head"]:
            TPLinear = TensorParallelLMhead(
                sub_m,
                num_kv_heads,
                num_heads,
                head_dim,
                rank,
                world_size,
                shard_by_col=shard_by_col,
            )
            setattr(model, name, TPLinear)
            return
        shard_lm_head_weights(
            sub_m,
            supported_model_class,
            num_heads,
            num_kv_heads,
            head_dim,
            rank,
            world_size,
        )


def update_heads_info(_model, rank, world_size):
    # update the head number of config after sharding
    num_heads = _model.config.num_attention_heads
    num_kv_heads = num_heads
    head_dim = _model.config.hidden_size // num_heads
    for name in ["num_key_value_heads"]:
        if hasattr(_model.config, name):
            num_kv_heads = getattr(_model.config, name)
    group_size = num_heads // num_kv_heads
    kv_head_per_rank = num_kv_heads // world_size
    assert world_size <= num_kv_heads
    kv_heads_range = [0]
    for i in range(world_size - 1, -1, -1):
        kv_heads_this_rank = kv_head_per_rank
        if i < num_kv_heads % world_size:
            kv_heads_this_rank += 1
        kv_heads_range.append(kv_heads_range[-1] + kv_heads_this_rank)

    def update(_model, group_size, kv_head_range):
        for _, sub_m in _model.named_children():
            # update number of query heads
            target_kv_head = kv_heads_range[rank + 1] - kv_heads_range[rank]
            for name in ["num_attention_heads", "num_heads"]:
                if hasattr(sub_m, "config") and hasattr(sub_m.config, name):
                    setattr(sub_m.config, name, group_size * target_kv_head)
                if hasattr(sub_m, name):
                    setattr(sub_m, name, group_size * target_kv_head)
            # update number of key/value heads
            for name in ["num_key_value_heads", "num_key_value_heads"]:
                if hasattr(sub_m, "config") and hasattr(sub_m.config, name):
                    setattr(sub_m.config, name, target_kv_head)
                if hasattr(sub_m, name):
                    setattr(sub_m, name, target_kv_head)
            # update hidden_size
            for name in ["hidden_size"]:
                if hasattr(sub_m, "config") and hasattr(sub_m.config, name):
                    setattr(sub_m.config, name, group_size * target_kv_head * head_dim)
                if hasattr(sub_m, name):
                    setattr(sub_m, name, group_size * target_kv_head * head_dim)
            update(sub_m, group_size, kv_head_range)

    update(_model, group_size, kv_heads_range)
