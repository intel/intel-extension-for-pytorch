import torch
import torch.utils._pytree as pytree
import copy

# import math
# from enum import Enum
# from collections import OrderedDict


def _prod(myList):
    ret = 1
    for x in myList:
        if x is None:
            return None
        ret = ret * x
    return ret


def get_vnni_blocking(dtype):
    if dtype == torch.float32:
        return 1
    elif dtype in [torch.bfloat16, torch.float16]:
        return 2
    elif dtype == torch.bfloat8:
        return 4
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


class BlockingManager(object):
    def __init__(self, orig_shape, blocking_factors=None, permute=None):
        dims = len(orig_shape)
        if blocking_factors is None:
            blocking_factors = [None] * dims
        else:
            assert len(blocking_factors) == dims
        self.orig_shape = orig_shape
        view_sizes = []
        for i in range(dims):
            if blocking_factors[i] is None:
                view_sizes.append(orig_shape[i])
            else:
                if isinstance(blocking_factors[i], int):
                    assert orig_shape[i] % blocking_factors[i] == 0, (
                        "Blocking factor doesn't divide dim evenly shape = %s, BF = %s"
                        % (orig_shape, blocking_factors)
                    )
                    view_sizes.append(orig_shape[i] // blocking_factors[i])
                    view_sizes.append(blocking_factors[i])
                elif isinstance(blocking_factors[i], (list, tuple)):
                    total_blocked = _prod(blocking_factors[i])
                    assert orig_shape[i] % total_blocked == 0
                    view_sizes.append(orig_shape[i] // total_blocked)
                    view_sizes.extend(blocking_factors[i])
                else:
                    raise ValueError(
                        "Unsupported blocking factor: %s" % (blocking_factors[i],)
                    )
        back_permute = None
        blocked_shape = view_sizes
        if permute is not None:
            assert isinstance(permute, (list, tuple))
            plen = len(permute)
            assert plen == len(view_sizes)
            assert all([i in permute for i in range(plen)])
            back_permute = [None] * plen
            blocked_shape = [None] * plen
            for i in range(plen):
                back_permute[permute[i]] = i
                blocked_shape[i] = view_sizes[permute[i]]

        self.permute = permute
        self.back_permute = back_permute
        self.view_shape = view_sizes
        self.blocked_shape = blocked_shape

    def block(self, input):
        assert input.shape == self.orig_shape
        output = input.view(self.view_shape)
        if self.permute:
            output = output.permute(self.permute).contiguous()
        return output

    def unblock(self, input):
        assert list(input.shape) == self.blocked_shape, "Shapes: %s, %s" % (
            input.shape,
            self.blocked_shape,
        )
        output = input
        if self.back_permute:
            output = output.permute(self.back_permute).contiguous()
        output = output.view(self.orig_shape)
        return output


def get_blocking_signature(plain_layout_str, blocked_layout_str):
    return [
        [j for j, d in enumerate(blocked_layout_str) if d == c]
        for i, c in enumerate(plain_layout_str)
    ]


def _get_block_sizes(blocked_shape, blocking_signeture, dim):
    return [blocked_shape[d] for d in blocking_signeture[dim]]


def _get_plain_size(blocked_shape, blocking_signeture, dim):
    return _prod(_get_block_sizes(blocked_shape, blocking_signeture, dim))


def _get_plain_shape(blocked_shape, blocking_signeture):
    return [
        _prod([blocked_shape[d] for d in dim_list]) for dim_list in blocking_signeture
    ]


def _get_permute_list(blocking_signeture):
    return [item for sublist in blocking_signeture for item in sublist]


class BlockedTensor(object):
    def __init__(self, data, blocking_signeture=None, plain_dtype=None, **kwargs):
        self._t = torch.as_tensor(data, **kwargs)
        self.blocking_signeture = blocking_signeture
        self.permute_list = None
        self.plain_shape = None
        self.plain_dtype = plain_dtype if plain_dtype else self._t.dtype

    def __repr__(self):
        return "Blocking_signature:\n{}\n\ndata:\n{}".format(
            self.blocking_signeture, self._t
        )

    def get_plain_shape(self, dim=None):
        if dim is None:
            if self.plain_shape:
                return self.plain_shape
            self.plain_shape = torch.Size(
                _get_plain_shape(self._t.shape, self.blocking_signeture)
            )
            return self.plain_shape
        else:
            return self.get_plain_shape()[dim]

    def get_permute_list(self):
        if self.permute_list:
            return self.permute_list
        self.permute_list = _get_permute_list(self.blocking_signeture)
        return self.permute_list

    def get_blocked_dim(self):
        return self._t.dim()

    def get_plain_dim(self):
        return len(self.blocking_signeture)

    def get_plain_size(self, dim):
        plain_shape = self.get_plain_shape()
        return plain_shape[dim]
        # return _get_plain_size(self._t.shape, self.blocking_signeture, dim)

    def get_plain_dtype(self):
        return self.plain_dtype

    def get_block_sizes(self, dim):
        return _get_block_sizes(self._t.shape, self.blocking_signeture, dim)

    def blocked_tensor(self):
        return self._t

    def unblocked_tensor(self):
        permute_list = self.get_permute_list()
        plain_shape = self.get_plain_shape()
        plain_dtype = self.get_plain_dtype()
        # print("BlockedTensor returning unblocked tensor with shape %s" % (plain_shape,))
        return (
            self._t.permute(permute_list).contiguous().view(plain_shape).to(plain_dtype)
        )

    def get_signature(self):
        return self.blocking_signeture

    def __getitem__(self, key):
        return self.unblocked_tensor().__getitem__(key)

    def __getattr__(self, attr):
        # print("requiested attr: %s" % attr)
        if attr == "shape":
            return torch.Size(self.get_plain_shape())
        if attr == "dtype":
            return self.get_plain_dtype()
        elif attr == "size":
            return self.get_plain_shape
        elif attr == "dim":
            return self.get_plain_dim
        elif attr == "mean":
            return getattr(self._t, attr)
        elif attr == "detach":
            return getattr(self.unblocked_tensor(), attr)
        elif attr == "view":
            return getattr(self.unblocked_tensor(), attr)
        # elif hasattr(self._t, attr): return getattr(self._t, attr)
        else:
            raise AttributeError("BlockedTensor doesn't support attr %s" % attr)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # args = [a._t if hasattr(a, '_t') else a for a in args]
        args = [
            a.unblocked_tensor() if isinstance(a, BlockedTensor) else a for a in args
        ]
        ret = func(*args, **kwargs)
        # return MetadataTensor(ret, metadata=self._metadata)
        return ret


class BlockedParameter(torch.nn.Parameter):
    def __new__(cls, data=None, requires_grad=True):
        return super(BlockedParameter, cls).__new__(
            cls, data=data, requires_grad=requires_grad
        )

    def __init__(self, data=None, requires_grad=True):
        self._data = data
        self.blocked = False
        self.blocking_param = None
        self.blocking_manager = None
        self.unblocked_dtype = None
        self.blocked_dtype = None

    def __repr__(self):
        return f"BlockedParameter({self._data})"

    def set_blocking_param(self, blocking_param):
        self.blocking_param = blocking_param

    def is_blocked(self):
        return self.blocked

    def block(self):
        if self.blocked:
            return
        if self.blocking_manager is None:
            if self.blocking_param is None:
                return
            self.unblocked_dtype = self.dtype
            self.blocked_dtype = (
                self.blocking_param[2] if len(self.blocking_param) > 2 else self.dtype
            )
            self.blocking_manager = BlockingManager(
                self._data.shape,
                blocking_factors=self.blocking_param[0],
                permute=self.blocking_param[1],
            )
        self._data = self.blocking_manager.block(self._data).to(self.blocked_dtype)
        if self.grad is not None:
            self.grad.data = self.blocking_manager.block(self.grad.data).to(
                self.blocked_dtype
            )
        self.blocked = True
        self.data = self._data

    def unblock(self):
        if not self.blocked:
            return
        assert self.blocking_manager is not None
        self._data = self.blocking_manager.unblock(self._data).to(self.unblocked_dtype)
        if self.grad is not None:
            self.grad.data = self.blocking_manager.unblock(self.grad.data).to(
                self.unblocked_dtype
            )
        self.blocked = False
        self.data = self._data

    def __tensor_flatten__(self):
        return ["_data"], [
            self.requires_grad,
            self.blocked,
            self.blocking_param,
            self.blocking_manager,
            self.unblocked_dtype,
            self.blocked_dtype,
        ]

    @staticmethod
    def __tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride):
        out = BlockedParameter(inner_tensors["_data"], requires_grad=ctx[0])
        out.blocked = ctx[1]
        out.blocking_param = ctx[2]
        out.blocking_manager = ctx[3]
        out.unblocked_dtype = ctx[4]
        out.blocked_dtype = ctx[5]
        return out

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        args_data = pytree.tree_map_only(BlockedParameter, lambda x: x._data, args)
        return func(*args_data, **kwargs)

    def __copy__(self):
        new_param = BlockedParameter(self._data, requires_grad=self.requires_grad)
        for k, v in self.__dict__.items():
            if k != "_data":
                setattr(new_param, k, copy.copy(v))
        return new_param

    def __deepcopy__(self, memo):
        new_param = BlockedParameter(
            copy.deepcopy(self._data, memo), requires_grad=self.requires_grad
        )
        for k, v in self.__dict__.items():
            setattr(new_param, k, copy.deepcopy(v, memo))
        return new_param


class BlockedModule(torch.nn.Module):
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        blocked_params = []
        for p in self.parameters(recurse=False):
            if isinstance(p, BlockedParameter) and p.is_blocked():
                p.unblock()
                blocked_params.append(p)
        super(BlockedModule, self)._save_to_state_dict(destination, prefix, keep_vars)
        for p in blocked_params:
            p.block()
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print("_save_to_state_dict Called - %s" % prefix)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        blocked_params = []
        for p in self.parameters(recurse=False):
            if isinstance(p, BlockedParameter) and p.is_blocked():
                p.unblock()
                blocked_params.append(p)
        super(BlockedModule, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for p in blocked_params:
            p.block()
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print("_load_from_state_dict Called - %s" % prefix)

    @staticmethod
    def default_blocking_factors(S):
        blocking_prio_list = (
            [64, 48, 32, 24, 16] + list(range(62, 11, -2)) + list(range(63, 10, -2))
        )
        for bs in blocking_prio_list:
            if S % bs == 0:
                return [S // bs, bs]
        return [1, S]

    @staticmethod
    def get_blocked_tensor(tensor, signature, blocking_factors=None):
        if isinstance(tensor, BlockedTensor):
            if tensor.get_signature() == signature:
                blocked_tensor = tensor.blocked_tensor()
                # print("Reusing blocked tensor with shape %s" % (blocked_tensor.shape,))
                return blocked_tensor
            else:
                raise TypeError("Blocked tensor signature doesn't match")
        else:
            # print("Converting to blocked tensor with shape %s" % (tensor.shape,))
            dim = tensor.dim()
            assert len(signature) == dim, "Tensor shape doesn't match with signature"
            if blocking_factors is None:
                blocking_factors = [None] * dim
            assert (
                len(blocking_factors) == dim
            ), "Tensor shape doesn't match with blocking_factors"
            view_shape = []
            back_permute = []
            plain_shape = tensor.shape
            for i, dl in enumerate(signature):
                back_permute += dl
                if len(dl) == 1:
                    view_shape.append(plain_shape[i])
                else:
                    nf = len(dl)
                    bf = None
                    if nf == 2:
                        if blocking_factors[i] is None:
                            bf = BlockedModule.default_blocking_factors(plain_shape[i])
                        else:
                            if isinstance(blocking_factors[i], int):
                                bf = [
                                    plain_shape[i] // blocking_factors[i],
                                    blocking_factors[i],
                                ]
                            else:
                                raise ValueError("blocking_factors is not Integer")
                    else:
                        raise ValueError(
                            "Blocking to more than 2 dims not supported yet"
                        )
                    view_shape += bf
            permute = [None] * len(back_permute)
            for i in range(len(back_permute)):
                permute[back_permute[i]] = i
            return tensor.view(view_shape).permute(permute).contiguous()


def block_model_params(model):
    for m in model.modules():
        if hasattr(m, "maybe_block_params"):
            m.maybe_block_params()


class TestModule(BlockedModule):
    def __init__(self):
        super(BlockedModule, self).__init__()
        self.param1 = BlockedParameter(torch.arange(10.0))
        self.param1.set_blocking_param(
            (
                [5],
                [1, 0],
            )
        )
        self.param2 = torch.nn.Parameter(torch.arange(3.0))

    def forward(self):
        print("Shape", self.param1.shape)
        self.param1.block()
        print("Blocked shape", self.param1.shape)
        return self.param1


if __name__ == "__main__":
    M = TestModule()

    print(list(M.parameters()))
    y = M()
    print(list(M.parameters()))
    y = M()

    # print(list(M.state_dict()))
    torch.save(M.state_dict(), "tmp.pth")

    print(list(M.parameters()))

    M.param1.data = M.param1.data * 2.0

    print(list(M.parameters()))

    state_dict = torch.load("tmp.pth")
    print("state_dict:", state_dict)
    M.load_state_dict(state_dict)

    print(list(M.parameters()))
    print(torch.load("tmp.pth"))
