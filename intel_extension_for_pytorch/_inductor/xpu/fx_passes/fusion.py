import torch
import functools
from torch.fx.experimental.symbolic_shapes import free_symbols
from torch._inductor.fx_passes.freezing_patterns import register_freezing_graph_pattern
from torch._inductor.lowering import lowerings as L
import torch._inductor.ir as torch_ir
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    filter_nodes,
    get_arg_value,
    KeywordArg,
)
from torch._inductor.virtualized import ops
from ..pattern_matcher import _register_lowering_pattern_post_grad_pre_pass
from typing import Tuple, Any

""" conv_relu_fusion
aten = torch.ops.aten
# input, weight, bias, padding, stride, dilation, groups
_conv_args = (Arg(), Arg(), Arg(), Arg(), Arg(), Arg(), Arg(), Arg(), Arg())
@register_lowering_pattern(
    CallFunction(
        aten.relu,
        CallFunction(aten.convolution.default,
            *_conv_args,
            _users=1
        )
    )
)
def conv_relu(match: Match, input, weight, bias, stride, padding, dilation, transpose, output_padding, groups):
    print("fuse conv2d with relu in pattern matcher")
    return xpu_ir.ConvolutionReLU.create(input, weight, bias, padding, stride, dilation, groups)
    # return torch.ops.torch_ipex.conv2d_relu(input, weight, bias, padding, stride, dilation, groups)
"""

aten = torch.ops.aten
torch_ipex = torch.ops.torch_ipex
prims = torch.ops.prims

_conv_args = [Arg() for _ in range(10)]
_linear_args = [Arg() for _ in range(6)]


def _conv_call(users=1):
    return CallFunction(
        torch_ipex._convolution_pointwise.default, *_conv_args, _users=users
    )


def _linear_call(users=1):
    return CallFunction(
        torch_ipex._linear_pointwise.default, *_linear_args, _users=users
    )


def _to_float(input_call, users=1):
    return CallFunction(
        prims.convert_element_type.default,
        input_call,
        KeywordArg("to_float"),
        _users=users,
    )


def _to_bf16(input_call):
    return CallFunction(
        prims.convert_element_type.default,
        input_call,
        KeywordArg("to_bf16"),
        _users=1,
    )


def _unary_fusion_pattern(unary_fusion, call_fn, users):
    # only insert to_dtype if is_bf16 is True
    computation_call = call_fn(users=users)
    out = unary_fusion(computation_call)
    return out


def _gelu_fusion_1(computation_call):
    return CallFunction(
        aten.mul,
        CallFunction(aten.mul, computation_call, 0.5),
        CallFunction(
            aten.add,
            CallFunction(
                aten.erf,
                CallFunction(aten.mul, computation_call, 0.7071067811865476),
            ),
            1,
        ),
    )


def _gelu_fusion_2(computation_call):
    return CallFunction(
        aten.mul,
        CallFunction(aten.mul, computation_call, 0.5),
        CallFunction(
            aten.add,
            CallFunction(
                aten.tanh,
                CallFunction(
                    aten.mul,
                    CallFunction(
                        aten.add,
                        computation_call,
                        CallFunction(
                            aten.mul,
                            CallFunction(
                                aten.mul,
                                CallFunction(
                                    aten.mul, computation_call, computation_call
                                ),
                                computation_call,
                            ),
                            0.044715,
                        ),
                    ),
                    0.7978845608028654,
                ),
            ),
            1,
        ),
    )


def _hardswish_fusion(computation_call):
    return CallFunction(
        aten.div,
        CallFunction(
            aten.mul,
            computation_call,
            CallFunction(
                aten.clamp_max,
                CallFunction(
                    aten.clamp_min, CallFunction(aten.add, computation_call, 3), 0
                ),
                6,
            ),
        ),
        6,
    )


def _silu_fusion(computation_call):
    return CallFunction(
        aten.mul, computation_call, CallFunction(aten.sigmoid, computation_call)
    )


def _hardsigmoid_fusion(computation_call):
    return CallFunction(
        aten.div,
        CallFunction(
            aten.clamp_max,
            CallFunction(
                aten.clamp_min, CallFunction(aten.add, computation_call, 3), 0
            ),
            6,
        ),
        6,
    )


def _leaky_relu_fusion(computation_call):
    return CallFunction(
        aten.where,
        CallFunction(aten.gt, computation_call, 0),
        computation_call,
        CallFunction(aten.mul, computation_call, KeywordArg("negative_slope")),
    )


def _hardtanh_fusion(computation_call):
    return CallFunction(
        aten.clamp_max,
        CallFunction(aten.clamp_min, computation_call, KeywordArg("min_value")),
        KeywordArg("max_value"),
    )


def _combined_fusion(computation_call, elementwise_op):
    return CallFunction(elementwise_op, computation_call)


# binary_op(other, computation_op)
def _binary_fusion_v1(computation_call, binary_fn):
    return CallFunction(binary_fn, KeywordArg("other"), computation_call)


# binary_op(computation_op, other)
def _binary_fusion_v2(computation_call, binary_fn):
    return CallFunction(binary_fn, computation_call, KeywordArg("other"))


def _is_single_computation_op(computation_op):
    def fn(match):
        computation_nodes = filter_nodes(match.nodes, computation_op)
        if len(computation_nodes) < 1:
            return False
        if any(n.args[-3] != "none" for n in computation_nodes):
            return False
        return True

    return fn


def _is_valid_computation_unary_fusion(computation_op):
    def fn(match):
        matched = _is_single_computation_op(computation_op)(match)
        computation_node = filter_nodes(match.nodes, computation_op)[0]
        return matched

    return fn


def _register_unary_fusion_lowering(pattern, unary_attr, computation_op):
    @_register_lowering_pattern_post_grad_pre_pass(
        pattern, extra_check=_is_valid_computation_unary_fusion(computation_op)
    )
    def fn(match, *args, **kwargs):
        computation_args = list(args)[:-3] + [
            unary_attr.op_name,
            unary_attr.scalars_attr,
            unary_attr.algorithm_attr,
        ]
        return L[computation_op](*computation_args)

    return fn


def _register_leaky_relu_fusion_lowering(pattern, computation_op):
    @_register_lowering_pattern_post_grad_pre_pass(
        pattern, extra_check=_is_single_computation_op(computation_op)
    )
    def fn(match, *args, **kwargs):
        negative_slope = kwargs.get("negative_slope")
        if isinstance(negative_slope, torch_ir.TensorBox):
            matched = False
        else:  # inp is a Number
            matched = True
        computation_args = list(args)
        if matched:
            computation_args = computation_args[:-3] + [
                "leaky_relu",
                [negative_slope],
                "",
            ]
            return L[computation_op](*computation_args)
        else:
            # computation_args += ["none", [], ""]
            out = L[computation_op](*computation_args)
            out = L[aten.where](
                L[aten.gt](out, 0),
                out,
                L[aten.mul](out, negative_slope),
            )
            return out

    return fn


def _register_hardtanh_fusion_lowering(pattern, computation_op):
    @_register_lowering_pattern_post_grad_pre_pass(
        pattern, extra_check=_is_single_computation_op(computation_op)
    )
    def fn(match, *args, **kwargs):
        min_value = kwargs.get("min_value")
        max_value = kwargs.get("max_value")
        if isinstance(min_value, torch_ir.TensorBox) or isinstance(
            max_value, torch_ir.TensorBox
        ):
            matched = False
        else:  # inp is a Number
            assert max_value is not None
            matched = min_value <= max_value
        computation_args = list(args)
        if matched:
            computation_args = computation_args[:-3] + [
                "hardtanh",
                [min_value, max_value],
                "",
            ]
            return L[computation_op](*computation_args)
        else:
            out = L[computation_op](*computation_args)
            out = L[aten.clamp_max](L[aten.clamp_min](out, min_value), max_value)
            return out

    return fn


_binary_attr = {
    aten.add: "add",
    ops.add: "add",
    aten.sub: "sub",
    ops.sub: "sub",
}


def _is_valid_binary(match, fn):
    binary_nodes = filter_nodes(match.nodes, fn)
    if len(binary_nodes) < 1:
        return False

    def get_meta_value(argument: torch.fx.node.Argument):
        # Only torch.fx.Node is expected to have meta.
        if isinstance(argument, torch.fx.Node):
            return argument.meta.get("val", None)
        return None

    if any(
        not isinstance(get_meta_value(n.args[0]), torch.Tensor)
        or not isinstance(get_meta_value(n.args[1]), torch.Tensor)
        for n in binary_nodes
    ):
        return False
    # check alpha is one.
    if any(
        get_arg_value(n, 2, kwarg_name="alpha") != 1.0
        and get_arg_value(n, 2, kwarg_name="alpha") is not None
        for n in binary_nodes
    ):
        return False
    if any(
        get_meta_value(n.args[0]).size() != get_meta_value(n.args[1]).size()
        or get_meta_value(n.args[0]).device != get_meta_value(n.args[1]).device
        or get_meta_value(n.args[0]).dtype != get_meta_value(n.args[1]).dtype
        for n in binary_nodes
    ):
        return False
    # check args[0] and args[1] is not same
    if any(n.args[0] == n.args[1] for n in binary_nodes):
        return False
    return True


def _is_valid_computation_binary(computation_op, binary_op, other_index=None):
    def fn(match):
        if not _is_single_computation_op(computation_op)(match):
            return False
        if not _is_valid_binary(match, binary_op):
            return False
        return True

    return fn


def _get_remaining_users(extra_input_node, compute_node):
    # Think about this pattern:
    #      ReLU
    #     /   \
    #  Conv1
    #   /      \
    # Conv2
    #   \      /
    #      Add
    # Although, the extra input node (ReLU) has more than 1 users: Conv1 and Add.
    # The Conv1 is the ancestor node of the current compute node (Conv2).
    # This indicates that the buffer of ReLU has completed all its usage,
    # So we can safely make changes to it now by doing Conv2->Add inplace fusion.
    # Take above case as example:
    # * extra_input_node: ReLU
    # * compute_node: Conv2
    # _get_remaining_users will return the users of extra_input_node which are not
    # ancestor node of compute_node.
    def _is_ancestor_node(_current_node, _ancestor_node):
        # Check whether _ancestor_node is the ancestor node of _current_node
        _node_list = [_current_node]
        _visited_nodes = set()
        while len(_node_list) != 0:
            _current_node = _node_list.pop(0)
            if _current_node not in _visited_nodes:
                _visited_nodes.add(_current_node)
                if _current_node == _ancestor_node:
                    return True
                elif isinstance(
                    _current_node, torch.fx.Node
                ) and _current_node.op not in ["placeholder", "output", "get_attr"]:
                    for input in _current_node.all_input_nodes:
                        _node_list.append(input)  # noqa: PERF402
        return False

    return [
        user
        for user in list(extra_input_node.users)
        if not _is_ancestor_node(compute_node, user)
    ]


def _is_valid_computation_binary_inplace(computation_op, binary_op, other_index):
    def fn(match):
        if not _is_valid_computation_binary(computation_op, binary_op)(match):
            return False
        binary_nodes = filter_nodes(match.nodes, binary_op)

        def _get_compute_node(_binary_node, _other_index):
            assert (
                len(_binary_node.all_input_nodes) == 2
            ), "Binary node should have 2 input nodes."
            _compute_index = 1 if (_other_index == 0) else 0
            return _binary_node.args[_compute_index]

        def _other_input_not_inplaceable(_binary_node, _other_index):
            _compute_node = _get_compute_node(_binary_node, _other_index)
            return (
                len(
                    _get_remaining_users(_binary_node.args[_other_index], _compute_node)
                )
                > 1
                or _binary_node.args[_other_index] == _compute_node.args[0]
            )

        if any(_other_input_not_inplaceable(n, other_index) for n in binary_nodes):
            return False
        if any(
            n.args[other_index].op in ["placeholder", "output"] for n in binary_nodes
        ):
            return False
        return True

    return fn


def _register_binary_unary_fusion_lowering(
    pattern,
    computation_op,
    binary_op,
    fusion_op,
    unary_attr=None,
):
    @_register_lowering_pattern_post_grad_pre_pass(
        pattern, extra_check=_is_valid_computation_binary(computation_op, binary_op)
    )
    def fn(match, *args, **kwargs):
        other = kwargs.get("other")
        assert isinstance(other, torch_ir.TensorBox)
        binary_attr = _binary_attr[binary_op]
        args_list = list(args)
        computation_args = [args_list[0], other] + args_list[1:-3] + [binary_attr]
        if len(args_list) > 6:
            if unary_attr is not None:
                computation_args += [
                    1.0,
                    unary_attr.op_name,
                    unary_attr.scalars_attr,
                    unary_attr.algorithm_attr,
                ]
            else:
                computation_args += [1.0, None, [], None]
        return L[fusion_op](*computation_args)

    return fn


computation_ops = [
    torch_ipex._convolution_pointwise.default,
    torch_ipex._linear_pointwise.default,
]


class UnaryAttr:
    def __init__(self, op_name: str, scalars_attr=None, algorithm_attr=None):
        self.op_name = op_name
        self.scalars_attr = scalars_attr if scalars_attr else []
        self.algorithm_attr = algorithm_attr if algorithm_attr else ""


def _register_unary_fusion():
    computation_call_fns = [_conv_call, _linear_call]

    def _unary_fusion_patterns():
        replacement_unary_fusion_patterns = {
            UnaryAttr("gelu", algorithm_attr="tanh"): [
                _unary_fusion_pattern(_gelu_fusion_2, call_fn, 4)
                for call_fn in computation_call_fns
            ],
            UnaryAttr("gelu", algorithm_attr="none"): [
                _unary_fusion_pattern(_gelu_fusion_1, call_fn, 2)
                for call_fn in computation_call_fns
            ],
            UnaryAttr("hardswish"): [
                _unary_fusion_pattern(_hardswish_fusion, call_fn, 2)
                for call_fn in computation_call_fns
            ],
            UnaryAttr("hardsigmoid"): [
                _unary_fusion_pattern(_hardsigmoid_fusion, call_fn, 1)
                for call_fn in computation_call_fns
            ],
            UnaryAttr("swish"): [
                _unary_fusion_pattern(_silu_fusion, call_fn, 2)
                for call_fn in computation_call_fns
            ],
        }
        call_user1 = [call_fn(users=1) for call_fn in computation_call_fns]
        replacement_unary_fusion_patterns.update(
            {
                UnaryAttr("relu"): [_combined_fusion(u, aten.relu) for u in call_user1],
                UnaryAttr("sigmoid"): [
                    _combined_fusion(u, aten.sigmoid) for u in call_user1
                ],
                UnaryAttr("tanh"): [_combined_fusion(u, aten.tanh) for u in call_user1],
            }
        )

        return replacement_unary_fusion_patterns

    replace_patterns = _unary_fusion_patterns()
    for unary_attr, patterns in replace_patterns.items():
        _register_unary_fusion_lowering(patterns[0], unary_attr, computation_ops[0])
        _register_unary_fusion_lowering(patterns[1], unary_attr, computation_ops[1])
    _leaky_relu_patterns = [
        _unary_fusion_pattern(_leaky_relu_fusion, call_fn, 3)
        for call_fn in computation_call_fns
    ]
    for pattern, computation_op in zip(_leaky_relu_patterns, computation_ops):
        _register_leaky_relu_fusion_lowering(pattern, computation_op)
    hardtanh_patterns = [
        _unary_fusion_pattern(_hardtanh_fusion, call_fn, 1)
        for call_fn in computation_call_fns
    ]
    for pattern, computation_op in zip(hardtanh_patterns, computation_ops):
        _register_hardtanh_fusion_lowering(pattern, computation_op)


def _can_be_inplace(_other):
    if isinstance(_other.data, torch_ir.View):
        return _can_be_inplace(_other.data)
    else:
        return not (
            isinstance(_other.data, torch_ir.ReinterpretView)
            or isinstance(
                _other.get_layout(), (torch_ir.MutationLayout, torch_ir.AliasedLayout)
            )
        )


def _register_binary_unary_maybe_inplace_fusion_lowering(
    pattern,
    computation_op,
    binary_op,
    inplace_fusion_op,
    outplace_fusion_op,
    unary_attr=None,
    other_index=None,
):
    @_register_lowering_pattern_post_grad_pre_pass(
        pattern,
        extra_check=_is_valid_computation_binary_inplace(
            computation_op, binary_op, other_index
        ),
    )
    def fn(match, *args, **kwargs):
        other = kwargs.get("other")
        assert isinstance(other, torch_ir.TensorBox)
        binary_attr = _binary_attr[binary_op]
        args_list = list(args)
        computation_args = [args_list[0], other] + args_list[1:-3] + [binary_attr]
        if len(args_list) > 6:
            if unary_attr is not None:
                computation_args += [
                    1.0,
                    unary_attr.op_name,
                    unary_attr.scalars_attr,
                    unary_attr.algorithm_attr,
                ]
            else:
                computation_args += [1.0, None, [], None]
        # Make sure the other is not an alias or mutation(fx side doesn't has such info).
        other.realize()
        if not _can_be_inplace(other):
            return L[outplace_fusion_op](*computation_args)
        return L[inplace_fusion_op](*computation_args)

    return fn


def _register_inplace_fusion():
    binary_ops = [aten.add, ops.add]
    inplace_fusion_op = torch_ipex._convolution_pointwise_.binary
    outplace_fusion_op = torch_ipex._convolution_pointwise.binary
    conv_call = _conv_call(users=1)
    conv_op = computation_ops[0]
    for binary_op in binary_ops:
        binary_v1 = _binary_fusion_v1(conv_call, binary_op)
        binary_unary_v1 = _combined_fusion(binary_v1, aten.relu)
        _register_binary_unary_maybe_inplace_fusion_lowering(
            binary_unary_v1,
            conv_op,
            binary_op,
            inplace_fusion_op,
            outplace_fusion_op,
            other_index=0,
            unary_attr=UnaryAttr("relu"),
        )
        _register_binary_unary_maybe_inplace_fusion_lowering(
            binary_v1,
            conv_op,
            binary_op,
            inplace_fusion_op,
            outplace_fusion_op,
            other_index=0,
        )
        binary_v2 = _binary_fusion_v2(conv_call, binary_op)
        binary_unary_v2 = _combined_fusion(binary_v2, aten.relu)
        _register_binary_unary_maybe_inplace_fusion_lowering(
            binary_unary_v2,
            conv_op,
            binary_op,
            inplace_fusion_op,
            outplace_fusion_op,
            other_index=1,
            unary_attr=UnaryAttr("relu"),
        )
        _register_binary_unary_maybe_inplace_fusion_lowering(
            binary_v2,
            conv_op,
            binary_op,
            inplace_fusion_op,
            outplace_fusion_op,
            other_index=1,
        )


def _register_binary_fusion():
    binary_ops = [aten.add, ops.add, aten.sub, ops.sub]
    fusion_ops = [
        torch_ipex._convolution_pointwise.binary,
        torch_ipex._linear_pointwise.binary,
    ]
    # TODO: Decide choose global computation ops or local for each pas
    computation_ops = [
        torch_ipex._convolution_pointwise.default,
        torch_ipex._linear_pointwise.default,
        None,  # Place holder for now, may be conv_transposed in future
    ]
    _computation_user_1 = [_conv_call(users=1), _linear_call(users=1)]
    for computation_call, computation_op, fusion_op in zip(
        _computation_user_1, computation_ops[:-1], fusion_ops
    ):
        for binary_op in binary_ops:
            pattern = _binary_fusion_v2(computation_call, binary_op)
            _register_binary_unary_fusion_lowering(
                pattern, computation_op, binary_op, fusion_op
            )

        for binary_op in [aten.add, ops.add]:
            pattern = _binary_fusion_v1(computation_call, binary_op)
            _register_binary_unary_fusion_lowering(
                pattern, computation_op, binary_op, fusion_op
            )


def _register_binary_unary_fusion():
    binary_ops = [aten.add, ops.add, aten.sub, ops.sub]
    fusion_ops = [torch_ipex._convolution_pointwise.binary]
    computation_ops = [
        torch_ipex._convolution_pointwise.default,
        torch_ipex._linear_pointwise.default,
        None,  # placeholder, override global computation_ops
    ]
    _computation_user_1 = [_conv_call(users=1)]
    for computation_call, computation_op, fusion_op in zip(
        _computation_user_1, computation_ops[:-1], fusion_ops
    ):
        for binary_op in binary_ops:
            pattern_v1 = _combined_fusion(
                _binary_fusion_v2(computation_call, binary_op), aten.relu
            )
            _register_binary_unary_fusion_lowering(
                pattern_v1,
                computation_op,
                binary_op,
                fusion_op,
                unary_attr=UnaryAttr("relu"),
            )
        for binary_op in [aten.add, ops.add]:
            pattern_v2 = _combined_fusion(
                _binary_fusion_v1(computation_call, binary_op), aten.relu
            )
            _register_binary_unary_fusion_lowering(
                pattern_v2,
                computation_op,
                binary_op,
                fusion_op,
                unary_attr=UnaryAttr("relu"),
            )


def _is_packable_convolution(match):
    """
    Check if the node is supported for MKLDNN convolution.
    """
    conv_node = match.output_node()
    input_meta_value = conv_node.args[0].meta.get("val")
    weight_meta_value = conv_node.args[1].meta.get("val")
    if input_meta_value is None or weight_meta_value is None:
        return False
    input_size = input_meta_value.shape
    if conv_node.args[1].op != "get_attr":
        return False
    for meta_value in [input_meta_value, weight_meta_value]:
        if (
            meta_value is None
            or meta_value.device.type != "xpu"
            or meta_value.dim() != 4
        ):
            return False
    is_transposed = conv_node.args[-3]
    if is_transposed:
        # TODO: add xpu support for deconv fusion
        return False
    return True


def _is_packable_linear(match):
    """
    Check if the node is supported for MKLDNN linear.
    """
    linear_node = match.output_node()
    # weight_idx is 1 for aten.mm and is 2 for aten.addmm
    weight_idx = 2 if linear_node.target == aten.addmm.default else 1
    if linear_node.args[weight_idx].op != "get_attr":
        return False
    input_meta_value = linear_node.args[weight_idx - 1].meta.get("val")
    weight_meta_value = linear_node.args[weight_idx].meta.get("val")
    if input_meta_value is None or weight_meta_value is None:
        return False
    batch_size = input_meta_value.shape[0]
    # for fp32, mkl should be enabled and batch_size should not be a free symbol.
    if free_symbols(batch_size):
        return False
    for meta_value in [input_meta_value, weight_meta_value]:
        if (
            meta_value is None
            or meta_value.device.type != "xpu"
            or meta_value.dim() != 2
        ):
            return False
    if weight_idx == 2:
        bias_meta_value = linear_node.args[0].meta.get("val")
        if (
            bias_meta_value is None
            or meta_value.device.type != "xpu"
            or bias_meta_value.dim() != 1
            or bias_meta_value.size(0) != weight_meta_value.size(1)
        ):
            return False

    return True


_aten_conv_args = (
    Arg(),
    Arg(),
    Arg(),
    Arg(),
    Arg(),
    Arg(),
    KeywordArg("is_transposed"),
    Arg(),
    Arg(),
)


def _register_weight_pack_pass():
    @register_freezing_graph_pattern(
        CallFunction(aten.convolution.default, *_aten_conv_args),
        extra_check=_is_packable_convolution,
    )
    def convolution(match, *args, **kwargs):
        is_transposed = kwargs.get("is_transposed")
        assert isinstance(is_transposed, bool)
        graph = match.graph
        conv_node = match.output_node()
        input_size = conv_node.args[0].meta.get("val").shape
        with graph.inserting_before(conv_node):
            constant_args = [args[4], args[3], args[5], args[-1]]
            packed_conv_op = torch_ipex._convolution_pointwise.default
            packed_weight_node = args[1]
            packed_conv_inputs = (
                (args[0], packed_weight_node, args[2])
                + tuple(constant_args)
                + ("none", [], "")
            )
            packed_conv_node = graph.create_node(
                "call_function", packed_conv_op, tuple(packed_conv_inputs)
            )
            conv_node.replace_all_uses_with(packed_conv_node)
            packed_conv_node.meta.update(conv_node.meta)
            graph.erase_node(conv_node)

    @register_freezing_graph_pattern(
        CallFunction(aten.addmm.default, Arg(), Arg(), Arg()),
        extra_check=_is_packable_linear,
    )
    @register_freezing_graph_pattern(
        CallFunction(aten.mm.default, Arg(), Arg()),
        extra_check=_is_packable_linear,
    )
    def linear(match, *args, **kwargs):
        graph = match.graph
        linear_node = match.output_node()
        input = args[0] if linear_node.target == aten.mm.default else args[1]
        bias = None if linear_node.target == aten.mm.default else args[0]
        weight = args[1] if linear_node.target == aten.mm.default else args[2]
        with graph.inserting_before(linear_node):
            weight_dtype = weight.meta.get("val").dtype
            batch_size = input.meta.get("val").shape[0]
            packed_linear_inputs: Tuple[Any, ...] = (input, weight)
            packed_linear_op = torch_ipex._linear_pointwise.default
            packed_linear_inputs += (bias, "none", [], "")
            packed_linear_node = graph.create_node(
                "call_function", packed_linear_op, packed_linear_inputs
            )
            linear_node.replace_all_uses_with(packed_linear_node)
            packed_linear_node.meta.update(linear_node.meta)
            graph.erase_node(linear_node)


@functools.lru_cache(None)
def _ipex_fusion_init():
    _register_unary_fusion()
    _register_inplace_fusion()
    _register_binary_unary_fusion()
    _register_binary_fusion()


@functools.lru_cache(None)
def _ipex_weight_pack_init():
    _register_weight_pack_pass()
