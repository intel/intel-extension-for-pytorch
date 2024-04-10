import enum
import json
from collections import OrderedDict
from typing import Callable, Optional
import inspect
import numbers

import torch
import torch.nn as nn
from torch import _VF
import torch.nn.functional as F
from torch.quantization.qconfig import QConfig
from intel_extension_for_pytorch.nn.functional import interaction

from ._quantization_state_utils import QTensorInfo
from ._smooth_quant import SmoothQuantActivationObserver, SmoothQuantWeightObserver
from ._qconfig import QConfigSmoothQuant
from intel_extension_for_pytorch.nn.modules import MergedEmbeddingBagWithCat

add_and_mul_ops = set(
    [
        torch.add,
        torch.Tensor.add,
        # TODO, mul op?
    ]
)

quantized_modules_has_weights = set(
    [
        torch.nn.Conv2d,
        torch.nn.Conv3d,
        torch.nn.Linear,
        torch.nn.EmbeddingBag,
        MergedEmbeddingBagWithCat,
        torch.nn.ConvTranspose2d,
        torch.nn.ConvTranspose3d,
        torch.nn.LSTM,
    ]
)

# those ops only support int8->int8, not int8->fp32/bf16
int8_int8_ops = set(
    [
        str(F.adaptive_avg_pool2d),
        str(F.adaptive_avg_pool3d),
        str(F.avg_pool2d),
        str(F.avg_pool3d),
        str(F.max_pool2d),
        str(F.max_pool3d),
        str(nn.MaxPool2d),
        str(nn.MaxPool3d),
        str(nn.AvgPool2d),
        str(nn.AvgPool3d),
        str(nn.AdaptiveAvgPool2d),
        str(nn.AdaptiveAvgPool3d),
        str(torch.Tensor.relu),
        str(torch.relu),
        str(F.relu),
        str(nn.ReLU),
        str(torch.flatten),
        str(torch.Tensor.flatten),
        str(torch.nn.Flatten),
        # the following op will be supported at next step.
        # str(torch.Tensor.sigmoid),
        # str(torch.sigmoid),
        # str(F.sigmoid),
        # str(nn.Sigmoid),
        # str(F.gelu),
        # str(nn.GELU),
        # ipex customer op
        str(interaction),
        str(torch.ops.torch_ipex.interaction_forward),
        str(torch.ops.torch_ipex.merged_embeddingbag_cat_forward),
        str(torch.embedding_bag),
        str(F.embedding_bag),
        str(torch.nn.EmbeddingBag),
        str(MergedEmbeddingBagWithCat),
        str(torch.nn.LSTM),
    ]
)


class OpQuantizeabilityType(enum.Enum):
    QUANTIZEABLE = 0
    NOT_QUANTIZEABLE = 1


class FuncOutputObsType(enum.Enum):
    NONE = 0
    NEW_OBS = 1
    REUSES_FIRST_INPUT_OBS = 2


# quantizeable modules
def is_leaf(
    m: torch.nn.Module,
) -> bool:
    # TODO(future PR): extend to the rest of the container classes
    container_classes = (
        torch.nn.Sequential,
        torch.nn.ModuleList,
    )
    return m.__module__.startswith("torch.nn") and (
        not isinstance(m, container_classes)
    )


def get_fqn_valid_for_module_dict_key(fqn: str) -> str:
    """
    Modifies `fqn`(fully qualified name) to make it a valid key to a ModuleDict.
    """
    if fqn == "":
        fqn = " "
    return fqn.replace(".", ":")


class HookType(enum.Enum):
    """
    Describes the various types of function and module hooks that are used
    to implement quantization syntax transforms.
    """

    # Hooks which are run before, during and after a quantizeable op.
    # Usually used for op input and output observation, subsituating
    # quantized kernels, and dynamically looking up arguments to quantized
    # kernels.
    OP_HOOKS = 0
    # Hooks which are run before or after a `torch.nn.Module` which
    # is a non-leaf. Usually used for dtype transforms if the user requests
    # that the inputs or outputs of a certain module are of some dtype.
    MODULE_IO_HOOKS = 1
    # Hooks which are run before a non-quantizeable op which requires
    # `torch.float` inputs. Any inputs which are not floats are converted
    # back to floats.
    ARG_DEQUANTS = 2
    # Everything else
    NONE = 2


def get_torch_function_hook_type(
    parent_module: Optional[torch.nn.Module],
    func: Callable,
) -> HookType:
    # the direct __dict__ accesses are for performance, because
    # the default `torch.nn.Module.__getattr__` has overhead.
    parent_module_has_qstate = (
        parent_module is not None and "_auto_quant_state" in parent_module.__dict__
    )
    needs_op_hooks = parent_module_has_qstate and parent_module.__dict__[
        "_auto_quant_state"
    ].cur_op_needs_hooks(
        func
    )  # type: ignore[union-attr, operator]

    if needs_op_hooks:
        return HookType.OP_HOOKS
    elif parent_module_has_qstate:
        return HookType.ARG_DEQUANTS
    else:
        return HookType.NONE


def get_module_hook_type(
    parent_module: Optional[torch.nn.Module],
    cur_module: torch.nn.Module,
) -> HookType:
    cached_hook_type = getattr(cur_module, "_auto_quant_module_hook_type", None)
    if cached_hook_type is not None:
        return cached_hook_type
    parent_module_has_qstate = (
        parent_module is not None and "_auto_quant_state" in parent_module.__dict__
    )
    needs_op_hooks = parent_module_has_qstate and parent_module.__dict__[
        "_auto_quant_state"
    ].cur_op_needs_hooks(
        cur_module
    )  # type: ignore[union-attr, operator]
    # We need IO hooks if
    # * we are calling forward on a module (always True here)
    # * that module has quant state
    # * that module does not need op hooks for the parent
    needs_io_hooks = "_auto_quant_state" in cur_module.__dict__ and (not needs_op_hooks)
    needs_arg_dequants = parent_module_has_qstate and not needs_op_hooks

    if needs_op_hooks:
        result = HookType.OP_HOOKS
    elif needs_io_hooks:
        result = HookType.MODULE_IO_HOOKS
    elif needs_arg_dequants:
        result = HookType.ARG_DEQUANTS
    else:
        result = HookType.NONE
    cur_module._auto_quant_module_hook_type = result  # type: ignore[assignment]
    return result


def attach_scale_zp_values_to_model(
    module: torch.nn.Module,
) -> None:
    """
    Calculates the scale and zero_point from each observer and attaches
    these values to the parent module. This is done to avoid recalculating
    these values at inference.
    """
    if hasattr(module, "_auto_quant_state"):
        qstate: AutoQuantizationState = module._auto_quant_state  # type: ignore[assignment]
        quantized_dtype = [torch.quint8, torch.qint8]
        for tensor_id, observer in qstate.tensor_id_to_observer.items():
            if observer.dtype in quantized_dtype:
                scale, zp = observer.calculate_qparams()
                qstate.tensor_id_to_scale_zp[int(tensor_id)] = (scale, zp)
            else:
                AssertionError(
                    False
                ), "The observer's dtype only can be torch.quint8 or torch.qint8"
        for tensor_id, observer in qstate.weight_tensor_id_to_observer.items():
            if observer.dtype in quantized_dtype:
                scale, zp = observer.calculate_qparams()
                qstate.weight_tensor_id_to_scale_zp[tensor_id] = (scale, zp)
            else:
                AssertionError(
                    False
                ), "The observer's dtype only can be torch.quint8 or torch.qint8"
        _attach_smooth_quant_scaling_factor_to_model(module)
        qstate.tensor_id_to_observer.clear()
        qstate.weight_tensor_id_to_observer.clear()

    for _, child in module.named_children():
        attach_scale_zp_values_to_model(child)


def _check_observer_has_run(observer):
    if observer.min_val.numel() == 0 or observer.max_val.numel() == 0:
        return False
    if (
        (observer.min_val.dim() == 0 or observer.max_val.dim() == 0)
        and observer.min_val == float("inf")
        and observer.max_val == float("-inf")
    ):
        return False
    return True


def check_model_obsever_has_run(
    module: torch.nn.Module,
) -> None:
    """
    This function is about check whether the module's observer has been run by checking the
        observer's min_value and max_max_value is the init value or not.
    Rules:
    - If no observer found, return false.
    - If any observer returns true or false, return that value.
    - To avoid wrong results in case no observer found at the beginning, only return false
        if all checks of the module and its children return false.
    """
    if hasattr(module, "_auto_quant_state"):
        qstate: AutoQuantizationState = module._auto_quant_state  # type: ignore[assignment]
        quantized_dtype = [torch.quint8, torch.qint8]
        for tensor_id, observer in qstate.tensor_id_to_observer.items():
            if observer.dtype in quantized_dtype:
                return _check_observer_has_run(observer)
            else:
                AssertionError(
                    False
                ), "The observer's dtype only can be torch.quint8 or torch.qint8"
        for tensor_id, observer in qstate.weight_tensor_id_to_observer.items():
            if observer.dtype in quantized_dtype:
                return _check_observer_has_run(observer)
            else:
                AssertionError(
                    False
                ), "The observer's dtype only can be torch.quint8 or torch.qint8"

    for _, child in module.named_children():
        if check_model_obsever_has_run(child):
            return True

    return False


def attach_op_convert_info_to_model(
    module: torch.nn.Module,
) -> None:
    """
    Calculates the info needed to convert each op and attaches
    it to the parent module. This is done to avoid recalculating these values
    at inference.
    """
    if hasattr(module, "_auto_quant_state"):
        qstate: AutoQuantizationState = module._auto_quant_state  # type: ignore[assignment]
        for _, seen_q_op_info in qstate.idx_to_seen_q_op_infos.items():
            qstate.idx_to_op_convert_info[seen_q_op_info.idx] = (
                qstate.calculate_op_convert_info(seen_q_op_info)
            )
            qstate.idx_to_op_weight_convert_info[seen_q_op_info.idx] = (
                qstate.calculate_op_weight_convert_info(seen_q_op_info)
            )
        _map_smooth_quant_info_to_idx(module)

    for _, child in module.named_children():
        attach_op_convert_info_to_model(child)


class Node:
    def __init__(
        self,
        op_infos,
        input_scale_zero=None,
        weight_scale_zero=None,
        output_scale_zero=None,
        qconfig=None,
    ):
        self.idx = op_infos.idx if hasattr(op_infos, "idx") else None
        self.type = op_infos.type
        self.fqn = op_infos.fqn
        self.input_tensor_infos = op_infos.input_tensor_infos
        self.input_tensor_force_inf_dtype = (
            [] if qconfig is None else op_infos.input_tensor_force_inf_dtype
        )
        self.output_tensor_infos = op_infos.output_tensor_infos
        self.insert_fake_quant_after_outputs = (
            [] if qconfig is None else op_infos.insert_fake_quant_after_outputs
        )
        self.weight_tensor_infos = (
            [] if qconfig is None else op_infos.weight_tensor_infos
        )
        self.qconfig = qconfig
        self.input_scale_zero = input_scale_zero
        self.weight_scale_zero = weight_scale_zero
        self.output_scale_zero = output_scale_zero
        self.pre_nodes = []
        self.post_nodes = []


from collections import namedtuple

ParentNode = namedtuple(typename="ParentNode", field_names=["output_info"])


def convert_quant_state_map_to_nodes(quant_state_map):
    nodes = []
    # step1: create nodes
    for _, v in quant_state_map.items():
        new_parent_node = ParentNode(v.output_qtensor_infos)
        nodes.append(new_parent_node)
        for idx, q_op_info in v.idx_to_seen_q_op_infos.items():
            input_scale_zero = {}
            weight_scale_zero = {}
            output_scale_zero = {}
            for tensor_info in q_op_info.input_tensor_infos:
                if (
                    tensor_info is not None
                    and tensor_info.id in v.tensor_id_to_scale_zp
                ):
                    input_scale_zero[tensor_info.id] = v.tensor_id_to_scale_zp[
                        tensor_info.id
                    ]
            for tensor_info in q_op_info.weight_tensor_infos:
                if tensor_info is not None and (
                    str(idx) + "_" + str(tensor_info.id)
                    in v.weight_tensor_id_to_scale_zp
                ):
                    weight_id = str(idx) + "_" + str(tensor_info.id)
                    weight_scale_zero[weight_id] = v.weight_tensor_id_to_scale_zp[
                        weight_id
                    ]
            for tensor_info in q_op_info.output_tensor_infos:
                if (
                    tensor_info is not None
                    and tensor_info.id in v.tensor_id_to_scale_zp
                ):
                    output_scale_zero[tensor_info.id] = v.tensor_id_to_scale_zp[
                        tensor_info.id
                    ]
            new_node = Node(
                q_op_info,
                input_scale_zero,
                weight_scale_zero,
                output_scale_zero,
                qconfig=q_op_info.qconfig,
            )
            nodes.append(new_node)
        for nonq_op_infos in v.seen_nonq_op_infos:
            new_node = Node(nonq_op_infos)
            nodes.append(new_node)
    # create connection between nodess
    for cur in nodes:
        if isinstance(cur, ParentNode):
            continue
        for n in nodes:
            if isinstance(n, ParentNode):
                continue
            # find pre_node:
            for input_info in cur.input_tensor_infos:
                if input_info in n.output_tensor_infos:
                    cur.pre_nodes.append(n)
            # find nex_node:
            for output_info in cur.output_tensor_infos:
                if output_info in n.input_tensor_infos:
                    cur.post_nodes.append(n)
    return nodes


def sync_pool_and_lstm_input_output_scale_zp(quant_state_map, nodes):
    pool_ops = [
        str(F.adaptive_avg_pool2d),
        str(F.adaptive_avg_pool3d),
        str(F.avg_pool2d),
        str(F.avg_pool3d),
        str(F.max_pool2d),
        str(F.max_pool3d),
        str(nn.MaxPool2d),
        str(nn.MaxPool3d),
        str(nn.AvgPool2d),
        str(nn.AvgPool3d),
        str(nn.AdaptiveAvgPool2d),
        str(nn.AdaptiveAvgPool3d),
    ]
    shape_ops = [str(torch.flatten), str(torch.nn.Flatten), str(torch.Tensor.flatten)]
    rnn_ops = [str(torch.nn.LSTM)]

    def _sync_scale_zp_given_id(quant_state_map, id, scale_zp):
        for _, v in quant_state_map.items():
            if id in v.tensor_id_to_scale_zp:
                v.tensor_id_to_scale_zp[id] = scale_zp

    def _find_sync_op_from_given_node(cur_node, ids):
        for next in cur_node.post_nodes:
            if next.type in (pool_ops + shape_ops + rnn_ops):
                ids.append(next.output_tensor_infos[0].id)
                _find_sync_op_from_given_node(next, ids)

    for node in nodes:
        if isinstance(node, ParentNode):
            continue
        if node.qconfig is not None and node.type in (pool_ops + rnn_ops):
            if node.input_scale_zero == node.output_scale_zero:
                continue
            sync_node_begin = node
            # fistly, find the fist sync op before the cur pooling(lstm) op.
            # like: pooling->pool->shape->cur_node,
            while len(sync_node_begin.pre_nodes) == 1 and sync_node_begin.pre_nodes[
                0
            ].type in (pool_ops + shape_ops + rnn_ops):
                sync_node_begin = sync_node_begin.pre_nodes[0]
            tensor_ids = [sync_node_begin.output_tensor_infos[0].id]
            scale_zp = sync_node_begin.input_scale_zero[
                sync_node_begin.input_tensor_infos[0].id
            ]
            # find all need sync op from sync_node_begin.
            _find_sync_op_from_given_node(sync_node_begin, tensor_ids)
            for id in tensor_ids:
                _sync_scale_zp_given_id(quant_state_map, id, scale_zp)


def _check_after_nodes_all_quantized_give_node(node):
    r"""
    This function is about check whether given node's post nodes are all quantized,
    """
    if len(node.post_nodes) == 0:
        return False
    else:
        # make sure all post nodes are quantizabled.
        for next in node.post_nodes:
            if next.qconfig is None:
                if next.type == str(nn.Identity):
                    return _check_after_nodes_all_quantized_give_node(next)
                else:
                    return False
            else:
                int8_int8_symmetric_ops = [
                    str(interaction),
                    str(torch.ops.torch_ipex.interaction_forward),
                    str(torch.ops.torch_ipex.merged_embeddingbag_cat_forward),
                    str(torch.embedding_bag),
                    str(F.embedding_bag),
                    str(torch.nn.EmbeddingBag),
                    str(MergedEmbeddingBagWithCat),
                ]
                if next.type in int8_int8_symmetric_ops:
                    if next.type in [
                        str(interaction),
                        str(torch.ops.torch_ipex.interaction_forward),
                        str(torch.ops.torch_ipex.merged_embeddingbag_cat_forward),
                    ]:
                        # node.input_tensor_infos may be set, we can use force_inf_dtype to check whether this op is quantizabled.
                        for force_inf_dtype in next.input_tensor_force_inf_dtype:
                            if force_inf_dtype != torch.qint8:
                                return False
                    else:
                        # embeddingBag
                        if next.weight_tensor_infos[0].inf_dtype != torch.qint8:
                            return False
                else:
                    for force_inf_dtype in next.input_tensor_force_inf_dtype:
                        if force_inf_dtype not in [torch.qint8, torch.quint8]:
                            return False
        # all post nodes are quantizabled.
        return True


def set_node_output_quantized(nodes):
    r"""
    # For interation, EmbeddingBag, MergedEmbWithCat, pooling and elt_wise,
    # we only support INT8->INT*, if those ops have quantized their inputs,
    # we need make sure their output also have falk quant to make them call in INT8 kernel.
    # this function will check whether the output inf dtype is int8 dtype if its' input is set to quantized, if the
    # output's infe dtype is not int8, set it and also set insert_fake_quant_after_output to True.
    """

    def _reset_post_node_input_infos(node):
        # make sure the post node will insert fake quant if we add fake quant by cur node' output
        if len(node.post_nodes) > 0:
            for post_node in node.post_nodes:
                if post_node.qconfig is not None:
                    for idx, tensor_info in enumerate(post_node.input_tensor_infos):
                        if tensor_info in node.output_tensor_infos:
                            post_node.input_tensor_force_inf_dtype[idx] = (
                                tensor_info.orig_dtype
                            )
                elif post_node.type == str(nn.Identity):
                    _reset_post_node_input_infos(post_node)

    for node in nodes:
        if isinstance(node, ParentNode):
            continue
        if node.qconfig is not None and node.type in int8_int8_ops:
            post_node_are_quantized = _check_after_nodes_all_quantized_give_node(node)
            if node.type in (
                str(torch.nn.EmbeddingBag),
                str(MergedEmbeddingBagWithCat),
            ):
                if (
                    node.weight_tensor_infos[0].inf_dtype == torch.qint8
                    and not post_node_are_quantized
                ):
                    node.output_tensor_infos[0].inf_dtype = torch.qint8
                    node.insert_fake_quant_after_outputs[0] = True
                    _reset_post_node_input_infos(node)
            elif node.type == str(F.embedding_bag):
                if (
                    node.input_tensor_force_inf_dtype[1] == torch.qint8
                    and not post_node_are_quantized
                ):
                    node.output_tensor_infos[0].inf_dtype = torch.qint8
                    node.insert_fake_quant_after_outputs[0] = True
                    _reset_post_node_input_infos(node)
            elif node.type in [
                str(interaction),
                str(torch.ops.torch_ipex.interaction_forward),
                str(torch.ops.torch_ipex.merged_embeddingbag_cat_forward),
            ]:
                if (
                    node.input_tensor_force_inf_dtype[0] == torch.qint8
                    and not post_node_are_quantized
                ):
                    node.output_tensor_infos[0].inf_dtype = torch.qint8
                    node.insert_fake_quant_after_outputs[0] = True
                    _reset_post_node_input_infos(node)
            else:
                # TODO: enable PackedSequence input for LSTM.
                if not (
                    node.type in [nn.LSTM]
                    and len(node.input_tensor_infos) > 2
                    and node.input_tensor_infos[1].orig_dtype == torch.int64
                ):
                    if (
                        node.input_tensor_force_inf_dtype[0]
                        in [torch.qint8, torch.quint8]
                        and not post_node_are_quantized
                    ):
                        node.output_tensor_infos[0].inf_dtype = (
                            node.input_tensor_force_inf_dtype[0]
                        )
                        node.insert_fake_quant_after_outputs[0] = True
                        _reset_post_node_input_infos(node)


qscheme_dict = {
    str(torch.per_tensor_affine): torch.per_tensor_affine,
    str(torch.per_tensor_symmetric): torch.per_tensor_symmetric,
    str(torch.per_channel_affine): torch.per_channel_affine,
    str(torch.per_channel_symmetric): torch.per_channel_symmetric,
}

dtype_dict = {
    str(torch.quint8): torch.quint8,
    str(torch.qint8): torch.qint8,
    str(torch.float32): torch.float32,
    str(torch.float64): torch.float64,
    str(torch.float16): torch.float16,
    str(torch.bfloat16): torch.bfloat16,
    str(torch.complex64): torch.complex64,
    str(torch.complex128): torch.complex128,
    str(torch.int16): torch.int16,
    str(torch.int32): torch.int32,
    str(torch.int64): torch.int64,
    str(torch.bool): torch.bool,
    str(torch.uint8): torch.uint8,
    str(torch.int8): torch.int8,
    str(torch.quint4x2): torch.quint4x2,
}

IPEX_OBSERVERS = {
    "SmoothQuantActivationObserver": SmoothQuantActivationObserver,
    "SmoothQuantWeightObserver": SmoothQuantWeightObserver,
}


def _get_observer_setting(observer):
    r"""
    Convert torch observer's args to dict for saving to json file.
    Because json only accept number or string, so we will convert some
    class type to string(dtype, qscheme, other class type?).
    """
    observer_setting = OrderedDict()
    observer_setting["name"] = observer.__class__.__name__
    # get observer arg name
    observer_args = inspect.signature(observer.__init__).parameters
    observer_dict = observer.__dict__
    # Now we only can save number or string to json file.
    for arg_name in observer_args.keys():
        if arg_name in observer_dict:
            if isinstance(observer_dict[arg_name], numbers.Number):
                observer_setting[arg_name] = observer_dict[arg_name]
            elif arg_name == "qscheme" or arg_name == "dtype":
                observer_setting[arg_name] = str(observer_dict[arg_name])
            elif (
                arg_name == "eps"
                and hasattr(observer, "eps")
                and isinstance(observer.eps, torch.Tensor)
                and observer.eps.numel() == 1
            ):
                observer_setting[arg_name] = observer.eps.item()
    return observer_setting


def _create_observer(setting):
    r"""
    Create torch observer according to user's setting.
    """
    if "qscheme" in setting:
        setting["qscheme"] = qscheme_dict[setting["qscheme"]]
    if "dtype" in setting:
        setting["dtype"] = dtype_dict[setting["dtype"]]

    if hasattr(torch.quantization.observer, setting["name"]):
        observer = getattr(torch.quantization.observer, setting["name"])
        setting.pop("name", None)
        return observer.with_args(**setting)
    elif setting["name"] in IPEX_OBSERVERS:
        observer = IPEX_OBSERVERS[setting["name"]]
        setting.pop("name", None)
        # SmoothQuant observers contain sub-observers
        smooth_quant_sub_obs_keys = [
            "act_observer",
            "act_ic_observer",
            "wei_observer",
            "wei_ic_observer",
        ]
        for key in smooth_quant_sub_obs_keys:
            if key in setting:
                setting[key] = _create_observer(setting[key])
        return observer.with_args(**setting)
    else:
        raise NameError("torch.quantization.observer %s not found" % setting["name"])


def save_quant_state(quant_state_map, configure_file):
    # save qparam's as json file for tunning
    quant_state_dict = OrderedDict()
    for k, v in quant_state_map.items():
        layer_infos = OrderedDict()
        if len(v.idx_to_seen_q_op_infos) == 0:
            layer_infos["q_op_infos"] = {}
        else:
            q_op_infos = OrderedDict()
            for q_k, op_info in v.idx_to_seen_q_op_infos.items():
                info = OrderedDict()
                info["op_type"] = op_info.type
                info["op_type_is_module"] = op_info.type_is_module
                info["fqn"] = op_info.fqn
                input_tensor_infos = []
                smooth_quant_enabled = False
                for tensor_info, force_dtype in zip(
                    op_info.input_tensor_infos, op_info.input_tensor_force_inf_dtype
                ):
                    cur_tensor_infos = {}

                    if tensor_info is not None:
                        cur_tensor_infos["id"] = tensor_info.id
                        cur_tensor_infos["orig_dtype"] = str(tensor_info.orig_dtype)
                        cur_tensor_infos["inf_dtype"] = str(tensor_info.inf_dtype)
                        cur_tensor_infos["force_dtype"] = str(force_dtype)
                        if tensor_info.id in v.tensor_id_to_scale_zp:
                            if isinstance(
                                v.tensor_id_to_scale_zp[tensor_info.id][0], torch.Tensor
                            ):
                                cur_tensor_infos["scale"] = v.tensor_id_to_scale_zp[
                                    tensor_info.id
                                ][0].tolist()
                                cur_tensor_infos["zero_point"] = (
                                    v.tensor_id_to_scale_zp[tensor_info.id][1].tolist()
                                )
                            else:
                                scales_dict = v.tensor_id_to_scale_zp[tensor_info.id][0]
                                zp_dict = v.tensor_id_to_scale_zp[tensor_info.id][1]
                                assert isinstance(scales_dict, dict) and isinstance(
                                    zp_dict, dict
                                )
                                scales_to_save = {}
                                zp_to_save = {}
                                for key, val in scales_dict.items():
                                    scales_to_save.update({key: val.tolist()})
                                for key, val in zp_dict.items():
                                    zp_to_save.update({key: val.tolist()})
                                cur_tensor_infos["scale"] = scales_to_save
                                cur_tensor_infos["zero_point"] = zp_to_save
                        if (
                            str(tensor_info.id)
                            in v.tensor_id_to_smooth_quant_scaling_factor
                            and v.tensor_id_to_smooth_quant_scaling_factor[
                                str(tensor_info.id)
                            ]
                            is not None
                        ):
                            scaling_factor_dict = (
                                v.tensor_id_to_smooth_quant_scaling_factor[
                                    str(tensor_info.id)
                                ]
                            )
                            assert isinstance(scaling_factor_dict, dict)
                            scaling_factors_to_save = {}
                            for key, val in scaling_factor_dict.items():
                                scaling_factors_to_save.update({key: val.tolist()})
                            cur_tensor_infos["smooth_quant_scaling_factor"] = (
                                scaling_factors_to_save
                            )
                            smooth_quant_enabled = True
                    input_tensor_infos.append(cur_tensor_infos)
                info["input_tensor_infos"] = input_tensor_infos
                # weight infos
                weight_tensor_infos = []
                for tensor_info in op_info.weight_tensor_infos:
                    cur_tensor_infos = {}
                    if tensor_info is not None:
                        cur_tensor_infos["orig_dtype"] = str(tensor_info.orig_dtype)
                        cur_tensor_infos["inf_dtype"] = str(tensor_info.inf_dtype)
                        weight_idx = str(op_info.idx) + "_" + str(tensor_info.id)
                        if weight_idx in v.weight_tensor_id_to_scale_zp:
                            cur_tensor_infos["scale"] = v.weight_tensor_id_to_scale_zp[
                                weight_idx
                            ][0].tolist()
                            cur_tensor_infos["zero_point"] = (
                                v.weight_tensor_id_to_scale_zp[weight_idx][1].tolist()
                            )
                        if (
                            weight_idx
                            in v.weight_tensor_id_to_smooth_quant_scaling_factor
                        ):
                            if (
                                v.weight_tensor_id_to_smooth_quant_scaling_factor[
                                    weight_idx
                                ]
                                is not None
                            ):
                                cur_tensor_infos["smooth_quant_scaling_factor"] = (
                                    v.weight_tensor_id_to_smooth_quant_scaling_factor[
                                        weight_idx
                                    ].tolist()
                                )
                    weight_tensor_infos.append(cur_tensor_infos)
                info["weight_tensor_infos"] = weight_tensor_infos
                # output infos
                output_tensor_infos = []
                for tensor_info in op_info.output_tensor_infos:
                    cur_tensor_infos = {}
                    if tensor_info is not None:
                        cur_tensor_infos["id"] = tensor_info.id
                        cur_tensor_infos["orig_dtype"] = str(tensor_info.orig_dtype)
                        cur_tensor_infos["inf_dtype"] = str(tensor_info.inf_dtype)
                        if tensor_info.id in v.tensor_id_to_scale_zp:
                            if isinstance(
                                v.tensor_id_to_scale_zp[tensor_info.id][0], torch.Tensor
                            ):
                                cur_tensor_infos["scale"] = v.tensor_id_to_scale_zp[
                                    tensor_info.id
                                ][0].tolist()
                                cur_tensor_infos["zero_point"] = (
                                    v.tensor_id_to_scale_zp[tensor_info.id][1].tolist()
                                )
                            else:
                                scales_dict = v.tensor_id_to_scale_zp[tensor_info.id][0]
                                zp_dict = v.tensor_id_to_scale_zp[tensor_info.id][1]
                                assert isinstance(scales_dict, dict) and isinstance(
                                    zp_dict, dict
                                )
                                scales_to_save = {}
                                zp_to_save = {}
                                for key, val in scales_dict.items():
                                    scales_to_save.update({key: val.tolist()})
                                for key, val in zp_dict.items():
                                    zp_to_save.update({key: val.tolist()})
                                cur_tensor_infos["scale"] = scales_to_save
                                cur_tensor_infos["zero_point"] = zp_to_save
                        if (
                            str(tensor_info.id)
                            in v.tensor_id_to_smooth_quant_scaling_factor
                        ):
                            scaling_factors = (
                                v.tensor_id_to_smooth_quant_scaling_factor[
                                    str(tensor_info.id)
                                ]
                            )
                            scaling_factors_to_save = None
                            if scaling_factors is not None:
                                assert isinstance(
                                    scaling_factors, dict
                                ), f"Expect scaling factors is a dict but found {type(scaling_factors)}"
                                scaling_factors_to_save = {}
                                for key, val in scaling_factors.items():
                                    scaling_factors_to_save.update({key: val.tolist()})
                            cur_tensor_infos["smooth_quant_scaling_factor"] = (
                                scaling_factors_to_save
                            )
                    output_tensor_infos.append(cur_tensor_infos)
                info["output_tensor_infos"] = output_tensor_infos
                # qconfig
                info["activation_observer"] = _get_observer_setting(
                    op_info.qconfig.activation()
                )
                if isinstance(
                    op_info.qconfig.activation(), SmoothQuantActivationObserver
                ):
                    info["activation_observer"][
                        "smooth_quant_enabled"
                    ] = smooth_quant_enabled
                    info["activation_observer"]["act_observer"] = _get_observer_setting(
                        op_info.qconfig.activation().act_obs
                    )
                    info["activation_observer"]["act_ic_observer"] = (
                        _get_observer_setting(op_info.qconfig.activation().ic_obs)
                    )
                    info["share_weight_observers"] = getattr(
                        op_info.qconfig, "share_weight_observers", True
                    )
                info["weight_observer"] = _get_observer_setting(
                    op_info.qconfig.weight()
                )
                if isinstance(op_info.qconfig.weight(), SmoothQuantWeightObserver):
                    info["weight_observer"][
                        "smooth_quant_enabled"
                    ] = smooth_quant_enabled
                    info["weight_observer"]["wei_observer"] = _get_observer_setting(
                        op_info.qconfig.weight().oc_obs
                    )
                    info["weight_observer"]["wei_ic_observer"] = _get_observer_setting(
                        op_info.qconfig.weight().ic_obs
                    )
                q_op_infos[q_k] = info
            layer_infos["q_op_infos"] = q_op_infos
        if len(v.seen_nonq_op_infos) == 0:
            layer_infos["nonq_op_infos"] = {}
        else:
            nonq_op_infos = OrderedDict()
            for non_q_k, op_info in enumerate(v.seen_nonq_op_infos):
                info = OrderedDict()
                info["op_type"] = op_info.type
                info["fqn"] = op_info.fqn
                input_tensor_infos = []
                for tensor_info in op_info.input_tensor_infos:
                    cur_tensor_infos = {}
                    if tensor_info is not None:
                        cur_tensor_infos["id"] = tensor_info.id
                        cur_tensor_infos["orig_dtype"] = str(tensor_info.orig_dtype)
                        cur_tensor_infos["inf_dtype"] = str(tensor_info.inf_dtype)
                    input_tensor_infos.append(cur_tensor_infos)
                info["input_tensor_infos"] = input_tensor_infos
                output_tensor_infos = []
                for tensor_info in op_info.output_tensor_infos:
                    cur_tensor_infos = {}
                    if tensor_info is not None:
                        cur_tensor_infos["id"] = tensor_info.id
                        cur_tensor_infos["orig_dtype"] = str(tensor_info.orig_dtype)
                        cur_tensor_infos["inf_dtype"] = str(tensor_info.inf_dtype)
                    output_tensor_infos.append(cur_tensor_infos)
                info["output_tensor_infos"] = output_tensor_infos
                nonq_op_infos[non_q_k] = info
            layer_infos["nonq_op_infos"] = nonq_op_infos
        layer_output_infos = []
        for tensor_info in v.output_qtensor_infos:
            cur_tensor_infos = {}
            if tensor_info is not None:
                cur_tensor_infos["id"] = tensor_info.id
                cur_tensor_infos["orig_dtype"] = str(tensor_info.orig_dtype)
                cur_tensor_infos["inf_dtype"] = str(tensor_info.inf_dtype)
                if tensor_info.id in v.tensor_id_to_scale_zp:
                    cur_tensor_infos["scale"] = v.tensor_id_to_scale_zp[tensor_info.id][
                        0
                    ].tolist()
                    cur_tensor_infos["zero_point"] = v.tensor_id_to_scale_zp[
                        tensor_info.id
                    ][1].tolist()
            layer_output_infos.append(cur_tensor_infos)
        layer_infos["layer_output_infos"] = layer_output_infos
        quant_state_dict[k] = layer_infos
    # save qparms as json file
    if configure_file is not None:
        with open(configure_file, "w") as fp:
            json.dump(quant_state_dict, fp, indent=4)


def load_qconf_summary_to_model(model, qconf_summary):
    """
    This function is about load the user given configure to origin model.
    """
    with open(qconf_summary, "r") as f:
        quant_state_dict = json.load(f)
    quant_state_map = model._fqn_to_auto_quant_state_map
    for k, v in quant_state_map.items():
        layer_info = quant_state_dict[k]
        user_q_op_infos = layer_info["q_op_infos"]
        for i, q_op_info in user_q_op_infos.items():
            fqn = q_op_info["fqn"]
            cur_fqn = v.idx_to_seen_q_op_infos[int(i)].fqn
            assert (
                int(i) in v.idx_to_seen_q_op_infos and cur_fqn == fqn
            ), "Loaded quantizable op info doesn't match the current model quantizable op info"
            input_tensor_infos = []
            input_force_dtype_infos = []
            for tensor_info in q_op_info["input_tensor_infos"]:
                if len(tensor_info) > 0:
                    input_tensor_infos.append(
                        QTensorInfo(
                            tensor_info["id"],
                            dtype_dict[tensor_info["orig_dtype"]],
                            dtype_dict[tensor_info["inf_dtype"]],
                        )
                    )
                    input_force_dtype_infos.append(
                        dtype_dict[tensor_info["force_dtype"]]
                    )
                    if "scale" in tensor_info:
                        if isinstance(tensor_info["scale"], list):
                            scale = torch.FloatTensor(tensor_info["scale"])
                            zp = torch.LongTensor(tensor_info["zero_point"])
                        else:
                            scale, zp = {}, {}
                            scale_to_load = tensor_info["scale"]
                            zp_to_load = tensor_info["zero_point"]
                            assert isinstance(scale_to_load, dict) and isinstance(
                                zp_to_load, dict
                            ), (
                                "Expect scales and zero points to load are dicts but "
                                f"found types {type(scale_to_load)} and {type(zp_to_load)}"
                            )
                            for key, val in scale_to_load.items():
                                s = torch.FloatTensor(val)
                                scale.update({key: s})
                            for key, val in zp_to_load.items():
                                z = torch.LongTensor(val)
                                zp.update({key: z})
                        v.tensor_id_to_scale_zp[tensor_info["id"]] = (scale, zp)
                    if "smooth_quant_scaling_factor" in tensor_info:
                        scaling_factors = {}
                        scaling_factors_to_load = tensor_info[
                            "smooth_quant_scaling_factor"
                        ]
                        if isinstance(scaling_factors_to_load, dict):
                            for key, val in scaling_factors_to_load.items():
                                scaling_factor = torch.FloatTensor(val)
                                scaling_factors.update({key: scaling_factor})
                        else:
                            # for backward compatibility
                            assert isinstance(scaling_factors_to_load, list), (
                                f"Expect scaling factors to load to be a dict or list "
                                f"but found type {type(scaling_factors_to_load)}"
                            )
                            scaling_factor = torch.FloatTensor(scaling_factors_to_load)
                            dummy_weight_key = "0_0"
                            scaling_factors.update({dummy_weight_key: scaling_factor})
                        v.tensor_id_to_smooth_quant_scaling_factor[
                            str(tensor_info["id"])
                        ] = scaling_factors
                else:
                    input_tensor_infos.append(None)
                    input_force_dtype_infos.append(None)
            weight_tensor_infos = []
            weight_idx = 0
            for tensor_info in q_op_info["weight_tensor_infos"]:
                if len(tensor_info) > 0:
                    weight_tensor_infos.append(
                        QTensorInfo(
                            weight_idx,
                            dtype_dict[tensor_info["orig_dtype"]],
                            dtype_dict[tensor_info["inf_dtype"]],
                        )
                    )
                    if "scale" in tensor_info:
                        scale = torch.FloatTensor(tensor_info["scale"])
                        zp = torch.LongTensor(tensor_info["zero_point"])
                        v.weight_tensor_id_to_scale_zp[
                            str(i) + "_" + str(weight_idx)
                        ] = (scale, zp)
                    if "smooth_quant_scaling_factor" in tensor_info:
                        scaling_factor = torch.FloatTensor(
                            tensor_info["smooth_quant_scaling_factor"]
                        )
                        v.weight_tensor_id_to_smooth_quant_scaling_factor[
                            str(i) + "_" + str(weight_idx)
                        ] = scaling_factor
                    weight_idx += 1
                else:
                    weight_tensor_infos.append(None)
            output_tensor_infos = []
            insert_fake_quant_after_outputs = []
            for tensor_info in q_op_info["output_tensor_infos"]:
                if len(tensor_info) > 0:
                    output_tensor_infos.append(
                        QTensorInfo(
                            tensor_info["id"],
                            dtype_dict[tensor_info["orig_dtype"]],
                            dtype_dict[tensor_info["inf_dtype"]],
                        )
                    )
                    insert_fake_quant_after_outputs.append(False)
                    if "scale" in tensor_info:
                        if isinstance(tensor_info["scale"], list):
                            scale = torch.FloatTensor(tensor_info["scale"])
                            zp = torch.LongTensor(tensor_info["zero_point"])
                        else:
                            scale, zp = {}, {}
                            scale_to_load = tensor_info["scale"]
                            zp_to_load = tensor_info["zero_point"]
                            assert isinstance(scale_to_load, dict) and isinstance(
                                zp_to_load, dict
                            ), (
                                "Expect scales and zero points to load are dicts but "
                                f"found types {type(scale_to_load)} and {type(zp_to_load)}"
                            )
                            for key, val in scale_to_load.items():
                                s = torch.FloatTensor(val)
                                scale.update({key: s})
                            for key, val in zp_to_load.items():
                                z = torch.LongTensor(val)
                                zp.update({key: z})
                        v.tensor_id_to_scale_zp[tensor_info["id"]] = (scale, zp)
                else:
                    output_tensor_infos.append(None)
            activation_observer = q_op_info["activation_observer"]
            weight_observer = q_op_info["weight_observer"]
            activation_obs = _create_observer(activation_observer)
            if isinstance(activation_obs(), SmoothQuantActivationObserver):
                share_weight_observers = q_op_info.get("share_weight_observers", True)
                qconfig = QConfigSmoothQuant(
                    activation=activation_obs,
                    weight=_create_observer(weight_observer),
                    share_weight_observers=share_weight_observers,
                )
            else:
                qconfig = QConfig(
                    activation=activation_obs,
                    weight=_create_observer(weight_observer),
                )
            # overide the cur model's info
            v.idx_to_seen_q_op_infos[int(i)].input_tensor_infos = input_tensor_infos
            v.idx_to_seen_q_op_infos[int(i)].input_tensor_force_inf_dtype = (
                input_force_dtype_infos
            )
            v.idx_to_seen_q_op_infos[int(i)].output_tensor_infos = output_tensor_infos
            v.idx_to_seen_q_op_infos[int(i)].insert_fake_quant_after_outputs = (
                insert_fake_quant_after_outputs
            )
            v.idx_to_seen_q_op_infos[int(i)].weight_tensor_infos = weight_tensor_infos
            v.idx_to_seen_q_op_infos[int(i)].qconfig = qconfig

        user_nonq_op_infos = layer_info["nonq_op_infos"]
        # v.seen_nonq_op_infos.clear()
        idx = 0
        for _, nonq_op_info in user_nonq_op_infos.items():
            fqn = nonq_op_info["fqn"]
            cur_fqn = v.seen_nonq_op_infos[idx].fqn
            assert (
                cur_fqn == fqn
            ), "Loaded none-quantizable op info doesn't match the current model none-quantizable op info"
            input_tensor_infos = []
            for tensor_info in nonq_op_info["input_tensor_infos"]:
                if len(tensor_info) > 0:
                    input_tensor_infos.append(
                        QTensorInfo(
                            tensor_info["id"],
                            dtype_dict[tensor_info["orig_dtype"]],
                            dtype_dict[tensor_info["inf_dtype"]],
                        )
                    )
                else:
                    input_tensor_infos.append(None)
            output_tensor_infos = []
            for tensor_info in nonq_op_info["output_tensor_infos"]:
                if len(tensor_info) > 0:
                    output_tensor_infos.append(
                        QTensorInfo(
                            tensor_info["id"],
                            dtype_dict[tensor_info["orig_dtype"]],
                            dtype_dict[tensor_info["inf_dtype"]],
                        )
                    )
                else:
                    output_tensor_infos.append(None)
            v.seen_nonq_op_infos[idx].input_tensor_infos = input_tensor_infos
            v.seen_nonq_op_infos[idx].output_tensor_infos = output_tensor_infos
            idx += 1

        layer_output_info = []
        for tensor_info in layer_info["layer_output_infos"]:
            if len(tensor_info) > 0:
                layer_output_info.append(
                    QTensorInfo(
                        tensor_info["id"],
                        dtype_dict[tensor_info["orig_dtype"]],
                        dtype_dict[tensor_info["inf_dtype"]],
                    )
                )
                if "scale" in tensor_info:
                    scale = torch.FloatTensor(tensor_info["scale"])
                    zp = torch.LongTensor(tensor_info["zero_point"])
                    v.tensor_id_to_scale_zp[tensor_info["id"]] = (scale, zp)
            else:
                layer_output_info.append(None)
        v.output_qtensor_infos = layer_output_info
    # insert observer according to user's setting.
    for _, v in model.named_modules():
        if hasattr(v, "_auto_quant_state"):
            v._auto_quant_state.tensor_id_to_observer.clear()
            v._auto_quant_state.weight_tensor_id_to_observer.clear()
            v._auto_quant_state.insert_observers(v)

    # update insert_fake_quant_after_output after load user setting, which for avoiding redundant fake quant.
    nodes = convert_quant_state_map_to_nodes(quant_state_map)
    set_node_output_quantized(nodes)


def _lstm_forward(module, input, hx, weights):
    r"""
    LSTM forward function.
    """
    orig_input = input
    # xxx: isinstance check needs to be in conditional for TorchScript to compile
    # batch_sizes = None
    if isinstance(orig_input, torch.nn.utils.rnn.PackedSequence):
        input, batch_sizes, sorted_indices, unsorted_indices = input
        max_batch_size = batch_sizes[0]
        max_batch_size = int(max_batch_size)
    else:
        batch_sizes = None
        is_batched = input.dim() == 3
        batch_dim = 0 if module.batch_first else 1
        if not is_batched:
            input = input.unsqueeze(batch_dim)
        max_batch_size = input.size(0) if module.batch_first else input.size(1)
        sorted_indices = None
        unsorted_indices = None
    if hx is None:
        num_directions = 2 if module.bidirectional else 1
        real_hidden_size = (
            module.proj_size if module.proj_size > 0 else module.hidden_size
        )
        h_zeros = torch.zeros(
            module.num_layers * num_directions,
            max_batch_size,
            real_hidden_size,
            dtype=input.dtype,
            device=input.device,
        )
        c_zeros = torch.zeros(
            module.num_layers * num_directions,
            max_batch_size,
            module.hidden_size,
            dtype=input.dtype,
            device=input.device,
        )
        hx = (h_zeros, c_zeros)
    else:
        if batch_sizes is None:  # If not PackedSequence input.
            if is_batched:
                if hx[0].dim() != 3 or hx[1].dim() != 3:
                    msg = (
                        "For batched 3-D input, hx and cx should "
                        f"also be 3-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors"
                    )
                    raise RuntimeError(msg)
            else:
                if hx[0].dim() != 2 or hx[1].dim() != 2:
                    msg = (
                        "For unbatched 2-D input, hx and cx should "
                        f"also be 2-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors"
                    )
                    raise RuntimeError(msg)
                hx = (hx[0].unsqueeze(1), hx[1].unsqueeze(1))

        # Each batch of the hidden state should match the input sequence that
        # the user believes he/she is passing in.
        hx = module.permute_hidden(hx, sorted_indices)

    module.check_forward_args(input, hx, batch_sizes)
    if batch_sizes is None:
        result = _VF.lstm(
            input,
            hx,
            weights,
            module.bias,
            module.num_layers,
            module.dropout,
            module.training,
            module.bidirectional,
            module.batch_first,
        )
    else:
        result = _VF.lstm(
            input,
            batch_sizes,
            hx,
            weights,
            module.bias,
            module.num_layers,
            module.dropout,
            module.training,
            module.bidirectional,
        )
    output = result[0]
    hidden = result[1:]
    # xxx: isinstance check needs to be in conditional for TorchScript to compile
    if isinstance(orig_input, torch.nn.utils.rnn.PackedSequence):
        output_packed = torch.nn.utils.rnn.PackedSequence(
            output, batch_sizes, sorted_indices, unsorted_indices
        )
        return output_packed, module.permute_hidden(hidden, unsorted_indices)
    else:
        if not is_batched:
            output = output.squeeze(batch_dim)
            hidden = (hidden[0].squeeze(1), hidden[1].squeeze(1))
        return output, module.permute_hidden(hidden, unsorted_indices)


def module_call_to_function_call(module, args, weights):
    r"""
    This function is a help function which replace nn.module call to funtion call, which implement
    the nn.module's forward function.
    """
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Conv3d):
        output = module._conv_forward(args[0], weights[0], module.bias)
    elif isinstance(module, torch.nn.Linear):
        output = F.linear(args[0], weights[0], module.bias)
    elif isinstance(module, torch.nn.EmbeddingBag):
        output = F.embedding_bag(
            args[0],
            weights[0],
            args[1],
            module.max_norm,
            module.norm_type,
            module.scale_grad_by_freq,
            module.mode,
            module.sparse,
            args[2] if len(args) == 3 else None,
            module.include_last_offset,
            module.padding_idx,
        )
    elif isinstance(module, MergedEmbeddingBagWithCat):
        output = torch.ops.torch_ipex.merged_embeddingbag_cat_forward(
            weights, args[0], args[1], args[2]
        )
    elif isinstance(module, torch.nn.ConvTranspose2d) or isinstance(
        module, torch.nn.ConvTranspose3d
    ):
        if module.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose2d"
            )
        assert isinstance(module.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        output_size = args[1] if len(args) == 2 else None
        # master code
        """
        num_spatial_dims = 2 if isinstance(module, torch.nn.ConvTranspose2d) else 3
        output_padding = module._output_padding(args[0], output_size,
                        module.stride, module.padding, module.kernel_size,
                        num_spatial_dims, module.dilation)
        """
        output_padding = module._output_padding(
            args[0],
            output_size,
            module.stride,
            module.padding,
            module.kernel_size,
            module.dilation,
        )
        # output_padding = module._output_padding(*arg_to)
        if isinstance(module, torch.nn.ConvTranspose2d):
            output = F.conv_transpose2d(
                args[0],
                weights[0],
                module.bias,
                module.stride,
                module.padding,
                output_padding,
                module.groups,
                module.dilation,
            )
        else:
            output = F.conv_transpose3d(
                args[0],
                weights[0],
                module.bias,
                module.stride,
                module.padding,
                output_padding,
                module.groups,
                module.dilation,
            )
    elif isinstance(module, torch.nn.LSTM):
        output = _lstm_forward(
            module, args[0], args[1] if len(args) == 2 else None, weights
        )
    return output


def _attach_smooth_quant_scaling_factor_to_model(module):
    """
    Get scaling factors for SmoothQuant from observers and
    store them in qstate
    """
    if not hasattr(module, "_auto_quant_state"):
        return
    qstate = module._auto_quant_state
    qconfig = qstate.qconfig
    if not isinstance(
        qconfig.activation(), SmoothQuantActivationObserver
    ) or not isinstance(qconfig.weight(), SmoothQuantWeightObserver):
        return
    if not qstate.tensor_id_to_observer:
        return
    for key, obs in qstate.tensor_id_to_observer.items():
        if key in qstate.tensor_id_to_smooth_quant_scaling_factor:
            continue
        scaling_factors = obs.get_scaling_factors()
        qstate.tensor_id_to_smooth_quant_scaling_factor[key] = scaling_factors

    for key, obs in qstate.weight_tensor_id_to_observer.items():
        if key in qstate.weight_tensor_id_to_smooth_quant_scaling_factor:
            continue
        scaling_factors = obs.get_scaling_factors()
        qstate.weight_tensor_id_to_smooth_quant_scaling_factor[key] = scaling_factors


def _map_smooth_quant_info_to_idx(module):
    """
    Map dict of {tensor id: smooth quant scaling factor} to
        dict of {idx: smooth quant scaling factor}.
    For nn.Linear module only.
    """
    if not hasattr(module, "_auto_quant_state"):
        return
    qstate: AutoQuantizationState = module._auto_quant_state  # type: ignore[assignment]
    qconfig = qstate.qconfig
    if not isinstance(
        qconfig.activation(), SmoothQuantActivationObserver
    ) or not isinstance(qconfig.weight(), SmoothQuantWeightObserver):
        return
    for _, seen_q_op_info in qstate.idx_to_seen_q_op_infos.items():
        if not seen_q_op_info.input_tensor_infos:
            continue
        # Linear has only one activation
        for input_arg in seen_q_op_info.input_tensor_infos:
            if input_arg is None:
                continue
            tensor_id = str(input_arg.id)
            if tensor_id in qstate.tensor_id_to_smooth_quant_scaling_factor:
                key = str(seen_q_op_info.idx)
                qstate.idx_to_smooth_quant_scaling_factor[key] = (
                    qstate.tensor_id_to_smooth_quant_scaling_factor[tensor_id]
                )
        # Linear has only one weight. Key is not changed.
        for weight_arg in seen_q_op_info.weight_tensor_infos:
            if weight_arg is None:
                continue
            tensor_id = str(seen_q_op_info.idx) + "_" + str(weight_arg.id)
            if tensor_id in qstate.weight_tensor_id_to_smooth_quant_scaling_factor:
                key = str(seen_q_op_info.idx) + "_" + str(weight_arg.id)
                qstate.idx_to_smooth_quant_scaling_factor[key] = (
                    qstate.weight_tensor_id_to_smooth_quant_scaling_factor[tensor_id]
                )
