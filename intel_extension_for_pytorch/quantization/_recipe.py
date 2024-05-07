import torch
import torch.nn as nn
import torch.nn.functional as F
from intel_extension_for_pytorch.nn.functional import interaction
from intel_extension_for_pytorch.nn.modules import MergedEmbeddingBagWithCat

from ._utils import ParentNode, set_node_output_quantized

add_inplace_ops = [str(torch.Tensor.add_)]
add_ops = [str(torch.add), str(torch.Tensor.add)]
elt_wise_q_ops = [str(torch.Tensor.relu), str(torch.relu), str(F.relu), str(nn.ReLU)]
elt_wise_noq_ops = [
    str(torch.relu_),
    str(torch.sigmoid_),
    str(nn.ReLU),
    str(torch.Tensor.relu_),
    str(torch.Tensor.sigmoid_),
    str(torch.nn.Hardtanh),
    str(F.hardtanh),
    str(F.hardtanh_),
    str(torch.nn.ELU),
    str(F.elu),
    str(F.elu_),
    str(nn.SiLU),
    str(F.silu),
    str(torch.Tensor.sigmoid),
    str(torch.sigmoid),
    str(F.sigmoid),
    str(nn.Sigmoid),
    str(F.gelu),
    str(nn.GELU),
]
conv_gemm_ops = [
    str(F.conv2d),
    str(nn.Conv2d),
    str(F.conv3d),
    str(nn.Conv3d),
    str(torch.conv2d),
    str(torch.conv3d),
    str(F.conv_transpose2d),
    str(torch.nn.ConvTranspose2d),
    str(F.conv_transpose3d),
    str(torch.nn.ConvTranspose3d),
    str(torch.conv_transpose2d),
    str(torch.conv_transpose2d),
    str(F.linear),
    str(nn.Linear),
    str(torch.matmul),
    str(torch.Tensor.matmul),
    str(torch.bmm),
    str(torch.Tensor.bmm),
]
conv_ops = [
    str(F.conv2d),
    str(nn.Conv2d),
    str(F.conv3d),
    str(nn.Conv3d),
    str(torch.conv2d),
    str(torch.conv3d),
    str(F.conv_transpose2d),
    str(torch.nn.ConvTranspose2d),
    str(F.conv_transpose3d),
    str(torch.nn.ConvTranspose3d),
    str(torch.conv_transpose2d),
    str(torch.conv_transpose2d),
]
rnn_ops = [str(torch.nn.LSTM)]

# Those ops only support s8->s8 path, and also require the qscheme is per_tensor_symmetric.
s8_s8_symmetric_ops = [
    str(interaction),
    str(torch.ops.torch_ipex.interaction_forward),
    str(torch.ops.torch_ipex.merged_embeddingbag_cat_forward),
    str(torch.embedding_bag),
    str(F.embedding_bag),
    str(torch.nn.EmbeddingBag),
]
conv_gemm_fs = [
    str(F.conv2d),
    str(F.conv3d),
    str(F.conv_transpose2d),
    str(F.conv_transpose3d),
    str(torch.conv2d),
    str(torch.conv3d),
    str(torch.conv_transpose2d),
    str(torch.conv_transpose2d),
    str(F.linear),
    str(torch._C._nn.linear),
]


def _default_recipe_init(nodes):
    r"""
    This function is about init default recipe: setting the quantizable op's inf dtype to qint8 or quint8 according the qconfig,
    there have some special cases, for some ops(interaction, EmbeddingBag), we only support some special \
    quantization path, so if the related qconfig
    doesn't meet the requirements, we will not set their inf dtype.
    """
    for node in nodes:
        if isinstance(node, ParentNode):
            continue
        if node.qconfig is not None:
            # Add q+dq before the quantizable op firstly.
            for idx, tensor_info in enumerate(node.input_tensor_infos):
                # only support fp32 tensor->int8 tensor
                if (
                    tensor_info is not None
                    and (tensor_info.orig_dtype == torch.float32)
                    and tensor_info.id in node.input_scale_zero
                ):
                    # gemm's weight
                    if node.type in conv_gemm_fs and idx == 1:
                        tensor_info.inf_dtype = node.qconfig.weight().dtype
                    else:
                        tensor_info.inf_dtype = node.qconfig.activation().dtype
                    node.input_tensor_force_inf_dtype[idx] = tensor_info.inf_dtype
            # For EmbeddingBag and interaction, we need to check the qconfig's setting, if not meet the requirements, \
            # reset the inputs'(or weight) inf dtype
            for tensor_info in node.weight_tensor_infos:
                # nn.EmbeddingBag use activation observer and only support torch.qint8 and torch.per_tensor_symmetric
                if (
                    tensor_info is not None
                    and (tensor_info.orig_dtype == torch.float32)
                    and (
                        str(node.idx) + "_" + str(tensor_info.id)
                        in node.weight_scale_zero
                    )
                ):
                    if (
                        node.type == str(torch.nn.EmbeddingBag)
                        and node.qconfig.activation().dtype == torch.qint8
                        and node.qconfig.activation().qscheme
                        == torch.per_tensor_symmetric
                    ) or node.type != str(torch.nn.EmbeddingBag):
                        tensor_info.inf_dtype = node.qconfig.weight().dtype
            # interaction only supports qint8 and torch.per_tensor_symmetric, if not meet the requirement,
            # reset the input's inf dtype.
            if node.type in s8_s8_symmetric_ops:
                if not (
                    node.qconfig.activation().dtype == torch.qint8
                    and node.qconfig.activation().qscheme == torch.per_tensor_symmetric
                ):
                    for idx, tensor_info in enumerate(node.input_tensor_infos):
                        if tensor_info is not None:
                            tensor_info.inf_dtype = tensor_info.orig_dtype
                            node.input_tensor_force_inf_dtype[idx] = (
                                tensor_info.inf_dtype
                            )

            # For LSTM, if it's input is a PackedSequence, we don't support ot now.
            # TODO: support PackedSequence input for quantization LSTM.
            if (
                node.type in rnn_ops
                and len(node.input_tensor_infos) > 2
                and node.input_tensor_infos[1].orig_dtype == torch.int64
            ):
                for idx, tensor_info in enumerate(node.input_tensor_infos):
                    if tensor_info is not None:
                        tensor_info.inf_dtype = tensor_info.orig_dtype
                        node.input_tensor_force_inf_dtype[idx] = tensor_info.inf_dtype
                for idx, tensor_info in enumerate(node.weight_tensor_infos):
                    if tensor_info is not None:
                        tensor_info.inf_dtype = tensor_info.orig_dtype


# TODO: making fusion pattern check more general.
def _find_fused_node_with_cur_elt_wise(node, ops):
    r"""
    Find a node before cur elt_wise which can be fused with cur elt_wise, which used by check
    whether has a op can be fused with elt_wise.
    """
    if len(node.pre_nodes) == 0:
        return None
    pre_node = node.pre_nodes[0]
    if pre_node is not None:
        if pre_node.type in ops:
            if len(pre_node.post_nodes) == 1:
                return pre_node
            elif len(node.post_nodes) == 1 and _find_conv_or_gemm_swish_fusion_node(
                node.post_nodes[0]
            ):
                # conv+sigmoid+mul
                return pre_node
            else:
                return None
        elif (
            pre_node.type in ([str(nn.Identity)] + elt_wise_q_ops + elt_wise_noq_ops)
            and len(pre_node.post_nodes) == 1
            and len(pre_node.pre_nodes) > 0
        ):
            return _find_fused_node_with_cur_elt_wise(pre_node.pre_nodes[0], ops)
        else:
            return None
    else:
        return None


def _find_fused_node_with_cur_add(node, ops):
    r"""
    Find a node before the cur node which can be fused with cur add node, which used to check
    whether has a node can be fused with add.
    """
    if len(node.pre_nodes) == 0:
        return None
    if len(node.pre_nodes) > 0:
        if (
            node.pre_nodes[0].type in ops
            and len(node.pre_nodes[0].post_nodes) == 1
            and node.pre_nodes[0].qconfig is not None
        ):
            return node.pre_nodes[0]
        elif (
            node.pre_nodes[0].type == str(nn.Identity)
            and len(node.pre_nodes[0].post_nodes) == 1
            and len(node.pre_nodes[0].pre_nodes) > 0
        ):
            fused_node = _find_fused_node_with_cur_add(node.pre_nodes[0], ops)
            if fused_node is not None:
                return node.pre_nodes[0]
            else:
                return None

        if len(node.pre_nodes) == 2:
            if (
                node.pre_nodes[1].type in ops
                and len(node.pre_nodes[1].post_nodes) == 1
                and node.pre_nodes[1].qconfig is not None
            ):
                return node.pre_nodes[1]
            elif (
                node.pre_nodes[1].type == str(nn.Identity)
                and len(node.pre_nodes[1].post_nodes) == 1
                and len(node.pre_nodes[1].pre_nodes) > 0
            ):
                fused_node = _find_fused_node_with_cur_add(node.pre_nodes[1], ops)
                if fused_node is not None:
                    return node.pre_nodes[1]
                else:
                    return None
        return None


def _find_conv_or_gemm_swish_fusion_node(node):
    r"""
    Check whether has conv/gemm_sigmoid_mul fusion before cur node(including).
        conv/gemm
          /  \
         /  sigmoid
         \     /
           mul(_)
    """
    mul_ops = [str(torch.mul), str(torch.Tensor.mul), str(torch.Tensor.mul_)]
    sigmoid_ops = [
        str(torch.Tensor.sigmoid),
        str(torch.Tensor.sigmoid_),
        str(torch.sigmoid),
        str(torch.sigmoid_),
        str(F.sigmoid),
        str(torch.nn.Sigmoid),
    ]
    if node.type in mul_ops and len(node.pre_nodes) == 2:
        if (
            node.pre_nodes[0].type in conv_gemm_ops
            and node.pre_nodes[1].type in sigmoid_ops
        ):
            if (
                len(node.pre_nodes[0].post_nodes) == 2
                and len(node.pre_nodes[1].post_nodes) == 1
                and node.pre_nodes[1] in node.pre_nodes[0].post_nodes
            ):
                return node.pre_nodes[0]
        elif (
            node.pre_nodes[1].type in conv_gemm_ops
            and node.pre_nodes[0].type in sigmoid_ops
        ):
            if (
                len(node.pre_node[1].post_nodes) == 2
                and len(node.pre_nodes[0].post_nodes) == 1
                and node.pre_nodes[0] in node.pre_node[1].post_nodes
            ):
                return node.pre_nodes[1]
    return None


def _check_has_quantizable_node_before_node(node):
    r"""
    This function is about check whether has a quantizable node before(including) the given node,
    which is used to check whether insert fake quant before one quantizable node or not. For example,
    given_node->quantizable_node, if the given node is a none-quantizable node(also not a fusion groups nodes),
    we can avoid inserting fake quant before this quantizable node.
    """
    if node.type == str(nn.Identity):
        if len(node.pre_nodes) > 0:
            return _check_has_quantizable_node_before_node(node.pre_nodes[0])
        else:
            return False
    else:
        # check whether has a qconfig
        if node.qconfig is None:
            if len(node.pre_nodes) == 0:
                return False
            # conv/gemm+add(_)+elt_wise
            if node.type in elt_wise_noq_ops:
                fused_elt_wise_node = _find_fused_node_with_cur_elt_wise(
                    node, conv_gemm_ops + add_ops + add_inplace_ops
                )
                if fused_elt_wise_node is not None:
                    # if fused_elt_wise_node is add_inplace_op, make sure it can also fused with conv/gemm.
                    if fused_elt_wise_node.type in add_inplace_ops:
                        fused_add_node = _find_fused_node_with_cur_add(
                            node, conv_gemm_ops
                        )
                        if (
                            fused_add_node is not None
                            and fused_add_node.qconfig is not None
                        ):
                            return True
                        else:
                            return False
                    else:
                        if fused_elt_wise_node.qconfig is not None:
                            return True
                        else:
                            return False
            elif node.type in add_inplace_ops:  # check gemm+add_
                fused_add_wise_node = _find_fused_node_with_cur_add(node, conv_gemm_ops)
                if (
                    fused_add_wise_node is not None
                    and fused_add_wise_node.qconfig is not None
                ):
                    return True
            # conv+sigmoid+mul(_)
            fused_conv_or_gemm_swish_node = _find_conv_or_gemm_swish_fusion_node(node)
            if (
                fused_conv_or_gemm_swish_node is not None
                and fused_conv_or_gemm_swish_node.qconfig is not None
            ):
                return True
            return False
        else:
            if node.type in s8_s8_symmetric_ops:
                if node.type in [
                    str(interaction),
                    str(torch.ops.torch_ipex.interaction_forward),
                    str(torch.ops.torch_ipex.merged_embeddingbag_cat_forward),
                ]:
                    for force_inf_dtype in node.input_tensor_force_inf_dtype:
                        if force_inf_dtype == torch.qint8:
                            return True
                    return False
                else:
                    # EmbeddingBag
                    if node.weight_tensor_infos[0].inf_dtype == torch.qint8:
                        return True
                    else:
                        return False
            else:
                # for none ipex customer op, if have a qconfig, we can say it is a quantizable op.
                return True


def _check_has_quantizable_node_after_node(node):
    r"""
    This function is about check whether all quantizable nodes after the given node,
    which is used to check whether insert fake quant before one quantizable node or not.
    """
    if len(node.post_nodes) > 0:
        output = True
        for i in range(len(node.post_nodes)):
            if node.post_nodes[i].qconfig is None:
                output = False
        return output
    else:
        return False


def _add_recipe(node):
    r"""
    Case1: add has pre gemm node.
    Given  gemm     op             gemm         op                gemm       op
             \     /                 \         /                   \       /
              \   /       ==>    fake_quant (fake_quant?)     ==>   \   (fake_quant?)
               \ /                     \    /                        \   /
               add                       add                          add

          gemm     fp32_op          gemm     quantizable_op
    ==>    \        /                \         /
            \      /          or      \     fake_quant
             \    /                    \    /
              add                       add

    Case2: add doesn't have pre conv/gemm node.
    For this case, if one add input has one none-quantizable op, we will don't insert fake quant before it.
    """

    def reset_input_inf_dtype_to_orig_dtype(node, input_idx):
        if node.input_tensor_infos[input_idx] is not None:
            if (
                node.input_tensor_infos[input_idx]
                in node.pre_nodes[0].output_tensor_infos
            ):
                pre_node = node.pre_nodes[input_idx]
            elif (
                len(node.pre_nodes) == 2
                and node.input_tensor_infos[input_idx]
                in node.pre_nodes[1].output_tensor_infos
            ):
                pre_node = node.pre_nodes[1]
            else:
                pre_node = None
            if pre_node is not None:
                add_quantize_add_input_idx = _check_has_quantizable_node_before_node(
                    pre_node
                )
            else:
                add_quantize_add_input_idx = False
            if not add_quantize_add_input_idx:
                node.input_tensor_infos[input_idx].inf_dtype = node.input_tensor_infos[
                    input_idx
                ].orig_dtype
                node.input_tensor_force_inf_dtype[input_idx] = node.input_tensor_infos[
                    input_idx
                ].inf_dtype

    def force_reset_input_inf_dtype_to_orig_dtype(node, input_idx):
        if node.input_tensor_infos[input_idx] is not None:
            node.input_tensor_infos[input_idx].inf_dtype = node.input_tensor_infos[
                input_idx
            ].orig_dtype
            node.input_tensor_force_inf_dtype[input_idx] = node.input_tensor_infos[
                input_idx
            ].inf_dtype

    conv_gemm_node = _find_fused_node_with_cur_add(node, conv_gemm_ops)
    conv_node = _find_fused_node_with_cur_add(node, conv_ops)
    if conv_gemm_node is None:
        #  If pre_nodes don't have gemm node, need to check whether have quantizable node before it,
        #  if does't have quantizable node before it, we will not insert fake quant before add.
        # hoping all input nodes are quantizable node.
        if len(node.pre_nodes) > 0:
            add_1_has_pre_quantizable_op = _check_has_quantizable_node_before_node(
                node.pre_nodes[0]
            )
            add_2_has_pre_quantizable_op = False
            if len(node.pre_nodes) == 2:
                add_2_has_pre_quantizable_op = _check_has_quantizable_node_before_node(
                    node.pre_nodes[1]
                )

            # Generally, if add connected to 2 quantizable node, we will keep the fake quant
            # in the input edges of this add.
            if not (add_1_has_pre_quantizable_op and add_2_has_pre_quantizable_op):
                for idx, tensor_info in enumerate(node.input_tensor_infos):
                    tensor_info.inf_dtype = tensor_info.orig_dtype
                    node.input_tensor_force_inf_dtype[idx] = tensor_info.inf_dtype

            # A corner case is
            # add1    add2
            #  \     /
            #    add3
            # which exists in GPT-J, in this case, we need to remove the fake quant in
            # 2 inputs edge of add3
            if (
                (len(node.pre_nodes) == 2)
                and (node.pre_nodes[0].type in [str(torch.Tensor.add), str(torch.add)])
                and (node.pre_nodes[1].type in [str(torch.Tensor.add), str(torch.add)])
            ):
                for idx, tensor_info in enumerate(node.input_tensor_infos):
                    tensor_info.inf_dtype = tensor_info.orig_dtype
                    node.input_tensor_force_inf_dtype[idx] = tensor_info.inf_dtype
        else:
            for idx, tensor_info in enumerate(node.input_tensor_infos):
                tensor_info.inf_dtype = tensor_info.orig_dtype
                node.input_tensor_force_inf_dtype[idx] = tensor_info.inf_dtype
    else:
        # add can fused with gemm.
        if (
            node.input_tensor_infos[0] is not None
            and node.input_tensor_infos[0] in conv_gemm_node.output_tensor_infos
        ):
            node.input_tensor_infos[0].inf_dtype = node.input_tensor_infos[0].orig_dtype
            node.input_tensor_force_inf_dtype[0] = node.input_tensor_infos[0].inf_dtype
            # TODO: set another input's dtype for conv nodes when oneDNN is ready.
            if conv_node is None or not _check_has_quantizable_node_after_node(node):
                # set another input's dtype, if another's input is from non-quantizable op, we can remove the fake quant.
                if conv_node is None:
                    # For linear_add pattern, force reset add's extra input's inf_dtype to orig_dtype
                    force_reset_input_inf_dtype_to_orig_dtype(node, 1)
                else:
                    reset_input_inf_dtype_to_orig_dtype(node, 1)
        elif (
            node.input_tensor_infos[1] is not None
            and node.input_tensor_infos[1] in conv_gemm_node.output_tensor_infos
        ):
            node.input_tensor_infos[1].inf_dtype = node.input_tensor_infos[1].orig_dtype
            node.input_tensor_force_inf_dtype[1] = node.input_tensor_infos[1].inf_dtype
            # TODO: set another input's dtype for conv nodes when oneDNN is ready.
            if conv_node is None or not _check_has_quantizable_node_after_node(node):
                # set another input's dtype, if another's input is from non-quantizable op, we can remove the fake quant.
                if conv_node is None:
                    # For linear_add pattern, force reset add's extra input's inf_dtype to orig_dtype
                    force_reset_input_inf_dtype_to_orig_dtype(node, 0)
                else:
                    reset_input_inf_dtype_to_orig_dtype(node, 0)


# get a default recipe
def get_default_recipe(nodes):
    r"""
    This function is about get default recipe which set where fake quant is inserted for the quantizable ops.
    """
    # step1: Quantization state init. Quantize inputs before quantizable node by setting their input's inf_dtype to
    # qconfig.activation().dtype, and also setting the weight's inf_dtype to
    # qconfig.weight().dtype if a module has a weight.
    _default_recipe_init(nodes)
    # step2: Optimization
    # 1. For conv, gemm, and LSTM,  we always quantize its' inputs and weight, so we keep them state.
    #    and for embedding_bag, which only has a weight, we always quantize it's weight to
    #    save memory space and bandwidth, we also keep it's state.
    # 2. For remaining quantizable ops (pooling, elt-wise op and add) which meet the following requirements, we will
    # update them inputs' quantization state.
    #   1. If it is a part of a quantized fusion pattern, don't need to quantize any inputs from inside the pattern.
    #   2. If any of its inputs outside the fusion pattern are from non-quantized op, don't quantize all inputs outside the pattern.
    #   3. If it is not part of a quantized fusion pattern, don't quantize all inputs if its one input from non-quantized op.
    # 3. For quantizable ops (pooling, relu, flatten, interation and embedding) forcing quantized output, need to \
    #    quantize its output if it is quantized.
    # 4. For interation and embedding, we only support s8->s8 symmetric quantization, so if doesn't meet the \
    #    requiresments, don't need to quantize its inputs.
    # Note: the fusion pattern we are supported is conv/gemm/add + elt-wise, conv/gemm + add, conv/gemm + add + elt-wise.
    # which means some ops can be combined with a single op to compute, but they are mathematically equivalent.
    embedding_bag_ops = [
        str(torch.embedding_bag),
        str(F.embedding_bag),
        str(torch.nn.EmbeddingBag),
        str(MergedEmbeddingBagWithCat),
        str(torch.ops.torch_ipex.merged_embeddingbag_cat_forward),
    ]
    for node in nodes:
        if isinstance(node, ParentNode):
            continue
        if node.qconfig is not None and node.type not in (
            conv_gemm_ops + rnn_ops + embedding_bag_ops
        ):
            if node.type in add_ops:
                # gemm+add fusion
                _add_recipe(node)
            elif node.type in elt_wise_q_ops:
                # don't have a pre_node, we can say it doesn't have a pre quantizable node.
                has_pre_quantized_node = True
                # If Has gemm(add) pre_op can be fused, not insert fake quant.
                if len(node.pre_nodes) > 0:
                    if (
                        _find_fused_node_with_cur_elt_wise(
                            node, conv_gemm_ops + add_ops + add_inplace_ops
                        )
                        is not None
                    ):
                        has_pre_quantized_node = False
                    else:
                        has_pre_quantized_node = (
                            _check_has_quantizable_node_before_node(node.pre_nodes[0])
                        )
                else:
                    has_pre_quantized_node = False
                if not has_pre_quantized_node:
                    node.input_tensor_infos[0].inf_dtype = node.input_tensor_infos[
                        0
                    ].orig_dtype
                    node.input_tensor_force_inf_dtype[0] = node.input_tensor_infos[
                        0
                    ].inf_dtype
            elif node.type == str(
                torch.Tensor.flatten
            ) and not _check_has_quantizable_node_after_node(node):
                # If the post op of flatten is not a quantizable node, force reset input's inf_dtype to orig_dtype
                node.input_tensor_infos[0].inf_dtype = node.input_tensor_infos[
                    0
                ].orig_dtype
                node.input_tensor_force_inf_dtype[0] = node.input_tensor_infos[
                    0
                ].inf_dtype
            else:
                # For other quantizable node, we don't need add fake quant before it if it's pre node is one none-quantizable op.
                # Now all other quantizable node only have one input info, so we can check the one pre input node info to check
                # whether has a pre quantizable node.
                has_pre_quantized_node = True
                if len(node.pre_nodes) == 1:
                    has_pre_quantized_node = _check_has_quantizable_node_before_node(
                        node.pre_nodes[0]
                    )
                elif len(node.pre_nodes) == 0:
                    has_pre_quantized_node = False
                # the node's pre node doesn't support int8 output.
                if not has_pre_quantized_node:
                    node.input_tensor_infos[0].inf_dtype = node.input_tensor_infos[
                        0
                    ].orig_dtype
                    node.input_tensor_force_inf_dtype[0] = node.input_tensor_infos[
                        0
                    ].inf_dtype

    set_node_output_quantized(nodes)
