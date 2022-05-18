import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from intel_extension_for_pytorch.nn.functional import interaction

from ._utils import ParentNode, check_node_in_give_op

add_inplace_op = [str(torch.Tensor.add_)]
add_op = [str(torch.add), str(torch.Tensor.add)]
elt_wise = [str(torch.relu), str(F.relu), str(nn.ReLU), str(torch.sigmoid), str(F.sigmoid), str(nn.Sigmoid), str(F.gelu), str(nn.GELU)]
gemm_op = [str(F.conv2d), str(nn.Conv2d), str(F.conv3d), str(nn.Conv3d), str(torch.conv2d), str(torch.conv3d), \
    str(F.conv_transpose2d), str(torch.nn.ConvTranspose2d), str(F.conv_transpose3d), str(torch.nn.ConvTranspose3d),
    str(torch.conv_transpose2d), str(torch.conv_transpose2d), str(F.linear), str(nn.Linear), str(torch.matmul)]
ipex_customer_op = [str(interaction), str(torch.ops.torch_ipex.interaction_forward), str(torch.embedding_bag), \
    str(F.embedding_bag), str(torch.nn.EmbeddingBag)]
gemm_f = [str(F.conv2d), str(F.conv3d), str(F.conv_transpose2d), str(F.conv_transpose3d), str(torch.conv2d), str(torch.conv3d), \
    str(torch.conv_transpose2d), str(torch.conv_transpose2d), str(F.linear), str(torch._C._nn.linear)]
lstm = [str(torch.nn.LSTM)]

def _default_recipe_init(nodes):
    r"""
    This funcition is about init default recipe: setting the quantizable op's inf dtype to qint8 or quint8 according the qconfig, there has some special case,
    for some ipex customer ops, we only support some special quantizaiton path, so if the qconfig doesn' meet the requiresments, we will not set their inf dtype.
    """
    for node in nodes:
        if isinstance(node, ParentNode):
            continue
        if node.qconfig is not None:
            # others, we will add q+dq before the quantizable op firstly.
            idx = 0
            for tensor_info in node.input_tensor_infos:
                # only support fp32 tensor->int8 tensor
                if tensor_info is not None and (tensor_info.orig_dtype == torch.float32) and tensor_info.id in node.input_scale_zero:
                    # gemm's weight
                    if node.type in gemm_f and idx == 1:
                        tensor_info.inf_dtype = node.qconfig.weight().dtype
                    else:
                        tensor_info.inf_dtype = node.qconfig.activation().dtype
                idx = idx + 1
            # ipex customer op, check the qconfig's setting, if not meet the requiresments, reset the inputs'(weight) inf dtype
            for tensor_info in node.weight_tensor_infos:
                # nn.EmbeddingBag use activation observer and only support torch.qint8 and torch.per_tensor_symmetric
                if tensor_info is not None and (tensor_info.orig_dtype == torch.float32) and (str(node.idx) + "_" + str(tensor_info.id) in node.weight_scale_zero):
                    if (node.type == str(torch.nn.EmbeddingBag) and node.qconfig.activation().dtype == torch.qint8 and \
                        node.qconfig.activation().qscheme == torch.per_tensor_symmetric) or node.type != str(torch.nn.EmbeddingBag):
                        tensor_info.inf_dtype = node.qconfig.weight().dtype
            # interaction only support qint8 and torch.per_tensor_symmetric, if not meet the requiresments,
            # reset the input's inf dtype.
            if check_node_in_give_op(node, ipex_customer_op):
                if not(node.qconfig.activation().dtype == torch.qint8 and node.qconfig.activation().qscheme == torch.per_tensor_symmetric):
                    for tensor_info in node.input_tensor_infos and tensor_info is not None:
                        tensor_info.inf_dtype = tensor_info.orig_dtype

def _find_gemm_node_before_given_node(node):
    r"""
    Find the gemm node from(including) cur node which can fused with elt_wise or add, which used by check
    whether has gemm node can be fused with elt_wise.
    """
    if node is not None:
        if check_node_in_give_op(node, gemm_op):
            return node
        elif check_node_in_give_op(node, [str(nn.Identity)]) and len(node.pre_nodes) > 0:
            return _find_gemm_node_before_given_node(node.pre_nodes[0])
        else:
            return None
    else:
        return None

def _find_add_node_before_given_node(node, ops):
    r"""
    Find the add node from(including) cur node which can fused with elt_wise , which used by check
    whether has add node can be fused with elt_wise.
    """
    if node is not None:
        if check_node_in_give_op(node, ops):
            return node
        elif check_node_in_give_op(node, [str(nn.Identity)]) and len(node.pre_nodes) > 0:
            return _find_add_node_before_given_node(node.pre_nodes[0], ops)
        else:
            return None
    else:
        return None

def _set_node_input_inf_dtype_to_orig_dtype(node):
    r"""
    This is function is about set the node's input inf dtype to orig_dtype, which can be used to
    avoid inserting fake quant befoe current node. 
    """
    for tensor_info in node.input_tensor_infos:
        if tensor_info is not None:
            tensor_info.inf_dtype = tensor_info.orig_dtype

def _check_has_quantized_node_before_node(node):
    r"""
    This function is about check whether has quantized op before(including) the given node,
    which is used to check whether insert fake quant before one quantized op(none-gemm op), for example:
    given_node->quantized_node, if the given node is fp32 node, we can avoid insert fake quant before the
    quantized_node for good performance(the post ops is also fp32 node).
    """
    if check_node_in_give_op(node, [str(nn.Identity)]):
        if len(node.pre_nodes) > 0:
            return _check_has_quantized_node_before_node(node.pre_nodes[0])
        else:
            return False
    else:
        # node is quantized node
        if node.qconfig is None:
            # check whether has fusion patten.
            elt_wise_inplace = [str(torch.relu_), str(torch.sigmoid_), str(nn.ReLU)]
            # support gemm(add)+multi-inplace ops.need to support such as gemm+relu_+relu_+quantizable op?
            if check_node_in_give_op(node, elt_wise_inplace) and len(node.pre_nodes) > 0 and \
                 (_find_gemm_node_before_given_node(node.pre_nodes[0]) is not None or \
                    _find_add_node_before_given_node(node.pre_nodes[0], add_op) is not None):
                 return True
            # check gemm+add_
            if check_node_in_give_op(node, add_inplace_op) and len(node.pre_nodes) > 0 and \
                (_find_gemm_node_before_given_node(node.pre_nodes[0]) is not None or \
                    (len(node.pre_nodes) == 2 and _find_gemm_node_before_given_node(node.pre_nodes[1]) is not None)):
                return True
            # gemm+add_+relu_(sigmoid_).
            if check_node_in_give_op(node, elt_wise_inplace) and len(node.pre_nodes) > 0 and \
                (_find_add_node_before_given_node(node.pre_nodes[0], add_inplace_op) is not None) and \
                len(node.pre_nodes[0].pre_nodes) > 0 and \
                    (_find_gemm_node_before_given_node(node.pre_nodes[0].pre_nodes[0]) is not None or \
                        (len(node.pre_nodes[0].pre_nodes) == 2 and \
                            _find_gemm_node_before_given_node(node.pre_nodes[0].pre_nodes[1]) is not None)):
                return True
            return False
        else:
            if check_node_in_give_op(node, ipex_customer_op):
                if check_node_in_give_op(node, [str(interaction), str(torch.ops.torch_ipex.interaction_forward)]):
                    for tensor_info in node.input_tensor_infos:
                        if tensor_info.inf_dtype  == torch.qint8:
                            return True
                    return False
                else:
                    # embeddingBag
                    if node.weight_tensor_infos[0].inf_dtype == torch.qint8:
                        return True
                    else:
                        return False
            else:
                # for none ipex customer op, we only need check has the qconfig even it's input inf dtype is
                # fp32(fused op).
                return True

def _check_has_quantized_node_after_node(node):
    r"""
    This function is about check whether has quantized op after(not including) the given node,
    which is used to check whether insert fake quant before one quantized op(none-gemm op), for example:
    quantized_node->after_nodel, if the after_nodel is fp32 node, we can avoid insert fake quant before the
    quantized_node for good performance(pre_op is also fp32 node).
    """
    if len(node.post_nodes) == 0:
        return False
    else:
        # make sure all post nodes are quantizabled. 
        for next in node.post_nodes:
            if next.qconfig is None:
                if check_node_in_give_op(next, [str(nn.Identity)]):
                    return _check_has_quantized_node_after_node(next)
                else:
                    return False
            else:
                if check_node_in_give_op(next, ipex_customer_op):
                    if check_node_in_give_op(next, [str(interaction), str(torch.ops.torch_ipex.interaction_forward)]):
                        for tensor_info in next.input_tensor_infos:
                            if tensor_info.inf_dtype != torch.qint8:
                                return False
                    else:
                        # embeddingBag
                        if next.weight_tensor_infos[0].inf_dtype != torch.qint8:
                            return False
        # all post nodes are quantizabled.
        return True

def _find_node_fused_with_elt_wise(cur_node):
    r"""
    Find gemm or add node which can fused with cur_node, which used by removing fake quant
    before elt_wise.
    gemm(add) -> fake quant-> elt_wise ==>  gemm(add) -> elt_wise.
    """
    if cur_node.pre_nodes == 0:
        return None
    else:
        if check_node_in_give_op(cur_node.pre_nodes[0], gemm_op + add_op + add_inplace_op):
            return cur_node.pre_nodes[0]
        elif check_node_in_give_op(cur_node.pre_nodes[0], [str(nn.Identity)]):
            return _find_node_fused_with_elt_wise(cur_node.pre_nodes[0])
        else:
            return None


def _add_recipe(node):
    '''
    Given  gemm     op             gemm       op                gemm       op
             \     /                 \       /                    \       /
              \   /       ==>      f_quant (f_quant?)     ==>      \    (f_quant?)
               \ /                     \   /                        \   /
               add                      add                          add
    '''
    gemm_node = None
    if check_node_in_give_op(node.pre_nodes[0], gemm_op):
        gemm_node = node.pre_nodes[0]
    if gemm_node is None and len(node.pre_nodes) == 2 and check_node_in_give_op(node.pre_nodes[1], gemm_op):
        gemm_node = node.pre_nodes[1]
    # if pre_nodes is nn.dentity, we should see the pre_node of pre_nodes
    if gemm_node is None and len(node.pre_nodes) > 0 and check_node_in_give_op(node.pre_nodes[0], [str(nn.Identity)]):
        pre_node = node.pre_nodes[0]
        if pre_node is not None and check_node_in_give_op(pre_node,  [str(nn.Identity)]):
           gemm_node =  _find_gemm_node_before_given_node(pre_node)
           if gemm_node is not None:
               # set gemm_node to node.pre_nodes[0], which is eaily to futher check.
               gemm_node = node.pre_nodes[0]
    if gemm_node is None and len(node.pre_nodes) == 2 and check_node_in_give_op(node.pre_nodes[1], [str(nn.Identity)]):
        pre_node = node.pre_nodes[1]
        if pre_node is not None and check_node_in_give_op(pre_node,  [str(nn.Identity)]):
           gemm_node =  _find_gemm_node_before_given_node(pre_node)
           if gemm_node is not None:
               gemm_node = node.pre_nodes[0]

    if gemm_node is None:
        # if pre_nodes don't have gemm nodel, need to check whether have quantizable op before or after it,
        #  if does't have quantizable op before or after it, we will not insert fake quant before add.
        has_post_quantized_op = _check_has_quantized_node_after_node(node)
        if len(node.pre_nodes) > 0:
            do_quantize_add_1 = True
            # case1: add(tensor, scalar)
            # case2: add(tensor, tensor), we only support int8+int8 case.
            if None not in node.input_tensor_infos and len(node.pre_nodes) == 1:
                do_quantize_add_1 = False
            else:
                has_pre_quantized_node = _check_has_quantized_node_before_node(node.pre_nodes[0])
                if not has_pre_quantized_node:
                    do_quantize_add_1 = False
            if len(node.pre_nodes) == 1:
                # add pre and post are all fp32 node.
                if not (do_quantize_add_1 or has_post_quantized_op):
                    _set_node_input_inf_dtype_to_orig_dtype(node)
            if len(node.pre_nodes) == 2:
                do_quantize_add_2 = True
                has_pre_quantized_node = _check_has_quantized_node_before_node(node.pre_nodes[1])
                if not has_pre_quantized_node:
                    do_quantize_add_2 = False
                # only support int8+int8 for add, if add pre and post node are fp32 nodes
                # set input inf dtype to orig dtype
                if not (do_quantize_add_1 and do_quantize_add_2 or has_post_quantized_op):
                    _set_node_input_inf_dtype_to_orig_dtype(node)
        else:
            if not has_post_quantized_op:
                _set_node_input_inf_dtype_to_orig_dtype(node)
    else:
        # add can fused with gemm.
        if node.input_tensor_infos[0] is not None and node.input_tensor_infos[0] in gemm_node.output_tensor_infos:
            node.input_tensor_infos[0].inf_dtype = node.input_tensor_infos[0].orig_dtype
            # set another input's dtype, if another's input is form fp32 op, we can remove the fake quant.
            if node.input_tensor_infos[1] is not None:
                if node.input_tensor_infos[1] in node.pre_nodes[0].output_tensor_infos:
                    pre_node = node.pre_nodes[0]
                elif len(node.pre_nodes) == 2 and node.input_tensor_infos[1] in node.pre_nodes[1].output_tensor_infos:
                    pre_node = node.pre_nodes[1]
                if not _check_has_quantized_node_before_node(pre_node):
                    node.input_tensor_infos[1].inf_dtype = node.input_tensor_infos[1].orig_dtype
        elif node.input_tensor_infos[1] is not None and node.input_tensor_infos[1] in gemm_node.output_tensor_infos:
            node.input_tensor_infos[1].inf_dtype = node.input_tensor_infos[1].orig_dtype
            # set another input's dtype, if another's input is form fp32 op, we can remove the fake quant.
            if node.input_tensor_infos[0] is not None:
                if node.input_tensor_infos[0] in node.pre_nodes[0].output_tensor_infos:
                    pre_node = node.pre_nodes[0]
                elif len(node.pre_nodes) == 2 and node.input_tensor_infos[0] in node.pre_nodes[1].output_tensor_infos:
                    pre_node = node.pre_nodes[1]
                if not _check_has_quantized_node_before_node(pre_node):
                    node.input_tensor_infos[0].inf_dtype = node.input_tensor_infos[0].orig_dtype

# get a default recipe
def get_defaut_recipe(nodes):
    r"""
    This function is about get default recipe which set where fake quant is inserted for the quantizable ops.
    """
    # step1: add q+dq before quantizabled op or module by setting the input's inf_dtype to 
    # qconfig.activation().dtype. and also setting the weight's inf_dtype to 
    # qconfig.weight().dtype if a module has a weight.
    _default_recipe_init(nodes)
    # step2: 1. remove q+dq for fusion op; 
    # 2. remove q+dq before none-gemm op from the quantized op's if its' pre ops are all none-quantizable op.
    for node in nodes:
        if isinstance(node, ParentNode):
            continue
        if node.qconfig is not None and not check_node_in_give_op(node, gemm_op + ipex_customer_op + lstm):
            # gemm+add fusion
            if check_node_in_give_op(node, add_op):
                _add_recipe(node)
            elif check_node_in_give_op(node, elt_wise):
                # gemm+elt_wise or add(_)+elt_wise fusion.
                has_post_quantized_op = _check_has_quantized_node_after_node(node)
                if len(node.pre_nodes) == 0:
                    if not has_post_quantized_op:
                        node.input_tensor_infos[0].inf_dtype = node.input_tensor_infos[0].orig_dtype
                else:
                    # has gemm(add) pre_op
                    if _find_node_fused_with_elt_wise(node) is not None:
                        node.input_tensor_infos[0].inf_dtype = node.input_tensor_infos[0].orig_dtype
                    else:
                        has_pre_quantized_node = _check_has_quantized_node_before_node(node.pre_nodes[0])
                        # if it's pre and post nodes are fp32 nodes, set input inf_dtype to orig_dtype.
                        if not (has_pre_quantized_node or has_post_quantized_op):
                             node.input_tensor_infos[0].inf_dtype = node.input_tensor_infos[0].orig_dtype
            else:
                # For none-quantizabled gemm and elt_wise, we don't need quantize it if it's prr and post ops
                # are all fp32 op, which can get a better performance.
                # Now all none-quantizabled gemm op only has one input info except for add:
                # so we can check the first input.
                # check whether has a pre qop.
                has_pre_quantized_node = True
                if len(node.pre_nodes) == 1:
                    has_pre_quantized_node = _check_has_quantized_node_before_node(node.pre_nodes[0])
                elif len(node.pre_nodes) == 0:
                    has_pre_quantized_node = False
                has_post_quantized_node = _check_has_quantized_node_after_node(node)
                # the node's pre and post node are all fp32 node.
                if not (has_pre_quantized_node or has_post_quantized_node):
                    node.input_tensor_infos[0].inf_dtype = node.input_tensor_infos[0].orig_dtype