import copy

import torch
import torch.fx.experimental.optimization as optimization
import warnings

from .nn import utils
from .optim._optimizer_utils import optimizer_fusion, IPEX_FUSED_OPTIMIZER_LIST
import intel_extension_for_pytorch._C as core


def _copy_model_and_optimizer(model, optimizer):
    new_model = copy.deepcopy(model)
    if optimizer is None:
        return new_model, optimizer
    else:
        new_optimizer = copy.deepcopy(optimizer)
        new_optimizer.state.clear()
        dic_param = {}
        for k, value in zip(model.parameters(), new_model.parameters()):
            dic_param[k] = value
        for group1, group2 in zip(optimizer.param_groups, new_optimizer.param_groups):
            for i, p in enumerate(group1['params']):
                new_model_param = dic_param[p]
                group2['params'][i] = new_model_param
                new_optimizer.state[new_model_param] = copy.deepcopy(optimizer.state[p])
        return new_model, new_optimizer

class _Properties(object):
    r"""
    This class is to establish a set of default properties.

    """
    def __init__(self):
        self.opt_level = None
        self.conv_bn_folding = None
        self.weights_prepack = None
        self.remove_dropout = None
        # optimizer opt conig
        self.split_master_weight_for_bf16 = None
        self.fuse_update_step = None
        self.auto_kernel_selection = None

# O0 properties
class _O0:
    def __call__(self, properties):
        properties.opt_level = "O0"
        properties.conv_bn_folding = False
        properties.weights_prepack = False
        properties.remove_dropout = False
        properties.split_master_weight_for_bf16 = False
        properties.fuse_update_step = False
        properties.auto_kernel_selection = False
        return properties


# O1 properties
class _O1:
    def __call__(self, properties):
        properties.opt_level = "O1"
        properties.conv_bn_folding = True
        properties.weights_prepack = True
        properties.remove_dropout = True
        properties.split_master_weight_for_bf16 = True
        properties.fuse_update_step = True
        properties.auto_kernel_selection = False
        return properties

opt_levels = {"O0": _O0(),
              "O1": _O1()}

def optimize(
    model,
    dtype=torch.float,
    optimizer=None,
    level="O1",
    inplace=False,
    conv_bn_folding=None,
    weights_prepack=None,
    remove_dropout=None,
    split_master_weight_for_bf16=None,
    fuse_update_step=None,
    auto_kernel_selection=None):
    r"""
    Convert user to ipex optimzied model, ther will be do conv+bn folding, model's parameters data dtype
    conversation for Convolution, Linear, Embedding. there also has a weight prepack for
    Convoluttion and Linear for better performance.

    Args:
        model: (torch.nn.Module): user model to do optimization.
        dtype: it can be torch.float or torch.bfloat16, it will do model's parameters data dtype cast if
            dtype is torch.bfloat16, the default value is torch.float.
        optimizer: (optim.Optimizer), user optimzizer to do optimization, suach as split-sgd, the default
            value is None, it means for inference case.
        level: can be 'O0' or 'O1', do nothing for 'O0', just return the origin model and optimizer,
            'O1' will do ipex optimization as abrove said, the default value is 'O1'.
        inplace: whether do inplace optimization, default value is None.
        conv_bn_folding: whether do conv_bn folding, it only works for inference model, the default value is None.
        weights_prepack: whether do weight prepack for convolution and linear to avoid OneDNN weight reorder.
            the default value is None(only workd for training model, the inference model is not optimized well).
        remove_dropout: whether remove dropout from model, it only works for inference model, the default value is None.
            the default value is True(only workd for training model, the inference model is not optimized well).
        split_master_weight_for_bf16: whether choose split master weight update for BF16 training which can
            save memory compare with master weight update solution, not support all optimizers
        fuse_update_step: whether choose fused params update for training which have better performance,
           not support all optimizers
        [experimental] auto_kernel_selection: Different backend may have different performance on different
            dtypes/shapes. Default value is False. IPEX will try to optimize the kernel selection for
            better performance if set this value to True. But may have regressions at current stage.

    """

    if model.training:
        assert optimizer is not None, "The optimizer should be given for training mode"
    else:
        assert optimizer is None, "The optimizer should not be given for inference mode"

    opt_properties = _Properties()
    if level not in opt_levels:
        raise RuntimeError(
            "Unexpected optimization level {}. ".format(level) +
            "Options are 'O0', 'O1'.")
    else:
        opt_properties = opt_levels[level](opt_properties)

    if level is not None:
        opt_properties.opt_level = level
    if conv_bn_folding is not None:
        opt_properties.conv_bn_folding = conv_bn_folding
    if weights_prepack is not None:
        opt_properties.weights_prepack = weights_prepack
    if remove_dropout is not None:
        opt_properties.remove_dropout = remove_dropout
    if split_master_weight_for_bf16 is not None:
        opt_properties.split_master_weight_for_bf16 = split_master_weight_for_bf16
    if fuse_update_step is not None:
        opt_properties.fuse_update_step = fuse_update_step
    if auto_kernel_selection is not None:
        opt_properties.auto_kernel_selection = auto_kernel_selection

    if inplace:
        optimized_model = model
        optimized_optimizer = optimizer
    else:
        optimized_model, optimized_optimizer = _copy_model_and_optimizer(model, optimizer)

    if not model.training:
        if opt_properties.conv_bn_folding:
            try:
                optimized_model = optimization.fuse(optimized_model, inplace=inplace)
            except:
                warnings.warn("Conv BatchNorm folding failed during the optimize process.")
        if opt_properties.remove_dropout:
            try :
                optimized_model = optimization.remove_dropout(optimized_model)
            except:
                warnings.warn("Failed to remove the Dropout module during the optimize process.")
        if dtype == torch.bfloat16:
            optimized_model = utils._model_convert.convert_module_data_type(optimized_model, torch.bfloat16)

    if opt_properties.split_master_weight_for_bf16 and dtype is torch.bfloat16:
        if not opt_properties.fuse_update_step:
            opt_properties.split_master_weight_for_bf16 = False
            warninig.warn(
                "IPEX does not non-fused split master weight for bf16 training," +
                "have reset split_master_weight_for_bf16 flag to False." +
                "If you want to use split_master_weight_for_bf16." +
                "Please set both split_master_weight_for_bf16 and fuse_update_step to True")
        elif type(optimizer) not in IPEX_FUSED_OPTIMIZER_LIST:
            opt_properties.split_master_weight_for_bf16 = False
            opt_properties.fuse_update_step = False
            warnings.warn(
                "IPEX does not support fused/fused split update for" + str(type(optimizer)) +
                "will use non-fused master weight update for bf16 training")

    # convert optimizer for training case.
    params_attr = {}
    if dtype == torch.bfloat16:
        optimized_model, optimized_optimizer, params_attr = utils._weight_cast.weight_dtype_convert_with_ipex(
            optimized_model, optimized_optimizer, params_attr, opt_properties.split_master_weight_for_bf16)
    if opt_properties.weights_prepack:
        optimized_model, optimized_optimizer, params_attr = utils._weight_prepack.weight_prepack_with_ipex(
          optimized_model, optimized_optimizer, params_attr, opt_properties.auto_kernel_selection)
    # TODO: model list, optimizer list.
    if optimizer is None:
        return optimized_model

    # with an optimizer
    if opt_properties.fuse_update_step:
        optimized_optimizer = optimizer_fusion(
            optimized_optimizer, opt_properties.split_master_weight_for_bf16)
    return optimized_model, optimized_optimizer

