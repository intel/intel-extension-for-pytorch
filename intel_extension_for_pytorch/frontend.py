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
        properties.replace_dropout_with_identity = False
        properties.optimize_lstm = False
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
        properties.replace_dropout_with_identity = True
        properties.optimize_lstm = True
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
    replace_dropout_with_identity=None,
    optimize_lstm=None,
    split_master_weight_for_bf16=None,
    fuse_update_step=None,
    auto_kernel_selection=None):
    r"""
    Apply optimizations at the python frontend to the given model (nn.Module) and optimizer.If the optimizer is given,
    optimization for training is assumed, otherwise, optimization for inference is assumed. The optimizations include
    conv+bn folding (for inference only), weight prepacking and a lot more.

    Args:
        model (torch.nn.Module): User model to do optimization.
        dtype (torch.dtype): It can be torch.float(torch.float32) or torch.bfloat16,
          it will do model's parameters data dtype cast if dtype is torch.bfloat16, the default value is torch.float.
        optimizer (torch.optim.Optimizer), User optimzizer to do optimization, such as sgd, the default
          value is None, it means for inference case.
        level (string): Can be "O0" or "O1", do nothing for "O0", just return the origin model and optimizer.
          "O1" will do ipex optimization: conv+bn folding, weights prepack, remove dropout(inferenc model),
          split master wieght and fuse optimizer update step(training model), the optimization options can be
          further overridden by explicit options below. The default value is "O1".
        inplace (bool): Whether do inplace optimization, default value False.
        conv_bn_folding (bool): Whether do conv_bn folding, it only works for inference model.
          The default value is None, if has value, it will override the level's setting.
        weights_prepack (bool): Whether do weight prepack for convolution and linear to avoid OneDNN weight reorder.
          For OneDNN deep neural network library, in order to achieve better vectorization and cache reuse, onednn will use
          blocked layout that splits one or several dimensions into the blocks of fixed size, so there will do prepack to avoid
          online weight data format convertion which will reduce momory copy consumption, see more details about EneDNN data
          mermory format: https://oneapi-src.github.io/oneDNN/dev_guide_understanding_memory_formats.html. The default value is None,
          if has value, it will override the level's setting.
        replace_dropout_with_identity (bool): Whether replace nn.Dropout with nn.Identity, if replaced, the aten::dropout
          won't be on the JIT graph, which may provide more fusion opportunites on the graph, it only works for inference
          model. The default value is None, if has value, it will override the level's setting.
        replace_lstm_with_ipex_lstm (bool): Whether replace nn.LSTM with IPEX LSTM which apply OneDNN kernel to get better
          performance. The default value is None, if has value, it will override the level's setting.
        split_master_weight_for_bf16 (bool): Whether choose split master weight update for BF16 training which can
            save memory compare with master weight update solution, not support all optimizers.
            The default value is None, if has value, it will override the level's setting.
        fuse_update_step (bool): Whether choose fused params update for training which have better performance,
           not support all optimizers. The default value is None, if has value, it will override the level's setting.
        [experimental] auto_kernel_selection (bool): Different backend may have different performance on different
            dtypes/shapes. Default value is False. IPEX will try to optimize the kernel selection for
            better performance if set this value to True. But may have regressions at current stage.
            The default value is None, if has value, it will override the level's setting.

    Returns:
        model and optimizer(given a optimizer) modified according to the 'level' or other user's setting. conv+bn folding may be
        happend and dropout may be replaced by identity if model have for inference case, for convolutuon, linear and lstm, they will
        be replaced by our custom ops(weight prepack for convolution and linear) for good performance. For bfloat16 case,
        the parameters of convolution and linear will be bfloat16 dtype. For training case, and the optimizer states will changed to the
        converted model's parameters to align with model's parameter's update at optimizer's step.

    .. warning::

        ipex.optimize deepcopy the origin model. If DDP comes before ipex.optimize, it just gets the origin model,
        which is not the same as the one ipex.optimize returns. Therefore, some ops in DDP like allreduce will not be called
        and may cause unpredictable accuracy loss in distributed model training.

    Examples::

        >>> # bfloat16 inference case.
        >>> model = ...
        >>> model.eval()
        >>> optimized_model = ipex.optimize(model, dtype=torch.bfloat16)
        >>> # running evaluation step.
        >>> # bfloat16 training case.
        >>> optimizer = ...
        >>> model.train()
        >>> optimized_model, optimized_optimizer = ipex.optimize(model, dtype=torch.bfloat16, optimizer=optimizer)
        >>> # running training step.

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
    if replace_dropout_with_identity is not None:
        opt_properties.replace_dropout_with_identity = replace_dropout_with_identity
    if optimize_lstm is not None:
        opt_properties.optimize_lstm = optimize_lstm
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
        if opt_properties.replace_dropout_with_identity:
            utils._model_convert.replace_dropout_with_identity(optimized_model)
        if dtype == torch.bfloat16:
            optimized_model = utils._model_convert.convert_module_data_type(optimized_model, torch.bfloat16)

    if opt_properties.optimize_lstm:
        utils._model_convert.replace_lstm_with_ipex_lstm(optimized_model)
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

def enable_onednn_fusion(enabled):
    if enabled:
        core.enable_jit_opt()
    else:
        core.disable_jit_opt()
