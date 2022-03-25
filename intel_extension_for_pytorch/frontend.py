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
    dtype=None,
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
    Apply optimizations at Python frontend to the given model (nn.Module), as
    well as the given optimizer (optional). If the optimizer is given,
    optimizations will be applied for training. Otherwise, optimization will be
    applied for inference. Optimizations include ``conv+bn`` folding (for
    inference only), weight prepacking and so on.

    Weight prepacking is a technique to accelerate performance of oneDNN
    operators. In order to achieve better vectorization and cache reuse, onednn
    uses a specific memory layout called ``blocked layout``. Although the
    calculation itself with ``blocked layout`` is fast enough, from memory usage
    perspective it has drawbacks. Running with the ``blocked layout``, oneDNN
    splits one or several dimensions of data into blocks with fixed size each
    time the operator is executed. More details information about oneDNN data
    mermory format is available at `oneDNN manual <https://oneapi-src.github.io/oneDNN/dev_guide_understanding_memory_formats.html>`_.
    To reduce this overhead, data will be converted to predefined block shapes
    prior to the execution of oneDNN operator execution. In runtime, if the data
    shape matches oneDNN operator execution requirements, oneDNN won't perform
    memory layout conversion but directly go to calculation. Through this
    methodology, called ``weight prepacking``, it is possible to avoid runtime
    weight data format convertion and thus increase performance.

    Args:
        model (torch.nn.Module): User model to apply optimizations on.
        dtype (torch.dtype): Only works for ``torch.bfloat16``.
            Model parameters will be casted to ``torch.bfloat16`` if dtype is set to
            ``torch.bfloat16``. The default value is None, meaning do nothing.
            Note: Data type conversion is only applied to ``nn.Conv2d``, ``nn.Linear``
            and ``nn.ConvTranspose2d`` for both training and inference cases. For
            inference mode, additional data type conversion is applied to the weights
            of ``nn.Embedding`` and ``nn.LSTM``.
        optimizer (torch.optim.Optimizer): User optimizer to apply optimizations
            on, such as SGD. The default value is ``None``, meaning inference case.
        level (string): ``"O0"`` or ``"O1"``. No optimizations are applied with
            ``"O0"``. The optimizer function just returns the original model and
            optimizer. With ``"O1"``, the following optimizations are applied:
            conv+bn folding, weights prepack, dropout removal (inferenc model),
            master weight split and fused optimizer update step (training model).
            The optimization options can be further overridden by setting the
            following options explicitly. The default value is ``"O1"``.
        inplace (bool): Whether to perform inplace optimization. Default value is
            ``False``.
        conv_bn_folding (bool): Whether to perform ``conv_bn`` folding. It only
            works for inference model. The default value is ``None``. Explicitly
            setting this knob overwrites the configuration set by ``level`` knob.
        weights_prepack (bool): Whether to perform weight prepack for convolution
            and linear to avoid oneDNN weights reorder. The default value is
            ``None``. Explicitly setting this knob overwrites the configuration
            set by ``level`` knob.
        replace_dropout_with_identity (bool): Whether to replace ``nn.Dropout``
            with ``nn.Identity``. If replaced, the ``aten::dropout`` won't be
            included in the JIT graph. This may provide more fusion opportunites
            on the graph. This only works for inference model. The default value
            is ``None``. Explicitly setting this knob overwrites the configuration
            set by ``level`` knob.
        optimize_lstm (bool): Whether to replace ``nn.LSTM`` with ``IPEX LSTM``
            which takes advantage of oneDNN kernels to get better performance.
            The default value is ``None``. Explicitly setting this knob
            overwrites the configuration set by ``level`` knob.
        split_master_weight_for_bf16 (bool): Whether to split master weights
            update for BF16 training. This saves memory comparing to master
            weight update solution. Split master weights update methodology
            doesn't support all optimizers. The default value is None. The
            default value is ``None``. Explicitly setting this knob overwrites
            the configuration set by ``level`` knob.
        fuse_update_step (bool): Whether to use fused params update for training
            which have better performance. It doesn't support all optimizers.
            The default value is ``None``. Explicitly setting this knob
            overwrites the configuration set by ``level`` knob.
        auto_kernel_selection (bool) [experimental]: Different backends may have
            different performances with different dtypes/shapes. Default value
            is False. Intel® Extension for PyTorch* will try to optimize the
            kernel selection for better performance if this knob is set to
            ``True``. There might be regressions at current stage. The default
            value is ``None``. Explicitly setting this knob overwrites the
            configuration set by ``level`` knob.

    Returns:
        Model and optimizer (if given) modified according to the ``level`` knob
        or other user settings. ``conv+bn`` folding may take place and
        ``dropout`` may be replaced by ``identity``. In inference scenarios,
        convolutuon, linear and lstm will be replaced with the optimized
        counterparts in Intel® Extension for PyTorch* (weight prepack for
        convolution and linear) for good performance. In bfloat16 scenarios,
        parameters of convolution and linear will be casted to bfloat16 dtype.

    .. warning::

        Please invoke ``optimize`` function AFTER loading weights to model via
        ``model.load_state_dict(torch.load(PATH))``.

    .. warning::

        Please invoke ``optimize`` function BEFORE invoking DDP in distributed
        training scenario.

        The ``optimize`` function deepcopys the original model. If DDP is invoked
        before ``optimize`` function, DDP is applied on the origin model, rather
        than the one returned from ``optimize`` function. In this case, some
        operators in DDP, like allreduce, will not be invoked and thus may cause
        unpredictable accuracy loss.

    Examples:

        >>> # bfloat16 inference case.
        >>> model = ...
        >>> model.load_state_dict(torch.load(PATH))
        >>> model.eval()
        >>> optimized_model = ipex.optimize(model, dtype=torch.bfloat16)
        >>> # running evaluation step.
        >>> # bfloat16 training case.
        >>> optimizer = ...
        >>> model.train()
        >>> optimized_model, optimized_optimizer = ipex.optimize(model, dtype=torch.bfloat16, optimizer=optimizer)
        >>> # running training step.

    """
    if isinstance(model, torch.jit.ScriptModule):
        if optimizer is None:
            return model
        return model, optimizer

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
    if model.training and opt_properties.split_master_weight_for_bf16 and dtype is torch.bfloat16:
        if not opt_properties.fuse_update_step:
            opt_properties.split_master_weight_for_bf16 = False
            warnings.warn(
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
    if dtype == torch.bfloat16 and model.training:
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
    r"""
    Enables or disables oneDNN fusion functionality. If enabled, oneDNN
    operators will be fused in runtime, when intel_extension_for_pytorch
    is imported.

    Args:
        enabled (bool): Whether to enable oneDNN fusion functionality or not.
            Default value is ``True``.

    Examples:

        >>> import intel_extension_for_pytorch as ipex
        >>> # to enable the oneDNN fusion
        >>> ipex.enable_onednn_fusion(True)
        >>> # to disable the oneDNN fusion
        >>> ipex.enable_onednn_fusion(False)
    """

    if enabled:
        core.enable_jit_opt()
    else:
        core.disable_jit_opt()
