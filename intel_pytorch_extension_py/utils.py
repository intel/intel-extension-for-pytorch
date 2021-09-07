import copy

import torch
import torch.fx.experimental.optimization as optimization
import warnings

from .ops.lstm import IpexLSTM
from .weight_prepack import _weight_prepack_with_ipex
from .weight_cast import _weight_dtype_convert_with_ipex
from .optimizer_utils import _ipex_optimizer
import _torch_ipex as core

def _replace_dropout_with_identity(model):
    # replace dropout with identity during inference, so that aten::dropout won't be on the JIT graph.
    # This optimization may provide more fusion opportunites on the graph.
    if not model.training:
        for child_name, child in model.named_children():
            if isinstance(child, torch.nn.Dropout):
                setattr(model, child_name, torch.nn.Identity())
            else:
                _replace_dropout_with_identity(child)

def _replace_lstm_with_ipex_lstm(model):
    # replace lstm with ipex lstm during inference
    # does not support the case where model itself is torch.nn.LSTM
    if not model.training:
        for child_name, child in model.named_children():
            if isinstance(child, torch.nn.LSTM):
                assert hasattr(child, "weight_ih_l0"), "torch.nn.LSTM should have weight_ih_l0"
                ipex_lstm = IpexLSTM(child.input_size, child.hidden_size,
                    child.num_layers, child.bias, child.batch_first,
                    child.dropout, child.bidirectional, child.proj_size,
                    child.weight_ih_l0.device, child.weight_ih_l0.dtype)
                ipex_lstm.__dict__ = copy.deepcopy(child.__dict__)
                setattr(model, child_name, ipex_lstm)
            else:
                _replace_lstm_with_ipex_lstm(child)

def _convert_module_data_type(module, dtype):
    # convert weights(bias) of module to dtype to reduce dtype reorder
    module_convert_list = [torch.nn.Conv2d,
                           torch.nn.Linear,
                           torch.nn.Embedding,
                           torch.nn.LayerNorm]
    for module_cls in module_convert_list:
        if isinstance(module, module_cls):
            weight_data = module.weight.detach().clone().to(dtype)
            module.weight.data = weight_data
            if hasattr(module, 'bias') and module.bias is not None:
                bias_data = module.bias.detach().clone().to(dtype)
                module.bias.data = bias_data
            break
    for child in module.children():
        _convert_module_data_type(child, dtype)
    return module

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
        self.opt_level=None,
        self.conv_bn_folding=None,
        self.weights_prepack=None,
        self.master_weight=None

# O0 properties
class _O0:
    def __call__(self, properties):
        properties.opt_level="O0"
        properties.conv_bn_folding=False
        properties.weights_prepack=False
        #properties.master_weight=False
        return properties


# O1 properties
class _O1:
    def __call__(self, properties):
        properties.opt_level="O1"
        properties.conv_bn_folding=True
        properties.weights_prepack=True
        #properties.master_weight=True
        return properties

opt_levels = {"O0": _O0(),
              "O1": _O1()}

def optimize(
    model,
    dtype=torch.bfloat16,
    optimizer=None,
    level="O1",
    inplace=False,
    conv_bn_folding=None,
    weights_prepack=None):
    r"""
    Convert user to ipex optimzied model, ther will be do conv+bn folding, model's parameters data dtype
    conversation for Convolution, Linear, Embedding, and layerNorm. there also has a weight prepack for
    Convoluttion and Linear for better performance.

    Args:
        model: (torch.nn.Module): user model to do optimization.
        dtype: it can be torch.float or torch.bfloat16, it will do model's parameters data dtype cast if
            dtype is torch.bfloat16, the default value is torch.bfloat16.
        optimizer: (optim.Optimizer), user optimzizer to do optimization, suach as split-sgd, the default
            value is None, it means for inference case.
        level: can be 'O0' or 'O1', do nothing for 'O0', just return the origin model and optimizer,
            'O1' will do ipex optimization as abrove said, the default value is 'O1'.
        inplace: whether do inplace optimization, default value if False.
        conv_bn_folding: whether do conv_bn folding, it only works for inference model, the default value is True.
        weights_prepack: whether do weight prepack for convolution and linear to avoid OneDNN weight reorder.
            the default value is True(only workd for training model, the inference model is not optimized well).

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
        if dtype == torch.bfloat16:
            optimized_model = _convert_module_data_type(optimized_model, torch.bfloat16)

    # convert optimizer for training case.
    params_attr = {}
    #TODO: add option master_weight for bf16 case
    if dtype == torch.bfloat16 and model.training:
        optimized_model, optimized_optimizer, params_attr = _weight_dtype_convert_with_ipex(optimized_model, optimized_optimizer, params_attr)

    #TODO enable inference weight prepack as default.
    if opt_properties.weights_prepack and model.training:
        optimized_model, optimized_optimizer, params_attr = _weight_prepack_with_ipex(optimized_model, optimized_optimizer, params_attr)

    # TODO: model list, optimizer list.
    if optimizer is None:
        return optimized_model
    else:
        return optimized_model, optimized_optimizer


VERBOSE_OFF = 0
VERBOSE_ON = 1
VERBOSE_ON_CREATION = 2
class verbose(object):
    def __init__(self, level):
        self.level = level

    def __enter__(self):
        if self.level == VERBOSE_OFF:
            return
        try:
            st = torch._C._verbose.mkldnn_set_verbose(self.level)
            assert bool(st), "Failed to set Verbose mode of MKLDNN in PyTorch. Please consider to disable this verbose scope."
        except:
            pass
        st = core.mkldnn_set_verbose(self.level)
        assert bool(st), "Failed to set Verbose mode of MKLDNN in IPEX. Please consider to disable this verbose scope."
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        core.mkldnn_set_verbose(VERBOSE_OFF)
        try:
            torch._C._verbose.mkldnn_set_verbose(VERBOSE_OFF)
        except:
            pass
        return False

try:
    verbose_torch=torch.backends.mkldnn.verbose
    torch.backends.mkldnn.verbose = verbose
except:
    pass
