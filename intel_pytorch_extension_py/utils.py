import copy

import torch
import warnings

from .ops.lstm import IpexLSTM
from .fx import *
from .weight_prepack import _weight_prepack_with_ipex
from .optimizer_utils import _ipex_optimizer

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

def optimize(model, dtype=torch.bfloat16, optimizer=None, level='O1', inplace=False):
    if inplace:
        optimized_model = model
        optimized_optimizer = optimizer
    else:
        optimized_model, optimized_optimizer = _copy_model_and_optimizer(model, optimizer)
    if level == 'O0':
        # will be removed after customer op can be traced with autocast,
        # see https://github.com/pytorch/pytorch/pull/60251.
        # after removed, will directly return original model and optimizer.
        if not model.training:
            try:
                optimized_model = conv_bn_fuse(optimized_model, inplace=inplace)
            except:
                warnings.warn("Conv BN folding failed during the optimize process.")
            # do weight data type convert for inference model.
            if dtype == torch.bfloat16:
                optimized_model = _convert_module_data_type(optimized_model, torch.bfloat16)
    elif level == 'O1':
        if not model.training:
            try:
                optimized_model = conv_bn_fuse(optimized_model, inplace=inplace)
            except:
                warnings.warn("Conv BN folding failed during the optimize process.")

        # Do weight prepack, and convert optimizer for training case.
        optimized_model, optimized_optimizer, weight_params_attr = _weight_prepack_with_ipex(optimized_model, optimized_optimizer, dtype)
        if optimizer is not None:
            assert model.training, "please call model.train() if you want to convert the optimizer to ipex optimizer."
            optimized_optimizer = _ipex_optimizer(optimized_optimizer, weight_params_attr, dtype)
    else:
        assert False, "Only support level O0 and O1 now for optimize"

    # TODO: model list, optimizer list.
    if optimizer is None:
        return optimized_model
    else:
        return optimized_model, optimized_optimizer
