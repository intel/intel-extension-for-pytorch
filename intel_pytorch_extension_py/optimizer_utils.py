import torch
import copy

def _optimizer_convert(model, optimized_model, optimizer, weight_params_attr):
    """
    Convert user's optimizer state to expected state, for example, some optimizer has
    momentum_buffer, need make sure the momentum_buffer is also preacked if the corresponding
    parameter has been prepacked.

    Args:
        see args in _ipex_optimizer.
    """
    new_optimizer = copy.deepcopy(optimizer)
    dic_param = {}
    for k, value in zip(model.parameters(), optimized_model.parameters()):
        dic_param[k] = value
    new_optimizer.state.clear()
    for group1, group2 in zip(optimizer.param_groups, new_optimizer.param_groups):
        for i, p in enumerate(group1['params']):
            # find the new tensor
            new_param = dic_param[p]
            group2['params'][i] = new_param
            # copy optimizer's state
            if p in optimizer.state:
                new_optimizer.state[new_param] = copy.deepcopy(optimizer.state[p])
                # for conv weight, preprack momentum_buffer using weight's attr.
                # TODO: Linear and LSTM.
                if new_param in weight_params_attr:
                    attr = weight_params_attr[new_para]
                    # TODO: Linear or other ops.
                    if attr['op'] is torch.nn.Conv2d:
                        new_optimizer.state[new_param]['momentum_buffer'] = torch.ops.torch_ipex.conv2d_weight_prepack(
                            new_optimizer.state[new_param]['momentum_buffer'],
                            attr['padding'],
                            attr['stride'],
                            attr['dilation'],
                            attr['groups'],
                            attr['dtype'])
                    if attr['op'] is torch.nn.Linear:
                        new_optimizer.state[new_param]['momentum_buffer'] = torch.ops.torch_ipex.linear_weight_prepack(
                            new_optimizer.state[new_param]['momentum_buffer'],
                            attr['out_features'],
                            attr['in_features'],
                            attr['dtype'])
    return new_optimizer

class _ipex_optimizer(object):
    """
    Convert user's optimizer to ipex optimizer, it is a temporary implementation,
    it will be removed by directly overwrite optimizer's methods.

    Args:
        model: original user model, it is a PyTorch model.
        optimized_model: a model which parameters has been prepacked from user model.
        optimizer: original optimizer defined by user, contains model's paramerter setting.
        weight_params_attr: the prepacked parameters' attrs, to do prepack for corresponding
            momentum_buffer or other state according those attrs.
    """

    def __init__(self, model, optimized_model, optimizer, weight_params_attr):
        if isinstance(optimizer, torch.optim.SGD):
            self.optimizer = _optimizer_convert(model, optimized_model, optimizer, weight_params_attr)
        else:
            # TODO: other optimizers
            assert False, "optimizer is not supported now for IPEX"
        self.weight_params_attr = weight_params_attr
        self.param_groups = self.optimizer.param_groups

    def state_dict(self):
        optimizer_temp = copy.deepcopy(self.optimizer)
        # need conver master weight's momentum to plain format.
        for (k1, _), (_, v2) in zip(self.optimizer.state.items(), optimizer_temp.state.items()):
            if k1 in self.weight_params_attr:
                #  for sgd optimizer, unpack momentum_buffer using weight's attr.
                if isinstance(self.optimizer, torch.optim.SGD):
                    if k1 in self.weight_params_attr:
                        weight_attr = self.weight_params_attr[k1]
                        if weight_attr['op'] is torch.nn.Conv2d:
                            # change optimizer_temp's momentum_buffer, origin optimizer should not be changed.
                            v2['momentum_buffer'] = torch.ops.torch_ipex.conv2d_weight_unpack(
                                v2['momentum_buffer'],
                                weight_attr['padding'],
                                weight_attr['stride'],
                                weight_attr['dilation'],
                                weight_attr['kernel_size'],
                                weight_attr['groups'],
                                weight_attr['out_channels'],
                                weight_attr['in_channels'],
                                weight_attr['weight_channels_last'],
                                weight_attr['dtype'])
                        if weight_attr['op'] is torch.nn.Linear:
                            # change optimizer_temp's momentum_buffer, origin optimizer should not be changed.
                            v2['momentum_buffer'] = torch.ops.torch_ipex.linear_weight_unpack(
                                v2['momentum_buffer'],
                                weight_attr['out_features'],
                                weight_attr['in_features'],
                                weight_attr['dtype'])
                else:
                    # TODO: other optimizer
                    assert False, "only sgd optmizer's state_dict is supported now for IPEX"
        return optimizer_temp.state_dict()

    def load_state_dict(self, state_dict):
        assert False, "_ipex_optimizer does not suppory load_state_dict"

    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad(set_to_none)

    def step(self, closure=None):
        return self.optimizer.step(closure)

