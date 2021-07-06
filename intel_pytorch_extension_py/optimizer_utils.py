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
            # find the new model's param.
            new_model_param = dic_param[p]
            # for bf16 path, replace bf16 weight with master weight.
            if new_model_param in weight_params_attr and new_model_param.dtype == torch.bfloat16:
                new_param = weight_params_attr[new_model_param]['master_weight']
            else:
                new_param = new_model_param

            group2['params'][i] = new_param
            # copy optimizer's state.
            if p in optimizer.state:
                new_optimizer.state[new_param] = copy.deepcopy(optimizer.state[p])
                # It covers both conv and linear now. TODO: LSTM or other ops.
                if new_model_param in weight_params_attr:
                    attr = weight_params_attr[new_model_param]
                    state = new_optimizer.state[new_param]
                    for state_key, state_value in state.items():
                        if isinstance(state_value, torch.Tensor):
                            assert state_value.size() == p.size(), \
                                "Only support the optimizer state's size has the same shape with model's parameter."
                            if attr['op'] is torch.nn.Conv2d:
                                value_temp = state_value.to(memory_format=torch.channels_last) \
                                    if attr['weight_channels_last'] else state_value
                                state[state_key] = torch.ops.torch_ipex.conv2d_weight_prepack(
                                    value_temp,
                                    attr['padding'],
                                    attr['stride'],
                                    attr['dilation'],
                                    attr['groups'],
                                    attr['dtype'])
                            elif attr['op'] is torch.nn.Linear:
                                state[state_key] = torch.ops.torch_ipex.linear_weight_prepack(
                                    state_value,
                                    attr['out_features'],
                                    attr['in_features'],
                                    attr['dtype'])
    return new_optimizer


class _ipex_optimizer(torch.optim.Optimizer):
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

    def __init__(self, model, optimized_model, optimizer, weight_params_attr, dtype):
        self.optimizer = _optimizer_convert(model, optimized_model, optimizer, weight_params_attr)
        self.weight_params_attr = weight_params_attr
        self.param_groups = self.optimizer.param_groups
        self.dtype = dtype

    def state_dict(self):
        optimizer_temp = copy.deepcopy(self.optimizer)
        weight_params_attr_ = {}
        # For bf16 path, the optimizer's params are master weight,
        # but self.weight_params_attr's keys are bf16 weight, it hard to
        # query the weight's attr, so recreate a dic which using master weight
        # as key for easily to query.
        if self.dtype == torch.bfloat16:
           for _, values in self.weight_params_attr.items():
                master_weight = values['master_weight']
                weight_params_attr_[master_weight] = values
        else:
            weight_params_attr_ = self.weight_params_attr

        for (k1, _), (_, v2) in zip(self.optimizer.state.items(), optimizer_temp.state.items()):
            # unpack tensor state using weight's attr.
            if k1 in weight_params_attr_:
                weight_attr = weight_params_attr_[k1]
                for state_key, state_value in v2.items():
                    if isinstance(state_value, torch.Tensor):
                        # It covers both conv and linear now. TODO: LSTM or other ops.
                        if weight_attr['op'] is torch.nn.Conv2d:
                            v2[state_key] = torch.ops.torch_ipex.conv2d_weight_unpack(
                                state_value,
                                weight_attr['padding'],
                                weight_attr['stride'],
                                weight_attr['dilation'],
                                weight_attr['kernel_size'],
                                weight_attr['groups'],
                                weight_attr['out_channels'],
                                weight_attr['in_channels'],
                                weight_attr['weight_channels_last'],
                                weight_attr['dtype'])
                        elif weight_attr['op'] is torch.nn.Linear:
                            v2[state_key] = torch.ops.torch_ipex.linear_weight_unpack(
                                state_value,
                                weight_attr['out_features'],
                                weight_attr['in_features'],
                                weight_attr['weight_transposed'],
                                weight_attr['dtype'])
        return optimizer_temp.state_dict()

    def load_state_dict(self, state_dict):
        assert False, "_ipex_optimizer does not suppory load_state_dict"

    def zero_grad(self, set_to_none: bool = False):
        if self.dtype == torch.bfloat16:
            for p in self.weight_params_attr:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()

        self.optimizer.zero_grad(set_to_none)

    def step(self, closure=None):
        if self.dtype == torch.bfloat16:
            # convert bf16 weight'grad to float.
            for k, value in self.weight_params_attr.items():
                value["master_weight"].grad = k.grad.detach().to(torch.float)
        loss = self.optimizer.step(closure)
        # sync mater weight to model's paramerter
        if self.dtype == torch.bfloat16:
            for k, value in self.weight_params_attr.items():
                torch.ops.torch_ipex.sync_master_weight_to_bf16(value["master_weight"], k)
        return loss

