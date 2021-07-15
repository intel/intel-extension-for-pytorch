import torch
import copy
class _ipex_optimizer(torch.optim.Optimizer):
    """
    Convert user's optimizer to ipex optimizer, it is a temporary implementation,
    it will be removed by directly overwrite optimizer's methods.

    Args:
        optimizer: optimized optimizer, contains optimized model's paramerter setting.
        weight_params_attr: the prepacked parameters' attrs, to do prepack for corresponding
            momentum_buffer or other state according those attrs.
        dtype: can be torch.bfloat16 or torch.float32(torch.float), determin doing bfloat16 training
            or float training.
    """

    def __init__(self, optimizer, weight_params_attr, dtype):
        self.optimizer = optimizer
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

