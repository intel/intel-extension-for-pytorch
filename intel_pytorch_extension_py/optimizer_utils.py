import torch
import copy
from .optim import *

IPEX_OPTIMIZER_MAPPING = {
  torch.optim.SGD: SplitSGD,
  torch.optim.Adagrad: FusedSplitAdagrad,
}

class _ipex_optimizer(torch.optim.Optimizer):
    """
    Convert user's optimizer to ipex optimizer, it is a temporary implementation,
    it will be removed by directly overwrite optimizer's methods.

    Args:
        optimizer: optimized optimizer, contains optimized model's paramerter setting.
        params_attr: the  parameters' attrs, to cat top_half and bottom(trail) half back to fp32

    """

    def __init__(self, optimizer, params_attr):
        if type(optimizer) in IPEX_OPTIMIZER_MAPPING:
            self.optimizer = IPEX_OPTIMIZER_MAPPING[type(optimizer)] (optimizer, params_attr)
            self.master_weight_split  = True
        else:
            self.optimizer = optimizer
            self.master_weight_split  = False
        self.params_attr = params_attr
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state

    def load_state_dict(self, state_dict):
        assert False, "_ipex_optimizer does not suppory load_state_dict"

    def zero_grad(self, set_to_none: bool = False):
        if not self.master_weight_split:
            for p in self.params_attr:
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
        if not self.master_weight_split:
            # convert bf16 weight'grad to float.
            for k, value in self.params_attr.items():
                value["master_param"].grad = k.grad.detach().to(torch.float)
        loss = self.optimizer.step(closure)
        # sync mater weight to model's paramerter
        if not self.master_weight_split:
            for k, value in self.params_attr.items():
                torch.ops.torch_ipex.sync_master_weight_to_bf16(value["master_param"], k)
        return loss

