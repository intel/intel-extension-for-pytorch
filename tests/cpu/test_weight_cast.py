import unittest
import copy

import torch
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.nn.utils._weight_cast import weight_dtype_convert_with_ipex as cast
from intel_extension_for_pytorch.nn.utils._weight_cast import IPEX_WEIGHT_CAST_MODULE as IPEX_WEIGHT_CAST_MODULE
from intel_extension_for_pytorch.optim._optimizer_utils import IPEX_FUSED_OPTIMIZER_LIST as IPEX_FUSED_OPTIMIZER_LIST

from torch.testing._internal.common_utils import TestCase
from torch.optim import Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, RMSprop, Rprop, SGD

class TestModule(torch.nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.linear = torch.nn.Linear(5, 10)
        self.conv1d = torch.nn.Conv1d(1, 10, 5, 1)
        self.conv2d = torch.nn.Conv2d(1, 10, 5, 1)
        self.conv3d = torch.nn.Conv3d(1, 10, 5, 1)
        self.transpose_conv1d = torch.nn.ConvTranspose1d(1, 10, 5, 1)
        self.transpose_conv2d = torch.nn.ConvTranspose2d(1, 10, 5, 1)
        self.transpose_conv3d = torch.nn.ConvTranspose3d(1, 10, 5, 1)
        self.bn = torch.nn.BatchNorm2d(num_features=10)
        self.embeddingbag = torch.nn.EmbeddingBag(10, 3, mode='sum')

    def forward(self, x):
        x = self.conv2d(x)
        return

class TestWeightCastCases(TestCase):
    def is_master_weight_solution(self, module, split_master_weight_for_bf16):
        return type(module) in IPEX_WEIGHT_CAST_MODULE and not split_master_weight_for_bf16

    def is_master_weight_split_solution(self, module, split_master_weight_for_bf16):
        return type(module) in IPEX_WEIGHT_CAST_MODULE and split_master_weight_for_bf16

    def is_fp32_weight_solution(self, module):
        return type(module) not in IPEX_WEIGHT_CAST_MODULE

    def test_weight_cast(self):
        M = TestModule()
        for optimizer in [Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, RMSprop, Rprop, SGD]:
            for split_master_weight_for_bf16 in [True, False]:
                if type(optimizer) not in IPEX_FUSED_OPTIMIZER_LIST:
                    split_master_weight_for_bf16 = False
                model = copy.deepcopy(M)
                optimizer = Adagrad(model.parameters(), lr=0.01)
                model, opt, _ = cast(model, optimizer, {}, split_master_weight_for_bf16)
                optimizer_params_list = opt.param_groups[0]['params']
                param_id = 0
                for _, sub_m in model.named_children():
                    if self.is_master_weight_solution(sub_m, split_master_weight_for_bf16):
                        self.assertTrue(hasattr(sub_m, 'master_weight'))
                        self.assertEqual(sub_m.weight.dtype, torch.bfloat16)
                        self.assertEqual(sub_m.master_weight.dtype, torch.float)
                        self.assertTrue(sub_m.master_weight is optimizer_params_list[param_id])
                        self.assertTrue(sub_m.weight is not optimizer_params_list[param_id])
                        param_id += 1
                        if hasattr(sub_m, 'bias'):
                            self.assertTrue(hasattr(sub_m, 'master_bias'))
                            self.assertEqual(sub_m.bias.dtype, torch.bfloat16)
                            self.assertEqual(sub_m.master_bias.dtype, torch.float)
                            self.assertTrue(sub_m.master_bias is optimizer_params_list[param_id])
                            self.assertTrue(sub_m.bias is not optimizer_params_list[param_id])
                            param_id += 1
                    elif self.is_master_weight_split_solution(sub_m, split_master_weight_for_bf16):
                        self.assertTrue(hasattr(sub_m, 'weight_trail'))
                        self.assertEqual(sub_m.weight.dtype, torch.bfloat16)
                        self.assertTrue(sub_m.weight is optimizer_params_list[param_id])
                        param_id += 1
                        if hasattr(sub_m, 'bias'):
                            self.assertTrue(hasattr(sub_m, 'bias_trail'))
                            self.assertEqual(sub_m.bias.dtype, torch.bfloat16)
                            self.assertTrue(sub_m.bias is optimizer_params_list[param_id])
                            param_id += 1
                    else:
                        self.assertTrue(self.is_fp32_weight_solution(sub_m))
                        for i, p in enumerate(sub_m.parameters()):
                            self.assertTrue(p is optimizer_params_list[param_id])
                            param_id += 1
                # For resume training, state_dict() should always return fp32 dtype
                origin_model_state = M.state_dict()
                ipex_model_state = model.state_dict()
                for var_name in ipex_model_state:
                    self.assertEqual(origin_model_state[var_name], ipex_model_state[var_name])
                    [module_name, param_name] = var_name.split('.')
                    if hasattr(getattr(model, module_name), 'master_' + param_name):
                        output_param = getattr(getattr(model, module_name), 'master_' + param_name)
                    elif hasattr(getattr(model, module_name), param_name + '_trail'):
                        trail = getattr(getattr(model, module_name), param_name + '_trail')
                        top_half = getattr(getattr(model, module_name), param_name)
                        output_param = torch.ops.torch_ipex.cat_bfloat16_float(top_half, trail)
                    else:
                        output_param = getattr(getattr(model, module_name), param_name)
                    self.assertEqual(ipex_model_state[var_name], output_param)
                origin_opt_state = optimizer.state_dict()
                ipex_opt_state = opt.state_dict()
                self.assertEqual(ipex_opt_state['state'], origin_opt_state['state'])


if __name__ == '__main__':
    test = unittest.main()
