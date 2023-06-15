import unittest
import copy

import torch
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.nn.utils._weight_cast import weight_dtype_convert_with_ipex as cast
from intel_extension_for_pytorch.nn.utils._weight_cast import IPEX_WEIGHT_CAST_MODULE as IPEX_WEIGHT_CAST_MODULE
from intel_extension_for_pytorch.optim._optimizer_utils import IPEX_FUSED_OPTIMIZER_LIST_CPU as IPEX_FUSED_OPTIMIZER_LIST
from intel_extension_for_pytorch.nn.modules import MergedEmbeddingBag

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
        self.embedding = torch.nn.Embedding(10, 3)
        table0 = torch.nn.EmbeddingBag(100, 16, mode='mean', sparse=False)
        table1 = torch.nn.EmbeddingBag(50, 32, mode='sum', sparse=False)
        self.merged = MergedEmbeddingBag.from_embeddingbag_list([table0, table1])

    def forward(self, x):
        x = self.conv2d(x)
        return

class TestWeightCastCases(TestCase):
    def is_master_weight_solution(self, module, split_master_weight):
        return type(module) in IPEX_WEIGHT_CAST_MODULE and not split_master_weight

    def is_master_weight_split_solution(self, module, split_master_weight):
        return type(module) in IPEX_WEIGHT_CAST_MODULE and split_master_weight

    def is_fp32_weight_solution(self, module):
        return type(module) not in IPEX_WEIGHT_CAST_MODULE
   
    def master_weight_test(self, m, param_id, cast_dtype, optimizer_params_list):
        for name, param in m.named_parameters():
            if hasattr(m, name):
                self.assertTrue(hasattr(m, 'master_{}'.format(name)))
                self.assertEqual(getattr(m, name).dtype, cast_dtype)
                self.assertEqual(getattr(m, 'master_{}'.format(name)).dtype, torch.float)
                self.assertTrue(getattr(m, 'master_{}'.format(name)) is optimizer_params_list[param_id])
                self.assertTrue(getattr(m, name) is not optimizer_params_list[param_id])
                param_id += 1
        return param_id 

    def master_weight_split_test(self, m, param_id, cast_dtype, optimizer_params_list):
        for name, param in m.named_parameters():
            if hasattr(m, name):
                self.assertTrue(hasattr(m, '{}_trail'.format(name)))
                self.assertEqual(getattr(m, name).dtype, cast_dtype)
                self.assertTrue(getattr(m, name) is optimizer_params_list[param_id])
                param_id += 1
        return param_id

    def test_weight_cast(self):
        M = TestModule()
        for pt_opt in [Adagrad, Adadelta, Adam, AdamW, Adamax, ASGD, RMSprop, Rprop, SGD]:
            for cast_dtype in [torch.bfloat16, torch.float16]:
                for split_master_weight_for_bf16 in [True, False]:
                    if pt_opt not in IPEX_FUSED_OPTIMIZER_LIST or cast_dtype == torch.float16:
                        split_master_weight_for_bf16 = False
                    model = copy.deepcopy(M)
                    optimizer = pt_opt(model.parameters(), lr=0.01)
                    model, opt, _ = cast(model, optimizer, {}, split_master_weight_for_bf16, cast_dtype)
                    optimizer_params_list = opt.param_groups[0]['params']
                    param_id = 0
                    for _, sub_m in model.named_children():
                        if self.is_master_weight_solution(sub_m, split_master_weight_for_bf16):
                            param_id = self.master_weight_test(sub_m, param_id, cast_dtype, optimizer_params_list)
                            for name, ssub_m in sub_m.named_children():
                                if isinstance(ssub_m, torch.nn.ParameterList):
                                    param_id = self.master_weight_test(ssub_m, param_id, cast_dtype, optimizer_params_list)
                        elif self.is_master_weight_split_solution(sub_m, split_master_weight_for_bf16):
                             param_id = self.master_weight_split_test(sub_m, param_id, cast_dtype, optimizer_params_list)
                             for name, ssub_m in sub_m.named_children():
                                if isinstance(ssub_m, torch.nn.ParameterList):
                                    param_id = self.master_weight_split_test(ssub_m, param_id, cast_dtype, optimizer_params_list)
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
                        module_name = var_name.split('.')[0]
                        param_name = var_name.split('.')[1]
                        sub_m = getattr(model, module_name) #e.g., merged.weights.0
                        if len(var_name.split('.')) == 3: # For param  in ParameterList
                            sub_m = getattr(sub_m, param_name)
                            param_name = var_name.split('.')[2]
                        if hasattr(sub_m, 'master_' + param_name):
                            output_param = getattr(sub_m, 'master_' + param_name)
                        elif hasattr(sub_m, param_name + '_trail'):
                            trail = getattr(sub_m, param_name + '_trail')
                            top_half = getattr(sub_m, param_name)
                            output_param = torch.ops.torch_ipex.cat_bfloat16_float(top_half, trail)
                        else:
                            output_param = getattr(sub_m, param_name)
                        #print(var_name, ipex_model_state[var_name].shape, output_param.shape)
                        self.assertEqual(ipex_model_state[var_name], output_param)
                    origin_opt_state = optimizer.state_dict()
                    ipex_opt_state = opt.state_dict()
                    self.assertEqual(ipex_opt_state['state'], origin_opt_state['state'])

if __name__ == '__main__':
    test = unittest.main()
