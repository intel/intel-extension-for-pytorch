import torch
import torch.nn.functional
from torch import nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
from intel_extension_for_pytorch.jit._trace import need_to_disable_check_trace_for_XPU
import pytest # noqa


class InferenceModel(nn.Module):
    def __init__(self):
        super(InferenceModel, self).__init__()
        self.m = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(4, eps=1e-05, momentum=0.1),
            nn.Dropout(p=0.01),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
            nn.Dropout(p=0.01),
        )
        # no use, only for checking model convert feature in torch.xpu.optimize
        self.emb = torch.nn.Embedding(256, 4),
        self.fc = nn.Linear(in_features=400, out_features=1000, bias=True)

    def forward(self, x):
        x = self.m(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TestTorchMethod(TestCase):
    def test_disable_check_trace_for_jit_trace(self, dtype=torch.float):
        example_inputs_list = []

        # create multi input situation, tensor, list of tensor, tuple of tensor and dict of tensor
        input_tensor1 = torch.randn(3, 3, device='xpu', dtype=dtype)
        input_tensor2 = torch.randn(3, 3, device='xpu', dtype=dtype)
        example_inputs_list.append(input_tensor1)

        input_list = [input_tensor1, input_tensor2]
        example_inputs_list.append(input_list)

        input_tuple = (input_tensor1, input_tensor2)
        example_inputs_list.append(input_tuple)

        input_double_tuple = (input_tensor1, input_tensor2, (input_tensor1, input_tensor2))
        example_inputs_list.append(input_double_tuple)

        input_dict = {'trace_input1' : input_tensor1, 'trace_input2' : input_tensor2}
        example_inputs_list.append(input_dict)

        # create the model
        module = InferenceModel().to(device='xpu')

        # check for each situation
        for example_input in example_inputs_list:
            if torch.xpu.utils.has_2d_block_array():
                # for the platform supports 2d block array, no need to disable the check trace default value
                self.assertFalse(need_to_disable_check_trace_for_XPU(module, example_input))
                self.assertFalse(need_to_disable_check_trace_for_XPU(func=module, example_inputs=example_input))
            else:
                # for the platform doesn't support 2d block array, need to disable the check trace default value
                self.assertTrue(need_to_disable_check_trace_for_XPU(module, example_input))
                self.assertTrue(need_to_disable_check_trace_for_XPU(func=module, example_inputs=example_input))
                # for the platform doesn't support 2d block array, no need to disable the check trace
                # if users explicitly manually set the check trace
                self.assertFalse(need_to_disable_check_trace_for_XPU(func=module, example_inputs=example_input, check_trace=True))
