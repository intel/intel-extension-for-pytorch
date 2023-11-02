import unittest
import copy

import torch
from intel_extension_for_pytorch.nn.utils._weight_cast import (
    weight_dtype_convert_with_ipex as cast,
)
from intel_extension_for_pytorch.nn.utils._parameter_wrapper import (
    IPEX_WEIGHT_CONVERT_MODULE_CPU as IPEX_WEIGHT_CONVERT_MODULE_CPU,
)
from intel_extension_for_pytorch.optim._optimizer_utils import (
    IPEX_FUSED_OPTIMIZER_LIST_CPU as IPEX_FUSED_OPTIMIZER_LIST,
)
from intel_extension_for_pytorch.nn.modules import MergedEmbeddingBag

from torch.testing._internal.common_utils import TestCase
from torch.optim import (
    Adadelta,
    Adagrad,
    Adam,
    AdamW,
    Adamax,
    ASGD,
    RMSprop,
    Rprop,
    SGD,
)


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
        self.embeddingbag = torch.nn.EmbeddingBag(10, 3, mode="sum")
        self.embedding = torch.nn.Embedding(10, 3)
        table0 = torch.nn.EmbeddingBag(100, 32, mode="sum", sparse=False)
        table1 = torch.nn.EmbeddingBag(50, 32, mode="sum", sparse=False)
        self.merged = MergedEmbeddingBag.from_embeddingbag_list([table0, table1])
        table2 = torch.nn.EmbeddingBag(100, 32, mode="mean", sparse=False)
        table3 = torch.nn.EmbeddingBag(50, 32, mode="mean", sparse=False)
        self.merged2 = MergedEmbeddingBag.from_embeddingbag_list([table0, table1])

    def forward(self, x):
        x = self.conv2d(x)
        return


class TestWeightCastCases(TestCase):
    def is_master_weight_solution(self, module, dtype, split_master_weight):
        return (
            type(module) in IPEX_WEIGHT_CONVERT_MODULE_CPU(False, dtype)
            and not split_master_weight
        )

    def is_master_weight_split_solution(self, module, split_master_weight):
        return (
            type(module) in IPEX_WEIGHT_CONVERT_MODULE_CPU(False, torch.bfloat16)
            and split_master_weight
        )

    def is_fp32_weight_solution(self, module, dtype):
        return type(module) not in IPEX_WEIGHT_CONVERT_MODULE_CPU(False, dtype)

    def master_weight_test(
        self, m, param_id, cast_dtype, optimizer_params_list, params_attr
    ):
        def found_wrapper(parameter, params_attr):
            for _, v in params_attr.items():
                if parameter is v.parameter:
                    return v
            # not found
            self.assertTrue(False)

        for name, param in m.named_parameters():
            if hasattr(m, name):
                param_wrapper = found_wrapper(param, params_attr)
                self.assertTrue(param_wrapper.master_parameter.dtype == torch.float)
                self.assertTrue(
                    param_wrapper.master_parameter is optimizer_params_list[param_id]
                )
                self.assertTrue(param_wrapper.parameter.dtype == cast_dtype)
                self.assertTrue(param_wrapper.parameter is getattr(m, name))
                param_id += 1
        return param_id

    def master_weight_split_test(
        self, m, param_id, cast_dtype, optimizer_params_list, params_attr
    ):
        for name, param in m.named_parameters():
            if hasattr(m, name):
                param_wrapper = params_attr[param]
                self.assertTrue(param_wrapper.parameter.dtype == torch.bfloat16)
                self.assertTrue(
                    param_wrapper.parameter is optimizer_params_list[param_id]
                )
                self.assertTrue(param_wrapper.parameter_trail.dtype == torch.bfloat16)
                self.assertTrue(param_wrapper.parameter is getattr(m, name))
                param_id += 1
        return param_id

    def test_weight_cast(self):
        M = TestModule()
        for pt_opt in [
            Adagrad,
            Adadelta,
            Adam,
            AdamW,
            Adamax,
            ASGD,
            RMSprop,
            Rprop,
            SGD,
        ]:
            for cast_dtype in [torch.bfloat16, torch.float16]:
                for split_master_weight_for_bf16 in [True, False]:
                    if (
                        pt_opt not in IPEX_FUSED_OPTIMIZER_LIST
                        or cast_dtype == torch.float16
                    ):
                        split_master_weight_for_bf16 = False
                    model = copy.deepcopy(M)
                    optimizer = pt_opt(model.parameters(), lr=0.01)
                    model, opt, params_attr = cast(
                        model, optimizer, {}, split_master_weight_for_bf16, cast_dtype
                    )
                    optimizer_params_list = opt.param_groups[0]["params"]
                    param_id = 0
                    for _, sub_m in model.named_children():
                        if self.is_master_weight_solution(
                            sub_m, cast_dtype, split_master_weight_for_bf16
                        ):
                            param_id = self.master_weight_test(
                                sub_m,
                                param_id,
                                cast_dtype,
                                optimizer_params_list,
                                params_attr,
                            )
                            for name, ssub_m in sub_m.named_children():
                                if isinstance(ssub_m, torch.nn.ParameterList):
                                    param_id = self.master_weight_test(
                                        ssub_m,
                                        param_id,
                                        cast_dtype,
                                        optimizer_params_list,
                                        params_attr,
                                    )
                        elif self.is_master_weight_split_solution(
                            sub_m, split_master_weight_for_bf16
                        ):
                            param_id = self.master_weight_split_test(
                                sub_m,
                                param_id,
                                cast_dtype,
                                optimizer_params_list,
                                params_attr,
                            )
                            for name, ssub_m in sub_m.named_children():
                                if isinstance(ssub_m, torch.nn.ParameterList):
                                    param_id = self.master_weight_split_test(
                                        ssub_m,
                                        param_id,
                                        cast_dtype,
                                        optimizer_params_list,
                                        params_attr,
                                    )
                        else:
                            self.assertTrue(
                                self.is_fp32_weight_solution(sub_m, cast_dtype)
                            )
                            for i, p in enumerate(sub_m.parameters()):
                                self.assertTrue(p is optimizer_params_list[param_id])
                                param_id += 1
                    # For resume training, state_dict() should always return fp32 dtype
                    origin_model_state = M.state_dict()
                    ipex_model_state = model.state_dict()
                    self.assertEqual(origin_model_state, ipex_model_state)
                    origin_opt_state = optimizer.state_dict()
                    ipex_opt_state = opt.state_dict()
                    self.assertEqual(ipex_opt_state["state"], origin_opt_state["state"])


if __name__ == "__main__":
    test = unittest.main()
