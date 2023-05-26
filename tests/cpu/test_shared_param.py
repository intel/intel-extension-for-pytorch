import unittest
import copy

import torch
import intel_extension_for_pytorch as ipex

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
from intel_extension_for_pytorch.optim._lamb import Lamb
import itertools


class TestParamSharing(TestCase):
    def test_param_shared(self):
        class SharedParaModel(torch.nn.Module):
            # from bart
            def __init__(self):
                super(SharedParaModel, self).__init__()
                self.shared = torch.nn.Embedding(3, 3)
                self.encoder = torch.nn.Embedding(3, 3)
                self.decoder = torch.nn.Embedding(3, 3)
                self.encoder.weight = self.shared.weight
                self.decoder.weight = self.shared.weight
                self.linear = torch.nn.Linear(3, 3)
                self.linear.weight = self.shared.weight

            def forward(self, x):
                x = self.encoder(x)
                x = self.decoder(x)
                x = self.linear(x) + self.shared(x)
                return x

        def check_shared_in_model(model, dtype):
            self.assertEqual(
                model.shared.weight.data_ptr(), model.encoder.weight.data_ptr()
            )
            self.assertEqual(
                model.shared.weight.data_ptr(), model.decoder.weight.data_ptr()
            )
            self.assertEqual(
                model.shared.weight.data_ptr(), model.linear.weight.data_ptr()
            )
            self.assertEqual(model.shared.weight.dtype, dtype)

        def check_shared_in_state_dict(state_dict):
            self.assertEqual(
                state_dict["shared.weight"].data_ptr(),
                state_dict["encoder.weight"].data_ptr(),
            )
            self.assertEqual(
                state_dict["shared.weight"].data_ptr(),
                state_dict["decoder.weight"].data_ptr(),
            )
            self.assertEqual(
                state_dict["shared.weight"].data_ptr(),
                state_dict["linear.weight"].data_ptr(),
            )
            self.assertEqual(state_dict["shared.weight"].dtype, torch.float)

        def test_inference(model):
            params_dict = {
                "dtype": [torch.float, torch.bfloat16],
                "level": ["O0", "O1"],
                "inplace": [True, False],
            }
            for dtype, level, inplace in list(itertools.product(*params_dict.values())):
                test_model = copy.deepcopy(model).eval()
                opt_M = ipex.optimize(
                    test_model, dtype=dtype, level=level, inplace=inplace
                )
                check_shared_in_model(opt_M, dtype)
                check_shared_in_state_dict(opt_M.state_dict())

        def test_training(model):
            params_dict = {
                "dtype": [torch.float, torch.bfloat16],
                "level": ["O0", "O1"],
                "inplace": [True, False],
                "optimizer": [
                    Lamb,
                    Adadelta,
                    Adagrad,
                    Adam,
                    AdamW,
                    Adamax,
                    ASGD,
                    RMSprop,
                    Rprop,
                    SGD,
                ],
            }
            for dtype, level, inplace, opt in list(
                itertools.product(*params_dict.values())
            ):
                test_model = copy.deepcopy(model)
                if opt == SGD:
                    optimizer = opt(model.parameters(), lr=10.01, momentum=0.1)
                else:
                    optimizer = opt(model.parameters(), lr=10.01)
                opt_M, _ = ipex.optimize(
                    test_model,
                    optimizer=optimizer,
                    dtype=dtype,
                    level=level,
                    inplace=inplace,
                )
                check_shared_in_model(opt_M, dtype)
                check_shared_in_state_dict(opt_M.state_dict())

        test_inference(SharedParaModel())
        test_training(SharedParaModel())

    def test_nocast_since_shared(self):
        class NoCastforSharingPara(torch.nn.Module):
            def __init__(self):
                super(NoCastforSharingPara, self).__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 3))
                self.no_cast_linear = torch.nn.Linear(3, 3)
                self.no_cast_linear.weight = self.param

            def forward(self, x):
                x = self.no_cast_linear(x)
                x = x + self.other
                return x

        model = NoCastforSharingPara()
        for level in ["O0", "O1"]:
            for train in [True, False]:
                test_model = copy.deepcopy(model)
                if train:
                    optimizer = SGD(model.parameters(), lr=10.01, momentum=0.1)
                    opt_M, _ = ipex.optimize(
                        test_model,
                        optimizer=optimizer,
                        dtype=torch.bfloat16,
                        level=level,
                    )
                else:
                    opt_M = ipex.optimize(
                        test_model.eval(), dtype=torch.bfloat16, level=level
                    )
                self.assertEqual(
                    opt_M.param.data_ptr(), opt_M.no_cast_linear.weight.data_ptr()
                )
                self.assertEqual(opt_M.no_cast_linear.weight.dtype, torch.float)
                self.assertEqual(opt_M.no_cast_linear.bias.dtype, torch.float)

    def test_noprepack_since_shared(self):
        class NoPrepackforSharingPara(torch.nn.Module):
            def __init__(self):
                super(NoPrepackforSharingPara, self).__init__()
                self.shared = torch.nn.Embedding(3, 3)
                self.no_prepack_linear = torch.nn.Linear(3, 3)
                self.no_prepack_linear.weight = self.shared.weight

            def forward(self, x):
                x = self.no_cast_linear(x)
                x = x + self.other
                return x

        model = NoPrepackforSharingPara()
        for level in ["O0", "O1"]:
            for train in [True, False]:
                test_model = copy.deepcopy(model)
                if train:
                    optimizer = SGD(model.parameters(), lr=10.01, momentum=0.1)
                    opt_M, _ = ipex.optimize(
                        test_model,
                        weights_prepack=True,
                        optimizer=optimizer,
                        dtype=torch.bfloat16,
                        level=level,
                    )
                else:
                    opt_M = ipex.optimize(
                        test_model.eval(),
                        weights_prepack=True,
                        dtype=torch.bfloat16,
                        level=level,
                    )
                self.assertEqual(
                    opt_M.shared.weight.data_ptr(),
                    opt_M.no_prepack_linear.weight.data_ptr(),
                )
                self.assertTrue(isinstance(opt_M.no_prepack_linear, torch.nn.Linear))


if __name__ == "__main__":
    test = unittest.main()
