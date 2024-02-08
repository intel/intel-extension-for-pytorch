import torch
import unittest

from intel_extension_for_pytorch.quantization.fp8 import (
    fp8_autocast,
    DelayedScaling,
    Format,
    prepare_fp8,
)
import intel_extension_for_pytorch._C as core

from torch.testing._internal.common_utils import TestCase
from torch.optim import SGD


class TestFP8Cases(TestCase):
    @unittest.skipIf(
        not core.onednn_has_fp8_support(),
        "IPEX FP8 is not supported on this CPU device",
    )
    def test_fp8_linear_base(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ln = torch.nn.LayerNorm(5, eps=1e-05)
                self.lin1 = torch.nn.Linear(5, 4, bias=False)
                self.lin2 = torch.nn.Linear(4, 3, bias=True)
                self.dropout = torch.nn.Dropout()

            def forward(self, x):
                x = self.ln(x)
                x = self.lin1(x)
                x = torch.nn.functional.gelu(x, approximate="tanh")
                x = self.lin2(x)
                z = self.dropout(x)
                return z

        torch.manual_seed(2024)

        my_linear = MyModel()
        my_linear.train()
        inp = torch.randn((10, 7, 3, 5), dtype=torch.float32)
        inp1 = inp.clone().requires_grad_(True)
        inp2 = inp.clone().requires_grad_(True)

        origin_optimizer = SGD(my_linear.parameters(), lr=0.01, momentum=0.9)
        fp8_linear, ipex_optimizer = prepare_fp8(my_linear, origin_optimizer)

        with fp8_autocast(
            enabled=True,
            fp8_recipe=DelayedScaling(fp8_format=Format.E4M3),
            device="cpu",
        ):
            for i in range(10):
                torch.manual_seed(2024)
                out = fp8_linear(inp2[i])
                ipex_optimizer.zero_grad()
                out.mean().backward()
                ipex_optimizer.step()

        for i in range(10):
            torch.manual_seed(2024)
            out_nn = my_linear(inp1[i])
            origin_optimizer.zero_grad()
            out_nn.mean().backward()
            origin_optimizer.step()

        self.assertEqual(out, out_nn, atol=0.05, rtol=0.1)
        self.assertEqual(inp1[-1].grad, inp2[-1].grad, atol=0.01, rtol=0.1)

        origin_model_state = my_linear.state_dict()
        ipex_model_state = fp8_linear.state_dict()
        for var_name in origin_model_state:
            self.assertEqual(
                origin_model_state[var_name],
                ipex_model_state[var_name],
                atol=0.01,
                rtol=0.1,
            )
        for name, _ in fp8_linear.named_children():
            if hasattr(getattr(my_linear, name), "weight"):
                if getattr(my_linear, name).weight is not None:
                    self.assertEqual(
                        getattr(my_linear, name).weight.grad,
                        getattr(fp8_linear, name).weight.grad,
                        atol=0.01,
                        rtol=0.1,
                    )
            if hasattr(getattr(my_linear, name), "bias"):
                if getattr(my_linear, name).bias is not None:
                    self.assertEqual(
                        getattr(my_linear, name).bias.grad,
                        getattr(fp8_linear, name).bias.grad,
                        atol=0.01,
                        rtol=0.1,
                    )

        origin_optimizer_state = origin_optimizer.state_dict()
        ipex_optimizer_state = ipex_optimizer.state_dict()
        for var_name in origin_optimizer_state:
            if var_name == "state":
                print(origin_optimizer_state[var_name])
                print(ipex_optimizer_state[var_name])
                self.assertEqual(
                    origin_optimizer_state[var_name],
                    ipex_optimizer_state[var_name],
                    atol=0.01,
                    rtol=0.1,
                )

    @unittest.skipIf(
        not core.onednn_has_fp8_support(),
        "IPEX FP8 is not supported on this CPU device",
    )
    def test_fp8_linear_calibration(self):
        class ClassA(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ln = torch.nn.LayerNorm(5, eps=1e-05)

            def forward(self, x):
                z = self.ln(x)
                return z

        class ClassC(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin2 = torch.nn.Linear(4, 3, bias=True)
                self.dropout = torch.nn.Dropout()

            def forward(self, x):
                x = self.lin2(x)
                z = self.dropout(x)
                return z

        class ClassB(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin1 = torch.nn.Linear(5, 4, bias=False)
                self.lin2_dropout = ClassC()

            def forward(self, x):
                x = self.lin1(x)
                x = torch.nn.functional.gelu(x, approximate="tanh")
                z = self.lin2_dropout(x)
                return z

        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ln = ClassA()
                self.lin1_gelu = ClassB()

            def forward(self, x):
                x = self.ln(x)
                z = self.lin1_gelu(x)
                return z

        # FP32 reference model
        my_linear = MyModel()
        my_linear.train()
        inp = torch.randn((5, 7, 3, 5), dtype=torch.float32)
        inp1 = inp.clone().requires_grad_(True)
        inp2 = inp.clone().requires_grad_(False)

        origin_optimizer = SGD(my_linear.parameters(), lr=0.01, momentum=0.9)

        for i in range(4):
            out_nn = my_linear(inp1[i])
            origin_optimizer.zero_grad()
            out_nn.mean().backward()
            origin_optimizer.step()

        torch.save(my_linear.state_dict(), "my_linear_inference.pt")
        my_linear_inference = MyModel()
        my_linear_inference.load_state_dict(torch.load("my_linear_inference.pt"))
        my_linear_inference.eval()
        out_nn_iter5 = my_linear_inference(inp1[4])

        fp8_linear_inference = prepare_fp8(my_linear_inference)
        # Do calibration to store amax of input and weight
        for i in range(4):
            with fp8_autocast(
                enabled=False,
                calibrating=True,
                fp8_recipe=DelayedScaling(fp8_format=Format.E4M3),
                device="cpu",
            ):
                _ = fp8_linear_inference(inp2[i])
        torch.save(fp8_linear_inference.state_dict(), "fp8_linear_inference.pt")

        # FP8 model with calibration
        fp8_linear_with_calibration = MyModel()
        fp8_linear_with_calibration = prepare_fp8(fp8_linear_with_calibration)
        fp8_linear_with_calibration.load_state_dict(
            torch.load("fp8_linear_inference.pt")
        )
        fp8_linear_with_calibration.eval()

        # Run model inference using calibration data
        with fp8_autocast(
            enabled=True,
            calibrating=False,
            fp8_recipe=DelayedScaling(fp8_format=Format.E4M3),
            device="cpu",
        ):
            out_fp8_iter5 = fp8_linear_with_calibration(inp2[4])
        self.assertEqual(out_fp8_iter5, out_nn_iter5, atol=0.01, rtol=0.1)

    @unittest.skipIf(
        not core.onednn_has_fp8_support(),
        "IPEX FP8 is not supported on this CPU device",
    )
    def test_fp8_non_contiguous_weight(self):
        nn_linear = torch.nn.Linear(2, 2)
        nn_linear.weight = torch.nn.Parameter(nn_linear.weight.transpose(0, 1))
        inp = torch.ones(3, 2)
        fp8_linear = prepare_fp8(nn_linear)
        with fp8_autocast(
            enabled=True,
            fp8_recipe=DelayedScaling(fp8_format=Format.E4M3),
            device="cpu",
        ):
            fp8_out = fp8_linear(inp)
        nn_out = nn_linear(inp)
        self.assertEqual(nn_out, fp8_out, atol=0.01, rtol=0.1)


if __name__ == "__main__":
    test = unittest.main()
