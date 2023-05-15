import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch  # noqa F401


class TestTorchMethod(TestCase):
    def test_topk_out(self, dtype=torch.float):
        x_cpu = torch.tensor(
            [
                [
                    -0.2911,
                    -1.3204,
                    -2.6425,
                    -2.4644,
                    -0.6018,
                    -0.0839,
                    -0.1322,
                    -0.4713,
                    -0.3586,
                    -0.8882,
                ]
            ],
            device=torch.device("cpu"),
            dtype=torch.float,
        )

        # create 1D tensor without any elements
        value = torch.randn([0])
        indices = torch.randn([0])

        value_cpu = value.new()
        indices_cpu = indices.long().new()

        value_xpu = value.new().to("xpu")
        indices_xpu = indices.long().new().to("xpu")

        x_dpcpp = x_cpu.to("xpu")
        print("x_dpcpp.dim ", x_dpcpp.dim())

        torch.topk(x_cpu, 2, out=(value_cpu, indices_cpu))
        torch.topk(x_dpcpp, 2, out=(value_xpu, indices_xpu))

        print("value_cpu: ", value_cpu)
        print("indices_cpu: ", indices_cpu)

        print("value_xpu: ", value_xpu.cpu())
        print("indices_xpu: ", indices_xpu.cpu())

        self.assertEqual(x_cpu, x_dpcpp.cpu())
        self.assertEqual(value_cpu, value_xpu.cpu())
        self.assertEqual(indices_cpu, indices_xpu.cpu())
