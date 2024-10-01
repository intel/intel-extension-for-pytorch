import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch  # noqa
import copy
import pytest

N, D_in, H, D_out = 32, 128, 32, 256
model = torch.nn.Sequential(torch.nn.Linear(D_in, H), torch.nn.Linear(H, D_out)).xpu()
loss_fn = torch.nn.MSELoss()
model_ref = copy.deepcopy(model)
loss_fn = torch.nn.MSELoss()


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(True, reason="compiler doesn't support sycl graph")
    def test_xpu_graph_linear(self, dtype=torch.float):
        # Placeholders used for capture
        static_input = torch.randn(N, D_in, device="xpu")
        static_target = torch.randn(N, D_out, device="xpu")
        static_input_ref = copy.deepcopy(static_input)
        static_target_ref = copy.deepcopy(static_target)

        # warmup
        s = torch.xpu.Stream()
        s.wait_stream(torch.xpu.current_stream())
        with torch.xpu.stream(s):
            for i in range(3):
                y_pred = model(static_input)
                loss = loss_fn(y_pred, static_target)
        torch.xpu.current_stream().wait_stream(s)

        # capture
        g = torch.xpu.XPUGraph()

        with torch.xpu.graph(g):
            static_y_pred = model(static_input)
            static_loss = loss_fn(static_y_pred, static_target)

        real_inputs = []
        real_targets = []
        real_inputs_ref = []
        real_targets_ref = []

        for _ in range(3):
            r_in = torch.rand_like(static_input)
            r_tg = torch.rand_like(static_target)
            real_inputs.append(r_in)
            real_targets.append(r_tg)
            real_inputs_ref.append(copy.deepcopy(r_in))
            real_targets_ref.append(copy.deepcopy(r_tg))

        for data, target in zip(real_inputs, real_targets):
            # Fills the graph's input memory with new data to compute on
            static_input.copy_(data)
            static_target.copy_(target)

            g.replay()
            print(static_loss)

        for data, target in zip(real_inputs_ref, real_targets_ref):
            y_pred_ref = model_ref(data)
            loss_ref = loss_fn(y_pred_ref, target)
            print(loss_ref)

        self.assertEqual(loss_ref, static_loss)
