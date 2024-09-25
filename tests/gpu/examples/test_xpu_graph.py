import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch  # noqa
import copy

N, C, H_in, H_out = 2, 2, 128, 64
model = torch.nn.AvgPool2d(2)
loss_fn = torch.nn.MSELoss()
model_ref = copy.deepcopy(model)
loss_fn = torch.nn.MSELoss()


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(True, reason="compiler doesn't support sycl graph")
    def test_xpu_graph(self, dtype=torch.float):
        # Placeholders used for capture
        static_input = torch.randn(N, C, H_in, H_in, device="xpu")
        static_target = torch.randn(N, C, H_out, H_out, device="xpu")
        static_input_ref = copy.deepcopy(static_input)
        static_target_ref = copy.deepcopy(static_target)
        # warmup
        s = torch.xpu.Stream()
        s.wait_stream(torch.xpu.current_stream())
        with torch.xpu.stream(s):
            for i in range(3):
                y_pred = model(static_input) * 2
                loss = loss_fn(y_pred, static_target)
        torch.xpu.current_stream().wait_stream(s)

        # capture
        g = torch.xpu.XPUGraph()
        # Sets grads to None before capture, so backward() will create
        # .grad attributes with allocations from the graph's private pool
        # optimizer.zero_grad()
        with torch.xpu.graph(g):
            static_y_pred = model(static_input) * 2
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
            # replay() includes forward, backward, and step.
            # You don't even need to call optimizer.zero_grad() between iterations
            # because the captured backward refills static .grad tensors in place.
            g.replay()
            print(static_loss)
            # Params have been updated. static_y_pred, static_loss, and .grad
            # attributes hold values from computing on this iteration's data.

        for data, target in zip(real_inputs_ref, real_targets_ref):
            y_pred_ref = model_ref(data) * 2
            loss_ref = loss_fn(y_pred_ref, target)
            print(loss_ref)

        self.assertEqual(loss_ref, static_loss)
