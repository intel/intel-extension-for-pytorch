import torch
import intel_extension_for_pytorch  # noqa
import torch.nn as nn
from torch.multiprocessing import Process

# torch.testing._internal.common_utils can trigger XPU runtime.
from expecttest import TestCase
import time


class mini50(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.Sequential(
            nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(
                2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
        )
        self.fc = nn.Linear(in_features=2048, out_features=1000, bias=True)

    def forward(self, x):
        x = self.m(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TestTorchMethod(TestCase):
    def test_lazy_init(self):
        def run_model():
            assert not torch.xpu.is_initialized()
            criterion = nn.CrossEntropyLoss().to("xpu")
            model = mini50()
            model = model.to(device="xpu", dtype=torch.float32)
            model.eval()

            bs = 2
            input = torch.randn(bs, 512, 7, 7, requires_grad=True)
            input = (
                input.detach().to(device="xpu", dtype=torch.float32).requires_grad_()
            )
            target = torch.empty(bs, dtype=torch.long).random_(1000).to("xpu")

            optimizer = torch.optim.SGD(
                model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
            )

            def compute_iter(input: torch.Tensor, if_train=False):
                if if_train:
                    output = model(input)
                    loss = criterion(output, target)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                else:
                    output = model(input)
                    loss = criterion(output, target)
                torch.xpu.synchronize()

            compute_iter(input)  # warm up
            time_list = []
            for iteration in range(10):
                start = time.time()
                compute_iter(input)
                end = time.time()
                time_list.append(end - start)
            print("[info] fps = ", round(bs / (sum(time_list) / len(time_list)), 3))

        def test_multi_process():
            p = Process(target=run_model)
            p.start()
            p.join(120)
            return p.exitcode

        self.assertFalse(torch.xpu.is_initialized())
        self.assertFalse(torch.xpu._is_in_bad_fork())
        self.assertEqual(test_multi_process(), 0)
        self.assertEqual(test_multi_process(), 0)
