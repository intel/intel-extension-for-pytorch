import tempfile

from numpy import identity
import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch  # noqa
import pytest
import os

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

batch_size = 128
class_num = 1000
input_channel = 512
hidden_channel = 2048
num_iter = 10
lr = 0.01
checkpoint_path_str = './_checkpoint.test.case.test_xpu_checkpoint_save_load_integrity_and_accuracy.pth.tar'

class TrainingModel(nn.Module):
    def __init__(self):
        super(TrainingModel, self).__init__()
        self.m = nn.Sequential(
            nn.Conv2d(input_channel, hidden_channel, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_channel, eps=1e-05, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
        )
        self.fc = nn.Linear(in_features=hidden_channel, out_features=class_num, bias=True)

    def forward(self, x, indentity_for_mul, indentity_for_add):
        x = self.m(x)
        x = x * indentity_for_mul
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x + indentity_for_add
        return x

class TestTorchMethod(TestCase):
    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_save_load(self):
        a = torch.ones([10], dtype=torch.float64)
        a = a.to(xpu_device)
        ckpt = tempfile.NamedTemporaryFile()
        torch.save(a, ckpt.name)
        b = torch.load(ckpt.name)
        assert torch.equal(a, b), "tensor saved & loaded not equal"

    def test_serialization_map_location(self):
        a = torch.randn(5)
        ckpt = tempfile.NamedTemporaryFile()
        torch.save(a, ckpt.name)
        b = torch.load(ckpt.name, map_location=lambda storage, loc: storage.xpu(0))
        self.assertEqual(a, b.to(cpu_device))

    @pytest.mark.skipif(torch.xpu.device_count() < 2, reason="doesn't support with one device")
    def test_serialization_multi_map_location(self):
        a = torch.randn(5, device='xpu:0')
        ckpt = tempfile.NamedTemporaryFile()
        torch.save(a, ckpt.name)
        b = torch.load(ckpt.name, map_location={'xpu:0':'xpu:1'})
        self.assertEqual(a.to(cpu_device), b.to(cpu_device))
        self.assertEqual(b.device.__str__(), 'xpu:1')

    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_xpu_checkpoint_save_load_integrity_and_accuracy(self, dtype=torch.bfloat16):
        # create model
        device = 'xpu'
        model_xpu = TrainingModel()
        model_xpu = model_xpu.to(device=device).train()
        optimizer_xpu = torch.optim.SGD(model_xpu.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        if os.path.exists(checkpoint_path_str):
            os.remove(checkpoint_path_str)

        # process torch.xpu.optimize
        model_xpu, optimizer_xpu = torch.xpu.optimize(model=model_xpu, dtype=dtype, optimizer=optimizer_xpu)

        def training_step(model_xpu, optimizer_xpu, criterion):
            input = torch.randn(batch_size, input_channel, 7, 7)
            target = torch.empty(batch_size, dtype=torch.long).random_(class_num)
            input_xpu = input.clone().to(device=device).requires_grad_()
            target_xpu = target.to(device)
            indentity_for_mul = torch.randn(batch_size, hidden_channel, 1, 1).to(device=device)
            indentity_for_add = torch.randn(batch_size, class_num).to(device=device)

            # forward
            with torch.xpu.amp.autocast(enabled=True, dtype=dtype):
                output_xpu = model_xpu(input_xpu, indentity_for_mul, indentity_for_add)
                loss_xpu = criterion(output_xpu, target_xpu)

            # optimizer
            optimizer_xpu.zero_grad()

            # backward
            loss_xpu.backward()

            # update
            optimizer_xpu.step()
            loss_xpu = loss_xpu.cpu()
            output_xpu = output_xpu.cpu()

        def save_checkpoint(state, filename=checkpoint_path_str):
            torch.save(state, filename)

        for _ in range(num_iter):
            training_step(model_xpu, optimizer_xpu, criterion)

        save_checkpoint({'model_state_dict': model_xpu.state_dict(), 'optimizer_state_dict': optimizer_xpu.state_dict()})
        if os.path.isfile(checkpoint_path_str):
            # load checkpoint
            checkpoint = torch.load(checkpoint_path_str, map_location='xpu')
            print('load checkpoint')

            # create model
            new_model = TrainingModel()
            new_model = new_model.to(device=device).train()
            print('create model')

            # create optimizer
            new_optimizer = torch.optim.SGD(new_model.parameters(), lr=lr)
            print('create model')

            # load state dict
            new_model.load_state_dict(checkpoint['model_state_dict'])
            new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('load state dict')

            # check
            print('checking...')
            self.assertEqual(model_xpu.state_dict(), new_model.state_dict(), atol=1e-6, rtol=1e-6)
            self.assertEqual(optimizer_xpu.state_dict(), new_optimizer.state_dict(), atol=1e-6, rtol=1e-6)
        else:
            assert False, "save checkpoint failed for xpu model" # noqa B011
