import tempfile

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch  # noqa
import torchvision.models as models
import pytest
import os

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

batch_size = 128
input_channel = 3
train_num_iter = 5
eval_num_iter = 3
lr = 0.01
checkpoint_path_str = "./_checkpoint.test.case.test_xpu_checkpoint_save_load_integrity_and_accuracy.pth.tar"


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_save_load(self):
        a = torch.ones([10], dtype=torch.float64)
        a = a.to(xpu_device)
        ckpt = tempfile.NamedTemporaryFile()
        with tempfile.NamedTemporaryFile(delete=False) as ckpt:
            torch.save(a, ckpt.name)
        b = torch.load(ckpt.name)
        assert torch.equal(a, b), "tensor saved & loaded not equal"

    def test_serialization_map_location(self):
        a = torch.randn(5)
        ckpt = tempfile.NamedTemporaryFile()
        with tempfile.NamedTemporaryFile(delete=False) as ckpt:
            torch.save(a, ckpt.name)
        b = torch.load(ckpt.name, map_location=lambda storage, loc: storage.xpu(0))
        self.assertEqual(a, b.to(cpu_device))

    @pytest.mark.skipif(
        torch.xpu.device_count() < 2, reason="doesn't support with one device"
    )
    def test_serialization_multi_map_location(self):
        a = torch.randn(5, device="xpu:0")
        ckpt = tempfile.NamedTemporaryFile()
        with tempfile.NamedTemporaryFile(delete=False) as ckpt:
            torch.save(a, ckpt.name)
        b = torch.load(ckpt.name, map_location={"xpu:0": "xpu:1"})
        self.assertEqual(a.to(cpu_device), b.to(cpu_device))
        self.assertEqual(b.device.__str__(), "xpu:1")

    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_xpu_checkpoint_save_load_integrity_and_accuracy(self):
        device = "xpu"

        def training_step(model_xpu, optimizer_xpu, criterion, dtype):
            input = torch.randn(batch_size, input_channel, 224, 224)
            target = torch.empty(batch_size, dtype=torch.long).random_(1000)
            input_xpu = input.clone().to(device=device).requires_grad_()
            target_xpu = target.to(device)

            # forward
            with torch.xpu.amp.autocast(enabled=True, dtype=dtype):
                output_xpu = model_xpu(input_xpu)
                loss_xpu = criterion(output_xpu, target_xpu)

            # optimizer
            optimizer_xpu.zero_grad()

            # backward
            loss_xpu.backward()

            # update
            optimizer_xpu.step()
            loss_xpu = loss_xpu.cpu()
            output_xpu = output_xpu.cpu()

        def eval_step(model_xpu, dtype):
            input = torch.randn(batch_size, input_channel, 224, 224)
            target = torch.empty(batch_size, dtype=torch.long).random_(1000)
            input_xpu = input.clone().to(device=device).requires_grad_()
            target_xpu = target.to(device)

            # forward
            with torch.xpu.amp.autocast(enabled=True, dtype=dtype):
                output_xpu = model_xpu(input_xpu)
                loss_xpu = criterion(output_xpu, target_xpu)

            loss_xpu = loss_xpu.cpu()
            output_xpu = output_xpu.cpu()

        def save_checkpoint(state, filename=checkpoint_path_str):
            torch.save(state, filename)

        for dtype in [torch.float32, torch.bfloat16]:
            for split_master_weight_for_bf16 in [True, False]:
                print("dtype = ", dtype)
                print("split master weight = ", dtype)

                # create model
                model_xpu = (
                    models.__dict__["resnet18"](pretrained=True).to(device=device).train()
                )
                optimizer_xpu = torch.optim.SGD(model_xpu.parameters(), lr=lr)
                criterion = nn.CrossEntropyLoss()

                if os.path.exists(checkpoint_path_str):
                    os.remove(checkpoint_path_str)

                # process torch.xpu.optimize
                model_xpu, optimizer_xpu = torch.xpu.optimize(
                    model=model_xpu, dtype=dtype, optimizer=optimizer_xpu,
                    split_master_weight_for_bf16=split_master_weight_for_bf16
                )

                # mimic model train, then eval
                for _ in range(train_num_iter):
                    training_step(model_xpu, optimizer_xpu, criterion, dtype)
                model_xpu.eval()
                for _ in range(eval_num_iter):
                    eval_step(model_xpu, dtype)
                torch.xpu.synchronize()

                save_checkpoint(
                    {
                        "model_state_dict": model_xpu.state_dict(),
                        "optimizer_state_dict": optimizer_xpu.state_dict(),
                    }
                )
                if os.path.isfile(checkpoint_path_str):
                    # load checkpoint
                    checkpoint = torch.load(checkpoint_path_str, map_location=device)
                    print("load checkpoint")

                    # create model
                    new_model = (
                        models.__dict__["resnet18"](pretrained=False)
                        .to(device=device)
                        .train()
                    )
                    print("create model")

                    # create optimizer
                    new_optimizer = torch.optim.SGD(new_model.parameters(), lr=lr)
                    print("create model")

                    # optimize
                    new_model, new_optimizer = torch.xpu.optimize(
                        model=new_model, dtype=dtype, optimizer=new_optimizer
                    )

                    # load state dict
                    new_model.load_state_dict(checkpoint["model_state_dict"])
                    new_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    print("load state dict")

                    # check
                    print("checking...")
                    self.assertEqual(
                        model_xpu.state_dict(), new_model.state_dict(), atol=1e-6, rtol=1e-6
                    )
                    self.assertEqual(
                        optimizer_xpu.state_dict(),
                        new_optimizer.state_dict(),
                        atol=1e-6,
                        rtol=1e-6,
                    )
                    os.remove(checkpoint_path_str)
                else:
                    assert False, "save checkpoint failed for xpu model"  # noqa B011
