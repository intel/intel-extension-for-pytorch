import torch
import torch.nn.functional
from torch import nn as nn
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch  # noqa
import pytest  # noqa

device = 'xpu'

TEST_MODULE_CONVERT_LIST = [torch.nn.Conv2d,
                            torch.nn.Conv3d,
                            torch.nn.ConvTranspose2d,
                            torch.nn.ConvTranspose3d,
                            torch.nn.Linear,
                            torch.nn.Embedding,
                            torch.nn.LSTM]

# TODO: for now, only support SGD and AdamW
SUPPORTED_FUSION_OPTIMIZER = ['SGD', 'AdamW']


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


# for testing fusion optimizer and weight prepack
batch_size = 256
class_num = 1000
input_channel = 512
hidden_channel = 2048
num_iter = 50


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

    def forward(self, x):
        x = self.m(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TestTorchMethod(TestCase):
    def test_convert_module_data_type(self):
        def check_dtype_for_module(module, dtype):
            for module_cls in TEST_MODULE_CONVERT_LIST:
                if isinstance(module, module_cls):
                    self.assertEqual(module.weight.dtype, dtype)
                    if hasattr(module, 'bias') and module.bias is not None:
                        self.assertEqual(module.bias.dtype, dtype)
            for child in module.children():
                check_dtype_for_module(child, dtype)
        for dtype in [torch.float, torch.bfloat16, torch.float16]:
            module = InferenceModel().to(device=device)
            # this feature is only for inference
            module.eval()
            optimized_module = torch.xpu.optimize(model=module, dtype=dtype, level="O0")
            check_dtype_for_module(optimized_module, dtype)

    def test_conv_bn_folding(self):
        def check_conv_bn_folding(module):
            # should not find the BatchNorm2d
            if isinstance(module, torch.nn.BatchNorm2d):
                # find bn, means conv bn failed for InferenceModel
                raise RuntimeError("conv bn folding failed")
            for child in module.children():
                check_conv_bn_folding(child)
        module = InferenceModel().to(device=device)
        # this feature is only for inference
        module.eval()
        optimized_module = torch.xpu.optimize(model=module, dtype=torch.float16, level="O0", conv_bn_folding=True)
        check_conv_bn_folding(optimized_module)

    def test_replace_dropout_for_inference(self):
        module = InferenceModel().to(device=device)
        # this feature is only for inference
        module.eval()

        module_pattern = {}

        def collect_module_pattern(module, pattern_pos):
            module_pattern[pattern_pos] = type(module)
            pattern_pos = pattern_pos + 1
            for child in module.children():
                collect_module_pattern(child, pattern_pos)

        def check_no_dropout_but_identity(module, check_pos):
            # should not find the Dropout but identity in original place
            if module_pattern[check_pos] != module and isinstance(module_pattern[check_pos], torch.nn.Dropout):
                self.assertEqual(type(module), torch.nn.identity)
            check_pos = check_pos + 1
            for child in module.children():
                check_no_dropout_but_identity(child, check_pos)

        pattern_pos = 0
        collect_module_pattern(module, pattern_pos)
        optimized_module = torch.xpu.optimize(model=module, dtype=torch.float16,
                                              level="O0", replace_dropout_with_identity=True)
        check_pos = 0
        check_no_dropout_but_identity(optimized_module, check_pos)

    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_master_weight_fusion_optimizer(self):
        lr = 0.01
        weight_decay = 0.01

        def create_model_optimizer(optimizer_string, mem_format, dtype):
            # create model
            model_xpu_no_fuse = TrainingModel()
            model_xpu = TrainingModel()

            model_xpu_no_fuse = model_xpu_no_fuse.to(memory_format=mem_format).to(device=device).train()
            model_xpu = model_xpu.to(memory_format=mem_format).to(device=device).train()

            # align the weight
            align_all(model_xpu, model_xpu_no_fuse)

            # optimizer
            if optimizer_string.lower() == 'adamw':
                beta1 = 0.9
                beta2 = 0.999
                adam_epsilon = 1e-6
                amsgrad = True
                optimizer_xpu_no_fuse = torch.optim.AdamW(model_xpu_no_fuse.parameters(),
                                                          lr=lr,
                                                          betas=(beta1, beta2),
                                                          eps=adam_epsilon,
                                                          weight_decay=weight_decay,
                                                          amsgrad=amsgrad)
                optimizer_xpu = torch.optim.AdamW(model_xpu.parameters(),
                                                  lr=lr,
                                                  betas=(beta1, beta2),
                                                  eps=adam_epsilon,
                                                  weight_decay=weight_decay,
                                                  amsgrad=amsgrad)
            elif optimizer_string.lower() == 'sgd':
                optimizer_xpu_no_fuse = torch.optim.SGD(model_xpu_no_fuse.parameters(), lr=lr)
                optimizer_xpu = torch.optim.SGD(model_xpu.parameters(), lr=lr)
            else:
                raise RuntimeError("found unknown optimizer {}".format(optimizer_string))

            # non-fusion
            model_xpu_no_fuse, optimizer_xpu_no_fuse = torch.xpu.optimize(model=model_xpu_no_fuse,
                                                                          dtype=dtype, 
                                                                          optimizer=optimizer_xpu_no_fuse,
                                                                          fuse_update_step=False)

            # fusion
            model_xpu, optimizer_xpu = torch.xpu.optimize(model=model_xpu, 
                                                          dtype=dtype, 
                                                          optimizer=optimizer_xpu,
                                                          fuse_update_step=True)
            return model_xpu_no_fuse, model_xpu, optimizer_xpu_no_fuse, optimizer_xpu

        def align_all(model_xpu, model_xpu_no_fuse):
            for layer1 in model_xpu.modules():
                for layer2 in model_xpu_no_fuse.modules():
                    if (isinstance(layer1, nn.BatchNorm2d) and isinstance(layer2, nn.BatchNorm2d)):
                        layer1.weight.data = layer2.weight.clone().data
                        layer1.bias.data = layer2.bias.clone().data
                        if layer1.weight.grad is not None:
                            layer1.weight.grad.data = layer2.weight.grad.clone().data
                        if layer1.bias.grad is not None:
                            layer1.bias.grad.data = layer2.bias.grad.clone().data

                    if (isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.Conv2d)):
                        layer1.weight.data = layer2.weight.clone().data
                        if layer1.weight.grad is not None:
                            layer1.weight.grad.data = layer2.weight.grad.clone().data
                        if hasattr(layer1, 'master_weight'):
                            layer1.master_weight.data = layer2.master_weight.clone().data

                    if (isinstance(layer1, nn.Linear) and isinstance(layer2, nn.Linear)):
                        layer1.weight.data = layer2.weight.clone().data
                        layer1.bias.data = layer2.bias.clone().data
                        if layer1.weight.grad is not None:
                            layer1.weight.grad.data = layer2.weight.grad.clone().data
                        if layer1.bias.grad is not None:
                            layer1.bias.grad.data = layer2.bias.grad.clone().data
                        if hasattr(layer1, 'master_weight'):
                            layer1.master_weight.data = layer2.master_weight.clone().data
                        if hasattr(layer1, 'master_bias'):
                            layer1.master_bias.data = layer2.master_bias.clone().data
            torch.xpu.synchronize()

        def training(input, target, mode, dtype, optimizer, mem_format):
            input_xpu = torch.empty_like(input)
            input_xpu.data = input.data
            input_xpu.data = input_xpu.contiguous(memory_format=mem_format).requires_grad_()
            target_xpu = torch.empty_like(target)
            target_xpu.data = target.data

            criterion = nn.CrossEntropyLoss()

            # forward
            with torch.xpu.amp.autocast(enabled=True, dtype=dtype):
                output_xpu = mode(input_xpu)
                loss_xpu = criterion(output_xpu, target_xpu)

            # optimizer
            optimizer.zero_grad(set_to_none=True)

            loss_xpu.backward()

            # fusion optimizer and no-fused optimizer update
            optimizer.step()
            torch.xpu.synchronize()

        for dtype in [torch.bfloat16, torch.float32]:
            print('checking dtype: ', dtype)
            checking_atol = 1e-3
            checking_rtol = 1.6e-2
            if dtype == torch.float32:
                checking_atol = 1e-5
                checking_rtol = 1.3e-6
            for optimizer_string in SUPPORTED_FUSION_OPTIMIZER:
                print('checking optimizer: ', optimizer_string)
                for mem_format in [torch.contiguous_format, torch.channels_last]:
                    print('checking memory format: ', mem_format)
                    model_xpu_no_fuse, model_xpu, optimizer_xpu_no_fuse, optimizer_xpu = create_model_optimizer(
                        optimizer_string, mem_format, dtype=dtype)
                    for i in range(num_iter):
                        print('checking iter: ', i, '. optimizer: ', optimizer_string, end='')
                        print('. memory format: ', mem_format, ' dtype: ', dtype)
                        input = torch.randn(batch_size, input_channel, 7, 7).to(device=device)
                        target = torch.empty(batch_size, dtype=torch.long).random_(class_num).to(device=device)
                        align_all(model_xpu, model_xpu_no_fuse)

                        training(input, target, model_xpu_no_fuse, dtype, optimizer_xpu_no_fuse, mem_format)
                        training(input, target, model_xpu, dtype, optimizer_xpu, mem_format)

                        # checking updated weight
                        for layer1 in model_xpu.modules():
                            for layer2 in model_xpu_no_fuse.modules():
                                if (isinstance(layer1, nn.BatchNorm2d) and isinstance(layer2, nn.BatchNorm2d)):
                                    self.assertEqual(layer1.weight.cpu(),
                                                     layer2.weight.cpu(),
                                                     atol=checking_atol,
                                                     rtol=checking_rtol)
                                    self.assertEqual(layer1.bias.cpu(),
                                                     layer2.bias.cpu(),
                                                     atol=checking_atol,
                                                     rtol=checking_rtol)
                                if (isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.Conv2d)):
                                    self.assertEqual(layer1.weight.cpu(),
                                                     layer2.weight.cpu(),
                                                     atol=checking_atol,
                                                     rtol=checking_rtol)
                                if (isinstance(layer1, nn.Linear) and isinstance(layer2, nn.Linear)):
                                    self.assertEqual(layer1.weight.cpu(),
                                                     layer2.weight.cpu(),
                                                     atol=checking_atol,
                                                     rtol=checking_rtol)
                                    self.assertEqual(layer1.bias.cpu(),
                                                     layer2.bias.cpu(),
                                                     atol=checking_atol,
                                                     rtol=checking_rtol)
