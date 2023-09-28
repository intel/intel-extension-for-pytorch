import torch
import torch.nn.functional
from torch import nn as nn
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch  # noqa
from intel_extension_for_pytorch.optim._lamb import Lamb
import pytest  # noqa
import os
import itertools
device = "xpu"

TEST_MODULE_CONVERT_LIST = [
    torch.nn.Conv2d,
    torch.nn.Conv3d,
    torch.nn.ConvTranspose2d,
    torch.nn.ConvTranspose3d,
    torch.nn.Linear,
    torch.nn.Embedding,
    torch.nn.LSTM,
]

SUPPORTED_FUSION_OPTIMIZER = ['Adam', 'SGD', 'AdamW', 'Lars', 'Lamb', 'splitSGD', 'Adagrad']
SUPPORTED_FUSED_ADAM = ['Adam', 'AdamW']

SUPPORTED_SPARSE_OPTIMIZER = ['Adagrad']

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
        self.emb = (torch.nn.Embedding(256, 4),)
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
            nn.Conv2d(
                input_channel,
                hidden_channel,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channel, eps=1e-05, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
        )
        self.fc = nn.Linear(
            in_features=hidden_channel, out_features=class_num, bias=True
        )

    def forward(self, x):
        x = self.m(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TrainingSparseModel(nn.Module):
    def __init__(self):
        super(TrainingSparseModel, self).__init__()
        self.emb = nn.EmbeddingBag(hidden_channel, 128, mode="sum", sparse=True)
        self.fc = nn.Linear(
            in_features=128, out_features=class_num, bias=True
        )

    def forward(self, x):
        offset = torch.tensor(list(range(batch_size)), dtype=torch.long).to(x.device)
        x = self.emb(x, offset)
        x = self.fc(x)
        return x


AUTO_CHANNELS_LAST_SCOPE = [
    torch.nn.Conv1d,
    torch.nn.Conv2d,
    torch.nn.Conv3d,
    torch.nn.ConvTranspose1d,
    torch.nn.ConvTranspose2d,
    torch.nn.ConvTranspose3d,
]


class ChannelsLastModel(nn.Module):
    def __init__(self):
        super(ChannelsLastModel, self).__init__()
        self.conv1d = nn.Conv1d(input_channel, hidden_channel, kernel_size=(1))
        self.conv2d = nn.Conv2d(input_channel, hidden_channel, kernel_size=(1, 1))
        self.conv3d = nn.Conv3d(input_channel, hidden_channel, kernel_size=(1, 1, 1))
        self.deconv1d = nn.ConvTranspose1d(
            input_channel, hidden_channel, kernel_size=(1)
        )
        self.deconv2d = nn.ConvTranspose2d(
            input_channel, hidden_channel, kernel_size=(1, 1)
        )
        self.deconv3d = nn.ConvTranspose3d(
            input_channel, hidden_channel, kernel_size=(1, 1, 1)
        )


class TestTorchMethod(TestCase):
    def test_convert_module_data_type(self):
        def check_dtype_for_module(module, dtype):
            for module_cls in TEST_MODULE_CONVERT_LIST:
                if isinstance(module, module_cls):
                    self.assertEqual(module.weight.dtype, dtype)
                    if hasattr(module, "bias") and module.bias is not None:
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
        optimized_module = torch.xpu.optimize(
            model=module, dtype=torch.float16, level="O0", conv_bn_folding=True
        )
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
            if module_pattern[check_pos] != module and isinstance(
                module_pattern[check_pos], torch.nn.Dropout
            ):
                self.assertEqual(type(module), torch.nn.identity)
            check_pos = check_pos + 1
            for child in module.children():
                check_no_dropout_but_identity(child, check_pos)

        pattern_pos = 0
        collect_module_pattern(module, pattern_pos)
        optimized_module = torch.xpu.optimize(
            model=module,
            dtype=torch.float16,
            level="O0",
            replace_dropout_with_identity=True,
        )
        check_pos = 0
        check_no_dropout_but_identity(optimized_module, check_pos)

    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_sparse_weight_optimizer(self):
        lr = 0.01
        weight_decay = 0.01

        def create_model_optimizer(optimizer_string, mem_format, dtype):
            model_optimizer_list = []
            # create model
            model_cpu = TrainingSparseModel()
            model_xpu = TrainingSparseModel()

            # align the weight
            align_all(model_cpu, model_xpu)

            model_cpu = model_cpu.train()
            model_xpu = model_xpu.to(device=device).train()


            # optimizer
            if optimizer_string.lower() == "adagrad":
                lr_decay = 0.01
                optimizer_cpu = torch.optim.Adagrad(model_cpu.parameters(), lr=lr, lr_decay=lr_decay, weight_decay=0)
                optimizer_xpu = torch.optim.Adagrad(model_xpu.parameters(), lr=lr, lr_decay=lr_decay, weight_decay=0)
                model_optimizer_list.append([optimizer_cpu, optimizer_xpu])
            else:
                raise RuntimeError(
                    "found unknown optimizer {}".format(optimizer_string)
                )

            model_and_optimzier_list = []
            model_xpu, optimizer_xpu = torch.xpu.optimize(model=model_xpu,
                                                          dtype=torch.float32, optimizer=model_optimizer_list[0][1])

            model_and_optimzier_list.append([model_cpu, model_xpu, optimizer_cpu, optimizer_xpu])
            return model_and_optimzier_list

        def align_all(model_cpu, model_xpu):
            for layer1 in model_xpu.modules():
                for layer2 in model_cpu.modules():
                    if isinstance(layer1, nn.BatchNorm2d) and isinstance(
                        layer2, nn.BatchNorm2d
                    ):
                        layer1.weight.data = layer2.weight.clone().data
                        layer1.bias.data = layer2.bias.clone().data
                        if layer1.weight.grad is not None:
                            layer1.weight.grad.data = layer2.weight.grad.clone().data
                        if layer1.bias.grad is not None:
                            layer1.bias.grad.data = layer2.bias.grad.clone().data

                    if isinstance(layer1, nn.EmbeddingBag) and isinstance(
                        layer2, nn.EmbeddingBag
                    ):
                        layer1.weight.data = layer2.weight.clone().data
                        if layer1.weight.grad is not None:
                            layer1.weight.grad.data = layer2.weight.grad.clone().data

                    if isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.Conv2d):
                        layer1.weight.data = layer2.weight.clone().data
                        if layer1.weight.grad is not None:
                            layer1.weight.grad.data = layer2.weight.grad.clone().data
                        if hasattr(layer1, "master_weight"):
                            layer1.master_weight.data = (
                                layer2.master_weight.clone().data
                            )

                    if isinstance(layer1, nn.Linear) and isinstance(layer2, nn.Linear):
                        layer1.weight.data = layer2.weight.clone().data
                        layer1.bias.data = layer2.bias.clone().data
                        if layer1.weight.grad is not None:
                            layer1.weight.grad.data = layer2.weight.grad.clone().data
                        if layer1.bias.grad is not None:
                            layer1.bias.grad.data = layer2.bias.grad.clone().data
                        if hasattr(layer1, "master_weight"):
                            layer1.master_weight.data = (
                                layer2.master_weight.clone().data
                            )
                        if hasattr(layer1, "master_bias"):
                            layer1.master_bias.data = layer2.master_bias.clone().data
            torch.xpu.synchronize()

        def training(input, target, mode, dtype, optimizer, mem_format):
            criterion = nn.CrossEntropyLoss()

            # forward
            loss = None
            if input.device == "xpu":
                with torch.xpu.amp.autocast(enabled=True, dtype=dtype):
                    output = mode(input)
                    loss = criterion(output, target)
            else:
                if dtype == torch.float32:
                    output = mode(input)
                    loss = criterion(output, target)
                else:
                    with torch.autocast(device_type="cpu", enabled=True, dtype=dtype):
                        output = mode(input)
                        loss = criterion(output, target)


            # optimizer
            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            # fusion optimizer and no-fused optimizer update
            optimizer.step()
            if input.device == "xpu":
                torch.xpu.synchronize()

        for optimizer_string in SUPPORTED_SPARSE_OPTIMIZER:
            print("checking optimizer: ", optimizer_string)
            support_dtype_list = [torch.float32]
            for dtype in support_dtype_list:
                print("checking dtype: ", dtype)
                if dtype == torch.bfloat16:
                    checking_atol = 1e-3
                    checking_rtol = 1.6e-2
                else:
                    checking_atol = 1e-3
                    checking_rtol = 1.6e-2
                for mem_format in [torch.contiguous_format, torch.channels_last]:
                    print('checking memory format: ', mem_format)
                    model_optimizer_list = create_model_optimizer(optimizer_string, mem_format, dtype=dtype)
                    for model_optimizer_item in model_optimizer_list:
                        model_cpu = model_optimizer_item[0]
                        model_xpu = model_optimizer_item[1]
                        optimizer_cpu = model_optimizer_item[2]
                        optimizer_xpu = model_optimizer_item[3]
                        for i in range(num_iter):
                            print('checking iter: ', i, '. optimizer: ', optimizer_string, end='')
                            print('. memory format: ', mem_format, ' dtype: ', dtype)
                            input = torch.randint(0, hidden_channel, (batch_size,))
                            target = torch.empty(batch_size, dtype=torch.long).random_(class_num)

                            align_all(model_cpu.to(device=device), model_xpu)
                            model_cpu.to("cpu")

                            training(input, target, model_cpu, dtype, optimizer_cpu, mem_format)
                            training(input.to(device), target.to(device), model_xpu, dtype, optimizer_xpu, mem_format)

                            # checking updated weight
                            for layer1 in model_xpu.modules():
                                for layer2 in model_cpu.modules():
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
                                    if (isinstance(layer1, nn.EmbeddingBag) and isinstance(layer2, nn.EmbeddingBag)):
                                        self.assertEqual(layer1.weight.cpu(),
                                                         layer2.weight.cpu(),
                                                         atol=checking_atol,
                                                         rtol=checking_rtol)


    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_master_weight_fusion_optimizer(self):
        # using large lr to fastern the error occurance
        lr = 10.0
        weight_decay = 0.01

        def create_model_optimizer(optimizer_string, dtype):
            model_optimizer_list = []

            def create_model():
                # create model
                model_xpu_no_fuse = TrainingModel()
                model_xpu = TrainingModel()

                model_xpu_no_fuse = (
                    model_xpu_no_fuse.to(device=device).train()
                )
                model_xpu = model_xpu.to(device=device).train()

                # align the weight
                align_all(model_xpu, model_xpu_no_fuse)
                return model_xpu_no_fuse, model_xpu

            # optimizer
            if optimizer_string.lower() == "adamw":
                beta1 = 0.9
                beta2 = 0.999
                adam_epsilon = 1e-6
                amsgrad = True
                weight_decay = 0.01
                lr = 0.05
                model_xpu_no_fuse, model_xpu = create_model()
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
                model_optimizer_list.append([model_xpu_no_fuse, model_xpu, optimizer_xpu_no_fuse, optimizer_xpu])
            elif optimizer_string.lower() == 'lamb':
                beta1 = 0.9
                beta2 = 0.999
                lamb_epsilon = 1e-6
                lr = 0.05
                model_xpu_no_fuse, model_xpu = create_model()
                optimizer_xpu_no_fuse = Lamb(model_xpu_no_fuse.parameters(), lr=lr, betas=(beta1, beta2), eps=lamb_epsilon)
                optimizer_xpu = Lamb(model_xpu.parameters(), lr=lr, betas=(beta1, beta2), eps=lamb_epsilon)
                model_optimizer_list.append([model_xpu_no_fuse, model_xpu, optimizer_xpu_no_fuse, optimizer_xpu])
            elif optimizer_string.lower() == 'sgd' or optimizer_string.lower() == 'splitsgd':
                lr = 5.0
                for momentum_value in [0.0, 0.9]:
                    model_xpu_no_fuse, model_xpu = create_model()
                    optimizer_xpu_no_fuse = torch.optim.SGD(model_xpu_no_fuse.parameters(), lr=lr, momentum=momentum_value)
                    optimizer_xpu = torch.optim.SGD(model_xpu.parameters(), lr=lr, momentum=momentum_value)
                    model_optimizer_list.append([model_xpu_no_fuse, model_xpu, optimizer_xpu_no_fuse, optimizer_xpu])
            elif optimizer_string.lower() == 'adagrad':
                lr = 5.0
                lr_decay = 0.01
                weight_decay = 0.05
                model_xpu_no_fuse, model_xpu = create_model()
                optimizer_xpu_no_fuse = torch.optim.Adagrad(model_xpu_no_fuse.parameters(),
                                                            lr=lr,
                                                            eps=0,
                                                            weight_decay=weight_decay,
                                                            lr_decay=lr_decay)
                optimizer_xpu = torch.optim.Adagrad(model_xpu.parameters(),
                                                    lr=lr,
                                                    eps=0,
                                                    weight_decay=weight_decay,
                                                    lr_decay=lr_decay)
            elif optimizer_string.lower() == 'adam':
                beta1 = 0.9
                beta2 = 0.999
                adam_epsilon = 1e-6
                amsgrad = True
                weight_decay = 0.01
                lr = 0.05
                model_xpu_no_fuse, model_xpu = create_model()
                optimizer_xpu_no_fuse = torch.optim.Adam(model_xpu_no_fuse.parameters(),
                                                         lr=lr,
                                                         betas=(beta1, beta2),
                                                         eps=adam_epsilon,
                                                         weight_decay=weight_decay,
                                                         amsgrad=amsgrad)
                optimizer_xpu = torch.optim.Adam(model_xpu.parameters(),
                                                 lr=lr,
                                                 betas=(beta1, beta2),
                                                 eps=adam_epsilon,
                                                 weight_decay=weight_decay,
                                                 amsgrad=amsgrad)
                model_optimizer_list.append([model_xpu_no_fuse, model_xpu, optimizer_xpu_no_fuse, optimizer_xpu])
            elif optimizer_string.lower() == 'lars':
                momentum = 0.9
                epsilon = 0.001
                lr = 5.0
                weight_decay = 0.01
                model_xpu_no_fuse, model_xpu = create_model()
                optimizer_xpu_no_fuse = torch.xpu.optim.Lars(model_xpu_no_fuse.parameters(),
                                                             lr=lr, weight_decay=weight_decay,
                                                             momentum=momentum,
                                                             epsilon=epsilon)
                optimizer_xpu = torch.xpu.optim.Lars(model_xpu.parameters(),
                                                     lr=lr, weight_decay=weight_decay,
                                                     momentum=momentum,
                                                     epsilon=epsilon)
                model_optimizer_list.append([model_xpu_no_fuse, model_xpu, optimizer_xpu_no_fuse, optimizer_xpu])
            else:
                raise RuntimeError(
                    "found unknown optimizer {}".format(optimizer_string)
                )

            # optimize with torch.xpu.optimize
            model_and_optimzier_list = []
            for model_optimzier_item in model_optimizer_list:
                # create non-fusion model and optimizer
                # [watch out] for split sgd, the fused optimizer is required in torch.xpu.optimize,
                # here model_xpu_no_fuse, as accuracy reference, it uses master weight training
                model_xpu_no_fuse, optimizer_xpu_no_fuse = torch.xpu.optimize(model=model_optimzier_item[0],
                                                                              dtype=dtype,
                                                                              optimizer=model_optimzier_item[2],
                                                                              fuse_update_step=False)

                # create fusion model and optimizer
                use_split_master_weight = True if 'split' in optimizer_string.lower() else False
                model_xpu, optimizer_xpu = torch.xpu.optimize(model=model_optimzier_item[1],
                                                              dtype=dtype,
                                                              optimizer=model_optimzier_item[3],
                                                              fuse_update_step=True,
                                                              split_master_weight_for_bf16=use_split_master_weight)
                model_and_optimzier_list.append([model_xpu_no_fuse, model_xpu, optimizer_xpu_no_fuse, optimizer_xpu])
            return model_and_optimzier_list

        def align_all(model_xpu, model_xpu_no_fuse):
            for layer1 in model_xpu.modules():
                for layer2 in model_xpu_no_fuse.modules():
                    if isinstance(layer1, nn.BatchNorm2d) and isinstance(
                        layer2, nn.BatchNorm2d
                    ):
                        layer1.weight.data = layer2.weight.clone().data
                        layer1.bias.data = layer2.bias.clone().data
                        if layer1.weight.grad is not None:
                            layer1.weight.grad.data = layer2.weight.grad.clone().data
                        if layer1.bias.grad is not None:
                            layer1.bias.grad.data = layer2.bias.grad.clone().data

                    if isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.Conv2d):
                        layer1.weight.data = layer2.weight.clone().data
                        if layer1.weight.grad is not None:
                            layer1.weight.grad.data = layer2.weight.grad.clone().data
                        if hasattr(layer1, "weight" + "_wrapper"):
                            if layer1.weight_wrapper.master_parameter is not None:
                                layer1.weight_wrapper.master_parameter.data = (
                                    layer2.weight_wrapper.master_parameter.clone().data
                                )
                            elif layer1.weight_wrapper.parameter_trail is not None:
                                # split the master weight from layer2 to align the layer1 weight trail
                                _, layer2_weight_trail = torch.ops.torch_ipex.split_float_bfloat16(
                                    layer2.weight_wrapper.master_parameter.clone().data
                                )
                                layer1.weight_wrapper.parameter_trail.data = layer2_weight_trail.data
                            else:
                                pass

                    if isinstance(layer1, nn.Linear) and isinstance(layer2, nn.Linear):
                        layer1.weight.data = layer2.weight.clone().data
                        layer1.bias.data = layer2.bias.clone().data
                        if layer1.weight.grad is not None:
                            layer1.weight.grad.data = layer2.weight.grad.clone().data
                        if layer1.bias.grad is not None:
                            layer1.bias.grad.data = layer2.bias.grad.clone().data
                        if hasattr(layer1, "weight" + "_wrapper"):
                            if layer1.weight_wrapper.master_parameter is not None:
                                layer1.weight_wrapper.master_parameter.data = (
                                    layer2.weight_wrapper.master_parameter.clone().data
                                )
                            elif layer1.weight_wrapper.parameter_trail is not None:
                                # split the master weight from layer2 to align the layer1 weight trail
                                _, layer2_weight_trail = torch.ops.torch_ipex.split_float_bfloat16(
                                    layer2.weight_wrapper.master_parameter.clone().data
                                )
                                layer1.weight_wrapper.parameter_trail.data = layer2_weight_trail.data
                            else:
                                pass

                        if hasattr(layer1, "bias" + "_wrapper"):
                            if layer1.bias_wrapper.master_parameter is not None:
                                layer1.bias_wrapper.master_parameter.data = (
                                    layer2.bias_wrapper.master_parameter.clone().data
                                )
                            elif layer1.bias_wrapper.parameter_trail is not None:
                                # split the master bias from layer2 to align the layer1 bias trail
                                _, layer2_bias_trail = torch.ops.torch_ipex.split_float_bfloat16(
                                    layer2.bias_wrapper.master_parameter.clone().data
                                )
                                layer1.bias_wrapper.parameter_trail.data = layer2_bias_trail.data
                            else:
                                pass

            torch.xpu.synchronize()

        def training_without_step(input, target, mode, dtype, optimizer):
            input_xpu = torch.empty_like(input)
            input_xpu.data = input.data
            input_xpu = input_xpu.requires_grad_(True)
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
            torch.xpu.synchronize()

        for optimizer_string in SUPPORTED_FUSION_OPTIMIZER:
            print("checking optimizer: ", optimizer_string)
            support_dtype_list = [torch.float32, torch.bfloat16]
            if optimizer_string.lower() == "adam" or optimizer_string.lower() == "sgd":
                support_dtype_list.append(torch.float64)

            # for split master weight training, the test for fp32 is ignored
            if 'split' in optimizer_string.lower():
                support_dtype_list.remove(torch.float32)
            for dtype in support_dtype_list:
                print("checking dtype: ", dtype)
                checking_atol = 1e-5
                checking_rtol = 1.3e-6
                if dtype == torch.bfloat16:
                    checking_atol = 1e-3
                    checking_rtol = 1.6e-2
                model_optimizer_list = create_model_optimizer(optimizer_string, dtype=dtype)
                for model_optimizer_item in model_optimizer_list:
                    model_xpu_no_fuse = model_optimizer_item[0]
                    model_xpu = model_optimizer_item[1]
                    optimizer_xpu_no_fuse = model_optimizer_item[2]
                    optimizer_xpu = model_optimizer_item[3]
                    for i in range(num_iter):
                        print('checking iter: ', i, '. optimizer: ', optimizer_string, end='')
                        print(' dtype: ', dtype)
                        input = torch.randn(batch_size, input_channel, 7, 7).to(device=device)
                        target = torch.empty(batch_size, dtype=torch.long).random_(class_num).to(device=device)

                        training_without_step(input, target, model_xpu_no_fuse, dtype, optimizer_xpu_no_fuse)
                        training_without_step(input, target, model_xpu, dtype, optimizer_xpu)
                        align_all(model_xpu, model_xpu_no_fuse)
                        optimizer_xpu_no_fuse.step()
                        optimizer_xpu.step()

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

    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_fused_adam_and_adamw(self):
        # using large lr to fastern the error occurance
        lr = 10.0
        weight_decay = 0.01

        def create_model_optimizer(optimizer_string, mem_format, dtype, lr_is_tensor):
            model_optimizer_list = []
            # create model
            model_xpu_no_fuse = TrainingModel()
            model_xpu = TrainingModel()

            model_xpu_no_fuse = (
                model_xpu_no_fuse.to(memory_format=mem_format).to(device=device).train()
            )
            model_xpu = model_xpu.to(memory_format=mem_format).to(device=device).train()

            # align the weight
            align_all(model_xpu, model_xpu_no_fuse)

            # optimizer
            if optimizer_string.lower() == "adamw":
                beta1 = 0.9
                beta2 = 0.999
                adam_epsilon = 1e-6
                amsgrad = True
                weight_decay = 0.01
                lr = torch.tensor(0.05) if lr_is_tensor else 0.05
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
                                                  amsgrad=amsgrad, fused=True)
                model_optimizer_list.append([optimizer_xpu_no_fuse, optimizer_xpu])
            elif optimizer_string.lower() == 'adam':
                beta1 = 0.9
                beta2 = 0.999
                adam_epsilon = 1e-6
                amsgrad = True
                weight_decay = 0.01
                lr = torch.tensor(0.05) if lr_is_tensor else 0.05
                optimizer_xpu_no_fuse = torch.optim.Adam(model_xpu_no_fuse.parameters(),
                                                         lr=lr,
                                                         betas=(beta1, beta2),
                                                         eps=adam_epsilon,
                                                         weight_decay=weight_decay,
                                                         amsgrad=amsgrad)
                optimizer_xpu = torch.optim.Adam(model_xpu.parameters(),
                                                 lr=lr,
                                                 betas=(beta1, beta2),
                                                 eps=adam_epsilon,
                                                 weight_decay=weight_decay,
                                                 amsgrad=amsgrad, fused=True)
                model_optimizer_list.append([optimizer_xpu_no_fuse, optimizer_xpu])
            else:
                raise RuntimeError(
                    "found unknown optimizer with fused {}".format(optimizer_string)
                )

            model_and_optimzier_list = []
            for model_optimzier_item in model_optimizer_list:
                # index 0 is non fuse optimizer, index 1 fuse optimizer
                use_split_master_weight = True if 'split' in optimizer_string.lower() else False
                # torch.xpu.adam(w)_fuse_step
                model_xpu_no_fuse, optimizer_xpu_no_fuse = torch.xpu.optimize(model=model_xpu_no_fuse,
                                                                              dtype=dtype,
                                                                              optimizer=model_optimzier_item[0],
                                                                              fuse_update_step=True,
                                                                              split_master_weight_for_bf16=use_split_master_weight)

                # aten::_fused_adam(w)
                model_xpu, optimizer_xpu = torch.xpu.optimize(model=model_xpu,
                                                              dtype=dtype,
                                                              optimizer=model_optimzier_item[1],
                                                              fuse_update_step=True,
                                                              split_master_weight_for_bf16=use_split_master_weight)
                model_and_optimzier_list.append([model_xpu_no_fuse, model_xpu, optimizer_xpu_no_fuse, optimizer_xpu])
            return model_and_optimzier_list

        def align_all(model_xpu, model_xpu_no_fuse):
            for layer1 in model_xpu.modules():
                for layer2 in model_xpu_no_fuse.modules():
                    if isinstance(layer1, nn.BatchNorm2d) and isinstance(
                        layer2, nn.BatchNorm2d
                    ):
                        layer1.weight.data = layer2.weight.clone().data
                        layer1.bias.data = layer2.bias.clone().data
                        if layer1.weight.grad is not None:
                            layer1.weight.grad.data = layer2.weight.grad.clone().data
                        if layer1.bias.grad is not None:
                            layer1.bias.grad.data = layer2.bias.grad.clone().data

                    if isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.Conv2d):
                        layer1.weight.data = layer2.weight.clone().data
                        if layer1.weight.grad is not None:
                            layer1.weight.grad.data = layer2.weight.grad.clone().data
                        if hasattr(layer1, "master_weight"):
                            layer1.master_weight.data = (
                                layer2.master_weight.clone().data
                            )

                    if isinstance(layer1, nn.Linear) and isinstance(layer2, nn.Linear):
                        layer1.weight.data = layer2.weight.clone().data
                        layer1.bias.data = layer2.bias.clone().data
                        if layer1.weight.grad is not None:
                            layer1.weight.grad.data = layer2.weight.grad.clone().data
                        if layer1.bias.grad is not None:
                            layer1.bias.grad.data = layer2.bias.grad.clone().data
                        if hasattr(layer1, "master_weight"):
                            layer1.master_weight.data = (
                                layer2.master_weight.clone().data
                            )
                        if hasattr(layer1, "master_bias"):
                            layer1.master_bias.data = layer2.master_bias.clone().data
            torch.xpu.synchronize()

        def training(input, target, mode, dtype, optimizer, mem_format):
            input_xpu = torch.empty_like(input)
            input_xpu.data = input.data
            input_xpu.data = input_xpu.contiguous(
                memory_format=mem_format
            ).requires_grad_()
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
            torch.xpu.synchronize()

        for optimizer_string in SUPPORTED_FUSED_ADAM:
            print("checking optimizer: ", optimizer_string)
            support_dtype_list = [torch.float32, torch.bfloat16]
            if optimizer_string.lower() == "adam" or optimizer_string.lower() == "sgd":
                support_dtype_list.append(torch.float64)
            for dtype in support_dtype_list:
                print("checking dtype: ", dtype)
                checking_atol = 1e-5
                checking_rtol = 1.3e-6
                if dtype == torch.bfloat16:
                    checking_atol = 1e-3
                    checking_rtol = 1.6e-2
                for mem_format in [torch.contiguous_format, torch.channels_last]:
                    print('checking memory format: ', mem_format)
                    for is_tensor in [True, False]:
                        model_optimizer_list = create_model_optimizer(
                            optimizer_string,
                            mem_format,
                            dtype=dtype,
                            lr_is_tensor=is_tensor
                        )
                        for model_optimizer_item in model_optimizer_list:
                            model_xpu_no_fuse = model_optimizer_item[0]
                            model_xpu = model_optimizer_item[1]
                            optimizer_xpu_no_fuse = model_optimizer_item[2]
                            optimizer_xpu = model_optimizer_item[3]
                            for i in range(num_iter):
                                print('checking iter: ', i, '. optimizer: ', optimizer_string, end='')
                                print('. memory format: ', mem_format, ' dtype: ', dtype)
                                input = torch.randn(batch_size, input_channel, 7, 7).to(device=device)
                                target = torch.empty(batch_size, dtype=torch.long).random_(class_num).to(device=device)

                                training(input, target, model_xpu_no_fuse, dtype, optimizer_xpu_no_fuse, mem_format)
                                training(input, target, model_xpu, dtype, optimizer_xpu, mem_format)
                                align_all(model_xpu, model_xpu_no_fuse)
                                optimizer_xpu_no_fuse.step()
                                optimizer_xpu.step()

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

    def test_xpu_auto_channels_last(self):
        def check_layout_for_module(module):
            if module in AUTO_CHANNELS_LAST_SCOPE:
                if torch.xpu.has_fp64_dtype():
                    self.assertTrue(
                        module.weight.is_contiguous(memory_format=torch.channels_last)
                    )
                else:
                    self.assertFalse(
                        module.weight.is_contiguous(memory_format=torch.channels_last)
                    )
            for child in module.children():
                check_layout_for_module(child)

        module = ChannelsLastModel().to(device=device).train()
        optimizer_xpu = torch.optim.SGD(module.parameters(), lr=0.1)
        optimized_module, optimizer_xpu = torch.xpu.optimize(
            model=module, optimizer=optimizer_xpu, dtype=torch.float32
        )
        check_layout_for_module(optimized_module)

    def test_load_state_dict(self):
        def training(model, optimizer):
            input_xpu = torch.randn(batch_size, input_channel, 7, 7).to(device=device)
            target_xpu = (
                torch.empty(batch_size, dtype=torch.long)
                .random_(class_num)
                .to(device=device)
            )

            criterion = nn.CrossEntropyLoss()

            # forward
            with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                output_xpu = model(input_xpu)
                loss_xpu = criterion(output_xpu, target_xpu)

            # optimizer
            optimizer.zero_grad(set_to_none=True)
            loss_xpu.backward()
            optimizer.step()
            torch.xpu.synchronize()

        model_xpu = TrainingModel()
        model_xpu = model_xpu.to(device=device).train()
        optimizer_xpu = torch.optim.SGD(model_xpu.parameters(), lr=0.1)
        model_xpu, optimizer_xpu = torch.xpu.optimize(
            model=model_xpu, dtype=torch.bfloat16, optimizer=optimizer_xpu
        )

        # training for some iterations
        for _ in range(10):
            training(model_xpu, optimizer_xpu)

        state = {
            "model_state_dict": model_xpu.state_dict(),
            "optimizer_state_dict": optimizer_xpu.state_dict(),
        }
        filename = "./_checkpoint_check_load_state_dict.pth.tar"
        if os.path.exists(filename):
            os.remove(filename)
        torch.save(state, filename)

        # load checkpoint
        new_model = TrainingModel()
        new_model = new_model.to(device=device).train()
        new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.1)
        new_model, new_optimizer = torch.xpu.optimize(
            model=new_model, dtype=torch.bfloat16, optimizer=new_optimizer
        )

        checkpoint = torch.load(filename)
        new_model.load_state_dict(checkpoint["model_state_dict"])
        new_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        load_state = {
            "model_state_dict": new_model.state_dict(),
            "optimizer_state_dict": new_optimizer.state_dict(),
        }

        for state_name in state["model_state_dict"]:
            original_model_state = state["model_state_dict"][state_name]
            loaded_model_state = load_state["model_state_dict"][state_name]
            if isinstance(original_model_state, torch.Tensor) and isinstance(
                loaded_model_state, torch.Tensor
            ):
                # For ATS-M, comparing xpu floating tensor will to double in comparing mechanism, so here to cpu
                self.assertEqual(original_model_state.cpu(), loaded_model_state.cpu())
            else:
                self.assertEqual(original_model_state, loaded_model_state)

        for state_name in state["optimizer_state_dict"]:
            original_optimizer_state = state["optimizer_state_dict"][state_name]
            loaded_optimizer_state = load_state["optimizer_state_dict"][state_name]
            if isinstance(original_optimizer_state, torch.Tensor) and isinstance(
                loaded_optimizer_state, torch.Tensor
            ):
                # For ATS-M, comparing xpu floating tensor will to double in comparing mechanism, so here to cpu
                self.assertEqual(
                    original_optimizer_state.cpu(), loaded_optimizer_state.cpu()
                )
            else:
                self.assertEqual(original_optimizer_state, loaded_optimizer_state)
        os.remove(filename)

    def test_reentrancy_of_torch_xpu_optimize(self):
        CALL_NUM = 3

        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.input = (
                    torch.randn(1, 3, 224, 224, device=device),
                    torch.randn(100, 100, device=device),
                    torch.randn(5, 5, 3, 3, device=device),
                )
                self.conv = torch.nn.Conv2d(
                    3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)
                )
                self.linear = torch.nn.Linear(100, 100)
                self.conv_transpose2d = torch.nn.ConvTranspose2d(5, 5, (3, 3))

            def forward(self, x1, x2, x3):
                return (
                    self.conv(x1).sum()
                    + self.linear(x2).sum()
                    + self.conv_transpose2d(x3)
                )

        def run_and_recursively_call_ipex_optimize(
            model_class, dtype, level, split_master_weight_for_bf16, fuse_update_step
        ):
            model = model_class().train().to(device=device)
            input = model.input
            optimizer = torch.optim.SGD(model.parameters(), lr=10.01)
            for _ in range(CALL_NUM):
                # recursively calling ipex.optimize CALL_NUM times
                model, optimizer = torch.xpu.optimize(
                    model,
                    dtype=dtype,
                    optimizer=optimizer,
                    level=level,
                    split_master_weight_for_bf16=split_master_weight_for_bf16,
                    fuse_update_step=fuse_update_step,
                )
                with torch.xpu.amp.autocast(enabled=True, dtype=dtype):
                    y = model(*input).sum()
                optimizer.zero_grad()
                y.backward()
                optimizer.step()

        params_dict = {
            "dtype": [torch.float32, torch.bfloat16],
            "level": ["O0", "O1"],
            "split_master_weight_for_bf16": [True, False],
            "fuse_update_step": [True, False],
        }

        for dtype, level, split_master_weight_for_bf16, fuse_update_step in list(
            itertools.product(*params_dict.values())
        ):
            run_and_recursively_call_ipex_optimize(
                Model, dtype, level, split_master_weight_for_bf16, fuse_update_step
            )
