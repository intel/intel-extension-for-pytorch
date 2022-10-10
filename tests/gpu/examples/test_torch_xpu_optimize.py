import torch
import torch.nn.functional
from torch import nn as nn
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch  # noqa
import pytest # noqa

device = 'xpu'

TEST_MODULE_CONVERT_LIST = [torch.nn.Conv2d,
                            torch.nn.Conv3d,
                            torch.nn.ConvTranspose2d,
                            torch.nn.ConvTranspose3d,
                            torch.nn.Linear,
                            torch.nn.Embedding,
                            torch.nn.LSTM]

# TODO: for now, only support SGD and AdamW
SUPPORTED_FUSION_OPTIMIZER = ['SGD',
                              'AdamW']


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
batch_size = 128
class_num = 1000
input_channel = 512
hidden_channel = 2048
num_iter = 10
checking_atol = 3e-3
checking_rtol = 3e-3


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

    def test_master_weight_fusion_optimizer(self):
        lr = 0.01
        weight_decay = 0.01
        dtype = torch.bfloat16

        def create_model_optimizer(optimizer_string, mem_format):
            # create model
            model_cpu = TrainingModel()
            model_cpu.train()
            model_xpu = TrainingModel()
            model_xpu.train()

            model_cpu = model_cpu.to(memory_format=mem_format)
            model_xpu = model_xpu.to(memory_format=mem_format).to(device=device)

            # align the weight in cpu model and xpu model in float32
            p_cpu_list = list(model_cpu.parameters())
            p_xpu_list = list(model_xpu.parameters())
            for k in range(len(p_cpu_list)):
                p_xpu_list[k].data = p_cpu_list[k].detach().clone().to(device=device).data
                torch.xpu.synchronize()

            # optimizer
            if optimizer_string.lower() == 'adamw':
                beta1 = 0.9
                beta2 = 0.999
                adam_epsilon = 1e-6
                amsgrad = True
                optimizer_cpu = torch.optim.AdamW(model_cpu.parameters(),
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
                optimizer_cpu = torch.optim.SGD(model_cpu.parameters(), lr=lr)
                optimizer_xpu = torch.optim.SGD(model_xpu.parameters(), lr=lr)
            else:
                raise RuntimeError("unknown optimizer {}".format(optimizer_string))

            # process torch.xpu.optimize
            model_xpu, optimizer_xpu = torch.xpu.optimize(model=model_xpu, dtype=dtype, optimizer=optimizer_xpu)
            return model_cpu, model_xpu, optimizer_cpu, optimizer_xpu

        def training_step(model_cpu, model_xpu, optimizer_cpu, optimizer_xpu, mem_format):
            input = torch.randn(batch_size, input_channel, 7, 7)
            target = torch.empty(batch_size, dtype=torch.long).random_(class_num)

            input_xpu = input.clone().to(device=device).requires_grad_()
            input_xpu = input_xpu.contiguous(memory_format=mem_format)
            target_xpu = target.to(device)
            input_cpu = input.clone().float().cpu().requires_grad_()
            input_cpu = input_cpu.contiguous(memory_format=mem_format)
            target_cpu = target.detach().clone().cpu()

            criterion = nn.CrossEntropyLoss()

            # forward
            output_cpu = model_cpu(input_cpu)
            with torch.xpu.amp.autocast(enabled=True, dtype=dtype):
                output_xpu = model_xpu(input_xpu)
                loss_xpu = criterion(output_xpu, target_xpu)
            torch.xpu.synchronize()

            # align output
            output_cpu.data = output_xpu.detach().clone().cpu().float().data
            torch.xpu.synchronize()

            # CPU loss
            loss_cpu = criterion(output_cpu, target_cpu)

            # optimizer
            optimizer_cpu.zero_grad()
            optimizer_xpu.zero_grad()

            # backward
            loss_cpu.backward()
            loss_xpu.backward()
            torch.xpu.synchronize()

            # align grad
            p_cpu_list = list(model_cpu.parameters())
            p_xpu_list = list(model_xpu.parameters())
            for k in range(len(p_cpu_list)):
                p_xpu_list[k].grad.data = p_cpu_list[k].grad.detach().clone().to(
                    device='xpu', dtype=p_xpu_list[k].dtype).data
                torch.xpu.synchronize()

            # update
            optimizer_cpu.step()
            optimizer_xpu.step()
            torch.xpu.synchronize()

        def check_result(model_xpu, model_cpu):
            # checking updated weight
            for layer1 in model_xpu.modules():
                for layer2 in model_cpu.modules():
                    if (isinstance(layer1, nn.BatchNorm2d) and isinstance(layer2, nn.BatchNorm2d)):
                        bn_xpu_weight = layer1.weight.clone().cpu().float()
                        bn_xpu_bias = layer1.bias.clone().cpu().float()
                        bn_cpu_weight = layer2.weight.clone().cpu().float()
                        bn_cpu_bias = layer2.bias.clone().cpu().float()

                        # checking
                        self.assertEqual(bn_cpu_weight, bn_xpu_weight.cpu().float(),
                                         atol=checking_atol, rtol=checking_rtol)
                        self.assertEqual(bn_cpu_bias, bn_xpu_bias.cpu().float(), atol=checking_atol, rtol=checking_rtol)

                    if (isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.Conv2d)):
                        conv_xpu_weight = layer1.master_weight.clone().cpu().float()
                        conv_cpu_weight = layer2.weight.clone().cpu().float()

                        # checking
                        self.assertEqual(conv_cpu_weight, conv_xpu_weight.cpu().float(),
                                         atol=checking_atol, rtol=checking_rtol)

                    if (isinstance(layer1, nn.Linear) and isinstance(layer2, nn.Linear)):
                        fc_xpu_weight = layer1.master_weight.clone().cpu().float()
                        fc_xpu_bias = layer1.master_bias.clone().cpu().float()
                        fc_cpu_weight = layer2.weight.clone().cpu().float()
                        fc_cpu_bias = layer2.bias.clone().cpu().float()

                        # checking
                        self.assertEqual(fc_cpu_weight, fc_xpu_weight.cpu().float(),
                                         atol=checking_atol, rtol=checking_rtol)
                        self.assertEqual(fc_cpu_bias, fc_xpu_bias.cpu().float(), atol=checking_atol, rtol=checking_rtol)

        for optimizer_string in SUPPORTED_FUSION_OPTIMIZER:
            print('checking optimizer: ', optimizer_string)
            for mem_format in [torch.contiguous_format, torch.channels_last]:
                print('checking memory format: ', mem_format)
                model_cpu, model_xpu, optimizer_cpu, optimizer_xpu = create_model_optimizer(
                    optimizer_string, mem_format)
                for i in range(num_iter):
                    print('checking iter: ', i, '. optimizer: ', optimizer_string, '. memory format: ', mem_format)
                    training_step(model_cpu, model_xpu, optimizer_cpu, optimizer_xpu, mem_format)
                    check_result(model_xpu, model_cpu)

    def test_weight_prepack_and_sample_input(self):
        num_iter = 5
        for use_sample_input in [True, False]:
            for dtype in [torch.float, torch.bfloat16, torch.float16]:
                use_autocast = False
                if dtype == torch.bfloat16 or dtype == torch.float16:
                    use_autocast = True

                module = InferenceModel().to(device=device)
                module_compare = InferenceModel()

                p_list = list(module.parameters())
                p_compare_list = list(module_compare.parameters())
                for k in range(len(p_list)):
                    p_compare_list[k].data = p_list[k].detach().clone().cpu().data
                    torch.xpu.synchronize()

                # this feature is only for inference
                module.eval()
                module_compare.eval()
                if use_sample_input:
                    optimized_module = torch.xpu.optimize(model=module, dtype=dtype,
                                                          level="O0", weights_prepack=True,
                                                          sample_input=torch.randn(16, 4, 16, 16).to(device=device))
                else:
                    optimized_module = torch.xpu.optimize(model=module, dtype=dtype, level="O0", weights_prepack=True)

                for i in range(num_iter):
                    dummy_input = torch.randn(16, 4, 16, 16).to(device=device)

                    # compute XPU
                    with torch.xpu.amp.autocast(enabled=use_autocast, dtype=dtype, cache_enabled=False):
                        output = optimized_module(dummy_input)

                    # compute CPU
                    output_compare = module_compare(dummy_input.cpu().float())

                    print('iter = ', i)
                    print('dtype = ', dtype)
                    print('use sample input = ', use_sample_input)
                    print('prepack output = ', output.cpu())
                    print('compare output = ', output_compare)
                    torch.xpu.synchronize()
                    if dtype == torch.float:
                        self.assertEqual(output_compare, output.cpu().float())
                    else:
                        self.assertEqual(output_compare, output.cpu().float(), atol=checking_atol, rtol=checking_rtol)
