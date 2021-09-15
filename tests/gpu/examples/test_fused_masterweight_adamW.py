import torch
import ipex
from torch.optim import Optimizer
from torch.testing._internal.common_utils import TestCase
import math

WARM = 10


class TestTorchMethod(TestCase):
    def test_fused_masterweight_adamW(self):
        # hpersparameters
        step = 0
        exp_avg = []
        exp_avg_sq = []
        beta1 = 1.0
        beta2 = 1.0
        correct_bias = False
        weight_decay = 0.1
        lr = 0.1
        eps = 0.1
        step_cpu = 0
        exp_avg_cpu = []
        exp_avg_sq_cpu = []

        # param_weight is used to store the parameters, which are same as Bert, BF16.
        param_weight = []
        param_weight_cpu = []

        # param_grad is used to store the parameters grad, which are BF16 grad.
        param_grad = []
        param_grad_cpu = []

        # param_master_weight is used to store the master weight parameters, which are same as Bert, FP32.
        param_master_weight = []

        param_master_weight.append(torch.randn([1024, 1024], dtype=torch.float32, device='xpu'))
        param_master_weight.append(torch.randn([1024, 4096], dtype=torch.float32, device='xpu'))
        param_master_weight.append(torch.randn([1024], dtype=torch.float32, device='xpu'))
        param_master_weight.append(torch.randn([2, 1024], dtype=torch.float32, device='xpu'))
        param_master_weight.append(torch.randn([30522, 1024], dtype=torch.float32, device='xpu'))
        param_master_weight.append(torch.randn([30522], dtype=torch.float32, device='xpu'))
        param_master_weight.append(torch.randn([4096, 1024], dtype=torch.float32, device='xpu'))
        param_master_weight.append(torch.randn([4096], dtype=torch.float32, device='xpu'))
        param_master_weight.append(torch.randn([512, 1024], dtype=torch.float32, device='xpu'))

        # param weight is original weight from model and has been convert to BF16
        for p in param_master_weight:
            param_weight.append(p.detach().clone().bfloat16())

        # param_grad create the according grad, BF16.
        param_grad.append(torch.randn([1024, 1024], dtype=torch.bfloat16, device='xpu'))
        param_grad.append(torch.randn([1024, 4096], dtype=torch.bfloat16, device='xpu'))
        param_grad.append(torch.randn([1024], dtype=torch.bfloat16, device='xpu'))
        param_grad.append(torch.randn([2, 1024], dtype=torch.bfloat16, device='xpu'))
        param_grad.append(torch.randn([30522, 1024], dtype=torch.bfloat16, device='xpu'))
        param_grad.append(torch.randn([30522], dtype=torch.bfloat16, device='xpu'))
        param_grad.append(torch.randn([4096, 1024], dtype=torch.bfloat16, device='xpu'))
        param_grad.append(torch.randn([4096], dtype=torch.bfloat16, device='xpu'))
        param_grad.append(torch.randn([512, 1024], dtype=torch.bfloat16, device='xpu'))

        # param_weight, BF16 weight
        # param_grad, BF16 grad
        # param_master_weight, FP32 master weight

        for p in param_master_weight:
            exp_avg.append(torch.zeros_like(p.data))
            exp_avg_sq.append(torch.zeros_like(p.data))

        # D2H
        for i in range(len(param_weight)):
            param_weight_cpu.append(param_weight[i].clone().cpu().float())
            param_grad_cpu.append(param_grad[i].clone().cpu().float())
            exp_avg_cpu.append(exp_avg[i].clone().cpu().float())
            exp_avg_sq_cpu.append(exp_avg_sq[i].clone().cpu().float())

        # xpu update - warmup
        for _ in range(WARM):
            for i in range(len(param_weight)):
                ipex._C.fused_adamW(torch.empty_like(param_master_weight[i]).data, torch.empty_like(param_weight[i]).data, torch.empty_like(param_grad[i]).data,
                                    torch.empty_like(exp_avg[i]).data, torch.empty_like(exp_avg_sq[i]).data, step,
                                    lr, eps, beta1, beta2, weight_decay, correct_bias)
        for i in range(len(param_weight)):
            step = step + 1
            print('shape: ', param_weight[i].shape)
            tmp_weight_decay = weight_decay
            if param_weight[i].dim() == 1:
                tmp_weight_decay = 0
            with torch.autograd.profiler.profile(use_xpu=True) as prof:
                ipex._C.fused_adamW(param_master_weight[i].data, param_weight[i].data, param_grad[i].data, exp_avg[i].data, exp_avg_sq[i].data, step,
                                    lr, eps, beta1, beta2, tmp_weight_decay, correct_bias)
            print(prof.key_averages().table(sort_by="self_cpu_time_total"))

        # cpu update
        for i in range(len(param_weight_cpu)):
            grad = param_grad_cpu[i].data
            p = param_weight_cpu[i]

            tmp_weight_decay = weight_decay
            if p.dim() == 1:
                tmp_weight_decay = 0

            step_cpu = step_cpu + 1

            exp_avg_cpu[i].mul_(beta1).add_(1.0 - beta1, grad)
            exp_avg_sq_cpu[i].mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
            denom = exp_avg_sq_cpu[i].sqrt().add_(eps)

            step_size = lr

            p.data.addcdiv_(-step_size, exp_avg_cpu[i], denom)

            if tmp_weight_decay > 0.0:
                p.data.add_(-lr * tmp_weight_decay, p.data)

        # verify
        for i in range(len(param_weight)):
            self.assertEqual(param_weight[i].cpu().float(), param_weight_cpu[i], atol=1e-2, rtol=1e-2)
            self.assertEqual(param_master_weight[i].cpu(), param_weight_cpu[i], atol=1e-2, rtol=1e-2)
