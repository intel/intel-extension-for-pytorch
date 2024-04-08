import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa


cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_adamw_fused_step_1(self, dtype=torch.float):
        param = torch.randn(2, device=xpu_device)
        exp_avg = torch.randn(2, device=xpu_device)
        exp_avg_sq = torch.randn(2, device=xpu_device)
        max_exp_avg_sq = torch.randn(2, device=xpu_device)
        grad = torch.randn(2, device=xpu_device)
        param2 = torch.randn(0, device=xpu_device)
        beta1 = 0.9
        beta2 = 0.999
        adam_epsilon = 1e-6
        amsgrad = True
        weight_decay = 0.01
        lr = 0.05
        step = 5
        res_xpu = torch.ops.torch_ipex.adamw_fused_step(
                    param_=param,
                    exp_avg_=exp_avg,
                    exp_avg_sq_=exp_avg_sq,
                    max_exp_avg_sq_=max_exp_avg_sq,
                    grad_=grad,
                    param2_=param2,
                    amsgrad=amsgrad,
                    step=step,
                    beta1=beta1,
                    beta2=beta2,
                    learning_rate=lr,
                    weight_decay=weight_decay,
                    eps=adam_epsilon,
                )

    def test_adamw_fused_step_2(self, dtype=torch.float):
        param = torch.randn(1024, device=xpu_device)
        exp_avg = torch.randn(1024, device=xpu_device)
        exp_avg_sq = torch.randn(1024, device=xpu_device)
        max_exp_avg_sq = torch.randn(0, device=xpu_device)
        grad = torch.randn(1024, device=xpu_device)
        param2 = torch.randn(0, device=xpu_device)
        beta1 = 0.9
        beta2 = 0.999
        adam_epsilon = 1e-6
        amsgrad = False
        weight_decay = 0.01
        lr = 0.05
        step = 5
        res_xpu = torch.ops.torch_ipex.adamw_fused_step(
                    param_=param,
                    exp_avg_=exp_avg,
                    exp_avg_sq_=exp_avg_sq,
                    max_exp_avg_sq_=max_exp_avg_sq,
                    grad_=grad,
                    param2_=param2,
                    amsgrad=amsgrad,
                    step=step,
                    beta1=beta1,
                    beta2=beta2,
                    learning_rate=lr,
                    weight_decay=weight_decay,
                    eps=adam_epsilon,
                )

    def test_adamw_fused_step_3(self, dtype=torch.float):
        param = torch.randn(1024, device=xpu_device)
        exp_avg = torch.randn(1024, device=xpu_device)
        exp_avg_sq = torch.randn(1024, device=xpu_device)
        max_exp_avg_sq = torch.randn(1024, device=xpu_device)
        grad = torch.randn(1024, device=xpu_device)
        param2 = torch.randn(0, device=xpu_device)
        beta1 = 0.9
        beta2 = 0.999
        adam_epsilon = 1e-6
        amsgrad = True
        weight_decay = 0.01
        lr = 0.05
        step = 5
        res_xpu = torch.ops.torch_ipex.adamw_fused_step(
                    param_=param,
                    exp_avg_=exp_avg,
                    exp_avg_sq_=exp_avg_sq,
                    max_exp_avg_sq_=max_exp_avg_sq,
                    grad_=grad,
                    param2_=param2,
                    amsgrad=amsgrad,
                    step=step,
                    beta1=beta1,
                    beta2=beta2,
                    learning_rate=lr,
                    weight_decay=weight_decay,
                    eps=adam_epsilon,
                )

    def test_adamw_fused_step_4(self, dtype=torch.float):
        param = torch.randn(4096, device=xpu_device)
        exp_avg = torch.randn(4096, device=xpu_device)
        exp_avg_sq = torch.randn(4096, device=xpu_device)
        max_exp_avg_sq = torch.randn(0, device=xpu_device)
        grad = torch.randn(4096, device=xpu_device)
        param2 = torch.randn(4096, device=xpu_device)
        beta1 = 0.9
        beta2 = 0.999
        adam_epsilon = 1e-6
        amsgrad = False
        weight_decay = 0.01
        lr = 0.05
        step = 5
        res_xpu = torch.ops.torch_ipex.adamw_fused_step(
                    param_=param,
                    exp_avg_=exp_avg,
                    exp_avg_sq_=exp_avg_sq,
                    max_exp_avg_sq_=max_exp_avg_sq,
                    grad_=grad,
                    param2_=param2,
                    amsgrad=amsgrad,
                    step=step,
                    beta1=beta1,
                    beta2=beta2,
                    learning_rate=lr,
                    weight_decay=weight_decay,
                    eps=adam_epsilon,
                )
    
    def test_adamw_fused_step_5(self, dtype=torch.float):
        param = torch.randn(30522, device=xpu_device)
        exp_avg = torch.randn(30522, device=xpu_device)
        exp_avg_sq = torch.randn(30522, device=xpu_device)
        max_exp_avg_sq = torch.randn(30522, device=xpu_device)
        grad = torch.randn(30522, device=xpu_device)
        param2 = torch.randn(0, device=xpu_device)
        beta1 = 0.9
        beta2 = 0.999
        adam_epsilon = 1e-6
        amsgrad = True
        weight_decay = 0.01
        lr = 0.05
        step = 5
        res_xpu = torch.ops.torch_ipex.adamw_fused_step(
                    param_=param,
                    exp_avg_=exp_avg,
                    exp_avg_sq_=exp_avg_sq,
                    max_exp_avg_sq_=max_exp_avg_sq,
                    grad_=grad,
                    param2_=param2,
                    amsgrad=amsgrad,
                    step=step,
                    beta1=beta1,
                    beta2=beta2,
                    learning_rate=lr,
                    weight_decay=weight_decay,
                    eps=adam_epsilon,
                )

    def test_adamw_fused_step_6(self, dtype=torch.float):
        param = torch.randn((2, 1024), device=xpu_device)
        exp_avg = torch.randn((2, 1024), device=xpu_device)
        exp_avg_sq = torch.randn((2, 1024), device=xpu_device)
        max_exp_avg_sq = torch.randn(0, device=xpu_device)
        grad = torch.randn((2, 1024), device=xpu_device)
        param2 = torch.randn((2, 1024), device=xpu_device)
        beta1 = 0.9
        beta2 = 0.999
        adam_epsilon = 1e-6
        amsgrad = False
        weight_decay = 0.01
        lr = 0.05
        step = 5
        res_xpu = torch.ops.torch_ipex.adamw_fused_step(
                    param_=param,
                    exp_avg_=exp_avg,
                    exp_avg_sq_=exp_avg_sq,
                    max_exp_avg_sq_=max_exp_avg_sq,
                    grad_=grad,
                    param2_=param2,
                    amsgrad=amsgrad,
                    step=step,
                    beta1=beta1,
                    beta2=beta2,
                    learning_rate=lr,
                    weight_decay=weight_decay,
                    eps=adam_epsilon,
                )

    def test_adamw_fused_step_7(self, dtype=torch.float):
        param = torch.randn((512, 1024), device=xpu_device)
        exp_avg = torch.randn((512, 1024), device=xpu_device)
        exp_avg_sq = torch.randn((512, 1024), device=xpu_device)
        max_exp_avg_sq = torch.randn((512, 1024), device=xpu_device)
        grad = torch.randn((512, 1024), device=xpu_device)
        param2 = torch.randn(0, device=xpu_device)
        beta1 = 0.9
        beta2 = 0.999
        adam_epsilon = 1e-6
        amsgrad = True
        weight_decay = 0.01
        lr = 0.05
        step = 5
        res_xpu = torch.ops.torch_ipex.adamw_fused_step(
                    param_=param,
                    exp_avg_=exp_avg,
                    exp_avg_sq_=exp_avg_sq,
                    max_exp_avg_sq_=max_exp_avg_sq,
                    grad_=grad,
                    param2_=param2,
                    amsgrad=amsgrad,
                    step=step,
                    beta1=beta1,
                    beta2=beta2,
                    learning_rate=lr,
                    weight_decay=weight_decay,
                    eps=adam_epsilon,
                )
    
    def test_adamw_fused_step_8(self, dtype=torch.float):
        param = torch.randn((1024, 1024), device=xpu_device)
        exp_avg = torch.randn((1024, 1024), device=xpu_device)
        exp_avg_sq = torch.randn((1024, 1024), device=xpu_device)
        max_exp_avg_sq = torch.randn(0, device=xpu_device)
        grad = torch.randn((1024, 1024), device=xpu_device)
        param2 = torch.randn((1024, 1024), device=xpu_device)
        beta1 = 0.9
        beta2 = 0.999
        adam_epsilon = 1e-6
        amsgrad = False
        weight_decay = 0.01
        lr = 0.05
        step = 5
        res_xpu = torch.ops.torch_ipex.adamw_fused_step(
                    param_=param,
                    exp_avg_=exp_avg,
                    exp_avg_sq_=exp_avg_sq,
                    max_exp_avg_sq_=max_exp_avg_sq,
                    grad_=grad,
                    param2_=param2,
                    amsgrad=amsgrad,
                    step=step,
                    beta1=beta1,
                    beta2=beta2,
                    learning_rate=lr,
                    weight_decay=weight_decay,
                    eps=adam_epsilon,
                )

    def test_adamw_fused_step_9(self, dtype=torch.float):
        param = torch.randn((1024, 4096), device=xpu_device)
        exp_avg = torch.randn((1024, 4096), device=xpu_device)
        exp_avg_sq = torch.randn((1024, 4096), device=xpu_device)
        max_exp_avg_sq = torch.randn((1024, 4096), device=xpu_device)
        grad = torch.randn((1024, 4096), device=xpu_device)
        param2 = torch.randn(0, device=xpu_device)
        beta1 = 0.9
        beta2 = 0.999
        adam_epsilon = 1e-6
        amsgrad = True
        weight_decay = 0.01
        lr = 0.05
        step = 5
        res_xpu = torch.ops.torch_ipex.adamw_fused_step(
                    param_=param,
                    exp_avg_=exp_avg,
                    exp_avg_sq_=exp_avg_sq,
                    max_exp_avg_sq_=max_exp_avg_sq,
                    grad_=grad,
                    param2_=param2,
                    amsgrad=amsgrad,
                    step=step,
                    beta1=beta1,
                    beta2=beta2,
                    learning_rate=lr,
                    weight_decay=weight_decay,
                    eps=adam_epsilon,
                )

    def test_adamw_fused_step_10(self, dtype=torch.float):
        param = torch.randn((4096, 1024), device=xpu_device)
        exp_avg = torch.randn((4096, 1024), device=xpu_device)
        exp_avg_sq = torch.randn((4096, 1024), device=xpu_device)
        max_exp_avg_sq = torch.randn(0, device=xpu_device)
        grad = torch.randn((4096, 1024), device=xpu_device)
        param2 = torch.randn((4096, 1024), device=xpu_device)
        beta1 = 0.9
        beta2 = 0.999
        adam_epsilon = 1e-6
        amsgrad = False
        weight_decay = 0.01
        lr = 0.05
        step = 5
        res_xpu = torch.ops.torch_ipex.adamw_fused_step(
                    param_=param,
                    exp_avg_=exp_avg,
                    exp_avg_sq_=exp_avg_sq,
                    max_exp_avg_sq_=max_exp_avg_sq,
                    grad_=grad,
                    param2_=param2,
                    amsgrad=amsgrad,
                    step=step,
                    beta1=beta1,
                    beta2=beta2,
                    learning_rate=lr,
                    weight_decay=weight_decay,
                    eps=adam_epsilon,
                )
    
    def test_adamw_fused_step_11(self, dtype=torch.float):
        param = torch.randn((30522, 1024), device=xpu_device)
        exp_avg = torch.randn((30522, 1024), device=xpu_device)
        exp_avg_sq = torch.randn((30522, 1024), device=xpu_device)
        max_exp_avg_sq = torch.randn((30522, 1024), device=xpu_device)
        grad = torch.randn((30522, 1024), device=xpu_device)
        param2 = torch.randn(0, device=xpu_device)
        beta1 = 0.9
        beta2 = 0.999
        adam_epsilon = 1e-6
        amsgrad = True
        weight_decay = 0.01
        lr = 0.05
        step = 5
        res_xpu = torch.ops.torch_ipex.adamw_fused_step(
                    param_=param,
                    exp_avg_=exp_avg,
                    exp_avg_sq_=exp_avg_sq,
                    max_exp_avg_sq_=max_exp_avg_sq,
                    grad_=grad,
                    param2_=param2,
                    amsgrad=amsgrad,
                    step=step,
                    beta1=beta1,
                    beta2=beta2,
                    learning_rate=lr,
                    weight_decay=weight_decay,
                    eps=adam_epsilon,
                )
