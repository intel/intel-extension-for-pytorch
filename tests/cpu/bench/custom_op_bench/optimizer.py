import torch
import intel_extension_for_pytorch as ipex
import time

a = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float)
b = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float)
def flush():
    global a, b
    a += b

def non_fused_sgd(param, grad, momentum_buf, momentum, lr, weight_decay, dampening, nesterov):
    if weight_decay != 0:
        grad = grad.add(param, alpha=weight_decay)

    if momentum != 0:
        buf = momentum_buf

        if buf is None:
            buf = torch.clone(grad).detach()
        else:
            buf.mul_(momentum).add_(grad, alpha=1 - dampening)

        if nesterov:
            grad = grad.add(buf, alpha=momentum)
        else:
            grad = buf
    param.add_(grad, alpha=-lr)

def non_fused_lamb(param, exp_avg, exp_avg_sq, grad, step, beta1, beta2, lr, weight_decay, eps):
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    # Decay the first and second moment running average coefficient
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    adam_step = (exp_avg / bias_correction1) / ((exp_avg_sq / bias_correction2).sqrt() + eps)
    if weight_decay != 0:
        adam_step.add_(param, alpha=weight_decay)
    weight_norm = param.norm(p=2)
    rtw_norm = adam_step.norm(p=2)
    true_ratio = weight_norm / rtw_norm
    param.add_(adam_step, alpha=-lr * true_ratio)

def non_fused_adagrad(param, grad, state_sum, step, lr, weight_decay, lr_decay, eps):
    if weight_decay != 0:
        grad = grad.add(param, alpha=weight_decay)
    clr = lr / (1 + (step - 1) * lr_decay)
    state_sum.addcmul_(grad, grad, value=1)
    std = state_sum.sqrt().add_(eps)
    param.addcdiv_(grad, std, value=-clr)

def run_bench(bench_name, func, *params):
    for _ in range(1000):
        flush()
        func(*params)

    start = time.time()
    flush_time = 0
    for i in range(1000):
        flush_start = time.time()
        flush()
        flush_time += time.time() - flush_start
        func(*params)
    end = time.time()
    avg_elapsed = (end - start - flush_time)
    print("Took {} ms on average to run {} update".format(avg_elapsed, bench_name))

def sgd_bench():
    print("Running benchmark for SGD update step")
    fused = torch.ops.torch_ipex.sgd_fused_step
    non_fused = non_fused_sgd

    learning_rate = 0.1
    weight_decay = 0.3
    momentum = 0.5
    dampening = 0.5
    nesterov = True

    for param_size in [1024, 512*1024, 8*1024*1024]:
        param = torch.randn(param_size)
        grad = torch.randn(param_size)
        momentum_buf = torch.randn(param_size)
        dummy_trail = torch.Tensor()
        trail = torch.randn(param_size).bfloat16()
  
        print("For parameter size", param_size)
        run_bench("fused sgd", fused, param, grad, momentum_buf, dummy_trail, momentum, learning_rate, weight_decay, dampening, nesterov)
        run_bench("fused split sgd", fused, param.bfloat16(), grad.bfloat16(), momentum_buf, trail, momentum, learning_rate, weight_decay, dampening, nesterov)
        run_bench("non fused sgd", non_fused, param, grad, momentum_buf, momentum, learning_rate, weight_decay, dampening, nesterov)

def lamb_bench():
    print("Running benchmark for Lamb update step")
    fused = torch.ops.torch_ipex.lamb_fused_step
    non_fused = non_fused_lamb

    step = 10
    beta1 = 0.8
    beta2 = 0.9
    learning_rate = 0.1
    weight_decay = 0.3
    eps = 0.001

    for param_size in [1024, 512*1024, 8*1024*1024]:
        param = torch.randn(param_size)
        grad = torch.randn(param_size)
        exp_avg = torch.randn(param_size).abs()
        exp_avg_sq = torch.randn(param_size).abs()
        dummy_trail = torch.Tensor()
        trail = torch.randn(param_size).bfloat16()

        print("For parameter size", param_size)
        run_bench("fused lamb", fused, param, exp_avg, exp_avg_sq, grad, dummy_trail, step, beta1, beta2, learning_rate, weight_decay, eps)
        run_bench("fused split lamb", fused, param.bfloat16(), exp_avg, exp_avg_sq, grad.bfloat16(), trail, step, beta1, beta2, learning_rate, weight_decay, eps)
        run_bench("non fused lamb", non_fused, param, exp_avg, exp_avg_sq, grad, step, beta1, beta2, learning_rate, weight_decay, eps)

def adagrad_bench():
    print("Running benchmark for Adagrad update step")
    fused = torch.ops.torch_ipex.adagrad_fused_step
    non_fused = non_fused_adagrad

    step = 10
    learning_rate = 0.1
    weight_decay = 0.3
    lr_decay = 0.01
    eps = 0.001

    for param_size in [1024, 512*1024, 8*1024*1024]:
        param = torch.randn(param_size)
        grad = torch.randn(param_size)
        state_sum = torch.randn(param_size)
        dummy_trail = torch.Tensor()
        trail = torch.randn(param_size).bfloat16()
  
        print("For parameter size", param_size)
        run_bench("fused adagrad", fused, param, grad, state_sum, dummy_trail, step, learning_rate, weight_decay, lr_decay, eps)
        run_bench("fused split adagrad", fused, param.bfloat16(), grad.bfloat16(), state_sum, trail, step, learning_rate, weight_decay, lr_decay, eps)
        run_bench("non fused adagrad", non_fused, param, grad, state_sum, step, learning_rate, weight_decay, lr_decay, eps)

def run():
    import argparse
    parser = argparse.ArgumentParser(
        description="benchmark for ipex optimizer"
    )
    parser.add_argument("--optimizer", type=str, choices=["sgd", "lamb", "adagrad"], default="sgd")
    args = parser.parse_args()
    if args.optimizer == "sgd":
        sgd_bench()
    elif args.optimizer == "lamb":
        lamb_bench()
    else:
        adagrad_bench()

if __name__ == "__main__":
    run()
