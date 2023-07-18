Optimizer Fusion
================

## Introduction
As with TorchScript, operation fusion reduces the number of operators that will be executed, and reduces overhead time. This methodology is also applied in ipex optimizer Optimization. We support Lamb/Adagrad/SGD fusion for both FP32/BF16(Split) at current stage.

Let's use [adagrad update](https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html?highlight=adagrad#torch.optim.Adagrad) as an example.

```python
    if weight_decay != 0:
        grad = grad.add(param, alpha=weight_decay)
    clr = lr / (1 + (step - 1) * lr_decay)
    state_sum.addcmul_(grad, grad, value=1)
    std = state_sum.sqrt().add_(eps)
    param.addcdiv_(grad, std, value=-clr)
```

## Operation Fusion

One problem of the native implementation above is that we need to access the whole storage of "grad", "parameters", and "state sum" several times. For example, we need to access the whole storage of "parameters" and "grad" at the first clause. For large topologies, it is possible that the "grad" and "parameters" cannot be stored on the onboard CPU cache. When we need to access the storage of "grad" again when executing the third clause, the processor must read data out from memory again instead of the more efficient onboard high speed CPU cache. This is a memory-bound bottle neck preventing good performance.

Fusion is the methodology to solve this problem. Since the 5 clauses in the pseudo code are all element-wise operations. We can fuse them into a single one, like the pseudo code below.

```python
   adagrad_fused_step(param, grad, state_sum, ...(other args))
```

 In our fused operators, we can separate the storage of  "grad", "parameters" and "state sum" in several groups and ensure each group is small enough to be stored in the cache. The pseudo code below illustrates our execution process.

```python
  grad = (grad0, grad1, ..., grad_n)
  param = (param, param, ..., param_n)
  state_sum = (state_sum, state_sum, ..., state_sum_n)
  for i in range(n):
    adagrad_step(grad_i, param_i, state_sum_i, ...(other_args))
```
