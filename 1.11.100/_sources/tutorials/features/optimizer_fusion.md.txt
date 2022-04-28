Optimizer Fusion
================

## Introduction
As the idea of TorchScript, operation fusion reduces number of operators that will be executed, and reduces overhead time. This methodology is also applied in ipex optimizer Optimization. We support Lamb/Adagrad/SGD fusion for both FP32/BF16(Split) at current stage.

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

One problem of the native implementation above is that we need to access the whole storage of "grad", "parameters" and "state sum" several times. For example, we need to access the whole storage of "parameters" and "grad" at the first clause. For large topologies, it is highly possible that the "grad" and "parameters" cannot be stored on CPU onboard cache. Thus when we need to access the storage of "grad" again when executing the third clause, processors need to read data out from memory again, rather than highly effeciently using CPU onboard high speed cache. This is a memory-bound bottle neck preventing us to get a good performance.

Fusion is the methodology to solve this problem. Since the 5 clauses in the pseudo code are all element-wise operations. We can fused them into a single one, like the pseudo code below.

```python
   adagrad_fused_step(param, grad, state_sum, ...(other args))
```

 In our fused opertors, we can seperate the storage of  "grad", "paramerters" and "state sum" in several groups and ensure each groups are small enough to be stored at cache. The pseudo code below illustrates our execution process.

```python
  grad = (grad0, grad1, ..., grad_n)
  param = (param, param, ..., param_n)
  state_sum = (state_sum, state_sum, ..., state_sum_n)
  for i in range(n):
    adagrad_step(grad_i, param_i, state_sum_i, ...(other_args))
```
