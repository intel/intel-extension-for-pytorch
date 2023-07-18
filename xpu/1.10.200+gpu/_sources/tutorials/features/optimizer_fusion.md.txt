Optimizer Fusion
================

## Introduction
As with TorchScript, operation fusion reduces the number of operators that will be executed, and reduces overhead time. This methodology is also applied in Intel® Extension for PyTorch\* optimizer optimization. We support SGD/AdamW fusion for both FP32/BF16 at current stage.

Let's examine the code in [sgd update](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html?highlight=sgd#torch.optim.SGD) as an example.

```python

    # original version
    if weight_decay != 0:
        grad = grad.add(param, alpha=weight_decay)
    if momentum != 0:
      buf = momentum_buffer_list[i]
      if buf is None:
          buf = torch.clone(grad).detach()
          momentum_buffer_list[i] = buf
      else:
          buf.mul_(momentum).add_(grad, alpha=1 - dampening)
    if nesterov:
        grad = grad.add(buf, alpha=momentum)
    else:
        grad = buf

    param.add_(grad, alpha=-lr)
```

## Operation Fusion

One problem of the native implementation above is that we need to access the storages of `grad`, `param`, and `buf` several times. For large topologies, `grad` and `parameters` might not be stored in the cache. When we need to access the storage of `grad` again when executing the remaining clauses, the processor must read data out of low speed memory again instead of the more efficient high speed cache. This is a memory-bound bottle neck preventing good performance.

Operation fusion is a way to solve this problem. The clauses in the pseudo code are all element-wise operations, so we can fuse them into a single operation, as in the pseudo code below.

```python
   # fused version
   sgd_fused_step(param, grad, buf, ...(other args))
```

After fusion, one operation `sgd_fused_step` can provide equivalent functionality but much better performance compared with original version of [sgd update](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html?highlight=sgd#torch.optim.SGD).
