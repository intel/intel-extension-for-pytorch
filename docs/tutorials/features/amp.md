Auto Mixed Precision (AMP)
==========================

## Introduction

`torch.cpu.amp` provides convenience for auto data type conversion at runtime. Deep learning workloads could benefit from lower precision floating point data types like `torch.float16` or `torch.bfloat16`, because of its lighter calculation workload and less memory usage. However, because of the nature character of lower precision floating point data types, accuracy is sacrificed. Using lower precision floating point data types is a trade-off between accuracy and performance. Thus, some operations need to keep in `torch.float32`, while others can be converted to lower precision floating point data types. The Auto Mixed Precision (AMP) feature automates the tuning of data type conversions over all operators.

Currently, `torch.cpu.amp` only supports `torch.bfloat16`. It is the default lower precision floating point data type when `torch.cpu.amp` is enabled. `torch.cpu.amp` primarily benefits on Intel CPU with BFloat16 instruction set support.

## Use Case

The following simple network should show a speedup with mixed precision.

```
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x):
        return self.conv(x)
```

### Default Precision

Without `torch.cpu.amp`, the network executes all operators with default precision (`torch.float32`).
```
model = SimpleNet()
x = torch.rand(64, 64, 224, 224)
y = model(x)
```

### Inference with Imperative Path

`torch.cpu.amp.autocast` is designed to be context managers that allow scopes of your script to run in mixed precision. In these scopes, operations run in a data type chosen by the `autocast` class to improve performance while maintaining accuracy. See the operations category section for details on what precision the `autocast` class chooses for each operator, and under what circumstances.

```
model = SimpleNet().eval()
x = torch.rand(64, 64, 224, 224)
with torch.cpu.amp.autocast():
    y = model(x)
```

### Inference with TorchScript Path

`torch.cpu.amp.autocast` can be used with `torch.jit.trace` to apply graph optimization.

```
model = SimpleNet().eval()
x = torch.rand(64, 64, 224, 224)
with torch.cpu.amp.autocast():
    model = torch.jit.trace(model, x)
    model = torch.jit.freeze(model)
    y = model(x)
```

### Training Support

`torch.cpu.amp.autocast` can be used in training to improve performance.

```
model = SimpleNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
for images, label in train_loader():
    with torch.cpu.amp.autocast():
        loss = criterion(model(images), label)
    loss.backward()
    optimizer.step()
```

## Autocast Op Reference

### Op Eligibility

Ops that run in `float64` or non-floating-point dtypes are not eligible, and will run in these types whether or not autocast is enabled.

Only out-of-place ops and Tensor methods are eligible. In-place variants and calls that explicitly supply an `out=...` Tensor
are allowed in autocast-enabled regions, but won't go through autocasting. For example, in an autocast-enabled region `a.addmm(b, c)` can autocast, but `a.addmm_(b, c)` and `a.addmm(b, c, out=d)` cannot. For best performance and stability, prefer out-of-place ops in autocast-enabled regions.

### Op-Specific Behavior

The following lists describe the behavior of eligible ops in autocast-enabled regions. These ops always go through autocasting whether they are invoked as part of a :class:`torch.nn.Module`, as a function, or as a :class:`torch.Tensor` method. If functions are exposed in multiple namespaces, they go through autocasting regardless of the namespace.

Ops not listed below do not go through autocasting. They run in the type defined by their inputs. However, autocasting may still change the type in which unlisted ops run if they're downstream from autocasted ops.

If an op is unlisted, we assume it's numerically stable in `bfloat16`. If you believe an unlisted op is numerically unstable in `bfloat16`, please file an issue.

#### Ops that can autocast to `bfloat16`

`conv1d`, `conv2d`, `conv3d`, `bmm`, `mm`, `baddbmm`, `addmm`, `addbmm`, `conv_transpose1d`, `conv_transpose2d`, `conv_transpose3d`, `linear`, `matmul`

#### Ops that can autocast to `float32`

`avg_pool3d`, `binary_cross_entropy`, `polar`, `fmod`, `prod`, `quantile`, `nanquantile`, `stft`, `cdist`, `cumprod`, `cumsum`, `diag`, `diagflat`, `histc`, `logcumsumexp`, `trace`, `vander`, `view_as_complex`, `cholesky`, `cholesky_inverse`, `cholesky_solve`, `inverse`, `lu_solve`, `matrix_rank`, `orgqr`, `ormqr`, `pinverse`, `max_pool3d`, `max_unpool2d`, `max_unpool3d`, `adaptive_avg_pool3d`, `reflection_pad1d`, `reflection_pad2d`, `replication_pad1d`, `replication_pad2d`, `replication_pad3d`, `group_norm`, `mse_loss`, `ctc_loss`, `kl_div`, `multilabel_margin_loss`, `fft_fft`, `fft_ifft`, `fft_fft2`, `fft_ifft2`, `fft_fftn`, `fft_ifftn`, `fft_rfft`, `fft_irfft`, `fft_rfft2`, `fft_irfft2`, `fft_rfftn`, `fft_irfftn`, `fft_hfft`, `fft_ihfft`, `conv_tbc`, `linalg_matrix_norm`, `linalg_cond`, `linalg_matrix_rank`, `linalg_solve`, `linalg_cholesky`, `linalg_svdvals`, `linalg_eigvals`, `linalg_inv`, `linalg_householder_product`, `linalg_tensorinv`, `linalg_tensorsolve`, `fake_quantize_per_tensor_affine`, `eig`, `geqrf`, `lstsq`, `_lu_with_info`, `qr`, `solve`, `svd`, `symeig`, `triangular_solve`, `fractional_max_pool2d`, `fractional_max_pool3d`, `adaptive_max_pool3d`, `multilabel_margin_loss_forward`, `linalg_qr`, `linalg_cholesky_ex`, `linalg_svd`, `linalg_eig`, `linalg_eigh`, `linalg_lstsq`, `linalg_inv_ex`

#### Ops that promote to the widest input type

These ops don't require a particular dtype for stability, but take multiple inputs and require that the inputs' dtypes match.  If all of the inputs are `bfloat16`, the op runs in `bfloat16`.  If any of the inputs is `float32`, autocast casts all inputs to `float32` and runs the op in `float32`.

`cat`, `stack`, `index_copy`

Some ops not listed here (e.g., binary ops like `add`) natively promote inputs without autocasting's intervention.  If inputs are a mixture of `bfloat16` and `float32`, these ops run in `float32` and produce `float32` output, regardless of whether autocast is enabled.

## Design Details

### Frontend API Design

`torch.cpu.amp` is designed to be context managers that allow scopes of your script to run in mixed precision. It takes input parameter `dtype`, which is `torch.bfloat16` by default.

### Dedicated Dispatch Key

`torch.cpu.amp` extends the design of the original pytorch `Auto Mixed Precision` using the dedicated dispatch key of `AutocastCPU`. Each tensor during creation will have an `Autocast` Dispatchkey corresponding to the device (`CUDA` or `CPU`). Thus, for every tensor on CPU, `AutocastCPU` exists along with the tensor. During the dispatch phase, operators with input tensors of `AutocastCPU` are dispatched to the `Autocast` layers. The `Autocast` layer decides what precision to chooses for each operator. `AutocastCPU` has higher dispatch priority comparing to `Autograd` which makes sure the `Autocast` layer runs before `Autograd`.

### Operations category

The operations are generally divided into 3 major categories and registered under Dispatch Key `AutocastCPU`:
* `lower_precision_fp` category: Computation bound operators that could get performance boost with BFloat16 data type through acceleration by Intel CPU BFloat16 instruction set. Inputs of them are casted into `torch.bfloat16` before execution. `convolutions` and `linear` are examples of this category.
* `fallthrough` category: Operators that support running with both Float32 and BFloat16 data types, but could not get performance boost with BFloat16 data type. `relu` and `max_pool2d` are examples of this category.
* `fp32` category: Operators that are not enabled with BFloat16 support yet. Inputs of them are casted into `float32` before execution. `max_pool3d` and `group_norm` are examples of this category.
