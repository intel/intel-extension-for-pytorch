Auto Mixed Precision (AMP)
==========================

## Introduction

`torch.cpu.amp` provides convenience for auto data type conversion at runtime. Deep learning workloads can benefit from lower-precision floating point data types such as `torch.float16` or `torch.bfloat16`, because of its lighter calculation workload and smaller memory usage. Accuracy is sacrificed when using lower-precision floating point data types so there's a trade-off between accuracy and performance. Thus, some operations should use the slower but more accurate`torch.float32`, while others can be converted to use the faster but less accurate `torch.float16` data type. The Auto Mixed Precision (AMP) feature automates the tuning of data type conversions over all operators.

`torch.cpu.amp` only supports `torch.bfloat16`. It is the default lower precision floating point data type when `torch.cpu.amp` is enabled. `torch.cpu.amp` primarily benefits when running on Intel CPU with BFloat16 instruction set support.

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

`torch.cpu.amp.autocast` is designed to be a context manager that allow scopes of your script to run with mixed precision. In these scopes, operations run in a data type chosen by the `autocast` class to improve performance while maintaining accuracy. See the operations category section for details on what precision the `autocast` class chooses for each operator, and under what circumstances.

```
model = SimpleNet().eval()
x = torch.rand(64, 64, 224, 224)
with torch.cpu.amp.autocast():
    y = model(x)
```

### Inference with TorchScript Path

`torch.cpu.amp.autocast` can be used with `torch.jit.trace` to apply graph optimization. Due to PyTorch limitation, only `torch.jit.trace` is supported.

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

Ops that run in `float64` or non-floating-point dtypes are not eligible for mixed precision, and will run in these types whether or not autocast is enabled.

Only out-of-place ops and Tensor methods are eligible for mixed precision. In-place variants and calls that explicitly supply an `out=...` Tensor
are allowed in autocast-enabled regions, but won't go through autocasting. For example, in an autocast-enabled region `a.addmm(b, c)` can autocast, but `a.addmm_(b, c)` and `a.addmm(b, c, out=d)` cannot. For best performance and stability, use out-of-place ops in autocast-enabled regions.

### Op-Specific Behavior

The following lists describe the behavior of eligible ops in autocast-enabled regions. These ops always go through autocasting whether they are invoked as part of a `torch.nn.Module`, as a function, or as a `torch.Tensor` method. If functions are exposed in multiple namespaces, they go through autocasting regardless of the namespace.

Ops not listed below do not go through autocasting. They run in the type defined by their inputs. However, autocasting may still change the type in which unlisted ops run if they're downstream from autocasted ops.

If an op is unlisted, we assume it's numerically stable in `bfloat16`. If you believe that an unlisted op is numerically unstable in `bfloat16`, file a [GitHub issue](https://github.com/intel/intel-extension-for-pytorch/issues).

#### Ops that can autocast to `bfloat16`

`conv1d`, `conv2d`, `conv3d`, `bmm`, `mm`, `baddbmm`, `addmm`, `addbmm`, `linear`, `matmul`, `conv_tbc`, `group_norm`, `_native_multi_head_attention`, `_transform_bias_rescale_qkv`

#### Ops that can autocast to `float32`

`conv_transpose1d`, `conv_transpose2d`, `conv_transpose3d`, `mish`, `avg_pool3d`, `binary_cross_entropy`, `grid_sampler`, `polar`, `prod`, `quantile`, `nanquantile`, `stft`, `cdist`, `trace`, `view_as_complex`, `cholesky`, `cholesky_inverse`, `cholesky_solve`, `inverse`, `lu_solve`, `orgqr`, `ormqr`, `pinverse`, `max_unpool2d`, `max_unpool3d`, `adaptive_avg_pool3d`, `reflection_pad1d`, `reflection_pad2d`, `replication_pad1d`, `replication_pad2d`, `replication_pad3d`, `mse_loss`, `cosine_embedding_loss`, `nll_loss`, `nll_loss2d`, `hinge_embedding_loss`, `poisson_nll_loss`, `smooth_l1_loss`, `cross_entropy_loss`, `l1_loss`, `huber_loss`, `margin_ranking_loss`, `soft_margin_loss`, `triplet_margin_loss`, `multi_margin_loss`, `ctc_loss`, `kl_div`, `multilabel_margin_loss`, `binary_cross_entropy_with_logits`, `fft_fft`, `fft_ifft`, `fft_fft2`, `fft_ifft2`, `fft_fftn`, `fft_ifftn`, `fft_rfft`, `fft_irfft`, `fft_rfft2`, `fft_irfft2`, `fft_rfftn`, `fft_irfftn`, `fft_hfft`, `fft_ihfft`, `linalg_matrix_norm`, `linalg_cond`, `linalg_matrix_rank`, `linalg_solve`, `linalg_cholesky`, `linalg_svdvals`, `linalg_eigvals`, `linalg_eigvalsh`, `linalg_inv`, `linalg_householder_product`, `linalg_tensorinv`, `linalg_tensorsolve`, `fake_quantize_per_tensor_affine`, `geqrf`, `_lu_with_info`, `qr`, `svd`, `symeig`, `triangular_solve`, `fractional_max_pool2d`, `fractional_max_pool3d`, `adaptive_max_pool3d`, `multilabel_margin_loss_forward`, `linalg_qr`, `linalg_cholesky_ex`, `linalg_svd`, `linalg_eig`, `linalg_eigh`, `linalg_lstsq`, `linalg_inv_ex`

#### Ops that promote to the widest input type

These ops don't require a particular dtype for stability, but take multiple inputs and require that the inputs' dtypes match.  If all of the inputs are `bfloat16`, the op runs in `bfloat16`.  If any of the inputs is `float32`, autocast casts all inputs to `float32` and runs the op in `float32`.

`cat`, `stack`, `index_copy`

Some ops not listed here (e.g., binary ops like `add`) natively promote inputs without autocasting's intervention.  If inputs are a mixture of `bfloat16` and `float32`, these ops run in `float32` and produce `float32` output, regardless of whether autocast is enabled.
