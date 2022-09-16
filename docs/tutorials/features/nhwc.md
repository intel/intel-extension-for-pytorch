# Channels Last

## What is Channels Last

**Note**: In PyTorch, **memory format** refers to data representation that describes how multidimensional arrays (nD) are stored in linear (1D) memory address space.

On CNN models, the canonical order of tensor dimensions is assigned with semantic meaning. For example the input tensor of 2D convolution is of NCHW by default on PyTorch - <batch_size, channels, height, width>. NHWC is an alternative way of describing the tensor dimensions - <batch_size, height, width, channels>.

Look at the following image of illustrating NCHW and NHWC when N=1. Actually when N=1, NHWC has the same format with BMP file image.
![fig-1-memory-layout](../../../images/channels_last/figure1_memory_layout.png)

PyTorch refers to NCHW as `torch.contiguous_format` (the default memory format) and to NHWC as `torch.channels_last`, which is a new feature as of the 1.5 release.

TensorFlow uses NHWC as the default memory format because NHWC has a performance advantage over NCHW. On XPU platforms, we propose to optimize Channels Last memory path for the following reasons:
* **Performance** - NHWC performance is not as good as blocked memory format (nChw16c), but it is close, and much better performance than NCHW.
* **User Experience** - Operator coverage of NHWC would be higher than blocked memory format, so user experience is better. To be specific, it is difficult to enable operators that manipulates `dim` on blocked format such as `sum(dim=?)`. You would need to convert tensor from blocked memory format back to NHWC, before feeding it into `sum()`. This is naturally supported on Channels Last memory format already.

## Memory Format Is All That Matters

On CNN models, memory format is almost the foundation of any upper level design. One important fact is that converting memory format could be very expensive. Thus, in case that multiple CNN operators are performed in sequence, e.g. `Conv2d -> ReLU -> Conv2d`, it's beneficial to transform them from different memory formats once, do computation and reorder them back.

On XPU, you can use 2 types of memory formats on CNN models:

### a. NCHW (default)

```python
import torch
import intel_extension_for_pytorch

input = torch.randn(1, 10, 32, 32).to("xpu")
model = torch.nn.Conv2d(10, 20, 1, 1).to("xpu")
output = model(input)
```

### b. NHWC

```python
import torch
import intel_extension_for_pytorch

input = torch.randn(1, 10, 32, 32).to("xpu")
model = torch.nn.Conv2d(10, 20, 1, 1).to("xpu")
## NB: convert to Channels Last memory format.
##   oneDNN supports NHWC for feature maps (input, output),
##   but weight still needs to be of blocked format.
##   Still we can save reorders for feature maps.
input = input.to(memory_format=torch.channels_last)
model = model.to(memory_format=torch.channels_last)
output = model(input)
```

## PyTorch Strided Layout

Before moving on, I feel it is necessary to explain how PyTorch organizes tensors in memory - the **layout**. Here we only focus on **dense** tensors, skip 'coo' layout of **sparse** tensor.

The question itself can be reinterpreted as for a tensor of size <N, C, H, W>, how does PyTorch access the element with index <n, c, h, w> from memory, the answer is **stride**:
```python
tensor: <N, C, H, W>
index: <n, c, h, w>
strides: <CHW, HW, W, 1>
offset(n,c,h,w) = stride_n * n + stride_c * c + stride_h * h + stride_w * w
                = CHW * n + HW * c + W * h + 1 * w
```

One merit of introducing **stride** is that it can express noncontiguous tensors, e.g. a slice of big tensor. For example, the 'Xs' in the following image have a stride of <n1+n2, 1>.

![fig-3-pytorch-strided-layout](../../../images/channels_last/figure3_strided_layout.png)

Keep in mind that PyTorch Tensor does not have an attribute called 'memory_format' or something else. The memory format expression completely relies on **size** and **stride**. The design principle can be found at reference: [RFC: Memory format (aka layout aka NHWC) support](https://github.com/pytorch/pytorch/issues/19092). No matter what the tensor's memory format is, we need a logical canonical order for the dimensions - that is **NCHW** on PyTorch. Thus, **size** and **stride** are ALWAYS described in the order of **NCHW**. Let's now look at the Channels Last case of the previous question:

```python
tensor: <N, C, H, W>
index: <n, c, h, w>
strides: <HWC, 1, WC, C>
offset(n,c,h,w) = stride_n * n + stride_c * c + stride_h * h + stride_w * w
                = HWC * n + 1 * c + WC * h + C * w
```

Actually, this pattern applies to ALL other memory formats as long as it is 4-dim, e.g. strides for CHWN would be <1, HWN, WN, N>.

## Channels Last Memory Format python APIs on XPU

### a. tensor creation

```python
x = torch.empty(N, C, H, W, memory_format=torch.channels_last).to("xpu")
```

### b. tensor conversion

```python
## .contiguous() transforms NHWC noncontiguous to NHWC contiguous.
## .to() converts NCHW tensor to NHWC one, it is outplace.
x = x.to("xpu")
x = x.contiguous(memory_format=torch.channels_last)
x = x.to(memory_format=torch.channels_last)

## contiguous check
x.is_contiguous(memory_format=torch.channels_last)
```

### c. model conversion

```python
## NB: tensor.to() is an outplace operation
##   model.to() is inplace. It calls _apply() which is inplace.
model = model.to("xpu").to(memory_format=torch.channels_last)
input = input.to("xpu").to(memory_format=torch.channels_last)
```

Some spontaneous questions:
* **How to tell whether this model or operator support Channels Last?** - This requires manual memory format check, aka. 'torch.channels_last' input and weight shall NOT generate 'torch.contiguous_format' output.
* **What if the model comprises of operator not supported Channels Last?** - No errors messages will be shown, the NHWC tensor will be handled by the operator as a non-contiguous NCHW tensor, so result might not be correct depending on the algorithm of this operator.

## Channels Last 1D support on XPU

Both stock PyTorch and Intel® Extension for PyTorch\* support Channels Last(2D) and Channels Last 3D, however, regarding Channels Last 1D, they are different. Stock PyTorch doesn't support Channels Last 1D, while XPU could supply limited support for Channels Last 1D.
Now, only support Channels Last 1D memory format in these operators: Conv1D, BatchNorm1D, MaxPool1D, Concat, binary add, binary div, upsample linear and upsample nearest.

The usage of Channels Last 1D on XPU is different from stock PyTorch Channels Last(2D) or Channels Last 3D. We use torch.xpu.to_channels_last_1d() to do conversation for both input tensor and model. See below:

```python
import torch
import intel_extension_for_pytorch

sycl_device = torch.device("xpu")


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm1d(3)
        )

    def forward(self, x):
        x = self.block(x)
        return x


model = Model()
test_input = torch.rand([2, 3, 4])
test_input_xpu = test_input.to(sycl_device)
test_input_xpu = torch.xpu.to_channels_last_1d(test_input_xpu) # Channels Last 1D conversation for tenor
model = model.to(sycl_device)
model = torch.xpu.to_channels_last_1d(model) # Channels Last 1D conversation for mode
xpu_res = model(test_input_xpu)

print(torch.xpu.is_contiguous_channels_last_1d(xpu_res))
```

### a. tensor conversion with Channels Last 1D

```python
input_xpu = torch.xpu.to_channels_last_1d(input_xpu)
```

### b. model conversion with Channels Last 1D

```python
model = torch.xpu.to_channels_last_1d(model)
```

### c. determine if in Channels Last 1D memory format

```python
print(torch.xpu.is_contiguous_channels_last_1d(input))
```

Note that because Meta doens't plan support Channels Last 1D feature now: [RFC: A suggestion of channels last memory format implementation for 3D tensor](https://github.com/pytorch/pytorch/issues/74935), expect Channels Last 1D APIs above, other APIs from stock PyTorch may be invalid. E.g.: If you want to use memory format corrsponding API for Channels Last 1D, it cannot work as you wish.
