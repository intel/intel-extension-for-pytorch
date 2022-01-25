## Microbench

#### Installation

```bash
pip install -r requirements.txt
rm -rf ./build
python setup.py install
```

#### Guide for Verbose Generation

```python
import torch
import microbench

microbench.enable_verbose()

# The following cmd will generate verbose
a = torch.randn(2)
a.requires_grad = True
b = torch.rand(2)
c = (a / b).sum()
c.backward()

microbench.disable_verbose()
```

#### Guide for Verbose Generation with Model

```python
import torch
try:
    import microbench
    enable_microbench = True
except ImportError:
    enable_microbench = False

with torch.autograd.profiler.profile(True, use_xpu=True, record_shapes=False) as prof:
    global enable_microbench
    if enable_microbench:
        microbench.enable_verbose()
    
    # generate 1-run verbose
    output = model(input)

    if enable_microbench:
        microbench.disable_verbose()
        enable_microbench = False
```

#### Verbose Example (saved as test.vb)

```vb
[normal_] self(Tensor &):float[2]Contiguous; mean(double):0; std(double):1; generator(c10::optional<Generator>):unknown; {mb_normal_}
[uniform_] self(Tensor &):float[2]Contiguous; from(double):0; to(double):1; generator(c10::optional<Generator>):unknown; {mb_uniform_}
[div] self(const Tensor &):float[2]Contiguous; other(const Tensor &):float[2]Contiguous; {mb_div_Tensor}
[sum] self(const Tensor &):float[2]Contiguous; dtype(c10::optional<ScalarType>):undef; {mb_sum}
[as_strided] self(const Tensor &):float[]Contiguous; size(IntArrayRef):[1]; stride(IntArrayRef):[0]; storage_offset(c10::optional<int64_t>):undef; {mb_as_strided}
[fill_] self(Tensor &):float[1]Contiguous; value(Scalar):0; {mb_fill__Scalar}
[ones_like] self(const Tensor &):float[]Contiguous; dtype(c10::optional<ScalarType>):Float; layout(c10::optional<Layout>):unknown; device(c10::optional<Device>):unknown; pin_memory(c10::optional<bool>):0; memory_format(c10::optional<MemoryFormat>):unknown; {mb_ones_like}
[empty_like] self(const Tensor &):float[]Contiguous; dtype(c10::optional<ScalarType>):Float; layout(c10::optional<Layout>):unknown; device(c10::optional<Device>):unknown; pin_memory(c10::optional<bool>):0; memory_format(c10::optional<MemoryFormat>):unknown; {mb_empty_like}
[fill_] self(Tensor &):float[]Contiguous; value(Scalar):1; {mb_fill__Scalar}
[expand] self(const Tensor &):float[]Contiguous; size(IntArrayRef):[2]; implicit(bool):0; {mb_expand}
[as_strided] self(const Tensor &):float[]Contiguous; size(IntArrayRef):[2]; stride(IntArrayRef):[0]; storage_offset(c10::optional<int64_t>):undef; {mb_as_strided}
[div] self(const Tensor &):float[2]Contiguous; other(const Tensor &):float[2]Contiguous; {mb_div_Tensor}
```

#### Run Microbench Verbose

```bash
python tools/runbench.py --log=test.vb --backend=cpu --dtype=fp32 --exclude=trivial+onednn
```

```jsonlines
{'op_class_name': 'normal_', 'caller': 'microbench.mb_normal_', 'inputs': ['float32[2]', 0, 1, None], 'time': 20.981, 'outputs': ['float32[2]'], 'params': ['self(Tensor &):float[2]Contiguous', 'mean(double):0', 'std(double):1', 'generator(c10::optional<Generator>):unknown'], 'backend': 'cpu'}
{'op_class_name': 'uniform_', 'caller': 'microbench.mb_uniform_', 'inputs': ['float32[2]', 0, 1, None], 'time': 18.559, 'outputs': ['float32[2]'], 'params': ['self(Tensor &):float[2]Contiguous', 'from(double):0', 'to(double):1', 'generator(c10::optional<Generator>):unknown'], 'backend': 'cpu'}
{'op_class_name': 'div', 'caller': 'microbench.mb_div_Tensor', 'inputs': ['float32[2]', 'float32[2]'], 'time': 19.689999999999998, 'outputs': ['float32[2]'], 'params': ['self(const Tensor &):float[2]Contiguous', 'other(const Tensor &):float[2]Contiguous'], 'backend': 'cpu'}
{'op_class_name': 'sum', 'caller': 'microbench.mb_sum', 'inputs': ['float32[2]', None], 'time': 33.459, 'outputs': ['float32[]'], 'params': ['self(const Tensor &):float[2]Contiguous', 'dtype(c10::optional<ScalarType>):undef'], 'backend': 'cpu'}
{'op_class_name': 'fill_', 'caller': 'microbench.mb_fill__Scalar', 'inputs': ['float32[1]', <microbench.Scalar object at 0x7f6180a49870>], 'time': 13.59, 'outputs': ['float32[1]'], 'params': ['self(Tensor &):float[1]Contiguous', 'value(Scalar):0'], 'backend': 'cpu'}
{'op_class_name': 'fill_', 'caller': 'microbench.mb_fill__Scalar', 'inputs': ['float32[]', <microbench.Scalar object at 0x7f6180a55530>], 'time': 14.573, 'outputs': ['float32[]'], 'params': ['self(Tensor &):float[]Contiguous', 'value(Scalar):1'], 'backend': 'cpu'}
{'op_class_name': 'expand', 'caller': 'microbench.mb_expand', 'inputs': ['float32[]', [2], False], 'time': 16.819, 'outputs': ['float32[2]'], 'params': ['self(const Tensor &):float[]Contiguous', 'size(IntArrayRef):[2]', 'implicit(bool):0'], 'backend': 'cpu'}
```

#### Run Microbench Verbose with CSV Report

```bash
python tools/report.py --log=test.vb --backend=xpu --dtype=fp32 --exclude=trivial+onednn --spec=./spec/Gen12-ATS-1T-480_1.4.csv
# -> test_fp32_report.csv

python tools/summary.py --dir=./
# -> summary.csv
```
