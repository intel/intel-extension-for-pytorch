Auto Channels Last
==================

Channels last memory format is known to have performance advantage over channels first memory format. Refer to [Channels Last](./nhwc.md) for details.
Intel® Extension for PyTorch\* automatically converts the model to channels last memory format by default when users optimize their model with `ipex.optimize(model)`.

## Ease-of-use auto channels last API
#### default
```python
model = ipex.optimize(model) # by default, model is channels last
```

#### enable
```python
ipex.enable_auto_channels_last()
model = ipex.optimize(model) # enable, model is channels last
```

#### disable
```python
ipex.disable_auto_channels_last()
model = ipex.optimize(model) # disable, model is channels first 
```

## Known issue 
For broad models, channels last memory format brings performance boost over channels first memory format. However, for few use cases, this may bring performance regression. If performance regression is observed, we recommend to feed sample input data to `ipex.optimize(model, sample_input=...)`.
```python
model = ipex.optimize(model, sample_input=...)
```
