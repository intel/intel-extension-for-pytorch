`ipex.optimize` Frontend API
======================================

The `ipex.optimize` API is designed to optimize PyTorch\* modules (`nn.modules`) and specific optimizers within Python modules. Its optimization options for Intel® GPU device include:

- Automatic Channels Last
- Fusing Convolutional Layers with Batch Normalization
- Fusing Linear Layers with Batch Normalization
- Replacing Dropout with Identity
- Splitting Master Weights
- Fusing Optimizer Update Step

The original python modules will be replaced to optimized versions automatically during model execution, if `ipex.optimize` is called in the model running script.

The following sections provide detailed descriptions for each optimization flag supported by **XPU** models on Intel® GPU. For CPU-specific flags, please refer to the [API Docs page](../api_doc.html#ipex.optimize).

### Automatic Channels Last

By default, `ipex.optimize` checks if current running GPU platform supports 2D Block Array Load or not. If it does, the `Conv*d` and `ConvTranspose*d` modules inside the model will be optimized for using channels last memory format. Use `ipex.enable_auto_channels_last` or `ipex.disable_auto_channels_last` before calling `ipex.optimize` to enable or disable this feature manually.

### `conv_bn_folding`

This flag is applicable for model inference. Intel® Extension for PyTorch\* tries to match all connected `nn.Conv(1/2/3)d` and `nn.BatchNorm(1/2/3)d` layers with matching dimensions in the model and fuses them to improve performance. If the fusion fails, the optimization process will be ended and the model will be executed automatically in normal path.

### `linear_bn_folding`

This flag is applicable for model inference. Intel® Extension for PyTorch\* tries to match all connected `nn.Linear` and `nn.BatchNorm(1/2/3)d` layers in the model and fuse them to improve performance. If the fusion fails, the optimization process will be ended and the model will be executed automatically in normal path.

### `replace_dropout_with_identity`

This flag is applicable for model inference. All instances of `torch.nn.Dropout` will be replaced with `torch.nn.Identity`. The `Identity` modules will be ignored during the static graph generation. This optimization could potentially create additional fusion opportunities for the generated graph.

### `split_master_weight_for_bf16`

This flag is applicable for model training. The optimization will be enabled once the following requirements are met:
- When calling `ipex.optimize`, the `dtype` flag must be set to `torch.bfloat16`.
- `fuse_update_step` must be enabled.

The optimization process is as follows:

- Wrap all parameters of this model with `ParameterWrapper`.
- Convert the parameters that meet the condition specified by `ipex.nn.utils._parameter_wrapper.can_cast_training`. This includes the original dtype `torch.float`, and module types defined in `ipex.nn.utils._parameter_wrapper.IPEX_WEIGHT_CONVERT_MODULE_XPU`.
- Convert the parameters wrapped by `ParameterWrapper` to the user-specified dtype. If **split master weight** is needed, the optimizer can only be SGD. The original parameters will be divided into top and bottom parts. The top part will be used for forward and backward computation. When updating weights, both the top and bottom parts will be updated simultaneously.

### fuse_update_step

This flag is used to specify whether to replace the original optimizer step with a fused step for better performance. The supported optimizers can be referenced from `IPEX_FUSED_OPTIMIZER_LIST_XPU` in `ipex.optim._optimizer_utils`. During the optimization, the original step is saved as `optimizer._original_step`, `optimizer.step` is replaced with a SYCL-written kernel, and the `optimizer.fused` parameter is set to `True`.
