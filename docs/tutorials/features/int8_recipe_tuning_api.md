INT8 Recipe Tuning API (Prototype)
=====================================

This [new API](../api_doc.html#ipex.quantization.autotune) `ipex.quantization.autotune` supports INT8 recipe tuning by using Intel® Neural Compressor as the backend in Intel® Extension for PyTorch\*. In general, we provid default recipe in Intel® Extension for PyTorch\*, and we still recommend users to try out the default recipe first without bothering tuning. If the default recipe doesn't bring about desired accuracy, users can use this API to tune for a more advanced receipe.

Users need to provide a fp32 model and some parameters required for tuning. The API will return a prepared model with tuned qconfig loaded.

## Usage Example
- Static Quantization
Please refer to [static_quant example](../../../examples/cpu/features/int8_recipe_tuning/imagenet_autotune.py).

- Smooth Quantization
Please refer to [llm sq example](../../../examples/cpu/inference/python/llm/single_instance/run_generation.py).

## Smooth Quantization Autotune
### Algorithm: Auto-tuning of $\alpha$.
SmoothQuant method aims to split the quantization difficulty of weight and activation by using a fixed-value $\alpha$ for an entire model. However, as the distributions of activation outliers vary not only across different models but also across different layers within a model, we hereby propose a method to obtain layer-wise optimal $\alpha$ values with the ability to tune automatically.
Currently, both layer-wise and block-wise auto-tuning methods are supported and the default option is layer-wise.
In block-wise auto-tuning, layers within one block (e.g an OPTDecoderLayer) would share the same alpha value; users could set *'do_blockwise': True* in *auto_alpha_args* to enable it.

Our proposed method consists of 8 major steps:

-    Hook input minimum and maximum values of layers to be smoothed using register_forward_hook.
-    Find a list of layers on which smoothquant could be performed.
-    Generate a list of $\alpha$ values of a user-defined range and set a default $\alpha$ value.
-    Calculate smoothing factor using default $\alpha$ value, adjust parameters accordingly and forward the adjusted model given an input sample.
-    Perform per-channel quantization_dequantization of weights and per-tensor quantization_dequantization of activations to predict output.
-    Calculate the layer-wise/block-wise loss with respect to FP32 output, iterate the previous two steps given each $\alpha$ value and save the layer-wise/block-wise loss per alpha.
-    Apply criterion on input LayerNorm op and obtain the optimal alpha values of a single input sample.
-    Iterate the previous three steps over a number of input samples and save the layer-wise/block-wise optimal $\alpha$ values.

Multiple criteria (e.g min, max and mean) are supported to determine the $\alpha$ value of an input LayerNorm op of a transformer block. Both alpha range and criterion could be configured in auto_alpha_args.

In our experiments, an $\alpha$ range of [0.0, 1.0] with a step_size of 0.1 is found to be well-balanced one for the majority of models.

### $\alpha$ Usage
There are two ways to apply smooth quantization: 1) using a fixed `alpha` for the entire model or 2) determining the `alpha` through auto-tuning.

#### Using a fixed `alpha`
To set a fixed alpha for the entire model, users can follow this example:
```python
import intel_extension_for_pytorch as ipex
smoothquant_args: {"alpha": 0.5, "folding": True}
tuned_model = ipex.quantization.autotune(
    model, calib_dataloader, eval_func, smoothquant_args=smoothquant_args,
)
```
`smoothquant_args` description:
"alpha": a float value. Default is 0.5.
"folding": whether to fold mul into the previous layer, where mul is required to update the input distribution during smoothing.
- True: Fold inserted `mul` into the previous layer in the model graph. IPEX will only insert `mul` for layers that can do folding. 
- False: Allow inserting `mul` to update the input distribution without folding in the graph explicitly. IPEX (version>=2.1) will fuse inserted `mul` automatically in the backend.

#### Determining the `alpha` through auto-tuning
Users can search for the best `alpha` at two levels: a) for the entire model, and b) for each layer/block.

1. Auto-tune the `alpha` for the entire model
The tuning process looks for the optimal `alpha` value from a list of `alpha` values provided by the user.
> Please note that, it may use a considerable amount of time as the tuning process applies each `alpha` to the entire model and uses the evaluation result on the entire dataset as the metric to determine the best `alpha`.
Here is an example:

```python
import numpy as np
smoothquant_args={"alpha": numpy.arange(0.0, 1.0, 0.1).tolist()}
```
2. Auto-tune the `alpha` for each layer/block
In this case, the tuning process searches the optimal `alpha` of each layer of the block by evaluating the loss with respect to FP32 output on a few batches of data.
Here is an example:

```python
smoothquant_args={
    "alpha": "auto",
    "auto_alpha_args"{
        "init_alpha": 0.8, # baseline alpha-value for auto-tuning
        "alpha_min": 0.8, # min value of auto-tuning alpha search space
        "alpha_max": 0.99, # max value of auto-tuning alpha search space
        "alpha_step": 0.01, # step_size of auto-tuning alpha search space
        "shared_criterion": "mean", # criterion for input LayerNorm op of a transformer block
        "enable_blockwise_loss": False, # whether to enable block-wise auto-tuning
    }
}
```

[//]: # (marker_feature_int8_autotune)
[//]: # (marker_feature_int8_autotune)
