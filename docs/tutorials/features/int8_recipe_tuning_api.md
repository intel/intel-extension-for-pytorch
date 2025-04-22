INT8 Recipe Tuning API (Prototype)
=====================================

This [new API](../api_doc.html#ipex.quantization.autotune) `ipex.quantization.autotune` supports INT8 recipe tuning by using Intel® Neural Compressor as the backend in Intel® Extension for PyTorch\*. In general, we provid default recipe in Intel® Extension for PyTorch\*, and we still recommend users to try out the default recipe first without bothering tuning. If the default recipe doesn't bring about desired accuracy, users can use this API to tune for a more advanced receipe.

Users need to provide a fp32 model and some parameters required for tuning. The API will return a prepared model with tuned qconfig loaded.

## Usage Example
- Static Quantization
Please refer to [static_quant example](https://github.com/intel/intel-extension-for-pytorch/tree/main/examples/cpu/features/int8_recipe_tuning/imagenet_autotune.py).

[//]: # (marker_feature_int8_autotune)
[//]: # (marker_feature_int8_autotune)
