Smooth Quant Recipe Tuning API (Prototype)
=============================================

Smooth Quantization is a popular method to improve the accuracy of int8 quantization. The [autotune API](../api_doc.html#ipex.quantization.autotune) allows automatic global alpha tuning, and automatic layer-by-layer alpha tuning provided by Intel® Neural Compressor for the best INT8 accuracy.

SmoothQuant will introduce alpha to calculate the ratio of input and weight updates to reduce quantization error. SmoothQuant arguments are as below:

|     Arguments    | Default Value |    Available Values   |                         Comments                          |
|:----------------:|:-------------:|:---------------------:|:-----------------------------------------------------------:|
|       alpha      |     'auto'    |     [0-1] / 'auto'    |   value to balance input and weight quantization error.   |
|   init_alpha  |      0.5      |     [0-1] / 'auto'    | value to get baseline quantization error for auto-tuning. |
|     alpha_min    |      0.0      |         [0-1]         |         min value of auto-tuning alpha search space         |
|     alpha_max    |      1.0      |         [0-1]         |         max value of auto-tuning alpha search space         |
|    alpha_step    |      0.1      |         [0-1]         |         step_size of auto-tuning alpha search space         |
| shared_criterion |     "mean"    | ["min", "mean","max"] |   criterion for input LayerNorm op of a transformer block.  |
|   enable_blockwise_loss   |     False     |     [True, False]     |          whether to enable block-wise auto-tuning          |

For LLM examples, please refer to [example](https://github.com/intel/intel-extension-for-pytorch/tree/v2.2.0%2Bcpu/examples/cpu/inference/python/llm).

**Note**: When defining dataloaders for calibration, please follow INC's dataloader [format](https://github.com/intel/neural-compressor/blob/master/docs/source/dataloader.md).
