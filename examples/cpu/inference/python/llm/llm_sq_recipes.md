## Smooth Quantization Autotune Feature:
SmoothQuant is a popular method to improve the accuracy of int8 quantization. The [autotune API](../../../../../docs/tutorials/features/sq_recipe_tuning_api.md) allows automatic global alpha tuning, and automatic layer-by-layer alpha tuning provided by Intel® Neural Compressor for the best INT8 accuracy.
```bash
# general command:
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run.py  --benchmark -m <MODEL_ID> --ipex-smooth-quant --alpha auto  --output-dir "saved_results"

# An example of llama2 7b model:
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  --benchmark -m meta-llama/Llama-2-7b-hf --ipex-smooth-quant --alpha auto
```

## Validated Model Tuning Recipes
| Model ID | Command |
|---|:---:|
| meta-llama/Llama-2-7b-hf | python run.py -m meta-llama/Llama-2-7b-hf --ipex-smooth-quant --batch_size 56 --calib_len 2048 --fallback_add --alpha auto --init-alpha 0.8 --alpha_min 0.8 --alpha_max 0.99 --alpha_step 0.01 --shared_criterion 'mean' |
| meta-llama/Llama-2-13b-hf | python run.py -m meta-llama/Llama-2-13b-hf --ipex-smooth-quant --batch_size 56 --calib_len 1024 --fallback_add --calib_shuffle --calib_padding --alpha auto --init-alpha 0.8 --alpha_min 0.75 --alpha_max 0.99 --alpha_step 0.01 |
| meta-llama/Llama-2-70b-hf | python run.py -m meta-llama/Llama-2-70b-hf --ipex-smooth-quant --batch_size 56 --calib_shuffle --fallback_add --alpha 0.8 |
| EleutherAI/gpt-j-6b | python run.py -m EleutherAI/gpt-j-6b --ipex-smooth-quant --batch_size 56 --calib_iters 100 --calib_shuffle --fallback_add --alpha 0.85 |
| tiiuae/falcon-40b | python run.py -m tiiuae/falcon-40b --ipex-smooth-quant --batch_size 56 --calib_iters 100 --calib_shuffle --alpha 0.9 |
| facebook/opt-30b | python run.py -m facebook/opt-30b --ipex-smooth-quant --batch_size 56 --calib_iters 100 --calib_shuffle |
| facebook/opt-1.3b | python run.py -m facebook/opt-1.3b --ipex-smooth-quant --batch_size 56 --calib_iters 100 --calib_shuffle --alpha 0.85 |
| baichuan-inc/Baichuan2-7B-Chat | python run.py -m baichuan-inc/Baichuan2-7B-Chat --ipex-smooth-quant --batch_size 56 --calib_iters 100 --calib_shuffle --alpha 0.95 |
| baichuan-inc/Baichuan2-13B-Chat | python run.py -m baichuan-inc/Baichuan2-13B-Chat --ipex-smooth-quant --batch_size 56 --calib_iters 100 -calib_shuffle --alpha 0.65 |
| THUDM/chatglm2-6b | python run.py -m THUDM/chatglm2-6b --ipex-smooth-quant --batch_size 56 --calib_iters 100 --calib_shuffle --alpha 0.75 |
| THUDM/chatglm3-6b | python run.py -m THUDM/chatglm3-6b --ipex-smooth-quant --batch_size 56 --calib_iters 100 --calib_shuffle --alpha 0.85 |

*Note*: Validated recipes above are from Intel® Neural Compressor, for details, please refer to [llm-recipes](https://github.com/intel/intel-extension-for-transformers/blob/main/examples/huggingface/pytorch/text-generation/quantization/llm_quantization_recipes.md).