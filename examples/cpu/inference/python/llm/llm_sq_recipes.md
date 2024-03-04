## Smooth Quantization Autotune Feature (Prototype):
SmoothQuant is a popular method to improve the accuracy of int8 quantization. The [autotune API](../../../../../docs/tutorials/features/sq_recipe_tuning_api.md) allows automatic global alpha tuning, and automatic layer-by-layer alpha tuning provided by Intel® Neural Compressor for the best accuracy. Below is the basic command to generate the qconfig summary files (and quantized model ".pt" file) with the SmoothQuant autotune API.

```bash
# general command:
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run.py  --benchmark -m <MODEL_ID> --ipex-smooth-quant --alpha auto  --output-dir "saved_results"

# An example of llama2 7b model:
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  --benchmark -m meta-llama/Llama-2-7b-hf --ipex-smooth-quant --alpha auto
```

## Example command for model tuning with AutoTune API
| Model ID | Command |
|---|:---:|
| meta-llama/Llama-2-7b-hf | python run.py -m meta-llama/Llama-2-7b-hf --ipex-smooth-quant --batch-size 56 --calib-len 2048 --fallback-add --alpha auto --init-alpha 0.8 --alpha-min 0.8 --alpha-max 0.99 --alpha-step 0.01 --shared-criterion 'mean' |
| meta-llama/Llama-2-70b-hf | python run.py -m meta-llama/Llama-2-70b-hf --ipex-smooth-quant --batch-size 56 --calib-shuffle --fallback-add --alpha 0.8 |
| EleutherAI/gpt-j-6b | python run.py -m EleutherAI/gpt-j-6b --ipex-smooth-quant --batch-size 56 --calib-iters 100 --calib-shuffle --fallback-add --alpha 0.85 |
| tiiuae/falcon-40b | python run.py -m tiiuae/falcon-40b --ipex-smooth-quant --batch-size 56 --calib-iters 100 --calib-shuffle --alpha 0.9 |
| facebook/opt-30b | python run.py -m facebook/opt-30b --ipex-smooth-quant --batch-size 56 --calib-iters 100 --calib-shuffle |
| facebook/opt-1.3b | python run.py -m facebook/opt-1.3b --ipex-smooth-quant --batch-size 56 --calib-iters 100 --calib-shuffle --alpha 0.85 |
| baichuan-inc/Baichuan2-7B-Chat | python run.py -m baichuan-inc/Baichuan2-7B-Chat --ipex-smooth-quant --batch-size 56 --calib-iters 100 --calib-shuffle --alpha 0.95 |
| baichuan-inc/Baichuan2-13B-Chat | python run.py -m baichuan-inc/Baichuan2-13B-Chat --ipex-smooth-quant --batch-size 56 --calib-iters 100 --calib-shuffle --alpha 0.65 |

*Note*: The above examples are validated with good accuracy on the "lamada_openai" dataset.
