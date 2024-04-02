Releases
=============

## 2.2.0

We are excited to announce the release of Intel® Extension for PyTorch\* 2.2.0+cpu which accompanies PyTorch 2.2. This release mainly brings in our latest optimization on Large Language Model (LLM) including new dedicated API set (`ipex.llm`), new capability for auto-tuning accuracy recipe for LLM, and a broader list of optimized LLM models, together with a set of bug fixing and small optimization. We want to sincerely thank our dedicated community for your contributions. As always, we encourage you to try this release and feedback as to improve further on this product.

### Highlights

- Large Language Model (LLM) optimization

  Intel® Extension for PyTorch\* provides a new dedicated module, `ipex.llm`, to host for Large Language Models (LLMs) specific APIs. With `ipex.llm`, Intel® Extension for PyTorch\* provides comprehensive LLM optimization cross various popular datatypes including FP32/BF16/INT8/INT4. Specifically for low precision, both SmoothQuant and Weight-Only quantization are supported for various scenarios. And user can also run Intel® Extension for PyTorch\* with Tensor Parallel to fit in the multiple ranks or multiple nodes scenarios to get even better performance.

  A typical API under this new module is `ipex.llm.optimize`, which is designed to optimize transformer-based models within frontend Python modules, with a particular focus on Large Language Models (LLMs). It provides optimizations for both model-wise and content-generation-wise. `ipex.llm.optimize` is an upgrade API to replace previous `ipex.optimize_transformers`, which will bring you more consistent LLM experience and performance. Below shows a simple example of `ipex.llm.optimize` for fp32 or bf16 inference:

  ```python
  import torch
  import intel_extension_for_pytorch as ipex
  import transformers

  model= transformers.AutoModelForCausalLM(model_name_or_path).eval()

  dtype = torch.float # or torch.bfloat16
  model = ipex.llm.optimize(model, dtype=dtype)

  model.generate(YOUR_GENERATION_PARAMS)
  ```

  More examples of this API can be found at [LLM optimization API](https://github.com/intel/intel-extension-for-pytorch/tree/v2.2.0%2Bcpu/docs/tutorials/llm/llm_optimize.md).

  Besides the new optimization API for LLM inference, Intel® Extension for PyTorch\* also provides new capability for users to auto-tune a good quantization recipe for running SmoothQuant INT8 with good accuracy. SmoothQuant is a popular method to improve the accuracy of int8 quantization. The new auto-tune API allows automatic global alpha tuning, and automatic layer-by-layer alpha tuning provided by Intel® Neural Compressor for the best INT8 accuracy. More details can be found at [SmoothQuant Recipe Tuning API Introduction](https://github.com/intel/intel-extension-for-pytorch/tree/v2.2.0%2Bcpu/docs/tutorials/features/sq_recipe_tuning_api.md).

  Intel® Extension for PyTorch\* newly optimized many more LLM models including more llama2 variance like llama2-13b/llama2-70b, encoder-decoder model like T5, code generation models like starcoder/codegen, and more like Baichuan, Baichuan2, ChatGLM2, ChatGLM3, mistral, mpt, dolly, etc.. A full list of optimized models can be found at [LLM Optimization](https://github.com/intel/intel-extension-for-pytorch/tree/v2.2.0%2Bcpu/examples/cpu/inference/python/llm).

- Bug fixing and other optimization

    - Further optimized the performance of LLMs [#2349](https://github.com/intel/intel-extension-for-pytorch/commit/d6d591938aefb9020a8a542a160abe4aeb6b238c) [#2412](https://github.com/intel/intel-extension-for-pytorch/commit/e0399108856c826ad609e5f421021945de30a4bf#diff-11f6a633ad677c6a8b6e8e4462afbe836a853a284e362ba794a8fcbceebc9dc5), [#2469](https://github.com/intel/intel-extension-for-pytorch/commit/aeaeba47bc722d9b18f13f8a78e02092c0a6bb5b), [#2476](https://github.com/intel/intel-extension-for-pytorch/commit/c95eb77398fa131e4ef60be65841ca09a284115d)
    - Optimized the Flash Attention Operator [#2317](https://github.com/intel/intel-extension-for-pytorch/commit/8d0426c1aebc85620fd417fa7fd4e0f1b357fa3d) [#2334](https://github.com/intel/intel-extension-for-pytorch/commit/efab335b427daf76e01836d520b1d7981de59595) [#2392](https://github.com/intel/intel-extension-for-pytorch/commit/5ed3a2413db5f0a5e53bcca0b3e84a814d87bb50) [#2480](https://github.com/intel/intel-extension-for-pytorch/commit/df2387e976461f6c42e0b90b3544ea76d3132694)
    - Fixed the static quantization of the ELSER model [#2491](https://github.com/intel/intel-extension-for-pytorch/commit/ac613a73fb395836b210710a6fefdf6d32df3386)
    - Switched deepspeed to the public release version on PyPI [#2473](https://github.com/intel/intel-extension-for-pytorch/commit/dba7b8c5fc9bfd8e7aa9431efe63499014acd722) [#2511](https://github.com/intel/intel-extension-for-pytorch/commit/94c31ecb3b6f6e77f595ce94dd6d6cbae1db1210)
    - Upgrade oneDNN to v3.3.4 [#2433](https://github.com/intel/intel-extension-for-pytorch/commit/af9b096070e81b46250172174bb9d12e3e1c6acf)

**Full Changelog**: https://github.com/intel/intel-extension-for-pytorch/compare/v2.1.100+cpu...v2.2.0+cpu

## 2.1.100

### Highlights

- Improved the performance of BF16 LLM generation inference: [#2253](https://github.com/intel/intel-extension-for-pytorch/commit/99aa54f757de6c7d98f704edc6f8a83650fb1541) [#2251](https://github.com/intel/intel-extension-for-pytorch/commit/1d5e83d85c3aaf7c00323d7cb4019b40849dd2ed) [#2236](https://github.com/intel/intel-extension-for-pytorch/commit/be349962f3362f8afde4f083ec04d335245992bb) [#2278](https://github.com/intel/intel-extension-for-pytorch/commit/066c3bff417df084fa8e1d48375c0e1404320e95)

- Added the optimization for Codegen: [#2257](https://github.com/intel/intel-extension-for-pytorch/commit/7c598e42e5b7899f284616c05c6896bf9d8bd2b8)

- Provided the dockerfile and updated the related doc to improve the UX for LLM users: [#2229](https://github.com/intel/intel-extension-for-pytorch/commit/11484c3ebad9f868d0179a46de3d1330d9011822) [#2195](https://github.com/intel/intel-extension-for-pytorch/commit/0cd25021952bddcf5a364da45dfbefd4a0c77af4) [#2299](https://github.com/intel/intel-extension-for-pytorch/commit/76a42e516a68539752a3a8ab9aeb814d28c44cf8) [#2315](https://github.com/intel/intel-extension-for-pytorch/commit/4091bb5c0bf5f3c3ce5fbece291b44159a7fbf5c) [#2283](https://github.com/intel/intel-extension-for-pytorch/commit/e5ed8270d4d89bf68757f967676db57292c71920)

- Improved the accuracy of the quantization path of LLMs: [#2280](https://github.com/intel/intel-extension-for-pytorch/commit/abc4c4e160cec3c792f5316e358173b8722a786e) [#2292](https://github.com/intel/intel-extension-for-pytorch/commit/4e212e41affa2ed07ffaf57bf10e9781113bc101) [#2275](https://github.com/intel/intel-extension-for-pytorch/commit/ed5957eb3b6190ad0be728656674f0a2a3b89158) [#2319](https://github.com/intel/intel-extension-for-pytorch/commit/1dae69de39408bc0ad245f4914d5f60e008a6eb3)

- Misc fix and enhancement: [#2198](https://github.com/intel/intel-extension-for-pytorch/commit/ed1deccb86403e12e895227045d558117c5ea0fe) [#2264](https://github.com/intel/intel-extension-for-pytorch/commit/5dedcd6eb7bbf70dc92f0c20962fb2340e42e76f) [#2290](https://github.com/intel/intel-extension-for-pytorch/commit/c6e46cecd899317acfd2bd2a44a3f17b3cc1ce69)

**Full Changelog**: https://github.com/intel/intel-extension-for-pytorch/compare/v2.1.0+cpu...v2.1.100+cpu

## 2.1.0

### Highlights

- **Large Language Model (LLM) optimization (Experimental)**: Intel® Extension for PyTorch\* provides a lot of specific optimizations for LLMs in this new release. In operator level, we provide highly efficient GEMM kernel to speedup Linear layer and customized operators to reduce the memory footprint. To better trade-off the performance and accuracy, different low-precision solutions e.g., smoothQuant for INT8 and weight-only-quantization for INT4 and INT8 are also enabled. Besides, tensor parallel can also be adopt to get lower latency for LLMs.

  A new API function, `ipex.optimize_transformers`, is designed to optimize transformer-based models within frontend Python modules, with a particular focus on Large Language Models (LLMs). It provides optimizations for both model-wise and content-generation-wise. You just need to invoke the `ipex.optimize_transformers` function instead of the `ipex.optimize` function to apply all optimizations transparently. More detailed information can be found at [Large Language Model optimizations overview](./llm.rst).

  Specifically, this new release includes the support of [SmoothQuant]( https://arxiv.org/abs/2211.10438) and weight only quantization (both INT8 weight and INT4 weight) as to provide better performance and accuracy for low precision scenarios.

  A typical usage of this new feature is quite simple as below:

  ```python
  import torch
  import intel_extension_for_pytorch as ipex
  ...
  model = ipex.optimize_transformers(model, dtype=dtype)
  ```

- **torch.compile backend optimization with PyTorch Inductor (Experimental)**: We optimized Intel® Extension for PyTorch to leverage PyTorch Inductor’s capability when working as a backend of torch.compile, which can better utilize torch.compile’s power of graph capture, Inductor’s scalable fusion capability, and still keep customized optimization from Intel® Extension for PyTorch.

- **performance optimization of static quantization under dynamic shape**: We optimized the static quantization performance of Intel® Extension for PyTorch for dynamic shapes. The usage is the same as the workflow of running static shapes while inputs of variable shapes could be provided during runtime.

- **Bug fixing and other optimization**
    - Optimized the runtime memory usage [#1563](https://github.com/intel/intel-extension-for-pytorch/commit/a821c0aef97ee6252d2bfbe6a75b6085f78bcc59)
    - Fixed the excessive size of the saved model [#1677](https://github.com/intel/intel-extension-for-pytorch/commit/39f2d0f4e91c6007cb58566b63e06b72d7b17ce4) [#1688](https://github.com/intel/intel-extension-for-pytorch/commit/58adee5b043a52e0c0a60320d48eae82de557074)
    - Supported shared parameters in `ipex.optimize` [#1664](https://github.com/intel/intel-extension-for-pytorch/commit/4fa37949385db88b854eb60ab6de7178706cdcfe)
    - Enabled the optimization of LARS fusion [#1695](https://github.com/intel/intel-extension-for-pytorch/commit/e5b169e8d1e06558bb366eeaf4c793a382bc2d62)
    - Supported dictionary input in `ipex.quantization.prepare` [#1682](https://github.com/intel/intel-extension-for-pytorch/commit/30b70e4b0bd8c3d1b2be55147ebd74fbfebe6093)
    - Updated oneDNN to v3.3 [#2137](https://github.com/intel/intel-extension-for-pytorch/commit/4dc4bb5f9d1cfb9f958893a410f7332be4b5f783)

## 2.0.100

### Highlights

- Enhanced the functionality of Intel® Extension for PyTorch as a backend of `torch.compile`: [#1568](https://github.com/intel/intel-extension-for-pytorch/commit/881c6fe0e6f8ab84a564b02216ddb96a3589363e) [#1585](https://github.com/intel/intel-extension-for-pytorch/commit/f5ce6193496ae68a57d688a3b3bbff541755e4ce) [#1590](https://github.com/intel/intel-extension-for-pytorch/commit/d8723df73358ae495ae5f62b5cdc90ae08920d27)
- Fixed the Stable Diffusion fine-tuning accuracy issue [#1587](https://github.com/intel/intel-extension-for-pytorch/commit/bc76ab133b7330852931db9cda8dca7c69a0b594) [#1594](https://github.com/intel/intel-extension-for-pytorch/commit/b2983b4d35fc0ea7f5bdaf37f6e269256f8c36c4)
- Fixed the ISA check on old hypervisor based VM [#1513](https://github.com/intel/intel-extension-for-pytorch/commit/a34eab577c4efa1c336b1f91768075bb490c1f14)
- Addressed the excessive memory usage in weight prepack [#1593](https://github.com/intel/intel-extension-for-pytorch/commit/ee7dc343790d1d63bab1caf71e57dd3f7affdce9)
- Fixed the weight prepack of convolution when `padding_mode` is not `'zeros'` [#1580](https://github.com/intel/intel-extension-for-pytorch/commit/02449ccb3a6b475643116532a4cffbe1f974c1d9)
- Optimized the INT8 LSTM performance [#1566](https://github.com/intel/intel-extension-for-pytorch/commit/fed42b17391fed477ae8adec83d920f8f8fb1a80)
- Fixed TransNetV2 calibration failure [#1564](https://github.com/intel/intel-extension-for-pytorch/commit/046f7dfbaa212389ac58ae219597c16403e66bad)
- Fixed BF16 RNN-T inference when `AVX512_CORE_VNNI` ISA is used [#1592](https://github.com/intel/intel-extension-for-pytorch/commit/023c104ab5953cf63b84efeb5176007d876015a2)
- Fixed the ROIAlign operator [#1589](https://github.com/intel/intel-extension-for-pytorch/commit/6beb3d4661f09f55d031628ebe9fa6d63f04cab1)
- Enabled execution on designated numa nodes with launch script [#1517](https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-cpu/commit/2ab3693d50d6edd4bfae766f75dc273396a79488)

**Full Changelog**: https://github.com/intel/intel-extension-for-pytorch/compare/v2.0.0+cpu...v2.0.100+cpu

## 2.0.0

We are pleased to announce the release of Intel® Extension for PyTorch\* 2.0.0-cpu which accompanies PyTorch 2.0. This release mainly brings in our latest optimization on NLP (BERT), support of PyTorch 2.0's hero API –- torch.compile as one of its backend, together with a set of bug fixing and small optimization.

### Highlights

- **Fast BERT optimization (Experimental)**: Intel introduced a new technique to speed up BERT workloads. Intel® Extension for PyTorch\* integrated this implementation, which benefits BERT model especially training. A new API `ipex.fast_bert` is provided to try this new optimization. More detailed information can be found at [Fast Bert Feature](./features/fast_bert.md).

- **MHA optimization with Flash Attention**: Intel optimized MHA module with Flash Attention technique as inspired by [Stanford paper](https://arxiv.org/abs/2205.14135). This brings less memory consumption for LLM, and also provides better inference performance for models like BERT, Stable Diffusion, etc.

- **Work with torch.compile as an backend (Experimental)**: PyTorch 2.0 introduces a new feature, `torch.compile`, to speed up PyTorch execution. We've enabled Intel® Extension for PyTorch as a backend of torch.compile, which can leverage this new PyTorch API's power of graph capture and provide additional optimization based on these graphs.
The usage of this new feature is quite simple as below: 

```python
import torch
import intel_extension_for_pytorch as ipex
...
model = ipex.optimize(model)
model = torch.compile(model, backend='ipex')
```

- **Bug fixing and other optimization**

  - Supported [RMSNorm](https://arxiv.org/abs/1910.07467) which is widely used in the t5 model of huggingface [#1341](https://github.com/intel/intel-extension-for-pytorch/commit/d1de1402a8d6b9ca49b9c9a45a92899f7566866a)
  - Optimized InstanceNorm [#1330](https://github.com/intel/intel-extension-for-pytorch/commit/8b97d2998567cc2fda6eb008194cd64f624e857f)
  - Fixed the quantization of LSTM [#1414](https://github.com/intel/intel-extension-for-pytorch/commit/a4f93c09855679d2b424ca5be81930e3a4562cef) [#1473](https://github.com/intel/intel-extension-for-pytorch/commit/5b44996dc0fdb5c45995d403e18a44f2e1a11b3d)
  - Fixed the correctness issue of unpacking non-contiguous Linear weight [#1419](https://github.com/intel/intel-extension-for-pytorch/commit/84d413d6c10e16c025c407b68652b1769597e016) 
  - oneDNN update [#1488](https://github.com/intel/intel-extension-for-pytorch/commit/fd5c10b664d19c87f8d94cf293077f65f78c3937)

### Known Issues

Please check at [Known Issues webpage](./known_issues.md).

## 1.13.100

### Highlights

- Quantization optimization with more fusion, op and auto channels last support [#1318](https://github.com/intel/intel-extension-for-pytorch/commit/5dd3a6ed9017197dea5c05c3af6d330336ed8eff) [#1353](https://github.com/intel/intel-extension-for-pytorch/commit/461b867021e1471c93a1a2a96255247c9d2ab45b) [#1328](https://github.com/intel/intel-extension-for-pytorch/commit/ff3f527025d2102898df9d02977df955e31ddf69) [#1355](https://github.com/intel/intel-extension-for-pytorch/commit/d21111565a179bb8f7ef6db3c04fafbe94871b61) [#1367](https://github.com/intel/intel-extension-for-pytorch/commit/2b898a935e597cfa92ee01a064a626763657c952) [#1384](https://github.com/intel/intel-extension-for-pytorch/commit/a81bd7023e9a119d1ce5f86307865b443034909e)
- Installation and build enhancement [#1295](https://github.com/intel/intel-extension-for-pytorch/commit/9da7844b75b7cf22d9f4f5401178948919c40914) [#1392](https://github.com/intel/intel-extension-for-pytorch/commit/ef12c70c3ed496e723ac087ea5703dae7df0358d)
- OneDNN graph and OneDNN update [#1376](https://github.com/intel/intel-extension-for-pytorch/commit/dab9dc18659da53e624637166283ccc8db1373f9)
- Misc fix and enhancement [#1373](https://github.com/intel/intel-extension-for-pytorch/commit/085ba5d93773ab283e954a4fce75468708b74d3a) [#1338](https://github.com/intel/intel-extension-for-pytorch/commit/0bdf4b27dc445eb8fd0d59f46d157949db597953) [#1391](https://github.com/intel/intel-extension-for-pytorch/commit/2e8289967472553a049158d55e60835371829925) [#1322](https://github.com/intel/intel-extension-for-pytorch/commit/f69492345eb8a9383a67d9416146c2b73de19d8d)

**Full Changelog**: https://github.com/intel/intel-extension-for-pytorch/compare/v1.13.0+cpu...v1.13.100+cpu

## 1.13.0

We are pleased to announce the release of Intel® Extension for PyTorch\* 1.13.0-cpu which accompanies PyTorch 1.13. This release is highlighted with quite a few usability features which help users to get good performance and accuracy on CPU with less effort. We also added a couple of performance features as always. Check out the feature summary below.
- Usability Features
1. **Automatic channels last format conversion**: Channels last conversion is now applied automatically to PyTorch modules with `ipex.optimize` by default. Users don't have to explicitly convert input and weight for CV models.
2. **Code-free optimization** (experimental): `ipex.optimize` is automatically applied to PyTorch modules without the need of code changes when the PyTorch program is started with the Intel® Extension for PyTorch\* launcher via the new `--auto-ipex` option.
3. **Graph capture mode** of `ipex.optimize` (experimental): A new boolean flag `graph_mode` (default off) was added to `ipex.optimize`, when turned on, converting the eager-mode PyTorch module into graph(s) to get the best of graph optimization.
4. **INT8 quantization accuracy autotune** (experimental): A new quantization API `ipex.quantization.autotune` was added to refine the default Intel® Extension for PyTorch\* quantization recipe via autotuning algorithms for better accuracy.
5. **Hypertune** (experimental) is a new tool added on top of Intel® Extension for PyTorch\* launcher to automatically identify the good configurations for best throughput via hyper-parameter tuning.
6. **ipexrun**: The counterpart of **torchrun**, is a shortcut added for invoking Intel® Extension for PyTorch\* launcher.
- Performance Features
1. Packed MKL SGEMM landed as the default kernel option for FP32 Linear, bringing up-to 20% geomean speedup for real-time NLP tasks.
2. DL compiler is now turned on by default with oneDNN fusion and gives additional performance boost for INT8 models.

### Highlights
* **Automatic channels last format conversion**: Channels last conversion is now applied to PyTorch modules automatically with `ipex.optimize` by default for both training and inference scenarios. Users don't have to explicitly convert input and weight for CV models.
  ```python
  import intel_extension_for_pytorch as ipex
  # No need to do explicitly format conversion
  # m = m.to(format=torch.channels_last)
  # x = x.to(format=torch.channels_last)
  # for inference
  m = ipex.optimize(m)
  m(x)
  # for training
  m, optimizer = ipex.optimize(m, optimizer)
  m(x)
  ```
* **Code-free optimization** (experimental): `ipex.optimize` is automatically applied to PyTorch modules without the need of code changes when the PyTorch program is started with the Intel® Extension for PyTorch\* launcher via the new `--auto-ipex` option.

  Example: QA case in HuggingFace

  ```bash
  # original command
  ipexrun --use_default_allocator --ninstance 2 --ncore_per_instance 28 run_qa.py \
    --model_name_or_path bert-base-uncased --dataset_name squad --do_eval \
    --per_device_train_batch_size 12 --learning_rate 3e-5 --num_train_epochs 2 \
    --max_seq_length 384 --doc_stride 128 --output_dir /tmp/debug_squad/
  
  # automatically apply bfloat16 optimization (--auto-ipex --dtype bfloat16)
  ipexrun --use_default_allocator --ninstance 2 --ncore_per_instance 28 --auto_ipex --dtype bfloat16 run_qa.py \
    --model_name_or_path bert-base-uncased --dataset_name squad --do_eval \
    --per_device_train_batch_size 12 --learning_rate 3e-5 --num_train_epochs 2 \
    --max_seq_length 384 --doc_stride 128 --output_dir /tmp/debug_squad/
  ```

* **Graph capture mode** of `ipex.optimize` (experimental): A new boolean flag `graph_mode` (default off) was added to `ipex.optimize`, when turned on, converting the eager-mode PyTorch module into graph(s) to get the best of graph optimization. Under the hood, it combines the goodness of both TorchScript tracing and TorchDynamo to get as max graph scope as possible. Currently, it only supports FP32 and BF16 inference. INT8 inference and training support are under way.
  ```python
  import intel_extension_for_pytorch as ipex
  model = ...
  model.load_state_dict(torch.load(PATH))
  model.eval()
  optimized_model = ipex.optimize(model, graph_mode=True)
  ```

* **INT8 quantization accuracy autotune** (experimental): A new quantization API `ipex.quantization.autotune` was added to refine the default Intel® Extension for PyTorch\* quantization recipe via autotuning algorithms for better accuracy. This is an optional API to invoke (after `prepare` and before `convert`) for scenarios when the accuracy of default quantization recipe of Intel® Extension for PyTorch\* cannot meet the requirement. The current implementation is powered by Intel® Neural Compressor.
  ```python
  import intel_extension_for_pytorch as ipex
  # Calibrate the model
  qconfig = ipex.quantization.default_static_qconfig
  calibrated_model = ipex.quantization.prepare(model_to_be_calibrated, qconfig, example_inputs=example_inputs)
  for data in calibration_data_set:
      calibrated_model(data)
  # Autotune the model
  calib_dataloader = torch.utils.data.DataLoader(...)
  def eval_func(model):
      # Return accuracy value
      ...
      return accuracy
  tuned_model = ipex.quantization.autotune(
                   calibrated_model, calib_dataloader, eval_func,
                   sampling_sizes=[100], accuracy_criterion={'relative': 0.01}, tuning_time=0
                )
  # Convert the model to jit model
  quantized_model = ipex.quantization.convert(tuned_model)
  with torch.no_grad():
      traced_model = torch.jit.trace(quantized_model, example_input)
      traced_model = torch.jit.freeze(traced_model)
  # Do inference
  y = traced_model(x)
  ```

* **Hypertune** (experimental) is a new tool added on top of Intel® Extension for PyTorch\* launcher to automatically identify the good configurations for best throughput via hyper-parameter tuning.
  ```bash
  python -m intel_extension_for_pytorch.cpu.launch.hypertune --conf_file <your_conf_file> <your_python_script> [args]
  ```

### Known Issues

Please check at [Known Issues webpage](./known_issues.md).

## 1.12.300

### Highlights

- Optimize BF16 MHA fusion to avoid transpose overhead to boost BERT-\* BF16 performance [#992](https://github.com/intel/intel-extension-for-pytorch/commit/7076524601f42a9b60402019af21b32782c2c203)
- Remove 64bytes alignment constraint for FP32 and BF16 AddLayerNorm fusion [#992](https://github.com/intel/intel-extension-for-pytorch/commit/7076524601f42a9b60402019af21b32782c2c203)
- Fix INT8 RetinaNet accuracy issue [#1032](https://github.com/intel/intel-extension-for-pytorch/commit/e0c719be8246041f8b7bc5feca9cf9c2f599210a)
- Fix `Cat.out` issue that does not update the `out` tensor (#1053) [#1074](https://github.com/intel/intel-extension-for-pytorch/commit/4381f9126bbb65aab2daf034299c3bf3d307e6e2)

**Full Changelog**: https://github.com/intel/intel-extension-for-pytorch/compare/v1.12.100...v1.12.300

## 1.12.100

This is a patch release to fix the AVX2 issue that blocks running on non-AVX512 platforms.

## 1.12.0

We are excited to bring you the release of Intel® Extension for PyTorch\* 1.12.0-cpu, by tightly following PyTorch [1.12](https://github.com/pytorch/pytorch/releases/tag/v1.12.0) release. In this release, we matured the automatic int8 quantization and made it a stable feature. We stabilized runtime extension and brought about a MultiStreamModule feature to further boost throughput in offline inference scenario. We also brought about various enhancements in operation and graph which are positive for performance of broad set of workloads.

Highlights include:
- Automatic INT8 quantization became a stable feature baking into a well-tuned default quantization recipe, supporting both static and dynamic quantization and a wide range of calibration algorithms.
- Runtime Extension, featured MultiStreamModule, became a stable feature, could further enhance throughput in offline inference scenario.
- More optimizations in graph and operations to improve performance of broad set of models, examples include but not limited to wave2vec, T5, Albert etc.
- Pre-built experimental binary with oneDNN Graph Compiler tuned on would deliver additional performance gain for Bert, Albert, Roberta in INT8 inference.

### Highlights

- Matured automatic INT8 quantization feature baking into a well-tuned default quantization recipe. We facilitated the user experience and provided a wide range of calibration algorithms like Histogram, MinMax, MovingAverageMinMax, etc. Meanwhile, We polished the static quantization with better flexibility and enabled dynamic quantization as well. Compared to the previous version, the brief changes are as follows. Refer to [tutorial page](features/int8_overview.md) for more details.

  <table align="center">
  <tbody>
  <tr>
  <td>v1.11.0-cpu</td>
  <td>v1.12.0-cpu</td>
  </tr>
  <tr>
  <td valign="top">
  
  ```python
  import intel_extension_for_pytorch as ipex
  # Calibrate the model
  qconfig = ipex.quantization.QuantConf(qscheme=torch.per_tensor_affine)
  for data in calibration_data_set:
      with ipex.quantization.calibrate(qconfig):
          model_to_be_calibrated(x)
  qconfig.save('qconfig.json')
  # Convert the model to jit model
  conf = ipex.quantization.QuantConf('qconfig.json')
  with torch.no_grad():
      traced_model = ipex.quantization.convert(model, conf, example_input)
  # Do inference 
  y = traced_model(x)
  ```
  
  </td>
  <td valign="top">
  
  ```python
  import intel_extension_for_pytorch as ipex
  # Calibrate the model
  qconfig = ipex.quantization.default_static_qconfig # Histogram calibration algorithm and 
  calibrated_model = ipex.quantization.prepare(model_to_be_calibrated, qconfig, example_inputs=example_inputs)
  for data in calibration_data_set:
      calibrated_model(data)
  # Convert the model to jit model
  quantized_model = ipex.quantization.convert(calibrated_model)
  with torch.no_grad():
      traced_model = torch.jit.trace(quantized_model, example_input)
      traced_model = torch.jit.freeze(traced_model)
  # Do inference 
  y = traced_model(x)
  ```
  
  </td>
  </tr>
  </tbody>
  </table>

- Runtime Extension, featured MultiStreamModule, became a stable feature. In this release, we enhanced the heuristic rule to further enhance throughput in offline inference scenario. Meanwhile, we also provide the `ipex.cpu.runtime.MultiStreamModuleHint` to custom how to split the input into streams and concat the output for each steam.

  <table align="center">
  <tbody>
  <tr>
  <td>v1.11.0-cpu</td>
  <td>v1.12.0-cpu</td>
  </tr>
  <tr>
  <td valign="top">
  
  ```python
  import intel_extension_for_pytorch as ipex
  # Create CPU pool
  cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
  # Create multi-stream model
  multi_Stream_model = ipex.cpu.runtime.MultiStreamModule(model, num_streams=2, cpu_pool=cpu_pool)
  ```
  
  </td>
  <td valign="top">
  
  ```python
  import intel_extension_for_pytorch as ipex
  # Create CPU pool
  cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
  # Optional
  multi_stream_input_hint = ipex.cpu.runtime.MultiStreamModuleHint(0)
  multi_stream_output_hint = ipex.cpu.runtime.MultiStreamModuleHint(0)
  # Create multi-stream model
  multi_Stream_model = ipex.cpu.runtime.MultiStreamModule(model, num_streams=2, cpu_pool=cpu_pool,
    multi_stream_input_hint,   # optional
    multi_stream_output_hint ) # optional
  ```
  
  </td>
  </tr>
  </tbody>
  </table>

- Polished the `ipex.optimize` to accept the input shape information which would conclude the optimal memory layout for better kernel efficiency.

  <table align="center">
  <tbody>
  <tr>
  <td>v1.11.0-cpu</td>
  <td>v1.12.0-cpu</td>
  </tr>
  <tr>
  <td valign="top">
  
  ```python
  import intel_extension_for_pytorch as ipex
  model = ...
  model.load_state_dict(torch.load(PATH))
  model.eval()
  optimized_model = ipex.optimize(model, dtype=torch.bfloat16)
  ```
  
  </td>
  <td valign="top">
  
  ```python
  import intel_extension_for_pytorch as ipex
  model = ...
  model.load_state_dict(torch.load(PATH))
  model.eval()
  optimized_model = ipex.optimize(model, dtype=torch.bfloat16, sample_input=input)
  ```
  
  </td>
  </tr>
  </tbody>
  </table>

- Provided more optimizations in graph and operations
  - Fuse Adam to improve training performance [#822](https://github.com/intel/intel-extension-for-pytorch/commit/d3f714e54dc8946675259ea7a445b26a2460b523)
  - Enable Normalization operators to support channels-last 3D [#642](https://github.com/intel/intel-extension-for-pytorch/commit/ae268ac1760d598a29584de5c99bfba46c6554ae)
  - Support Deconv3D to serve most models and implement most fusions like Conv
  - Enable LSTM to support static and dynamic quantization [#692](https://github.com/intel/intel-extension-for-pytorch/commit/2bf8dba0c380a26bbb385e253adbfaa2a033a785)
  - Enable Linear to support dynamic quantization [#787](https://github.com/intel/intel-extension-for-pytorch/commit/ff231fb55e33c37126a0ef7f0e739cd750d1ef6c)
  - Fusions.
    - Fuse `Add` + `Swish` to accelerate FSI Riskful model [#551](https://github.com/intel/intel-extension-for-pytorch/commit/cc855ff2bafd245413a6111f3d21244d0bcbb6f6)
    - Fuse `Conv` + `LeakyReLU` [#589](https://github.com/intel/intel-extension-for-pytorch/commit/dc6ed1a5967c644b03874fd1f8a503f0b80be6bd)
    - Fuse `BMM` + `Add` [#407](https://github.com/intel/intel-extension-for-pytorch/commit/d1379aa565cc84b4a61b537ba2c9a046b7652f1a)
    - Fuse `Concat` + `BN` + `ReLU` [#647](https://github.com/intel/intel-extension-for-pytorch/commit/cad3f82f6b7efed0c08b2f0c11117a4720f58df4)
    - Optimize `Convolution1D` to support channels last memory layout and fuse `GeLU` as its post operation. [#657](https://github.com/intel/intel-extension-for-pytorch/commit/a0c063bdf4fd1a7e66f8a23750ac0c2fe471a559)
    - Fuse `Einsum` + `Add` to boost Alphafold2 [#674](https://github.com/intel/intel-extension-for-pytorch/commit/3094f346a67c81ad858ad2a80900fab4c3b4f4e9)
    - Fuse `Linear` + `Tanh` [#711](https://github.com/intel/intel-extension-for-pytorch/commit/b24cc530b1fd29cb161a76317891e361453333c9)

### Known Issues
- `RuntimeError: Overflow when unpacking long` when a tensor's min max value exceeds int range while performing int8 calibration. Please customize QConfig to use min-max calibration method.
- Calibrating with quantize_per_tensor, when benchmarking with 1 OpenMP\* thread, results might be incorrect with large tensors (find more detailed info [here](https://github.com/pytorch/pytorch/issues/80501). Editing your code following the pseudocode below can workaround this issue, if you do need to explicitly set OMP_NUM_THREAEDS=1 for benchmarking. However, there could be a performance regression if oneDNN graph compiler prototype feature is utilized.

  Workaround pseudocode:
  ```
  # perform convert/trace/freeze with omp_num_threads > 1(N)
  torch.set_num_threads(N)
  prepared_model = prepare(model, input)
  converted_model = convert(prepared_model)
  traced_model = torch.jit.trace(converted_model, input)
  freezed_model = torch.jit.freeze(traced_model)
  # run freezed model to apply optimization pass
  freezed_model(input)

  # benchmarking with omp_num_threads = 1
  torch.set_num_threads(1)
  run_benchmark(freezed_model, input)
  ```
- Low performance with INT8 support for dynamic shapes
  The support for dynamic shapes in Intel® Extension for PyTorch\* INT8 integration is still work in progress. When the input shapes are dynamic, for example inputs of variable image sizes in an object detection task or of variable sequence lengths in NLP tasks, the Intel® Extension for PyTorch\* INT8 path may slow down the model inference. In this case, use stock PyTorch INT8 functionality.
  **Note**: Using Runtime Extension feature if batch size cannot be divided by number of streams, because mini batch size on each stream are not equivalent, scripts run into this issues.
- BF16 AMP(auto-mixed-precision) runs abnormally with the extension on the AVX2-only machine if the topology contains `Conv`, `Matmul`, `Linear`, and `BatchNormalization`
- Runtime extension of MultiStreamModule doesn't support DLRM inference, since the input of DLRM (EmbeddingBag specifically) can't be simplely batch split.
- Runtime extension of MultiStreamModule has poor performance of RNNT Inference comparing with native throughput mode. Only part of the RNNT models (joint_net specifically) can be jit traced into graph. However, in one batch inference, `joint_net` is invoked multi times. It increases the overhead of MultiStreamModule as input batch split, thread synchronization and output concat.
- Incorrect Conv and Linear result if the number of OMP threads is changed at runtime
  The oneDNN memory layout depends on the number of OMP threads, which requires the caller to detect the changes for the # of OMP threads while this release has not implemented it yet.
- Low throughput with DLRM FP32 Train
  A 'Sparse Add' [PR](https://github.com/pytorch/pytorch/pull/23057) is pending on review. The issue will be fixed when the PR is merged.
- If inference is done with a custom function, `conv+bn` folding feature of the `ipex.optimize()` function doesn't work.
  ```
  import torch
  import intel_pytorch_extension as ipex
  class Module(torch.nn.Module):
      def __init__(self):
          super(Module, self).__init__()
          self.conv = torch.nn.Conv2d(1, 10, 5, 1)
          self.bn = torch.nn.BatchNorm2d(10)
          self.relu = torch.nn.ReLU()
      def forward(self, x):
          x = self.conv(x)
          x = self.bn(x)
          x = self.relu(x)
          return x
      def inference(self, x):
          return self.forward(x)
  if __name__ == '__main__':
      m = Module()
      m.eval()
      m = ipex.optimize(m, dtype=torch.float32, level="O0")
      d = torch.rand(1, 1, 112, 112)
      with torch.no_grad():
        m.inference(d)
  ```
  This is a PyTorch FX limitation. You can avoid this error by calling `m = ipex.optimize(m, level="O0")`, which doesn't apply ipex optimization, or disable `conv+bn` folding by calling `m = ipex.optimize(m, level="O1", conv_bn_folding=False)`.

## 1.11.200

### Highlights

- Enable more fused operators to accelerate particular models.
- Fuse `Convolution` and `LeakyReLU` ([#648](https://github.com/intel/intel-extension-for-pytorch/commit/d7603133f37375b3aba7bf744f1095b923ba979e))
- Support [`torch.einsum`](https://pytorch.org/docs/stable/generated/torch.einsum.html) and fuse it with `add` ([#684](https://github.com/intel/intel-extension-for-pytorch/commit/b66d6d8d0c743db21e534d13be3ee75951a3771d))
- Fuse `Linear` and `Tanh` ([#685](https://github.com/intel/intel-extension-for-pytorch/commit/f0f2bae96162747ed2a0002b274fe7226a8eb200))
- In addition to the original installation methods, this release provides Docker installation from [DockerHub](https://hub.docker.com/).
- Provided the <a class="reference external" href="installation.html#installation_onednn_graph_compiler">evaluation wheel packages</a> that could boost performance for selective topologies on top of oneDNN graph compiler prototype feature.
***NOTE***: This is still at an early development stage and not fully mature yet, but feel free to reach out through [GitHub issues](https://github.com/intel/intel-extension-for-pytorch/issues) if you have any suggestions.

**[Full Changelog](https://github.com/intel/intel-extension-for-pytorch/compare/v1.11.0...v1.11.200)**

## 1.11.0

We are excited to announce Intel® Extension for PyTorch\* 1.11.0-cpu release by tightly following PyTorch 1.11 release. Along with extension 1.11, we focused on continually improving OOB user experience and performance. Highlights include:

* Support a single binary with runtime dynamic dispatch based on AVX2/AVX512 hardware ISA detection
* Support install binary from `pip` with package name only (without the need of specifying the URL)
* Provide the C++ SDK installation to facilitate ease of C++ app development and deployment
* Add more optimizations, including graph fusions for speeding up Transformer-based models and CNN, etc
* Reduce the binary size for both the PIP wheel and C++ SDK (2X to 5X reduction from the previous version)

### Highlights
- Combine the AVX2 and AVX512 binary as a single binary and automatically dispatch to different implementations based on hardware ISA detection at runtime. The typical case is to serve the data center that mixtures AVX2-only and AVX512 platforms. It does not need to deploy the different ISA binary now compared to the previous version

    ***NOTE***:  The extension uses the oneDNN library as the backend. However, the BF16 and INT8 operator sets and features are different between AVX2 and AVX512. Refer to [oneDNN document](https://oneapi-src.github.io/oneDNN/dev_guide_int8_computations.html#processors-with-the-intel-avx2-or-intel-avx-512-support) for more details. 

    > When one input is of type u8, and the other one is of type s8, oneDNN assumes the user will choose the quantization parameters so no overflow/saturation occurs. For instance, a user can use u7 [0, 127] instead of u8 for the unsigned input, or s7 [-64, 63] instead of the s8 one. It is worth mentioning that this is required only when the Intel AVX2 or Intel AVX512 Instruction Set is used.

- The extension wheel packages have been uploaded to [pypi.org](https://pypi.org/project/intel-extension-for-pytorch/). The user could directly install the extension by `pip/pip3` without explicitly specifying the binary location URL.

<table align="center">
<tbody>
<tr>
<td>v1.10.100-cpu</td>
<td>v1.11.0-cpu</td>
</tr>
<tr>
<td>

```python
python -m pip install intel_extension_for_pytorch==1.10.100 -f https://software.intel.com/ipex-whl-stable
```
</td>
<td>

```python
pip install intel_extension_for_pytorch
```
</td>
</tr>
</tbody>
</table>

- Compared to the previous version, this release provides a dedicated installation file for the C++ SDK. The installation file automatically detects the PyTorch C++ SDK location and installs the extension C++ SDK files to the PyTorch C++ SDK. The user does not need to manually add the extension C++ SDK source files and CMake to the PyTorch SDK. In addition to that, the installation file reduces the C++ SDK binary size from ~220MB to ~13.5MB. 

<table align="center">
<tbody>
<tr>
<td>v1.10.100-cpu</td>
<td>v1.11.0-cpu</td>
</tr>
<tr>
<td>

```python
intel-ext-pt-cpu-libtorch-shared-with-deps-1.10.0+cpu.zip (220M)
intel-ext-pt-cpu-libtorch-cxx11-abi-shared-with-deps-1.10.0+cpu.zip (224M)
```
</td>
<td>

```python
libintel-ext-pt-1.11.0+cpu.run (13.7M)
libintel-ext-pt-cxx11-abi-1.11.0+cpu.run (13.5M)
```
</td>
</tr>
</tbody>
</table>

- Add more optimizations, including more custom operators and fusions.
    - Fuse the QKV linear operators as a single Linear to accelerate the Transformer\*(BERT-\*) encoder part  - [#278](https://github.com/intel/intel-extension-for-pytorch/commit/0f27c269cae0f902973412dc39c9a7aae940e07b).
    - Remove Multi-Head-Attention fusion limitations to support the 64bytes unaligned tensor shape. [#531](https://github.com/intel/intel-extension-for-pytorch/commit/dbb10fedb00c6ead0f5b48252146ae9d005a0fad)
    - Fold the binary operator to Convolution and Linear operator to reduce computation. [#432](https://github.com/intel/intel-extension-for-pytorch/commit/564588561fa5d45b8b63e490336d151ff1fc9cbc) [#438](https://github.com/intel/intel-extension-for-pytorch/commit/b4e7dacf08acd849cecf8d143a11dc4581a3857f) [#602](https://github.com/intel/intel-extension-for-pytorch/commit/74aa21262938b923d3ed1e6929e7d2b629b3ff27)
    - Replace the outplace operators with their corresponding in-place version to reduce memory footprint. The extension currently supports the operators including `sliu`, `sigmoid`, `tanh`, `hardsigmoid`, `hardswish`, `relu6`, `relu`, `selu`, `softmax`. [#524](https://github.com/intel/intel-extension-for-pytorch/commit/38647677e8186a235769ea519f4db65925eca33c)
    - Fuse the Concat + BN + ReLU as a single operator. [#452](https://github.com/intel/intel-extension-for-pytorch/commit/275ff503aea780a6b741f04db5323d9529ee1081)
    - Optimize Conv3D for both imperative and JIT by enabling NHWC and pre-packing the weight. [#425](https://github.com/intel/intel-extension-for-pytorch/commit/ae33faf62bb63b204b0ee63acb8e29e24f6076f3)
- Reduce the binary size. C++ SDK is reduced from ~220MB to ~13.5MB while the wheel packaged is reduced from ~100MB to ~40MB.
- Update oneDNN and oneDNN graph to [2.5.2](https://github.com/oneapi-src/oneDNN/releases/tag/v2.5.2) and [0.4.2](https://github.com/oneapi-src/oneDNN/releases/tag/graph-v0.4.2) respectively.

### What's Changed
**Full Changelog**: https://github.com/intel/intel-extension-for-pytorch/compare/v1.10.100...v1.11.0

## 1.10.100

This release is meant to fix the following issues:
- Resolve the issue that the PyTorch Tensor Expression(TE) did not work after importing the extension.
- Wraps the BactchNorm(BN) as another operator to break the TE's BN-related fusions. Because the BatchNorm performance of PyTorch Tensor Expression can not achieve the same performance as PyTorch ATen BN.
- Update the [documentation](https://intel.github.io/intel-extension-for-pytorch/)
    - Fix the INT8 quantization example issue #205
    - Polish the installation guide

## 1.10.0

The Intel® Extension for PyTorch\* 1.10 is on top of PyTorch 1.10. In this release, we polished the front end APIs. The APIs are more simple, stable, and straightforward now. According to PyTorch community recommendation, we changed the underhood device from `XPU` to `CPU`. With this change, the model and tensor does not need to be converted to the extension device to get performance improvement. It simplifies the model changes.

Besides that, we continuously optimize the Transformer\* and CNN models by fusing more operators and applying NHWC. We measured the 1.10 performance on Torchvison and HugginFace. As expected, 1.10 can speed up the two model zones.

### Highlights

- Change the package name to `intel_extension_for_pytorch` while the original package name is `intel_pytorch_extension`. This change targets to avoid any potential legal issues.

<table align="center">
<tbody>
<tr>
<td>v1.9.0-cpu</td>
<td>v1.10.0-cpu</td>
</tr>
<tr>
<td>

```
import intel_extension_for_pytorch as ipex
```
</td>
<td>

```
import intel_extension_for_pytorch as ipex
```
</td>
</tr>
</tbody>
</table>

- The underhood device is changed from the extension-specific device(`XPU`) to the standard CPU device that aligns with the PyTorch CPU device design, regardless of the dispatch mechanism and operator register mechanism. The means the model does not need to be converted to the extension device explicitly.

<table align="center">
<tbody>
<tr>
<td>v1.9.0-cpu</td>
<td>v1.10.0-cpu</td>
</tr>
<tr>
<td>

```
import torch
import torchvision.models as models

# Import the extension
import intel_extension_for_pytorch as ipex

resnet18 = models.resnet18(pretrained = True)

# Explicitly convert the model to the extension device
resnet18_xpu = resnet18.to(ipex.DEVICE)
```
</td>
<td>

```
import torch
import torchvision.models as models

# Import the extension
import intel_extension_for_pytorch as ipex

resnet18 = models.resnet18(pretrained = True)
```
</td>
</tr>
</tbody>
</table>

- Compared to v1.9.0, v1.10.0 follows PyTorch AMP API(`torch.cpu.amp`) to support auto-mixed-precision. `torch.cpu.amp` provides convenience for auto data type conversion at runtime. Currently, `torch.cpu.amp` only supports `torch.bfloat16`. It is the default lower precision floating point data type when `torch.cpu.amp` is enabled. `torch.cpu.amp` primarily benefits on Intel CPU with BFloat16 instruction set support.

```
import torch
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x):
        return self.conv(x)
```

<table align="center">
<tbody>
<tr>
<td>v1.9.0-cpu</td>
<td>v1.10.0-cpu</td>
</tr>
<tr>
<td>

```
# Import the extension
import intel_extension_for_pytorch as ipex

# Automatically mix precision
ipex.enable_auto_mixed_precision(mixed_dtype = torch.bfloat16)

model = SimpleNet().eval()
x = torch.rand(64, 64, 224, 224)
with torch.no_grad():
    model = torch.jit.trace(model, x)
    model = torch.jit.freeze(model)
    y = model(x)
```
</td>
<td>

```
# Import the extension
import intel_extension_for_pytorch as ipex

model = SimpleNet().eval()
x = torch.rand(64, 64, 224, 224)
with torch.cpu.amp.autocast(), torch.no_grad():
    model = torch.jit.trace(model, x)
    model = torch.jit.freeze(model)
    y = model(x)
```
</td>
</tr>
</tbody>
</table>

- The 1.10 release provides the INT8 calibration as an experimental feature while it only supports post-training static quantization now. Compared to 1.9.0, the fronted APIs for quantization is more straightforward and ease-of-use.

```
import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(10, 10, 3)

    def forward(self, x):
        x = self.conv(x)
        return x

model = MyModel().eval()

# user dataset for calibration.
xx_c = [torch.randn(1, 10, 28, 28) for i in range(2))
# user dataset for validation.
xx_v = [torch.randn(1, 10, 28, 28) for i in range(20))
```
  - Clibration
<table align="center">
<tbody>
<tr>
<td>v1.9.0-cpu</td>
<td>v1.10.0-cpu</td>
</tr>
<tr>
<td>

```
# Import the extension
import intel_extension_for_pytorch as ipex

# Convert the model to the Extension device
model = Model().to(ipex.DEVICE)

# Create a configuration file to save quantization parameters.
conf = ipex.AmpConf(torch.int8)
with torch.no_grad():
    for x in xx_c:
        # Run the model under calibration mode to collect quantization parameters
        with ipex.AutoMixPrecision(conf, running_mode='calibration'):
            y = model(x.to(ipex.DEVICE))
# Save the configuration file
conf.save('configure.json')
```
</td>
<td>

```
# Import the extension
import intel_extension_for_pytorch as ipex

conf = ipex.quantization.QuantConf(qscheme=torch.per_tensor_affine)
with torch.no_grad():
    for x in xx_c:
        with ipex.quantization.calibrate(conf):
            y = model(x)

conf.save('configure.json')
```
</td>
</tr>
</tbody>
</table>

 - Inference
 <table align="center">
<tbody>
<tr>
<td>v1.9.0-cpu</td>
<td>v1.10.0-cpu</td>
</tr>
<tr>
<td>

```
# Import the extension
import intel_extension_for_pytorch as ipex

# Convert the model to the Extension device
model = Model().to(ipex.DEVICE)
conf = ipex.AmpConf(torch.int8, 'configure.json')
with torch.no_grad():
    for x in cali_dataset:
        with ipex.AutoMixPrecision(conf, running_mode='inference'):
            y = model(x.to(ipex.DEVICE))
```
</td>
<td>

```
# Import the extension
import intel_extension_for_pytorch as ipex

conf = ipex.quantization.QuantConf('configure.json')

with torch.no_grad():
    trace_model = ipex.quantization.convert(model, conf, example_input)
    for x in xx_v:
        y = trace_model(x)
```
</td>
</tr>
</tbody>
</table>


- This release introduces the `optimize` API at python front end to optimize the model and optimizer for training. The new API both supports FP32 and BF16, inference and training.

- Runtime Extension (Experimental) provides a runtime CPU pool API to bind threads to cores. It also features async tasks. **Note**: Intel® Extension for PyTorch\* Runtime extension is still in the **experimental** stage. The API is subject to change. More detailed descriptions are available in the extension documentation.

### Known Issues

- `omp_set_num_threads` function failed to change OpenMP threads number of oneDNN operators if it was set before.

  `omp_set_num_threads` function is provided in Intel® Extension for PyTorch\* to change the number of threads used with OpenMP. However, it failed to change the number of OpenMP threads if it was set before.

  pseudo-code:

  ```
  omp_set_num_threads(6)
  model_execution()
  omp_set_num_threads(4)
  same_model_execution_again()
  ```

  **Reason:** oneDNN primitive descriptor stores the omp number of threads. Current oneDNN integration caches the primitive descriptor in IPEX. So if we use runtime extension with oneDNN based pytorch/ipex operation, the runtime extension fails to change the used omp number of threads.

- Low performance with INT8 support for dynamic shapes

  The support for dynamic shapes in Intel® Extension for PyTorch\* INT8 integration is still work in progress. When the input shapes are dynamic, for example, inputs of variable image sizes in an object detection task or of variable sequence lengths in NLP tasks, the Intel® Extension for PyTorch\* INT8 path may slow down the model inference. In this case, use stock PyTorch INT8 functionality.

- Low throughput with DLRM FP32 Train

  A 'Sparse Add' [PR](https://github.com/pytorch/pytorch/pull/23057) is pending review. The issue will be fixed when the PR is merged.

### What's Changed
**Full Changelog**: https://github.com/intel/intel-extension-for-pytorch/compare/v1.9.0...v1.10.0+cpu-rc3

## 1.9.0

### What's New

* Rebased the Intel Extension for Pytorch from PyTorch-1.8.0 to the official PyTorch-1.9.0 release.
* Support binary installation.

  `python -m pip install torch_ipex==1.9.0 -f https://software.intel.com/ipex-whl-stable`
* Support the C++ library. The third party App can link the Intel-Extension-for-PyTorch C++ library to enable the particular optimizations.

## 1.8.0

### What's New

* Rebased the Intel Extension for Pytorch from Pytorch -1.7.0 to the official Pytorch-1.8.0 release. The new XPU device type has been added into Pytorch-1.8.0(49786), don’t need to patch PyTorch to enable Intel Extension for Pytorch anymore
* Upgraded the oneDNN from v1.5-rc to v1.8.1
* Updated the README file to add the sections to introduce supported customized operators, supported fusion patterns, tutorials, and joint blogs with stakeholders

## 1.2.0

### What's New

* We rebased the Intel Extension for pytorch from Pytorch -1.5rc3 to the official Pytorch-1.7.0 release. It will have performance improvement with the new Pytorch-1.7 support.
* Device name was changed from DPCPP to XPU.

  We changed the device name from DPCPP to XPU to align with the future Intel GPU product for heterogeneous computation.
* Enabled the launcher for end users.
* We enabled the launch script that helps users launch the program for training and inference, then automatically setup the strategy for multi-thread, multi-instance, and memory allocator. Refer to the launch script comments for more details.

### Performance Improvement

* This upgrade provides better INT8 optimization with refined auto mixed-precision API.
* More operators are optimized for the int8 inference and bfp16 training of some key workloads, like MaskRCNN, SSD-ResNet34, DLRM, RNNT.

### Others

* Bug fixes
  * This upgrade fixes the issue that saving the model trained by Intel extension for PyTorch caused errors.
  * This upgrade fixes the issue that Intel extension for PyTorch was slower than pytorch proper for Tacotron2.
* New custom operators

  This upgrade adds several custom operators: ROIAlign, RNN, FrozenBatchNorm, nms.
* Optimized operators/fusion

  This upgrade optimizes several operators: tanh, log_softmax, upsample, and embeddingbad and enables int8 linear fusion.
* Performance

  The release has daily automated testing for the supported models: ResNet50, ResNext101, Huggingface Bert, DLRM, Resnext3d, MaskRNN, SSD-ResNet34. With the extension imported, it can bring up to 2x INT8 over FP32 inference performance improvements on the 3rd Gen Intel Xeon scalable processors (formerly codename Cooper Lake).

### Known issues

* Multi-node training still encounter hang issues after several iterations. The fix will be included in the next official release.

## 1.1.0

### What's New

* Added optimization for training with FP32 data type & BF16 data type. All the optimized FP32/BF16 backward operators include:
  * Conv2d
  * Relu
  * Gelu
  * Linear
  * Pooling
  * BatchNorm
  * LayerNorm
  * Cat
  * Softmax
  * Sigmoid
  * Split
  * Embedding_bag
  * Interaction
  * MLP
* More fusion patterns are supported and validated in the release, see table:

  |Fusion Patterns|Release|
  |--|--|
  |Conv + Sum|v1.0|
  |Conv + BN|v1.0|
  |Conv + Relu|v1.0|
  |Linear + Relu|v1.0|
  |Conv + Eltwise|v1.1|
  |Linear + Gelu|v1.1|

* Add docker support
* [Alpha] Multi-node training with oneCCL support.
* [Alpha] INT8 inference optimization.

### Performance

* The release has daily automated testing for the supported models: ResNet50, ResNext101, [Huggingface Bert](https://github.com/huggingface/transformers), [DLRM](https://github.com/intel/optimized-models/tree/master/pytorch/dlrm), [Resnext3d](https://github.com/XiaobingSuper/Resnext3d-for-video-classification), [Transformer](https://github.com/pytorch/fairseq/blob/master/fairseq/models/transformer.py). With the extension imported, it can bring up to 1.2x~1.7x BF16 over FP32 training performance improvements on the 3rd Gen Intel Xeon scalable processors (formerly codename Cooper Lake).

### Known issue

* Some workloads may crash after several iterations on the extension with [jemalloc](https://github.com/jemalloc/jemalloc) enabled.

## 1.0.2

* Rebase torch CCL patch to PyTorch 1.5.0-rc3

## 1.0.1-Alpha

* Static link oneDNN library
* Check AVX512 build option
* Fix the issue that cannot normally invoke `enable_auto_optimization`

## 1.0.0-Alpha

### What's New

* Auto Operator Optimization

  Intel Extension for PyTorch will automatically optimize the operators of PyTorch when importing its python package. It will significantly improve the computation performance if the input tensor and the model is converted to the extension device.

* Auto Mixed Precision
  Currently, the extension has supported bfloat16. It streamlines the work to enable a bfloat16 model. The feature is controlled by `enable_auto_mix_precision`. If you enable it, the extension will run the operator with bfloat16 automatically to accelerate the operator computation.

### Performance Result

We collected the performance data of some models on the Intel Cooper Lake platform with 1 socket and 28 cores. Intel Cooper Lake introduced AVX512 BF16 instructions that could improve the bfloat16 computation significantly. The detail is as follows (The data is the speedup ratio and the baseline is upstream PyTorch).

||Imperative - Operator Injection|Imperative - Mixed Precision|JIT- Operator Injection|JIT - Mixed Precision|
|:--:|:--:|:--:|:--:|:--:|
|RN50|2.68|5.01|5.14|9.66|
|ResNet3D|3.00|4.67|5.19|8.39|
|BERT-LARGE|0.99|1.40|N/A|N/A|

We also measured the performance of ResNeXt101, Transformer-FB, DLRM, and YOLOv3 with the extension. We observed that the performance could be significantly improved by the extension as expected.

### Known issue

* [#10](https://github.com/intel/intel-extension-for-pytorch/issues/10) All data types have not been registered for DPCPP
* [#37](https://github.com/intel/intel-extension-for-pytorch/issues/37) MaxPool can't get nan result when input's value is nan

### NOTE

The extension supported PyTorch v1.5.0-rc3. Support for other PyTorch versions is working in progress.
