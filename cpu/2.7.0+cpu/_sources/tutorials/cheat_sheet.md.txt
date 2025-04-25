Cheat Sheet
===========

Get started with Intel® Extension for PyTorch\* using the following commands:

|Description    | Command |
| -------- | ------- |
| Basic CPU Installation | `python -m pip install intel_extension_for_pytorch`    |
| Import Intel® Extension for PyTorch\*    | `import intel_extension_for_pytorch as ipex`|
| Capture a Verbose Log (Command Prompt)    | `export ONEDNN_VERBOSE=1`   |
| Optimization During Training   | `model = ...`<br>`optimizer = ...`<br>`model.train()`<br>`model, optimizer = ipex.optimize(model, optimizer=optimizer)`|
| Optimization During Inference  | `model = ...`<br>`model.eval()`<br>`model = ipex.optimize(model)`   |
| Optimization Using the Low-Precision Data Type bfloat16 <br>During Training (Default FP32) | `model = ...`<br>`optimizer = ...`<br>`model.train()`<br/><br/>`model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)`<br/><br/>`with torch.no_grad():`<br>`    with torch.cpu.amp.autocast():`<br>`        model(data)`   |
| Optimization Using the Low-Precision Data Type bfloat16 <br>During Inference (Default FP32) | `model = ...`<br>`model.eval()`<br/><br/>`model = ipex.optimize(model, dtype=torch.bfloat16)`<br/><br/>`with torch.cpu.amp.autocast():`<br>`    model(data)`
| [Prototype] Fast BERT Optimization | `from transformers import BertModel`<br>`model = BertModel.from_pretrained("bert-base-uncased")`<br>`model.eval()`<br/><br/>`model = ipex.fast_bert(model, dtype=torch.bfloat16)`|
| Run CPU Launch Script (Command Prompt): <br>Automate Configuration Settings for Performance | `ipexrun [knobs] <your_pytorch_script> [args]`|
| [Prototype] Run HyperTune to perform hyperparameter/execution configuration search | `python -m intel_extension_for_pytorch.cpu.hypertune --conf-file <your_conf_file> <your_python_script> [args]`|
| [Prototype] Enable Graph capture | `model = …`<br>`model.eval()`<br>`model = ipex.optimize(model, graph_mode=True)`|
| Post-Training INT8 Quantization (Static)  | `model = …`<br>`model.eval()`<br>`data = …`<br/><br/>`qconfig = ipex.quantization.default_static_qconfig`<br/><br/>`prepared_model = ipex.quantization.prepare(model, qconfig, example_inputs=data, anyplace=False)`<br/><br/>`for d in calibration_data_loader():`<br>`  prepared_model(d)`<br/><br/>`converted_model = ipex.quantization.convert(prepared_model)`|
| Post-Training INT8 Quantization (Dynamic) | `model = …`<br>`model.eval()`<br>`data = …`<br/><br/>`qconfig = ipex.quantization.default_dynamic_qconfig`<br/><br/>`prepared_model = ipex.quantization.prepare(model, qconfig, example_inputs=data)`<br/><br/>`converted_model = ipex.quantization.convert(prepared_model)` |
| [Prototype] Post-Training INT8 Quantization (Tuning Recipe): | `model = …`<br>`model.eval()`<br>`data = …`<br/><br/>`qconfig = ipex.quantization.default_static_qconfig`<br/><br/>`prepared_model = ipex.quantization.prepare(model, qconfig, example_inputs=data, inplace=False)`<br/><br/>`tuned_model = ipex.quantization.autotune(prepared_model, calibration_data_loader, eval_function, sampling_sizes=[100],`<br>`    accuracy_criterion={'relative': .01}, tuning_time=0)`<br/><br/>`convert_model = ipex.quantization.convert(tuned_model)`|
