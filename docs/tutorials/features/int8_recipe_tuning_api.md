INT8 Recipe Tuning API (Experimental)
=====================================

This [new API](../api_doc.html#ipex.quantization.autotune) `ipex.quantization.autotune` supports INT8 recipe tuning by using Intel® Neural Compressor as the backend in Intel® Extension for PyTorch\*. In general, we provid default recipe in Intel® Extension for PyTorch\*, and we still recommend users to try out the default recipe first without bothering tuning. If the default recipe doesn't bring about desired accuracy, users can use this API to tune for a more advanced receipe.

Users need to provide a prepared model and some parameters required for tuning. The API will return a tuned model with advanced recipe.

### Usage Example

```python
model = torchvision.models.resnet50(pretrained=True)
model.eval()
data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

# prepare model, do conv+bn folding, and init model quant_state.
qconfig = ipex.quantization.default_static_qconfig
data = torch.randn(1, 3, 224, 224)
prepared_model = ipex.quantization.prepare(model, qconfig, example_inputs=data, inplace=False)

######################## recipe tuning with INC ########################
def eval(prepared_model):
    # return accuracy value
    return evaluate(prepared_model, data_loader)
tuned_model = ipex.quantization.autotune(prepared_model, data_loader, eval, sampling_size=[100], 
        accuracy_criterion={'relative': 0.01}, tuning_time=0)
########################################################################

# run tuned model
convert_model = ipex.quantization.convert(tuned_model)
with torch.no_grad():
    traced_model = torch.jit.trace(convert_model, data)
    traced_model = torch.jit.freeze(traced_model)
    traced_model(data)

# save tuned qconfig file
tuned_model.save_qconf_summary(qconf_summary = "tuned_conf.json")
```
