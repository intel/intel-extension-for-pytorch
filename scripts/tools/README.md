# Introduction of Cosim Tool

## What Does This Tool Use For

This is a tool for comparing target model with a cosim model layer by layer to see if any error occurred unexpectedly. The target model may be customized with user's device and dtype. And the cosim model will has the same structure as target model but runs on CPU as default device and has float32 as default dtype. For each layer in given model, target model and cosim model will process on same inputs and calculate for independent outputs. Then they compare each output to see if its error are toleranted.

## Install

Before use this tool, you need to install it. Just change your path to this directory and run:

```python
python setup.py install
```

After successful installation, you may have the ability to `import cosim` in current Python environment.

## Use Case

To enable this tool, you need to modify your model script. See below code as an example:

```python
# your imports here
import torch
import intel_extension_for_pytorch
# necessary cosim import here
import cosim

# your model definition
class MyModel(...):
  ...

# your inputs and other configures
inputs = ...

# your original model instance
model = MyModel()
# wrap your model with CosimModule
model = cosim.CosimModule(model)

# run forward and backward as normal
outputs = model(inputs)
loss = your_loss_function(outputs)
loss.backward()

# plot cosim results to pdf if needed
model.plot_result(file='cosim_outputs/#/')
```

In a word, to use cosim tool, you need to import cosim module and wrap your original model with `cosim.CosimModule`. Other code in model script can be remained as normal.

## Known Issues

* Current cosim tool cannot support inplace operations such as `torch.nn.ReLU(inplace=True)`. Please manually correct these operations.

* Current cosim tool cannot support any graph fusion or folding as well as jit-script. Please wait for further developing.

* Current cosim tool cannot support customizing parameters, such as device, dtype and tolerance, of cosim model. If needed, you must modify the correlated fields in file `cosim.py` by yourself. Or you can wait for further developing.

* Some machines complain a memory lack issue. The optimize for this issue is in progress.

## Author and Contact

Xunsong, Huang (xunsong.huang@intel.com)
