import os
import json
import torch
import intel_extension_for_pytorch._C as core

qscheme_dict ={torch.per_tensor_affine:0,
               torch.per_channel_affine:1,
               torch.per_tensor_symmetric:2,
               torch.per_channel_symmetric:3,
               torch.torch.per_channel_affine_float_qparams:4}

class QuantConf(object):
    r"""
    Configure setting for INT8 quantization flow.

    Args:
        configure_file (string): The INT8 configure file(.json file) to be
            loaded or saved.
        qscheme (torch.qscheme): quantization scheme to be used(activation)

    Available configurations in the *configure_file* are:

        * id (int): The number of quantized ops in the model running flow.  Note: only limited ops are reordered, such as convolution, linear or other ops.
        * name (string): Quantized OP's name.
        * algorithm (string): observe method for activation tensors during calibration. Only support min-max now, more methods will be support in future.
        * weight_granularity (Qscheme): Qscheme for weight quantizer for convolution and linear, can be per_channel or per_tesor, user can manually set it before load existed configure file. The default value is uint8.
        * input_scales: Scales for inputs.
        * input_zero_points: Zero points for inputs.
        * output_scales": Scales for outputs.
        * output_zero_points: Zero points for outputs.
        * weight_scales: Scales for Weights.
        * input_quantized_dtypes: Quantized dtypes fot inputs, can be uint8 or int8, user can manually set it before load existed configure file.  The default value is uint8.
        * output_quantized_dtypes: Quantized dtypes fot ouputs, can be uint8 or int8, user can manually set it before load existed configure file.  The default value is uint8.
        * inputs_quantized: Whether inputs need quantized, can be true or false, user can manually set it before load existed configure file.
        * outputs_quantized: Whether output need quantized, can be true or false, user can manually set it before load existed configure file.
        * inputs_flow: Where the inputs are from, beacuse we only record limited ops, we can know which ops are adjacent by compare one inputs flow with others' output flow.
        * outputs_flow: Outputs flag for current op, which can be used to check which ops are adjacent.

    .. warning::

        :attr:`qscheme` can only take one of the following options:

        - ``torch.per_tensor_affine``
        - ``torch.per_tensor_symmetric``

    .. note::

        The loaded or saved json file will be has the content like:

        | [
        |     {
        |         "id": 0,
        |         "name": "conv2d",
        |         "algorithm": "min_max",
        |         "weight_granularity": "per_channel",
        |         "input_scales": [
        |             0.01865844801068306
        |         ],
        |         "input_zero_points": [
        |             114
        |         ],
        |         "output_scales": [
        |             0.05267734453082085
        |         ],
        |         "output_zero_points": [
        |             132
        |         ],
        |         "weight_scales": [
        |             [
        |                 0.0006843071314506233,
        |                 0.0005326663958840072,
        |                 0.00016389577649533749,
        |             ]
        |         ],
        |         "input_quantized_dtypes": [
        |             "uint8"
        |         ],
        |         "output_quantized_dtypes": [
        |             "uint8"
        |         ],
        |         "inputs_quantized": [
        |             true
        |         ],
        |         "outputs_quantized": [
        |             false
        |         ],
        |         "inputs_flow": [
        |             "conv2d.0.input"
        |         ],
        |         "outputs_flow": [
        |             "conv2d.0.output"
        |        ]
        |     }
        | ]

    """

    def __init__(self, configure_file=None, qscheme=torch.per_tensor_affine):
        self.configure_file = configure_file

        core.clear_indicators()
        assert qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric], \
            "qscheme is only support torch.per_tensor_affine and torch.per_tensor_symmetric now"
        core.set_int8_qscheme(qscheme_dict[qscheme])

        # if user provides an existing configuration file, load it
        if self.configure_file != None:
            if os.path.exists(self.configure_file) and os.stat(self.configure_file).st_size != 0:
                with open(self.configure_file, 'r') as f:
                    configures = json.load(f)
                    core.load_indicators_file(configures)
            else:
                assert False, 'Can not load a empty file or none existed file, plese first do calibartion step'

    def save(self, configure_file):
        configures = core.get_int8_configures()
        with open(configure_file, 'w') as fp:
            json.dump(configures, fp, indent = 4)
        # clear indicators after saved
        core.clear_indicators()
