import torch
import functools
import warnings
import copy
import numpy as np
import intel_extension_for_pytorch._C as core
from .. import nn

def _get_default_recipe(configures):
    # For int8 quantization, will save the date after doing calibration step,
    # if default_recipe is True, it will make some progress the data, such as
    # remove redundant quantizer between two ops.
    elt_wise = ['relu', 'sigmoid', 'gelu']
    inplace_ops = ['relu_', 'add_']
    shape_ops = ['flatten']
    # get default recipe,
    # q+dq+conv+q+dq+relu => q+dq+conv+relu
    # q+dq+op1+q+dq+q+dq+op2+q+dq => q+dq+op1+q+dq+op2+q+dq
    default_configures = configures
    num_ops = len(default_configures)
    for cur_id in range(num_ops):
        cur_op = default_configures[cur_id]['name']
        if cur_op == 'dropout':
            continue
        inputs = default_configures[cur_id]['inputs_flow']
        num_input = len(inputs)
        pre_ops = {}
        for i_num in range(num_input):
            inp = inputs[i_num]
            for pre_id in range(cur_id):
                pre_op = default_configures[pre_id]['name']
                pre_out = default_configures[pre_id]['outputs_flow']
                num_out= len(pre_out)
                for o_num in range(num_out):
                    # pre_op+qu+dequ+qu+dequ+cur_op+qu+dequ -> pre_op+qu+dequ+cur_op+qu+dequ.
                    # for relu, sigmoid or other elt_wise ops, id pre_op is conv, linear, then
                    # remove qu+dequ between them for fusion: pre_op+cur_op+qu_dequ.
                    if pre_out[o_num] == inp:
                        if (cur_op not in inplace_ops) \
                                or (cur_op in inplace_ops and \
                                    (pre_op == 'conv2d' or pre_op == 'conv3d' or pre_op == 'linear')):
                            if pre_op not in inplace_ops and pre_op != 'dropout':
                                default_configures[pre_id]['outputs_quantized'][o_num] = False
                        if cur_op in elt_wise \
                                and (pre_op == 'conv2d' or pre_op == 'conv3d' or pre_op == 'linear' or pre_op == 'add'):
                            default_configures[cur_id]['inputs_quantized'][i_num] = False
                        if cur_op == 'add':
                            pre_ops[i_num] = pre_op
                        if cur_op in shape_ops:
                            # for pooling case, the input and output always has same scale and zero point,
                            # if the pooling's post ops is flatten, need sync flatten's input and output's
                            # scale and zero point to pooling.
                            if pre_op in ['max_pool2d', 'adaptive_avg_pool2d']:
                                default_configures[cur_id]['input_scales'][i_num] = default_configures[pre_id]['output_scales'][o_num]
                                default_configures[cur_id]['input_zero_points'][i_num] = default_configures[pre_id]['output_zero_points'][o_num]
                                default_configures[cur_id]['output_scales'][i_num] = default_configures[pre_id]['output_scales'][o_num]
                                default_configures[cur_id]['output_zero_points'][i_num] = default_configures[pre_id]['output_zero_points'][o_num]
                        if pre_op in shape_ops:
                            # if pre op is flatten, sync the input's scale and zero point to flatten.
                            default_configures[cur_id]['input_scales'][i_num] = default_configures[pre_id]['output_scales'][o_num]
                            default_configures[cur_id]['input_zero_points'][i_num] = default_configures[pre_id]['output_zero_points'][o_num]
        # conv            op        conv         op
        #    \            /          \           /
        #     q          q            \         q
        #      \        /      =>      \       /
        #       dq     dq               \     dq
        #         \   /                  \   /
        #          add                    add
        if len(pre_ops) > 0:
            for key, value in pre_ops.items():
                if value == 'conv2d' or value == 'conv3d' or value == 'linear':
                    default_configures[cur_id]['inputs_quantized'][key] = False
                    break
        # if add pre_op hasn't conv and linear, not need add q, dq for accuracy.
        pre_inputs = pre_ops.values()
        if cur_op == 'add' and \
                ('conv2d' not in pre_inputs and 'conv3d' not in pre_inputs and 'linear' not in pre_inputs):
            default_configures[cur_id]['inputs_quantized'][0] = False
            default_configures[cur_id]['inputs_quantized'][1] = False
    # post process for add, linear, if cur op hasn't post quantized op, i.e. 'outputs_quantized' is True,
    # for good perfromance, the default recipe:
    # int8_input -> op -> q -> dq will converted to int8_input -> op.
    ops_remove_q_dq_after = ['add', 'linear', 'conv2d', 'matmul']
    # post process for flatten, if flatten's pre-pop and post op are fp32 op, don't need add q and dq
    # before and after it.
    ops_remove_q_dq_before_after = ['flatten']
    for cur_id in range(num_ops):
        cur_op = default_configures[cur_id]['name']
        if cur_op in ops_remove_q_dq_after and default_configures[cur_id]['outputs_quantized'][0]:
            default_configures[cur_id]['outputs_quantized'][0] = False
        if cur_op in ops_remove_q_dq_before_after and default_configures[cur_id]['inputs_quantized'][0] \
                and default_configures[cur_id]['outputs_quantized'][0]:
            default_configures[cur_id]['inputs_quantized'][0] = False
            default_configures[cur_id]['outputs_quantized'][0] = False
    return default_configures

class calibrate(object):
    r"""

    Enable quantization  calibration scope which will collect the scales and zero
    points of ops to be quantized, such as convolution, linear.

    Args:
        conf (quantization.QuantConf): Quantization's setting.
        default_recipe (bool): Whether produce a default Quantization's setting,
            which will do a post-process to remove reduent quantizer, which can
            be True of False. For example, conv+relu, quantizers are inserted
            before and after conv and relu, the data flow will be like: quant->
            dequant->conv->quant->dequant->quant->dequant->relu->quant->dequant,
            if default_recipe is True, the data flow will be converted to: quant
            ->dequant->conv->relu->quant->dequant, the quantizers will not be
            inserted after conv's output and before relu's input. The deault
            value is ``True``.

    """
    def __init__(self, conf, default_recipe=True):
        self.conf = conf
        self.default_recipe = default_recipe

    def __enter__(self):
        self.prev = torch.is_autocast_cpu_enabled()
        self.pre_quantization_state = core.is_quantization_enabled()
        self.pre_calibration_state = core.get_int8_calibration()

        torch.set_autocast_cpu_enabled(True)
        core.set_quantization_enabled(True)
        core.enable_int8_calibration()
        core.autocast_increment_nesting()

    def __exit__(self, *args):
        # Drop the cache when we exit to a nesting level that's outside any instance of autocast.
        if core.autocast_decrement_nesting() == 0:
            core.clear_autocast_cache_int8()
        torch.set_autocast_cpu_enabled(self.prev)
        core.set_quantization_enabled(self.pre_quantization_state)
        core.calibration_reset()
        if self.pre_calibration_state:
            core.enable_int8_calibration()
        else:
            core.disable_int8_calibration()

        # compute the scales and zero_points.
        core.add_indicators()
        if self.default_recipe:
            # get default recipe
            configures = core.get_int8_configures()
            configures = _get_default_recipe(configures)
            core.clear_indicators()
            core.load_indicators_file(configures)
        return False

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_autocast(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return decorate_autocast

class _quantization_int8(object):
    def __enter__(self):
        self.prev = torch.is_autocast_cpu_enabled()
        self.pre_quantization_state = core.is_quantization_enabled()
        self.pre_calibration_state = core.get_int8_calibration()

        torch.set_autocast_cpu_enabled(True)
        core.set_quantization_enabled(True)
        core.disable_int8_calibration()

    def __exit__(self, *args):
        torch.set_autocast_cpu_enabled(self.prev)
        core.set_quantization_enabled(self.pre_quantization_state)
        if self.pre_calibration_state:
            core.enable_int8_calibration()
        else:
            core.disable_int8_calibration()
        return False

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_autocast(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return decorate_autocast

def convert(model, conf, inputs, inplace=False):
    r"""
    Convert an FP32 torch.nn.Module model to a quantized JIT ScriptModule
    according to the given quantization recipes in the quantization
    configuration conf.

    The function will conduct a JIT trace with the given inputs. It will fail
    if the given model doesn't support JIT trace.

    Args:
        model (torch.nn.Module): The FP32 model to be convert.
        conf (quantization.QuantConf): Quantization's setting.
        inputs (tuple or torch.Tensor): A tuple of example inputs that are used
            for JIT trace of the given model.
        inplace (bool): Whether or not to do inplace model convert.

    Returns:
        torch.jit.ScriptModule

    """

    assert isinstance(model, torch.nn.Module), "Only support nn.Module convert for quantization path"

    if inplace:
        model_ = model
    else:
        model_ = copy.deepcopy(model)
 
    # pre-conver model's parameters dtype if it has conv, linear
    # and Embedding for bfloat16 path.
    if torch.is_autocast_cpu_enabled() and core.get_autocast_dtype() == torch.bfloat16:
        model_ = nn.utils._model_convert.convert_module_data_type(model_, torch.bfloat16)

    with torch.no_grad(), _quantization_int8():
        trace_model = torch.jit.trace(model_, inputs, check_trace=False)
    trace_model = torch.jit.freeze(trace_model)

    return trace_model

