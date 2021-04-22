import os
import json
import torch
import _torch_ipex as core

class AmpConf(object):
    def __init__(self, mixed_dtype = torch.bfloat16, configure_file = None):
        self.dtype = mixed_dtype
        self.configure_file = configure_file

        if self.dtype == torch.int8:
            core.clear_indicators()
        # for int8 path, if user give a exited configure file, load it.
        if self.configure_file != None and self.dtype == torch.int8:
            if os.path.exists(self.configure_file) and os.stat(self.configure_file).st_size != 0:
                with open(self.configure_file, 'r') as f:
                    configures = json.load(f)
                    core.load_indicators_file(configures)
            else:
                assert False, 'Can not load a empty file or none existed file, plese first do calibartion step'

    # for int8 quantization, will save the date after doing calibration step.
    def save(self, configure_file, default_recipe=True):
        core.add_indicators()
        configures = core.get_int8_configures()
        if default_recipe:
            configures = self.get_default_recipe(configures)
        with open(configure_file, 'w') as fp:
            json.dump(configures, fp, indent = 4)

    def get_default_recipe(self, configures):
        elt_wise = ['relu', 'sigmoid']
        inplace_ops = ['relu_', 'add_']
        # get default recipe,
        # q+dq+conv+q+dq+relu => q+dq+conv+relu
        # q+dq+op1+q+dq+q+dq+op2+q+dq => q+dq+op1+q+dq+op2+q+dq
        default_configures = configures
        num_ops = len(default_configures)
        add_ids = []
        for cur_id in range(num_ops):
            cur_op = default_configures[cur_id]['name']
            if cur_op == 'dropout':
                continue
            if cur_op == 'add':
                add_ids.append(cur_id)
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
            # conv            op        conv         op
            #    \            /          \           /
            #     q          q            \         q
            #      \        /      =>      \       /
            #       dq     dq               \     dq
            #         \   /                  \   /
            #          add                    add
            if len(pre_ops) > 0:
                for key, value in pre_ops.items():
                    if value == 'conv2d' or value == 'conv3d':
                        default_configures[cur_id]['inputs_quantized'][key] = False
                        break

            # if add pre_op hasn't conv, not need add q, dq for accuracy.
            pre_inputs = pre_ops.values()
            if cur_op == 'add' and ('conv2d' not in pre_inputs and 'conv3d' not in pre_inputs):
                default_configures[cur_id]['inputs_quantized'][0] = False
                default_configures[cur_id]['inputs_quantized'][1] = False

        for add_id in add_ids:
            # if add hasn't post quantized op, i.e. 'outputs_quantized' is true.
            # not need add q, dq for output.
            if default_configures[add_id]['outputs_quantized'][0]:
                default_configures[add_id]['outputs_quantized'][0] = False

        return default_configures
