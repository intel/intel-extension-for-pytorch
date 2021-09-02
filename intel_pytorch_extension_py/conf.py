import os
import json
import torch
import _torch_ipex as core

qscheme_dict ={torch.per_tensor_affine:0,
               torch.per_channel_affine:1,
               torch.per_tensor_symmetric:2,
               torch.per_channel_symmetric:3,
               torch.torch.per_channel_affine_float_qparams:4}

class QuantConf(object):
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
