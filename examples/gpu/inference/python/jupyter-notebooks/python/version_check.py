#import importlib
from importlib import util
tensorflow_found = util.find_spec("tensorflow") is not None
pytorch_found = util.find_spec("torch") is not None
pytorch_ext_found = util.find_spec("intel_extension_for_pytorch") is not None
tensorflow_ext_found = util.find_spec("intel_extension_for_tensorflow") is not None
xgboost_found = util.find_spec("xgboost") is not None
sklearn_found = util.find_spec("sklearn") is not None
sklearnex_found = util.find_spec("sklearnex") is not None
inc_found = util.find_spec("neural_compressor") is not None
modin_found = util.find_spec("modin") is not None
torchccl_found = util.find_spec("oneccl_bindings_for_pytorch") is not None
dpctl_found = util.find_spec("dpctl") is not None
numba_dpex_found = util.find_spec("numba_dpex") is not None
dpnp_found = util.find_spec("dpnp") is not None

import warnings
warnings.filterwarnings('ignore')

class arch_checker:

    def __init__(self):
        cpuinfo_found = util.find_spec("cpuinfo") is not None
        if cpuinfo_found == False:
            self.arch = 'None'
            print("please install py-cpuinfo")
            return
        from cpuinfo import get_cpu_info
        info = get_cpu_info()
        flags = info['flags']
        arch_list = ['SPR', 'CPX',"ICX|CLX", "SKX", "BDW|CORE|ATOM"]
        isa_list = [['amx_bf16', 'amx_int8', 'amx_tile'],['avx512_bf16'],['avx512_vnni'],['avx512'],['avx2']]
        index = len(arch_list) - 1
        for flag in flags:
            for idx, isa_sublist in enumerate(isa_list):
                for isa in isa_sublist:
                    if isa in flag:
                        if idx < index:
                            index = idx
        self.arch = arch_list[index]
        return

if tensorflow_found == True:

    import tensorflow as tf

    import os

    def get_mkl_enabled_flag():

        mkl_enabled = False
        major_version = int(tf.__version__.split(".")[0])
        minor_version = int(tf.__version__.split(".")[1])
        if major_version >= 2:
            onednn_enabled = 0
            if minor_version < 5:
                from tensorflow.python import _pywrap_util_port
            else:
                from tensorflow.python.util import _pywrap_util_port
                onednn_enabled = int(os.environ.get('TF_ENABLE_ONEDNN_OPTS', '0'))
            mkl_enabled = _pywrap_util_port.IsMklEnabled() or (onednn_enabled == 1)
        else:
            mkl_enabled = tf.pywrap_tensorflow.IsMklEnabled()
        return mkl_enabled

    print ("TensorTlow version: ", tf.__version__)
    print("MKL enabled :", get_mkl_enabled_flag())
    if tensorflow_ext_found == True:
        import intel_extension_for_tensorflow as itex
        print("itex_version : ", itex.__version__)

if pytorch_found == True:
    import torch
    print("PyTorch Version: ", torch.__version__)
    mkldnn_enabled = torch.backends.mkldnn.is_available()
    mkl_enabled = torch.backends.mkl.is_available()
    openmp_enabled = torch.backends.openmp.is_available()
    print('mkldnn : {0},  mkl : {1}, openmp : {2}'.format(mkldnn_enabled, mkl_enabled, openmp_enabled))
    print(torch.__config__.show())

    if pytorch_ext_found == True:
        import intel_extension_for_pytorch as ipex
        print("ipex_verion : ",ipex.__version__)

if xgboost_found == True:
    import xgboost as xgb
    print("XGBoost Version: ", xgb.__version__)

if modin_found == True:
    import modin
    import modin.config as cfg
    major_version = int(modin.__version__.split(".")[0])
    minor_version = int(modin.__version__.split(".")[1])
    print("Modin Version: ", modin.__version__)
    cfg_engine = ''
    if minor_version > 12 and major_version == 0:
        cfg_engine = cfg.StorageFormat.get()

    else:
        cfg_engine = cfg.Engine.get()
    print("Modin Engine: ", cfg_engine)

if sklearn_found == True:
    import sklearn
    print("scikit learn Version: ", sklearn.__version__)
    if sklearnex_found == True:
        import sklearnex
        print("have scikit learn ext 2021.4 : ", sklearnex._utils.get_sklearnex_version((2021, 'P', 400)))

if inc_found == True:
    import neural_compressor as inc
    print("neural_compressor version {}".format(inc.__version__))

if torchccl_found == True:
    import oneccl_bindings_for_pytorch as torchccl
    print("oneCCL Bindings version {}".format(torchccl.__version__))

if dpctl_found == True:
    import dpctl as dpctl
    print("DPCTL version {}".format(dpctl.__version__))

if numba_dpex_found == True:
    import numba_dpex as dpex
    print("numba_dpex version {}".format(dpex.__version__))

if dpnp_found == True:
    import dpnp as np
    print("dpnp version {}".format(np.__version__))

checker = arch_checker()
print("Arch : ", checker.arch)
