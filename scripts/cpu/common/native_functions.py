#!/usr/bin/python

import collections
import os
import re
import string
import sys

from .cpp_sig_parser import CPPSig
from .utils import *

class NativeFunctions(object):
    def __init__(self, func_file_path):
        self._func_file_path = func_file_path
        self._native_sigs_str = []
        self._func_data = ''
        self._err_info = []

        with open(self._func_file_path, 'r') as ff:
            self._func_data = ff.read()

        for line in open(self._func_file_path, 'r'):
            m = re.match(r'TORCH_API *(.*); *', line)
            if not m:
                continue
            native_cpp_sig_str = m.group(1).replace('at::', '').replace('c10::', '').replace('Reduction::', '')
            # Remove ={xxx}
            native_cpp_sig_str = re.sub("\=\{.*?\}\,", ",", native_cpp_sig_str)
            # Remove =xxx,
            native_cpp_sig_str = re.sub("\=.*?\,", ",", native_cpp_sig_str)
            # Remove =),
            native_cpp_sig_str = re.sub("\=.*?\)", ")", native_cpp_sig_str)
            if not is_tensor_api(native_cpp_sig_str):
                continue
            self._native_sigs_str.append(native_cpp_sig_str)

    def is_tensor_member_function(self, func_name):
        if self._func_data.find(' {}('.format(func_name)) >= 0:
            return False
        else:
            return True

    def query(self, cpp_sig):
        cnt = 0
        cur_native_cpp_sig_str = ''
        ret_native_cpp_sig = None
        try:
            for native_sig_str_item in self._native_sigs_str:
                target_str = ' {}('.format(cpp_sig.def_name)
                if native_sig_str_item.find(target_str) >= 0:
                    cur_native_cpp_sig_str = native_sig_str_item
                    native_cpp_sig = CPPSig(cur_native_cpp_sig_str)
                    params1 = [param.ipex_name if param.ipex_name != '' else param.name for param in native_cpp_sig.input_params]
                    params2 = [param.ipex_name if param.ipex_name != '' else param.name for param in cpp_sig.input_params]
                    if compare_params(params1, params2):
                        cnt = cnt + 1
                        ret_native_cpp_sig = native_cpp_sig
        except Exception as e:
            self._err_info.append((cur_native_cpp_sig_str, str(e)))
            print('[NativeFunctions] Error parsing "{}": {}'.format(cur_native_cpp_sig_str, e), file=sys.stderr)

        if cnt == 0:
            raise Exception("Cannot the function:{} in Functions.h".format(cpp_sig.def_name))
        return ret_native_cpp_sig
