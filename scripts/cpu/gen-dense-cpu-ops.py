from __future__ import print_function

import argparse
import collections
import lark
import os
import re
import string
import sys
import json

from common.codegen import write_or_skip
from common.cpp_sig_parser import CPPSig
from common.aten_sig_parser import AtenSig
import common.utils as utils

_FN_BYPASS_REGEX = [
    # ATEN CUDA functions
    r'[^(]*cudnn',
    r'[^(]*cufft',
    r'[^(]*mkldnn',
    r'[^(]*_amp',
    r'[^(]*_test_',
]

_FN_DNNL_FUNCS_WITH_SIMPLE_ATEN_SIG = [
    'aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor',
    'aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)',
    'aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)',
    'aten::mul.Tensor(Tensor self, Tensor other) -> Tensor',
    'aten::mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)',
    'aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor',
    'aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)',
    'aten::native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)',
    'aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor',
    'aten::avg_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor',
    'aten::avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor',
    'aten::avg_pool3d_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor',
    'aten::relu(Tensor self) -> Tensor',
    'aten::relu_(Tensor(a!) self) -> Tensor(a!)',
    'aten::threshold_backward(Tensor grad_output, Tensor self, Scalar threshold) -> Tensor',
    'aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor',
    'aten::_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor',
    'aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor',
    'aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor',
    'aten::sigmoid(Tensor self) -> Tensor',
    'aten::sigmoid_(Tensor(a!) self) -> Tensor(a!)',
    'aten::sigmoid_backward(Tensor grad_output, Tensor output) -> Tensor',
    'aten::tanh(Tensor self) -> Tensor',
    'aten::tanh_(Tensor(a!) self) -> Tensor(a!)',
    'aten::tanh_backward(Tensor grad_output, Tensor output) -> Tensor',
    'aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)',
    'aten::cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::cat(Tensor[] tensors, int dim=0) -> Tensor',
    'aten::split_with_sizes(Tensor(a) self, int[] split_sizes, int dim=0) -> Tensor(a)[]',
    'aten::split.Tensor(Tensor(a) self, int split_size, int dim=0) -> Tensor(a)[]',
    'aten::bmm(Tensor self, Tensor mat2) -> Tensor',
    'aten::bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::mm(Tensor self, Tensor mat2) -> Tensor',
    'aten::mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor',
    'aten::baddbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)',
    'aten::baddbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)',
    'aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor',
    'aten::addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)',
    'aten::addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)',
    'aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor',
    'aten::addbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)',
    'aten::addbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)',
    'aten::convolution_overrideable(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor',
    'aten::convolution_backward_overrideable(Tensor grad_output, Tensor input, Tensor weight, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)',
    'aten::size.int(Tensor self, int dim) -> int',
    'aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor',
    'aten::gelu(Tensor self) -> Tensor',
    'aten::gelu_backward(Tensor grad, Tensor self) -> Tensor',
    'aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=0, int? end=9223372036854775807, int step=1) -> Tensor(a)',
    'aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)',
    'aten::select.Dimname(Tensor(a) self, Dimname dim, int index) -> Tensor(a)',
    'aten::unbind.int(Tensor(a) self, int dim=0) -> Tensor(a)[]',
    'aten::unbind.Dimname(Tensor(a) self, Dimname dim) -> Tensor(a)[]',
    'aten::view(Tensor(a) self, int[] size) -> Tensor(a)',
    'aten::index_select(Tensor self, int dim, Tensor index) -> Tensor',
    'aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor',
    'aten::_unsafe_view(Tensor self, int[] size) -> Tensor',
    'aten::native_layer_norm(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)',
    'aten::native_layer_norm_backward(Tensor grad_out, Tensor input, Tensor mean, Tensor rstd, Tensor? weight, int M, int N, bool[3] output_mask) -> (Tensor, Tensor, Tensor)',
    # 'aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)',
    # 'aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor',
    'aten::native_layer_norm_backward(Tensor grad_out, Tensor input, int[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias, bool[3] output_mask) -> (Tensor, Tensor, Tensor)',
    # 'aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)',
    'aten::_pack_padded_sequence(Tensor input, Tensor lengths, bool batch_first) -> (Tensor, Tensor)',
    'aten::upsample_nearest1d(Tensor self, int[1] output_size, float? scales=None) -> Tensor',
    'aten::upsample_nearest1d_backward(Tensor grad_output, int[1] output_size, int[3] input_size, float? scales=None) -> Tensor',
    'aten::upsample_nearest2d(Tensor self, int[2] output_size, float? scales_h=None, float? scales_w=None) -> Tensor',
    'aten::upsample_nearest2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, float? scales_h=None, float? scales_w=None) -> Tensor',
    'aten::upsample_nearest3d(Tensor self, int[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor',
    'aten::upsample_nearest3d_backward(Tensor grad_output, int[3] output_size, int[5] input_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor',
    'aten::upsample_linear1d(Tensor self, int[1] output_size, bool align_corners, float? scales=None) -> Tensor',
    'aten::upsample_linear1d_backward(Tensor grad_output, int[1] output_size, int[3] input_size, bool align_corners, float? scales=None) -> Tensor',
    'aten::upsample_bilinear2d(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor',
    'aten::upsample_bilinear2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor',
    'aten::upsample_trilinear3d(Tensor self, int[3] output_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor',
    'aten::upsample_trilinear3d_backward(Tensor grad_output, int[3] output_size, int[5] input_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor',
    'aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)',
    'aten::div.Tensor(Tensor self, Tensor other) -> Tensor',
    'aten::div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)',
    'aten::div_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)',
    'aten::div.Scalar(Tensor self, Scalar other) -> Tensor',
    'aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)',
]

_FN_IPEX_FUNCS_WITH_SIMPLE_ATEN_SIG = [
    'aten::index_select(Tensor self, int dim, Tensor index) -> Tensor',
    'aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor',
    # 'aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)',
    'aten::_pack_padded_sequence(Tensor input, Tensor lengths, bool batch_first) -> (Tensor, Tensor)',
    'aten::div.Tensor(Tensor self, Tensor other) -> Tensor',
    'aten::div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)',
    'aten::div_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)',
    'aten::div.Scalar(Tensor self, Scalar other) -> Tensor',
    'aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)',
]

_SHALLOW_FALLBACK_TO_CPU_TENSOR_LIST = 'shallowFallbackToCPUTensorList'
_SHALLOW_FALLBACK_TO_CPU_TENSOR = 'shallowFallbackToCPUTensor'
_SHALLOW_UPGRADE_TO_DPCPP_TENSOR = 'shallowUpgradeToDPCPPTensor'
_SHALLOW_UPGRADE_TO_DPCPP_TENSOR_VEC = 'shallowUpgradeToDPCPPTensorVec'
_SHALLOW_UPGRADE_TO_DPCPP_TENSOR_A = 'shallowUpgradeToDPCPPTensorA'
_SHALLOW_UPGRADE_TO_DPCPP_TENSOR_AW = 'shallowUpgradeToDPCPPTensorAW'

_REG_PATTERN =  """
    m.impl("{}", static_cast<{}>(&{}));"""

_REG_BLOCK = """
namespace {{
  TORCH_LIBRARY_IMPL(aten, XPU, m) {{
    {reg_ops}
  }}
}}"""

_H_HEADER = """// Autogenerated file by {gen}. Do not edit directly!
#pragma once

#include <ATen/ATen.h>

namespace torch_ipex {{
namespace cpu {{

class AtenIpexCPUDefault {{
 public:
{hfuncs}
}};

}}  // namespace cpu

}}  // namespace torch_ipex
"""

_CPP_HEADER = """// Autogenerated file by {gen}. Do not edit directly!
#include "DenseOPs.h"

#include <ATen/Context.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/record_function.h>
#include <c10/core/Layout.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <torch/csrc/autograd/function.h>
#include <torch/library.h>

#include "aten_ipex_bridge.h"
#include "utils.h"
#include "DevOPs.h"
#include "dbl/DNNLChecker.h"

namespace torch_ipex {{
namespace cpu {{

{funcs}

{regs}

}}  // namespace cpu
}}  // namespace torch_ipex
"""

_RESULT_NAME = '_ipex_result'
_IPEX_OP_FUNC_NS = 'AtenIpexCPUDefault'

class DenseOPCodeGen(object):
    def __init__(self, reg_dec_file_path, func_file_path, op_h_file_path, op_cpp_file_path):
        self._reg_dec_file_path = reg_dec_file_path
        self._func_file_path = func_file_path
        self._op_h_file_path = op_h_file_path
        self._op_cpp_file_path = op_cpp_file_path
        self._sigs = []
        self._native_sigs = []
        self._native_sigs_str = []
        self._err_info = []
        self._func_data = ''

    def is_tensor_api(self, func_name):
        m = re.search(r'\bTensor\b', func_name)
        return m is not None

    def is_tensor_member_function(self, func_name):
        if self._func_data.find(' {}('.format(func_name)) >= 0:
            return False
        else:
            return True

    def is_void_func(self, cpp_sig):
        ret_params = cpp_sig.ret_params
        assert len(ret_params) == 1
        ret_param = ret_params[0]
        if ret_param.core_type == 'void' and not ret_param.is_pointer:
            return True
        return False

    def is_dnnl_func(self, simple_aten_sig):
        stripped_str = simple_aten_sig.replace(' ', '')
        for item in _FN_DNNL_FUNCS_WITH_SIMPLE_ATEN_SIG:
            if stripped_str == item.replace(' ', ''):
                return True
        return False

    def is_ipex_func(self, simple_aten_sig):
        stripped_str = simple_aten_sig.replace(' ', '')
        for item in _FN_IPEX_FUNCS_WITH_SIMPLE_ATEN_SIG:
            if stripped_str == item.replace(' ', ''):
                return True
        return False

    def is_bypass_func(self, cpp_sig):
        for frx in _FN_BYPASS_REGEX:
            if re.match(frx, cpp_sig.def_name):
                return True
        return False

    def cross_correct_sig(self, cpp_sig, aten_sig):
        for cpp_input_param in cpp_sig.input_params:
            for aten_sig_param in aten_sig.input_params:
                if cpp_input_param.name == aten_sig_param.name:
                    cpp_input_param.is_to_be_written = aten_sig_param.is_to_be_written
                    cpp_input_param.is_alias = aten_sig_param.is_alias

    def parse_native_functions(self):
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
            if not self.is_tensor_api(native_cpp_sig_str):
                continue
            self._native_sigs_str.append(native_cpp_sig_str)

    def compare_params(self, params1, params2):
        if len(params1) != len(params2):
            return False

        for param_item in params1:
            if param_item not in params2:
                return False
        return True

    def get_native_functions(self, cpp_sig):
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
                    if self.compare_params(params1, params2):
                        cnt = cnt + 1
                        ret_native_cpp_sig = native_cpp_sig
        except Exception as e:
            self._err_info.append((cur_native_cpp_sig_str, str(e)))
            print('[NativeFunctions] Error parsing "{}": {}'.format(cur_native_cpp_sig_str, e), file=sys.stderr)

        if cnt == 0:
            raise Exception("Cannot the function:{} in Functions.h".format(cpp_sig.def_name))
        return ret_native_cpp_sig

    def prepare_functions(self):
        # Get all functions in Functions.h
        self.parse_native_functions()

        for line in open(self._reg_dec_file_path, 'r'):
            m = re.match(r'\s*([^\s].*); //\s+(.*)', line)
            if not m:
                continue
            cpp_func_sig = m.group(1).replace('at::', '').replace('c10::', '')
            aten_func_sig_literal = m.group(2)

            aten_func_sig = aten_func_sig_literal
            if "schema" in aten_func_sig_literal and "dispatch" in aten_func_sig_literal:
                res = json.loads(aten_func_sig_literal)
                aten_func_sig = res["schema"]

            if not self.is_tensor_api(cpp_func_sig):
                continue

            try:
                cpp_sig = CPPSig(cpp_func_sig)
                if self.is_bypass_func(cpp_sig):
                    continue

                native_cpp_sig = None
                if utils.is_out_func(cpp_sig.def_name):
                    native_cpp_sig = self.get_native_functions(cpp_sig)

                aten_sig = AtenSig(aten_func_sig)

                self.cross_correct_sig(cpp_sig, aten_sig)

                self._sigs.append((cpp_sig, aten_sig, native_cpp_sig, cpp_func_sig, aten_func_sig))
            except Exception as e:
                self._err_info.append((cpp_func_sig, str(e)))
                print('[RegistrationDeclarations] Error parsing "{}": {}'.format(cpp_func_sig, e), file=sys.stderr)

        print('Extracted {} functions ({} errors) from {}'.format(
              len(self._sigs),
              len(self._err_info),
              self._reg_dec_file_path),
            file=sys.stderr)
        assert len(self._err_info) == 0

    def get_alias_tensor_by_index(self, cpp_sig, idx):
        alias_tensors = cpp_sig.get_alias_tensors()
        assert len(alias_tensors) > idx
        return alias_tensors[idx]

    def get_ret_type_str(self, cpp_func_str):
        cpp_func_str = utils.add_ns(cpp_func_str)

        m = re.search(r'(.*) (\b\S*)\(', cpp_func_str)
        assert m
        return m.group(1)

    def get_func_dec(self, cpp_sig):
        func_dec_str = cpp_sig.sig_str.replace(cpp_sig.def_name + '(', ' (*)(')
        return utils.add_ns(func_dec_str)

    def gen_func_signature(self, cpp_func_str, old_func_name, new_func_name):
        cpp_func_str_h = utils.add_ns(cpp_func_str.replace(old_func_name + '(', new_func_name + '('))
        func_name_with_ns = "{}::{}".format(_IPEX_OP_FUNC_NS, new_func_name)
        cpp_func_str_cpp = cpp_func_str_h.replace(new_func_name + '(', func_name_with_ns + '(')

        return cpp_func_str_h, cpp_func_str_cpp

    def gen_dnnl_code(self, cpp_sig, native_cpp_sig, aten_func_sig_str):
        code = ''

        if not self.is_dnnl_func(aten_func_sig_str):
            return code

        param_vars = []
        dnnl_tensor_param_vars = []

        input_params = cpp_sig.input_params
        # Reorder the input parameters
        if native_cpp_sig is not None:
            params1_name = [param.name for param in cpp_sig.input_params]
            params2_name = [param.name for param in native_cpp_sig.input_params]
            new_idxs = utils.reorder_params_idx(params1_name, params2_name)
            input_params = [cpp_sig.input_params[new_idxs[idx]] for idx in range(len(new_idxs))]

        for param in input_params:
            if param.core_type == 'Tensor':
                dnnl_tensor_param_vars.append(param)

            if param.core_type == 'Tensor' and param.is_optional:
                param_vars.append("{}.has_value() ? {}.value() : at::Tensor()".format(param.name, param.name))
            else:
                param_vars.append(param.name)

        code += '  try {\n'

        code += '    if (check_auto_dnnl()) {\n'

        if not self.is_ipex_func(aten_func_sig_str):
            # There are two different kind of DevOPs in IPEX
            #    1. DNNL Operator
            #    2. CPU BF16/INT8 Operator in Vanilla PyTorch. IPEX itegrates this kind of operators in IPEX for
            #       mixture precision.
            # For the type 2, IPEX does not need to check if DNNL supports these tensors.
            code += '      std::vector<at::Tensor> dnnl_input_tensors;\n'
            if len(dnnl_tensor_param_vars) > 0:
                for dnnl_tensor_param_var in dnnl_tensor_param_vars:
                    if dnnl_tensor_param_var.is_optional:
                        code += '      if ({}.has_value()) dnnl_input_tensors.push_back({}.value());\n'.format(dnnl_tensor_param_var.name, dnnl_tensor_param_var.name)
                    else:
                        code += '      dnnl_input_tensors.push_back({});\n'.format(dnnl_tensor_param_var.name)

        fname = cpp_sig.def_name
        if fname.endswith('_'):
            assert len(dnnl_tensor_param_vars) > 0
            if self.is_ipex_func(aten_func_sig_str):
                code += self.gen_ipex_func_code(fname, param_vars)
            else:
                code += '      if (dbl::chk::dnnl_inplace_support_the_tensors(dnnl_input_tensors)) {\n'
                code += '        return AtenIpexCPUDev::dil_{}({});\n'.format(fname, ', '.join(list(param_vars)))
                code += '      }\n' # Check support tensors
        else:
            param_seq_str_vec = []
            for param_var in param_vars:
                param_seq_str = param_var
                param_seq_str_vec.append(param_seq_str)

            if self.is_ipex_func(aten_func_sig_str):
                code += self.gen_ipex_func_code(fname, param_seq_str_vec)
            else:
                code += '      if (dbl::chk::dnnl_support_the_tensors(dnnl_input_tensors)) {\n'
                code += '        return AtenIpexCPUDev::dil_{}({});\n'.format(fname, ', '.join(param_seq_str_vec))
                code += '      }\n' # Check support tensors
        code += '    }\n' # Check auto dnnl
        code += '  } catch (std::exception& e) {\n'
        code += '#if defined(_DEBUG)\n'
        code += '    TORCH_WARN(e.what());\n'
        code += '#endif\n'
        code += '  }\n\n'


        return code

    def gen_ipex_func_code(self, fname, param_vars):
        code = ''
        code += '        auto _result = AtenIpexCPUDev::dil_{}({});\n'.format(fname, ', '.join(param_vars))
        code += '        if (is_ipex_func_success()) {\n'
        code += '          return _result;\n'
        code += '        } else {\n'
        code += '          reset_ipex_func_status();\n'
        code += '        }\n'
        return code

    def gen_fallback_prepare_code(self, cpp_sig):
        code = ''
        op_check_code = ''
        for param in cpp_sig.input_params:
            if param.core_type == 'TensorList' or param.core_type == 'List':
                ipex_name = '_ipex_{}'.format(param.name)
                code += ('  auto&& {} = bridge::{}({});\n').format(ipex_name, _SHALLOW_FALLBACK_TO_CPU_TENSOR_LIST, param.name)
                param.ipex_name = ipex_name
            elif param.core_type == 'TensorOptions':
                ipex_name = '_ipex_{}'.format(param.name)
                param.ipex_name = ipex_name
                check_cond = '{}.device().type() == at::DeviceType::XPU'.format(param.name)
                op_check_code += '  TORCH_INTERNAL_ASSERT_DEBUG_ONLY({});\n'.format(check_cond)
                code += '  at::TensorOptions {} = {}.device(at::DeviceType::CPU);\n'.format(ipex_name, param.name)
            elif param.core_type == 'Storage':
                code += '  TORCH_INTERNAL_ASSERT_DEBUG_ONLY({}.device_type() == c10::DeviceType::XPU);\n'.format(param.name)
            elif param.core_type == 'MemoryFormat':
                if param.is_optional:
                    check_cond = '{}.value_or(c10::MemoryFormat::Contiguous) != c10::MemoryFormat::Contiguous'.format(param.name)
                else:
                    check_cond = '{} != c10::MemoryFormat::Contiguous'.format(param.name)
                #op_check_code += '  if ({})\n'.format(check_cond)
                #op_check_code += '      TORCH_WARN({});\n'.format(check_cond)
            elif param.core_type != 'Tensor':
                None
            # Tensor
            else:
                assert param.core_type == 'Tensor'
                ipex_name = '_ipex_{}'.format(param.name)
                check_cond = ''
                if param.is_optional:
                    check_cond = '(!({}.has_value())) || ({}->layout() == c10::kStrided)'.format(param.name, param.name)
                else:
                    check_cond = '{}.layout() == c10::kStrided'.format(param.name)
                op_check_code += '  TORCH_INTERNAL_ASSERT_DEBUG_ONLY({});\n'.format(check_cond)

                if param.is_optional:
                    code += '  auto&& {} = c10::optional<at::Tensor>();\n'.format(ipex_name)
                    code += '  if ({}.has_value()) {{\n'.format(param.name)
                    code += '    {} = c10::optional<at::Tensor>(bridge::{}({}.value()));\n'.format(ipex_name, _SHALLOW_FALLBACK_TO_CPU_TENSOR, param.name)
                    code += '  }\n'
                else:
                    code += '  auto&& {} = bridge::{}({});\n'.format(ipex_name, _SHALLOW_FALLBACK_TO_CPU_TENSOR, param.name)
                param.ipex_name = ipex_name
        return op_check_code + code

    def gen_fallback_code(self, cpp_sig, native_cpp_sig):
        func_name = cpp_sig.def_name

        for param in cpp_sig.input_params:
            assert param.name

        if native_cpp_sig is None:
            params_name = [param.ipex_name if param.ipex_name != '' else param.name for param in cpp_sig.input_params]
        else:
            params1_name = [param.name for param in cpp_sig.input_params]
            params2_name = [param.name for param in native_cpp_sig.input_params]
            new_idxs = utils.reorder_params_idx(params1_name, params2_name)
            input_params = cpp_sig.input_params
            params_name = [input_params[new_idxs[idx]].ipex_name if input_params[new_idxs[idx]].ipex_name != '' else input_params[new_idxs[idx]].name for idx in range(len(new_idxs))]

        code = ''
        # Wrap the input parameters as tensor option
        start_idx, end_idx = utils.query_tensor_options(cpp_sig.input_params)
        if start_idx >= 0 and end_idx > start_idx:
            # assert bool((end_idx - start_idx + 1) == 4)
            wrapped_options = 'ipex_wrapped_options'
            code += '  auto&& {} = at::TensorOptions().dtype(dtype).device(at::DeviceType::CPU).layout(layout).pinned_memory(pin_memory);\n'
            code = code.format(wrapped_options)
            # Remove original param name
            params_name = params_name[:start_idx] + [wrapped_options] + params_name[end_idx + 1:]

        if self.is_tensor_member_function(func_name):
            assert "_ipex_self" in params_name
            params_name.remove('_ipex_self')
            if self.is_void_func(cpp_sig):
                code += '  {}.{}({});\n'.format('_ipex_self', cpp_sig.def_name, ', '.join(params_name))
            else:
                code += '  auto&& {} = {}.{}({});\n'.format(_RESULT_NAME, '_ipex_self', cpp_sig.def_name, ', '.join(params_name))
        else:
            if self.is_void_func(cpp_sig):
                code += '  at::{}({});\n'.format(cpp_sig.def_name, ', '.join(params_name))
            else:
                code += '  auto&& {} = at::{}({});\n'.format(_RESULT_NAME, cpp_sig.def_name, ', '.join(params_name))

        return code

    def gen_fallback_post_code(self, cpp_sig):
        code = ''

        if self.is_void_func(cpp_sig):
            for param in cpp_sig.get_output_tensors():
                if param.is_tensor:
                    code += '  bridge::{}({}, {});\n'.format(_SHALLOW_UPGRADE_TO_DPCPP_TENSOR_AW,
                                                             param.name,
                                                             param.ipex_name)
            return code

        # current OP is in-place or out OP
        if cpp_sig.contain_output_tensor:
            #assert cpp_sig.def_name.endswith('_') or cpp_sig.def_name.endswith('out')
            for param in cpp_sig.input_params:
                if param.is_tensor and param.is_to_be_written:
                    code += '  bridge::{}({}, {});\n'.format(_SHALLOW_UPGRADE_TO_DPCPP_TENSOR_AW,
                                                             param.name,
                                                             param.ipex_name)

        ret_params = cpp_sig.ret_params
        assert len(ret_params) == 1
        ret_param = ret_params[0]
        if ret_param.core_type == 'std::tuple':
            assert len(ret_param.sub_params) > 0
            tuple_items = []
            for i, sub_param in enumerate(ret_param.sub_params):
                tuple_item = 'std::get<{}>({})'.format(i, _RESULT_NAME)
                tuple_item_final_str = tuple_item
                if sub_param.core_type == 'Tensor':
                    if sub_param.is_ref:
                        i_th_alias_tensor = self.get_alias_tensor_by_index(cpp_sig, i)
                        assert i_th_alias_tensor.name
                        tuple_item_final_str = i_th_alias_tensor.name
                    else:
                        tuple_item_final_str = 'bridge::{}({})'.format(_SHALLOW_UPGRADE_TO_DPCPP_TENSOR, tuple_item)

                tuple_items.append(tuple_item_final_str)

            code += '  static_cast<void>({}); // Avoid warnings in case not used\n'.format(_RESULT_NAME)
            code += '  return {}({});\n'.format(self.get_ret_type_str(cpp_sig.sig_str), ', '.join(tuple_items))
            return code

        if ret_param.core_type == 'std::vector':
            code += '  static_cast<void>({}); // Avoid warnings in case not used\n'.format(_RESULT_NAME)
            code += '  return bridge::{}({});\n'.format(_SHALLOW_UPGRADE_TO_DPCPP_TENSOR_VEC, _RESULT_NAME)
            return code

        if ret_param.core_type == 'Tensor':
            code += '  static_cast<void>({}); // Avoid warnings in case not used\n'.format(_RESULT_NAME)

            if cpp_sig.contain_output_tensor:
                output_params = cpp_sig.get_output_tensors()
                # NOTE: We cannot assume that only one input tensor can be modified during execution according to
                # the aten signature. ex. aten::_linalg_inv_out_helper_(Tensor(a!) self, Tensor(b!) infos_lu, Tensor(c!) infos_getri) -> Tensor(a!)
                # assert len(output_params) == 1
                code += '  return {};\n'.format(output_params[0].name)
                return code
            else:
                if cpp_sig.contain_alias_tensor:
                    alias_tensors = cpp_sig.get_alias_tensors()
                    assert len(alias_tensors) == 1
                    alias_tensor = alias_tensors[0]
                    assert alias_tensor.name
                    assert alias_tensor.ipex_name
                    code += '  bridge::{}({}, {});\n'.format(_SHALLOW_UPGRADE_TO_DPCPP_TENSOR_A, alias_tensor.name, alias_tensor.ipex_name)
                code += '  return bridge::{}({});\n'.format(_SHALLOW_UPGRADE_TO_DPCPP_TENSOR, _RESULT_NAME)
                return code

        # Else: other return types
        code += '  static_cast<void>({}); // Avoid warnings in case not used\n'.format(_RESULT_NAME)
        code += '  return {};\n'.format(_RESULT_NAME)
        return code

    def gen_head_dec_code(self, cpp_func_str_h):
        return '  static {};\n'.format(cpp_func_str_h)

    def gen_cpu_ops_shard(self, func_defs, cpp_path, header_path, num_shards=1):
        head_file_content = _H_HEADER.format(gen=os.path.basename(sys.argv[0]), hfuncs=''.join([f['dec'] for f in func_defs]))
        write_or_skip(header_path, head_file_content)

        shards = [[] for _ in range(num_shards)]
        for idx, func in enumerate(func_defs):
            shards[idx % num_shards].append(func)

        for idx, shard in enumerate(shards):
            regs_code = _REG_BLOCK.format(reg_ops=''.join([f['reg'] for f in shard]))
            defs_code = ''.join([f['def'] for f in shard])

            filename, ext = os.path.splitext(cpp_path)
            shard_filepath = '%s_%s%s' % (filename, idx, ext)
            shard_content = _CPP_HEADER.format(gen=os.path.basename(sys.argv[0]), funcs=defs_code, regs=regs_code)
            write_or_skip(shard_filepath, shard_content)

    def gen_code(self):
        self.prepare_functions()
        assert len(self._err_info) == 0

        def is_conv_overrideable_func(fname):
            return fname in ['convolution_overrideable', 'convolution_backward_overrideable']

        func_defs = []
        for cpp_sig, aten_sig, native_cpp_sig, cpp_func_sig_str, aten_func_sig_str in self._sigs:
            # The operator name should be unique because the new registration mechanism of PyTorch 1.7
            new_cpp_func_name = aten_sig.def_name.replace('.', '_')
            cpp_func_str_h, cpp_func_str_cpp = self.gen_func_signature(cpp_func_sig_str, cpp_sig.def_name, new_cpp_func_name)

            # Gen declaration code for head file
            func_dec = self.gen_head_dec_code(cpp_func_str_h)

            func_reg = _REG_PATTERN.format(aten_sig.def_name, self.get_func_dec(cpp_sig), _IPEX_OP_FUNC_NS + "::" + new_cpp_func_name)

            # Gen definition code for cpp file
            code = '{} {{\n'.format(cpp_func_str_cpp)

            # Gen OP Name
            code += '#if defined(IPEX_DISP_OP)\n'
            code += '  printf("{}::{}\\n");\n'.format(_IPEX_OP_FUNC_NS, cpp_sig.def_name)
            code += '#endif\n'

            # Gen profile info
            profiler_inputs = []
            for param in cpp_sig.input_params:
                if param.core_type in ['Tensor', 'Scalar']:
                    profiler_inputs.append(param.name)
            code += '#if defined(IPEX_PROFILE_OP)\n'
            code += '  RECORD_FUNCTION("{ns}::{name}", std::vector<c10::IValue>({{{input_names}}}));\n'.format(ns=_IPEX_OP_FUNC_NS, name=cpp_sig.def_name, input_names=', '.join(profiler_inputs))
            code += '#endif\n'

            if is_conv_overrideable_func(cpp_sig.def_name):
                code += '  return AtenIpexCPUDev::dil_{}({});\n'.format(cpp_sig.def_name, ', '.join([param.name for param in cpp_sig.input_params]))
            else:
                code += self.gen_dnnl_code(cpp_sig, native_cpp_sig, aten_func_sig_str)
                code += self.gen_fallback_prepare_code(cpp_sig)
                code += self.gen_fallback_code(cpp_sig, native_cpp_sig)
                code += self.gen_fallback_post_code(cpp_sig)

            code += '}\n\n'

            func_defs.append({'dec': func_dec, 'reg': func_reg, 'def': code})

        self.gen_cpu_ops_shard(func_defs,
                               cpp_path=self._op_cpp_file_path,
                               header_path=self._op_h_file_path,
                               num_shards=8)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        'ipex_cpu_ops_head',
        type=str,
        metavar='IPEX_CPU_OPS_HEAD_FILE',
        help='The path to the IPEX cpu ATEN overrides head file')
    arg_parser.add_argument(
        'ipex_cpu_ops_cpp',
        type=str,
        metavar='IPEX_CPU_OPS_CPP_FILE',
        help='The path to the IPEX cpu ATEN overrides cpp file')
    arg_parser.add_argument(
        'reg_dec',
        type=str,
        metavar='REG_DEC_FILE',
        help='The path to the RegistrationDeclarations.h file')
    arg_parser.add_argument(
        'functions',
        type=str,
        metavar='FUNCTIONS_FILE',
        help='The path to the Functions.h file')
    args, files = arg_parser.parse_known_args()
    des_code_gen = DenseOPCodeGen(
        args.reg_dec,
        args.functions,
        args.ipex_cpu_ops_head,
        args.ipex_cpu_ops_cpp)
    des_code_gen.gen_code()
