from __future__ import print_function

import argparse
import collections
import lark
import os
import re
import string
import sys
import json

from common.cpp_sig_parser import CPPSig
from common.aten_sig_parser import AtenSig

_FN_BYPASS_REGEX = [
    # ATEN CUDA functions
    r'[^(]*cudnn',
    r'[^(]*cufft',
    r'[^(]*mkldnn',
    r'[^(]*_amp'
]

_FN_DNNL_FUNCS_WITH_SIMPLE_ATEN_SIG = [
    # 'aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor',
    # 'aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)',
    # 'aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)',
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
    'aten::sigmoid(Tensor self) -> Tensor',
    'aten::sigmoid_(Tensor(a!) self) -> Tensor(a!)',
    'aten::sigmoid_backward(Tensor grad_output, Tensor output) -> Tensor',
    'aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)',
    'aten::cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)',
    'aten::cat(Tensor[] tensors, int dim=0) -> Tensor',
    'aten::split_with_sizes(Tensor self, int[] split_sizes, int dim=0) -> Tensor[]',
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
    'aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor',
]

_SHALLOW_FALLBACK_TO_CPU_TENSOR_LIST = 'shallowFallbackToCPUTensorList'
_SHALLOW_FALLBACK_TO_CPU_TENSOR = 'shallowFallbackToCPUTensor'
_SHALLOW_UPGRADE_TO_DPCPP_TENSOR = 'shallowUpgradeToDPCPPTensor'
_SHALLOW_UPGRADE_TO_DPCPP_TENSOR_VEC = 'shallowUpgradeToDPCPPTensorVec'
_SHALLOW_UPGRADE_TO_DPCPP_TENSOR_A = 'shallowUpgradeToDPCPPTensorA'
_SHALLOW_UPGRADE_TO_DPCPP_TENSOR_AW = 'shallowUpgradeToDPCPPTensorAW'

_TYPE_NSMAP = {
    'Tensor': 'at::Tensor', # Cover TensorList, TensorOptions and Tensor
    'Scalar': 'at::Scalar', # Cover ScalarType and Scalar
    'Storage': 'at::Storage',
    'IntList': 'at::IntList',
    'IntArrayRef': 'at::IntArrayRef',
    'Generator': 'at::Generator',
    'SparseTensorRef': 'at::SparseTensorRef',
    'Device': 'c10::Device',
    'optional': 'c10::optional',
    'MemoryFormat': 'at::MemoryFormat',
    'QScheme': 'at::QScheme',
    'ConstQuantizerPtr': 'at::ConstQuantizerPtr',
    'Dimname': 'at::Dimname',  # Cover DimnameList and Dimname
}

_REG_PATTERN =  """
    .op(torch::RegisterOperators::options().schema("{}")
      .impl_unboxedOnlyKernel<{}, &{}>(at::DispatchKey::DPCPPTensorId)
      .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))"""
_H_HEADER = """// Autogenerated file by {gen}. Do not edit directly!
#pragma once

#include <ATen/Tensor.h>

namespace torch_ipex {{
namespace cpu {{

class AtenIpexCPUDefault {{
 public:
{hfuncs}
}};

void RegisterIpexDenseOPs();

}}  // namespace cpu

}}  // namespace torch_ipex
"""

_CPP_HEADER = """// Autogenerated file by {gen}. Do not edit directly!
#include "DenseOPs.h"

#include <ATen/Context.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/CPUGenerator.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>

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
        self._op_h_file = None
        self._op_cpp_file_path = op_cpp_file_path
        self._op_cpp_file = None
        self._sigs = []
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

    def prepare_functions(self):
        for line in open(self._reg_dec_file_path, 'r'):
            m = re.match(r'\s*([^\s].*); //\s+(.*)', line)
            if not m:
                continue
            cpp_func_sig = m.group(1).replace('at::', '').replace('c10::', '')
            aten_func_sig_literal = m.group(2)

            aten_func_sig = aten_func_sig_literal
            if "schema" in aten_func_sig_literal and "compound" in aten_func_sig_literal:
                res = json.loads(aten_func_sig_literal)
                aten_func_sig = res["schema"]

            if not self.is_tensor_api(cpp_func_sig):
                continue

            try:
                cpp_sig = CPPSig(cpp_func_sig)
                if self.is_bypass_func(cpp_sig):
                    continue

                aten_sig = AtenSig(aten_func_sig)

                self.cross_correct_sig(cpp_sig, aten_sig)

                self._sigs.append((cpp_sig, aten_sig, cpp_func_sig, aten_func_sig))
            except Exception as e:
                self._err_info.append((cpp_func_sig, str(e)))
                print('Error parsing "{}": {}'.format(cpp_func_sig, e), file=sys.stderr)

        with open(self._func_file_path, 'r') as ff:
            self._func_data = ff.read()

        self._op_h_file = open(self._op_h_file_path, 'w')
        self._op_cpp_file = open(self._op_cpp_file_path, 'w')

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
        for key in _TYPE_NSMAP:
            cpp_func_str = cpp_func_str.replace(key, _TYPE_NSMAP[key])

        m = re.search(r'(.*) (\b\S*)\(', cpp_func_str)
        assert m
        return m.group(1)

    def get_func_dec(self, cpp_sig):
        ret_params = cpp_sig.ret_params
        assert len(cpp_sig.ret_params) == 1
        assert cpp_sig.ret_params[0].core_type_temp_ins != ''
        input_params_type = [input_param.core_type_temp_ins for input_param in cpp_sig.input_params]
        func_dec_str = ret_params[0].core_type_temp_ins +  '(' + ", ".join(input_params_type) + ')'
        for key in _TYPE_NSMAP:
            func_dec_str = func_dec_str.replace(key, _TYPE_NSMAP[key])
        return func_dec_str

    def gen_func_signature(self, cpp_func_str):
        cpp_func_str_h = cpp_func_str
        for key in _TYPE_NSMAP:
            cpp_func_str_h = cpp_func_str_h.replace(key, _TYPE_NSMAP[key])

        m = re.search(r'(\b\S*)\(', cpp_func_str_h)
        assert m
        orig_func_name = m.group(1)
        func_name_with_ns = "{}::{}".format(_IPEX_OP_FUNC_NS, orig_func_name)
        cpp_func_str_cpp = cpp_func_str_h.replace(orig_func_name + '(', func_name_with_ns + '(')

        return (cpp_func_str_h, cpp_func_str_cpp)

    def gen_dnnl_code(self, cpp_sig, aten_func_sig_str):
        code = ''

        def is_out_func(fname):
            return fname.endswith("_out")

        if not self.is_dnnl_func(aten_func_sig_str):
            return code

        param_vars = []
        dnnl_tensor_param_vars = []
        for param in cpp_sig.input_params:
            if param.core_type == 'Tensor':
                dnnl_tensor_param_vars.append(param.name)
            param_vars.append(param.name)

        code += '  try {\n'

        code += '    if (check_auto_dnnl()) {\n'
        code += '      std::vector<at::Tensor> dnnl_input_tensors;\n'
        if len(dnnl_tensor_param_vars) > 0:
            for dnnl_tensor_param_var in dnnl_tensor_param_vars:
                code += '      dnnl_input_tensors.push_back({});\n'.format(dnnl_tensor_param_var)

        fname = cpp_sig.def_name
        if fname.endswith('_'):
            assert len(dnnl_tensor_param_vars) > 0
            code += '      if (dbl::chk::dnnl_inplace_support_the_tensors(dnnl_input_tensors))\n'
            code += '        return AtenIpexCPUDev::dil_{}({});\n'.format(fname, ', '.join(list(param_vars)))
        else:
            param_seq_str_vec = []
            for param_var in param_vars:
                param_seq_str = param_var
                if param_var in dnnl_tensor_param_vars:
                    if param_var == 'out' and is_out_func(fname):
                        code += '      TORCH_INTERNAL_ASSERT({}.is_contiguous());\n'.format(param_var)
                    else:
                        # param_seq_str = '{}.is_contiguous() ? {} : {}.contiguous()'.format(param_var, param_var, param_var)
                        None
                param_seq_str_vec.append(param_seq_str)
            code += '      if (dbl::chk::dnnl_support_the_tensors(dnnl_input_tensors))\n'
            code += '        return AtenIpexCPUDev::dil_{}({});\n'.format(fname, ', '.join(param_seq_str_vec))

        code += '    }\n'

        code += '  } catch (std::exception& e) {\n'
        code += '  }\n\n'


        return code

    def gen_fallback_prepare_code(self, cpp_sig):
        code = ''
        op_check_code = ''
        for param in cpp_sig.input_params:
            if param.core_type == 'TensorList':
                ipex_name = '_ipex_{}'.format(param.name)
                code += ('  auto&& {} = bridge::{}({});\n').format(ipex_name, _SHALLOW_FALLBACK_TO_CPU_TENSOR_LIST, param.name)
                param.ipex_name = ipex_name
            elif param.core_type == 'TensorOptions':
                ipex_name = '_ipex_{}'.format(param.name)
                param.ipex_name = ipex_name
                check_cond = '{}.device().type() == at::DeviceType::DPCPP'.format(param.name)
                op_check_code += '  TORCH_INTERNAL_ASSERT({});\n'.format(check_cond)
                code += '  at::TensorOptions {} = {}.device(at::DeviceType::CPU);\n'.format(ipex_name, param.name)
            elif param.core_type == 'Storage':
                code += '  TORCH_INTERNAL_ASSERT({}.device_type() == c10::DeviceType::DPCPP);\n'.format(param.name)
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
                check_cond = '{}.layout() == c10::kStrided'.format(param.name)
                op_check_code += '  TORCH_INTERNAL_ASSERT({});\n'.format(check_cond)
                code += '  auto&& {} = bridge::{}({});\n'.format(ipex_name, _SHALLOW_FALLBACK_TO_CPU_TENSOR, param.name)
                param.ipex_name = ipex_name
        return op_check_code + code

    def gen_fallback_code(self, cpp_sig):
        func_name = cpp_sig.def_name

        for param in cpp_sig.input_params:
            assert param.name
        params_name = [param.ipex_name if param.ipex_name != '' else param.name for param in cpp_sig.input_params]

        if self.is_tensor_member_function(func_name):
            assert "_ipex_self" in params_name
            params_name.remove('_ipex_self')
            if self.is_void_func(cpp_sig):
                return '  {}.{}({});\n'.format('_ipex_self', cpp_sig.def_name, ', '.join(params_name))
            else:
                return '  auto&& {} = {}.{}({});\n'.format(_RESULT_NAME, '_ipex_self', cpp_sig.def_name, ', '.join(params_name))
        else:
            if self.is_void_func(cpp_sig):
                return '  at::{}({});\n'.format(cpp_sig.def_name, ', '.join(params_name))
            else:
                return '  auto&& {} = at::{}({});\n'.format(_RESULT_NAME, cpp_sig.def_name, ', '.join(params_name))

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
                assert len(output_params) == 1
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

    def gen_code(self):
        self.prepare_functions()
        assert len(self._err_info) == 0

        def is_conv_overrideable_func(fname):
            return fname in ['convolution_overrideable', 'convolution_backward_overrideable']

        func_decs = []
        func_regs = []
        func_defs = []
        for cpp_sig, aten_sig, cpp_func_sig_str, aten_func_sig_str in self._sigs:
            cpp_func_str_h, cpp_func_str_cpp = self.gen_func_signature(cpp_func_sig_str)
            # Gen declaration code for head file
            func_decs.append(self.gen_head_dec_code(cpp_func_str_h))

            func_regs.append(_REG_PATTERN.format(aten_func_sig_str, self.get_func_dec(cpp_sig), "AtenIpexCPUDefault::" + cpp_sig.def_name))

            # Gen definition code for cpp file
            code = '{} {{\n'.format(cpp_func_str_cpp)

            if is_conv_overrideable_func(cpp_sig.def_name):
                code += '  return AtenIpexCPUDev::dil_{}({});\n'.format(cpp_sig.def_name, ', '.join([param.name for param in cpp_sig.input_params]))
            else:
                code += self.gen_dnnl_code(cpp_sig, aten_func_sig_str)
                code += self.gen_fallback_prepare_code(cpp_sig)
                code += self.gen_fallback_code(cpp_sig)
                code += self.gen_fallback_post_code(cpp_sig)

            code += '}\n'

            code += '\n'

            func_defs.append(code)

        head_file_content = _H_HEADER.format(gen=os.path.basename(sys.argv[0]), hfuncs=''.join(func_decs))

        regs_code = 'void RegisterIpexDenseOPs() {\n'
        regs_code += '  static auto dispatch = torch::RegisterOperators()\n'
        regs_code += ''.join(func_regs)
        regs_code += ';\n}\n'

        source_file_content = _CPP_HEADER.format(gen=os.path.basename(sys.argv[0]), funcs=''.join(func_defs), regs=regs_code)
        print(head_file_content, file=self._op_h_file)
        print(source_file_content, file=self._op_cpp_file)


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
