#!/usr/bin/python

import argparse
import collections
import lark
import os
import re
import string
import sys

from .param import Param
from .sig_parser import SigParser

_CPP_SIG_GRAMMAR = r"""
    start: type fnname "(" params ")"
    type: CONST? core_type refspec?
    fnname: CNAME
    refspec: REF
           | PTR
    core_type: template
             | TNAME
    template: TNAME "<" typelist ">"
    typelist: type
            | type "," typelist
    REF: "&"
    PTR: "*"
    CONST: "const"
    TNAME: /[a-zA-Z0-9_:]+/
    HEXNUMBER: /0x[0-9a-fA-F]+/
    params: param
          | param "," params
    param: type param_name param_defval?
    param_name: CNAME

    param_defval: "=" init_value
    init_value: "true"
              | "false"
              | "{}"
              | NUMBER
              | SIGNED_NUMBER
              | HEXNUMBER
              | ESCAPED_STRING
              | CNAME

    %import common.CNAME -> CNAME
    %import common.NUMBER -> NUMBER
    %import common.SIGNED_NUMBER -> SIGNED_NUMBER
    %import common.ESCAPED_STRING -> ESCAPED_STRING
    %import common.WS
    %ignore WS
    """

_CPP_SIG_PARSER = lark.Lark(_CPP_SIG_GRAMMAR, parser='lalr', propagate_positions=True)


class CPPSig(SigParser):
    def __init__(self, cpp_sig):
        super(CPPSig, self).__init__(cpp_sig, _CPP_SIG_PARSER)

        self._def_name = self.__get_function_name(self._sig_tree)

    def __param_name(self, t):
        assert isinstance(t, lark.tree.Tree)
        c = t.children[1]
        assert isinstance(c, lark.tree.Tree)
        assert c.data == 'param_name'
        token = c.children[0]
        assert isinstance(token, lark.lexer.Token)
        return token.value

    def __param_type(self, t):
        assert isinstance(t, lark.tree.Tree)
        c = t.children[0]
        assert isinstance(c, lark.tree.Tree)
        return c

    def __get_function_name(self, t):
        assert isinstance(t, lark.tree.Tree)
        fname = t.children[1]
        assert isinstance(fname, lark.tree.Tree)
        assert fname.data == 'fnname'
        return fname.children[0].value

    def __extract_list(self, t, l):
        assert isinstance(t, lark.tree.Tree)
        l.append(t.children[0])
        if len(t.children) == 2:
            c = t.children[1]
            if isinstance(c, lark.tree.Tree) and c.data == t.data:
                self.__extract_list(c, l)
        return l

    def __get_parameters(self):
        assert isinstance(self._sig_tree, lark.tree.Tree)
        c = self._sig_tree.children[2]
        assert isinstance(c, lark.tree.Tree)
        assert c.data == 'params'
        params = []
        self.__extract_list(c, params)
        return params

    def __type_core(self, t):
        assert isinstance(t, lark.tree.Tree)
        for c in t.children:
            if isinstance(c, lark.tree.Tree) and c.data == 'core_type':
                c = c.children[0]
                if isinstance(c, lark.lexer.Token):
                    return c.value
                assert isinstance(c, lark.tree.Tree) and c.data == 'template'
                if c.children[0].value == 'optional':
                    type_list = c.children[1]
                    assert isinstance(type_list, lark.tree.Tree) and type_list.data == 'typelist'
                    return self.__type_core(type_list.children[0])
                return c.children[0].value
        raise RuntimeError('Not a type tree: {}'.format(t))

    def __type_is_optional(self, t):
        assert isinstance(t, lark.tree.Tree)
        for c in t.children:
            if isinstance(c, lark.tree.Tree) and c.data == 'core_type':
                c = c.children[0]
                if isinstance(c, lark.lexer.Token):
                    return False
                assert isinstance(c, lark.tree.Tree) and c.data == 'template'
                if c.children[0].value == 'optional':
                    return True
                else:
                    return False
        raise RuntimeError('Not a type tree: {}'.format(t))

    def __type_is_const(self, t):
        assert isinstance(t, lark.tree.Tree)
        c = t.children[0]
        return isinstance(c, lark.lexer.Token) and c.value == 'const'

    def __type_is_refptr(self, t, kind):
        assert isinstance(t, lark.tree.Tree)
        c = t.children[-1]
        if not isinstance(c, lark.tree.Tree) or c.data != 'refspec':
            return False
        c = c.children[0]
        return isinstance(c, lark.lexer.Token) and c.value == kind

    def __tuple_type_list(self, t):
        assert isinstance(t, lark.tree.Tree)
        c = t.children[0]
        assert isinstance(c, lark.tree.Tree) and c.data == 'core_type'
        c = c.children[0]
        assert isinstance(c, lark.tree.Tree) and c.data == 'template'
        types = []
        return self.__extract_list(c.children[1], types)

    def __get_return_type_str(self, t, orig_sig):
        assert isinstance(t, lark.tree.Tree)
        fname = t.children[1]
        assert isinstance(fname, lark.tree.Tree)
        assert fname.data == 'fnname'
        token = fname.children[0]
        assert isinstance(token, lark.lexer.Token)
        return orig_sig[0:token.column - 2]

    def get_all_input_params(self):
        params = self.__get_parameters()
        for param in params:
            _param_ins = Param()

            ptype = self.__param_type(param)
            _param_ins.name = self.__param_name(param)
            _param_ins.core_type = self.__type_core(ptype)
            _param_ins.core_type_temp_ins = self.sig_str[(ptype.column-1):(ptype.end_column-1)]

            if self.__type_is_const(ptype):
                _param_ins.is_const = True
            if self.__type_is_optional(ptype):
                _param_ins.is_optional = True

            self.input_params.append(_param_ins)

    def get_all_return_params(self):
        ret_type = self._sig_tree.children[0]
        core_type = self.__type_core(ret_type)

        cur_param = Param()
        cur_param.core_type = core_type
        cur_param.core_type_temp_ins = self.sig_str[(ret_type.column-1):(ret_type.end_column-1)]
        if core_type == 'std::tuple' or core_type == 'std::vector' or core_type == 'std::array':
            cur_param.is_std_tuple = True
            types = self.__tuple_type_list(ret_type)
            for _, sub_param_type in enumerate(types):
                sub_param = Param()
                sub_param.core_type = self.__type_core(sub_param_type)
                if self.__type_is_refptr(sub_param_type, '*'):
                    sub_param.is_pointer = True
                if self.__type_is_refptr(sub_param_type, '&'):
                    sub_param.is_ref = True
                cur_param.sub_params.append(sub_param)

        if self.__type_is_refptr(ret_type, '*'):
            # In case return type is (void *)
            cur_param.is_pointer = True

        if self.__type_is_refptr(ret_type, '&'):
            # In case return type is (void *)
            cur_param.is_ref = True

        self.ret_params.append(cur_param)


if __name__ == '__main__':
    sigs = [
        "Tensor data(const Tensor & self)",
        "bool is_leaf(const Tensor & self)",
        "int64_t output_nr(const Tensor & self)",
        "bool _use_cudnn_ctc_loss(const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank)",
        "std::tuple<Tensor,Tensor> _cudnn_ctc_loss(const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, bool deterministic, bool zero_infinity)",
        "std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor>> _cudnn_rnn_backward(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, const Tensor & output, const Tensor & grad_output, const Tensor & grad_hy, const Tensor & grad_cy, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const Tensor & dropout_state, const Tensor & reserve, std::array<bool,4> output_mask)",
        "Tensor im2col_backward(const Tensor & grad_output, IntArrayRef input_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride)",
    ]

    for sig in sigs:
        cpp_sig = CPPSig(sig.replace("c10::", '').replace("at::", ''))
        print(">>>>>>>>>")
        print(cpp_sig.contain_alias_tensor)
        print(cpp_sig.contain_output_tensor)
        for param in cpp_sig.input_params:
            print(param.name)
        print("***")
        for param in cpp_sig.ret_params:
            print(param.core_type)
            for sub_param in param.sub_params:
                print(sub_param.core_type)
        print("<<<<<<<<<<")
