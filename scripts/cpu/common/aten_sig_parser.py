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

_ATEN_SIG_GRAMMAR = r"""
    start:aten_ns fnname "(" (params)* ")" "->" return_type
    aten_ns: ATEN_NS
    fnname: CNAME "."* (CNAME)*
    params: param
          | param "," params
    param: type is_optional is_vec is_optional param_name (param_defval)?
          | "*"
    type: CNAME
        | CNAME "(" is_alias is_w ")"
    param_name: CNAME
    param_defval: "=" init_value
    init_value: "True"
              | "False"
              | "None"
              | "[" (array_list)* "]"
              | "{}"
              | NUMBER
              | FLOAT
              | SIGNED_NUMBER
              | ESCAPED_STRING
              | CNAME

    array_list: array_val
              | array_val "," array_list
    array_val: NUMBER*
             | CNAME
    return_type: "("* ret_param_list ")"*
               | "()"

    ret_param_list: ret_param
                  | ret_param "," ret_param_list

    ret_param: ret_tensor
             | non_tensor_type

    ret_tensor: const_tensor_param
              | alias_tensor_param
              | alias_w_tensor_param

    const_tensor_param: const_tensor_type (tensor_name)*
    alias_tensor_param: alias_tensor_type (tensor_name)*
    alias_w_tensor_param: alias_w_tensor_type (tensor_name)*

    const_tensor_type: TENSOR_TYPE is_vec
    alias_tensor_type: TENSOR_TYPE "(" ALIAS ")" is_vec
    alias_w_tensor_type: TENSOR_TYPE "(" ALIAS W_SYM ")" is_vec

    tensor_name: CNAME
    non_tensor_type: CNAME

    is_vec: (vec_type)*
    is_optional: (OPTIONAL_SYM)*
    is_alias: (ALIAS)*
    is_w: (W_SYM)*

    vec_type: "[" vec_number "]"
    vec_number: (NUMBER)*

    ATEN_NS: "aten::"
    VEC: "[" (NUMBER)* "]"
    TENSOR_TYPE: "Tensor"
    W_SYM: "!"
    OPTIONAL_SYM: "?"
    ALIAS: LETTER
    TENSOR_NAME: CNAME
    TYPE_NAME: LETTER+

    %import common.LETTER -> LETTER
    %import common.CNAME -> CNAME
    %import common.NUMBER -> NUMBER
    %import common.FLOAT -> FLOAT
    %import common.SIGNED_NUMBER -> SIGNED_NUMBER
    %import common.ESCAPED_STRING -> ESCAPED_STRING
    %import common.WS
    %ignore WS
    """

_ATEN_SIG_PARSER = lark.Lark(_ATEN_SIG_GRAMMAR, parser='lalr', propagate_positions=True)
_X_ATEN_SIG_PARSER = lark.Lark(_ATEN_SIG_GRAMMAR, parser='lalr', propagate_positions=True, keep_all_tokens=True)

class AtenSig(SigParser):
    def __init__(self, aten_sig):
        super(AtenSig, self).__init__(aten_sig, _ATEN_SIG_PARSER)

    def __extract_all_params(self, params_root_tree):
        for params_tree in params_root_tree.children:
            if params_tree.data == 'param':
                cur_param = Param()
                for param_item in params_tree.children:
                    if param_item.data == 'type':
                        token = param_item.children[0]
                        assert isinstance(token, lark.lexer.Token)
                        cur_param.core_type = token.value
                        if cur_param.core_type == 'Tensor':
                            cur_param.is_tensor = True

                        for param_alias_w in param_item.children[1:]:
                            if param_alias_w.data == 'is_alias':
                                cur_param.is_alias = True
                            elif param_alias_w.data == 'is_w':
                                if len(param_alias_w.children) > 0:
                                    cur_param.is_to_be_written = True
                            else:
                                print("** {} **".format(param_alias_w.data))
                                assert False
                    elif param_item.data == 'is_optional':
                        if len(param_item.children) > 0:
                            cur_param.is_optional = True
                    elif param_item.data == 'is_vec':
                        if len(param_item.children) > 0:
                            cur_param.is_vec = True
                    elif param_item.data == 'is_alias':
                        if len(param_item.children) > 0:
                            cur_param.is_alias = True
                    elif param_item.data == 'is_w':
                        if len(param_item.children) > 0:
                            cur_param.is_to_be_written = True
                    elif param_item.data == 'param_name':
                        token = param_item.children[0]
                        assert isinstance(token, lark.lexer.Token)
                        cur_param.name = token.value
                    elif param_item.data == 'param_defval':
                        None
                    else:
                        print( param_item.data )
                        assert False

                if cur_param.core_type != None and cur_param.core_type != '':
                    self.input_params.append(cur_param)
            else:
                assert params_tree.data == 'params'
                return self.__extract_all_params(params_tree)

    def get_all_input_params(self):
        for sub_tree in self._sig_tree.children:
            if sub_tree.data != "params":
                continue

            return self.__extract_all_params(sub_tree)

    def get_all_return_params(self):
        None

if __name__ == '__main__':
    sigs = [
    "aten::abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
    "aten::abs_(Tensor(a!) self) -> Tensor(a!)",
    "aten::angle(Tensor self) -> Tensor",
    "aten::acos_(Tensor(a!) self) -> Tensor(a!)",
    "aten::adaptive_max_pool1d(Tensor self, int[1] output_size) -> (Tensor, Tensor)",
    "aten::median.dim_values(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)",
    "aten::min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)",
    ]

    for sig in sigs:
        aten_sig = AtenSig(sig, _ATEN_SIG_PARSER)
        print("------------------")
        print(aten_sig.contain_alias_tensor)
        print(aten_sig.contain_output_tensor)
        print("<<<<<")
        for param in aten_sig.input_params:
            print(param.core_type)
        print(">>>>>")
