from __future__ import print_function

import argparse
import collections
import lark
import os
import re
import string
import sys

_ATEN_SIG_GRAMMAR = r"""
    start:aten_ns fnname "(" (params)* ")" "->" return_type
    aten_ns: ATEN_NS
    fnname: CNAME "."* (CNAME)*
    params: param
          | param "," params
    param: type is_optional is_vec param_name (param_defval)?
          | "*"
    type: TENSOR_NAME
        | TENSOR_NAME "(" is_alias is_w ")"
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

_PARSER = lark.Lark(_ATEN_SIG_GRAMMAR, parser='lalr', propagate_positions=True)
_XPARSER = lark.Lark(_ATEN_SIG_GRAMMAR, parser='lalr', propagate_positions=True, keep_all_tokens=True)

sigs = [
    "aten::col2im(Tensor self, int[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor"
]

for sig in sigs:
    print(sig)
    xtree = _PARSER.parse(sig)
    print(xtree.pretty())