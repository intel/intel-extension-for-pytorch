import sys
import argparse
import copy
from copy import deepcopy
import os
import yaml
import re
import ast
try:
    # use faster C loader if available
    from yaml import CLoader as YamlLoader
except ImportError:
    from yaml import Loader as YamlLoader


MICROBENCH_DISPATCH_KEY = 'PrivateUse3'


def get_simple_type(arg):
    simple_type = arg['type']
    simple_type = simple_type.replace(' &', '').replace('const ', '')
    simple_type = simple_type.replace('Generator *', 'Generator')
    opt_match = re.match(r'c10::optional<(.+)>', simple_type)
    if opt_match:
        simple_type = '{}?'.format(opt_match.group(1))
    return simple_type


def has_tensoroptions_argument(declaration):
    for argument in declaration['arguments']:
        if 'TensorOptions' == argument['dynamic_type']:
            return True
    return False


def process_schema_order_arg(schema_order_arg):
    if schema_order_arg == 'dtype':
        return 'optTypeMetaToScalarType(options.dtype_opt())'
    elif schema_order_arg == 'layout':
        return 'options.layout_opt()'
    elif schema_order_arg == 'device':
        return 'options.device_opt()'
    elif schema_order_arg == 'pin_memory':
        return 'options.pinned_memory_opt()'
    elif schema_order_arg == 'memory_format':
        return 'c10::impl::check_tensor_options_and_extract_memory_format(options, memory_format)'
    else:
        return schema_order_arg


def format_return_type(returns):
    if len(returns) == 0:
        return 'void'
    elif len(returns) == 1:
        return returns[0]['type']
    else:
        return_types = [r['type'] for r in returns]
        return 'std::tuple<{}>'.format(','.join(return_types))


def load_aten_declarations(path):
    with open(path, 'r') as f:
        declarations = yaml.load(f, Loader=YamlLoader)

    # enrich declarations with additional information
    selected_declarations = []
    for declaration in declarations:
        if declaration.get('deprecated'):
            continue

        for arg in declaration['arguments']:
            arg['simple_type'] = get_simple_type(arg)
        for ret in declaration['returns']:
            ret['simple_type'] = get_simple_type(ret)

        declaration['formals'] = [arg['type'] + ' ' + arg['name']
                                  for arg in declaration['arguments']]
        declaration['schema_order_formals'] = [arg['type'] + ' ' + arg['name']
                                               for arg in declaration['schema_order_arguments']]
        declaration['args'] = [arg['name'] for arg in declaration['arguments']]
        declaration['schema_order_args'] = [arg['name'] for arg in declaration['schema_order_arguments']]
        if has_tensoroptions_argument(declaration):
            declaration['schema_order_args'] = [process_schema_order_arg(
                arg) for arg in declaration['schema_order_args']]
        declaration['api_name'] = declaration['name']
        if declaration.get('overload_name'):
            declaration['type_wrapper_name'] = "{}_{}".format(
                declaration['name'], declaration['overload_name'])
        else:
            declaration['type_wrapper_name'] = declaration['name']
        declaration['operator_name_with_overload'] = declaration['schema_string'].split('(')[0]
        declaration['unqual_operator_name_with_overload'] = declaration['operator_name_with_overload'].split('::')[1]
        declaration['return_type'] = format_return_type(declaration['returns'])

        declaration['base_name'] = declaration['name']
        declaration['type_method_definition_dispatch'] = {}
        selected_declarations.append(declaration)

    return selected_declarations


def load_exclude_file(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if len(line) > 2]
    return lines


def get_bench_verbose(option, name_outer=['']):
    def wrap_(s):
        return "\"" + s + "\""
    type_wrapper_name = option['type_wrapper_name']
    op_class_name = option['name']
    if op_class_name in name_outer:
        return ''
    data = "  std::cout<< \"[" + op_class_name + "] \";"
    for argument in option['declaration_formals']:
        arg_type = argument[:argument.rfind(' ')].strip()
        arg_name = argument[argument.rfind(' ') + 1:].strip()
        data += " argprint({0}, {1}, {2});".format(arg_name, wrap_(arg_name), wrap_(arg_type))
    func_name = 'mb_' + type_wrapper_name
    return data + " std::cout<<\"{" + func_name + '}\"<<std::endl;\n'


def gen_code(aten_path, out_file):
    exclude_set = ['choose_qparams_optimized']
    full_aten_decls = load_aten_declarations(aten_path)
    output = "#pragma once\n#include <torch/extension.h>\n#include <iostream>\nusing namespace at;\n\nnamespace microbench {\n\n"
    wrapper_metas = []
    for declaration in full_aten_decls:
        name = declaration['name']
        operator_name = declaration['operator_name']
        if declaration.get('overload_name'):
            overload_name = declaration['overload_name']
            wrapper_name = "{}_{}".format(name, overload_name)
        else:
            overload_name = ''
            wrapper_name = name
        if wrapper_name in exclude_set:
            continue
        return_type = declaration['return_type']
        make_unboxed_only = False
        if declaration['use_c10_dispatcher'] == 'full':
            declaration_formals = declaration['schema_order_formals']
        elif declaration['use_c10_dispatcher'] == 'with_codegenerated_unboxing_wrapper':
            make_unboxed_only = True
            declaration_formals = declaration['formals']
        else:
            assert declaration['use_c10_dispatcher'] == 'hacky_wrapper_for_legacy_signatures'
            declaration_formals = declaration['schema_order_formals']
        arg_types = []
        arg_names = []
        for argument in declaration_formals:
            arg_types.append(argument[:argument.rfind(' ')].strip())
            arg_names.append(argument[argument.rfind(' ') + 1:].strip())
        op_output = "{0} {1}({2})\n".format(return_type, wrapper_name, ", ".join(declaration_formals)) + "{\n"

        option = {
            'type_wrapper_name': wrapper_name,
            'name': name,
            'declaration_formals': declaration_formals,
        }
        op_output += get_bench_verbose(option)

        op_type_signature = "{0}({1})".format(
            return_type, ", ".join(arg_types))

        op_output += "  auto op = c10::Dispatcher::singleton().findSchemaOrThrow(\"{0}\", \"{1}\").typed<{2}>();\n  ".format(
            'aten::' + operator_name, overload_name, op_type_signature)
        all_types = return_type + ', ' + ", ".join(arg_types)
        if all_types.endswith(', '):
            all_types = all_types[:-2]
        if len(arg_names) > 0:
            op_output += "return c10::Dispatcher::singleton().redispatch<{0}>(op, c10::DispatchKey::{1}, {2});\n".format(
                all_types, MICROBENCH_DISPATCH_KEY, ", ".join(arg_names))
        else:
            op_output += "return c10::Dispatcher::singleton().redispatch<{0}>(op, c10::DispatchKey::{1});\n".format(
                all_types, MICROBENCH_DISPATCH_KEY)
        op_output += '}\n\n'
        output += op_output
        wrapper_metas.append([name, operator_name, overload_name, wrapper_name,
                              make_unboxed_only, arg_types, arg_names, op_type_signature])
    output += "}\n\nTORCH_LIBRARY_IMPL(aten, " + MICROBENCH_DISPATCH_KEY + ", m)\n{\n"
    for item in wrapper_metas:
        name, operator_name, overload_name, wrapper_name, make_unboxed_only, _, _, _ = item
        aten_name = ".".join([operator_name, overload_name]) if len(overload_name) > 0 else name
        aten_name = 'aten::' + aten_name
        microbench_impl = 'microbench::' + wrapper_name
        if not make_unboxed_only:
            output += "  m.impl(\"{0}\", TORCH_FN({1}));\n".format(aten_name, microbench_impl)
        else:
            output += "  m.impl(\"{0}\", torch::dispatch(c10::DispatchKey::{1}, torch::CppFunction::makeUnboxedOnly(&{2})));\n".format(
                aten_name, MICROBENCH_DISPATCH_KEY, microbench_impl)
    output += "}\n\n"
    output += "#define MICRO_BENCH_REGISTER \\\n"
    output += "py::class_<at::Scalar>(m, \"Scalar\").def(py::init<>()); \\\n"
    output += "m.def(\"bench_scalar_slow\", [](py::handle t) { return scalar_slow(t.ptr());}); \\\n"
    for item in wrapper_metas:
        name, operator_name, overload_name, wrapper_name, make_unboxed_only, arg_types, arg_names, op_type_signature = item
        args = []
        for t, n in zip(arg_types, arg_names):
            args.append(t.strip() + " " + n.strip())
        args = ", ".join(args)
        func_for_call = "mb_" + wrapper_name
        aten_operator_name = 'aten::' + operator_name
        output += "m.def(\"" + func_for_call + "\", [](" + args + \
            "){{ auto op = c10::Dispatcher::singleton().findSchemaOrThrow(\"{0}\", \"{1}\").typed<{2}>(); return op.call(".format(
                aten_operator_name, overload_name, op_type_signature) + ", ".join(arg_names) + "); }); \\\n"
    with open(out_file, 'w') as f:
        f.write(output)


def main():
    parser = argparse.ArgumentParser(
        description='Generate microbench code dispatch register script')
    parser.add_argument('--declarations-path',
                        help='path to Declarations.yaml')
    parser.add_argument('--out', help='path to output')
    args = parser.parse_args()
    gen_code(args.declarations_path, args.out)


if __name__ == '__main__':
    main()
