import os
import yaml
import argparse
try:
    # use faster C loader if available
    from yaml import CLoader as YamlLoader
except ImportError:
    from yaml import Loader as YamlLoader


MICROBENCH_DISPATCH_KEY = 'PrivateUse3'


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
    selected_declarations = []
    for declaration in declarations:
        if declaration.get('deprecated'):
            continue
        declaration['schema_order_formals'] = \
            [arg['type'] + ' ' + arg['name']
                for arg in declaration['schema_order_arguments']]
        declaration['return_type'] = format_return_type(declaration['returns'])
        selected_declarations.append(declaration)
    return selected_declarations


def get_bench_verbose(name, wrapper_name, arg_names, arg_types):
    def wrap_(s):
        return "\"" + s + "\""
    data = "  std::cout<< \"[" + name + "] \";"
    for arg_name, arg_type in zip(arg_names, arg_types):
        data += " argprint({0}, {1}, {2});".format(arg_name,
                                                   wrap_(arg_name), wrap_(arg_type))
    return data + " std::cout<<\"{mb_" + wrapper_name + '}\"<<std::endl;\n'


def gen_code(aten_path, out_file):
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
        if wrapper_name in ['choose_qparams_optimized']:
            continue
        return_type = declaration['return_type']
        declaration_formals = declaration['schema_order_formals']
        arg_types = []
        arg_names = []
        for argument in declaration_formals:
            arg_types.append(argument[:argument.rfind(' ')].strip())
            arg_names.append(argument[argument.rfind(' ') + 1:].strip())
        if len(declaration_formals) > 0:
            op_output = "{0} {1}({2})\n".format(return_type, wrapper_name,
                                                "c10::DispatchKeySet ks, " + ", ".join(declaration_formals)) + "{\n"
        else:
            op_output = "{0} {1}({2})\n".format(return_type, wrapper_name, "c10::DispatchKeySet ks") + "{\n"
        op_output += get_bench_verbose(name,
                                       wrapper_name, arg_names, arg_types)
        op_type_signature = "{0}({1})".format(
            return_type, ", ".join(arg_types))
        aten_operator_name = 'aten::' + operator_name
        op_output += "  auto op = c10::Dispatcher::singleton().findSchemaOrThrow(\"{0}\", \"{1}\").typed<{2}>();\n".format(
            aten_operator_name, overload_name, op_type_signature)
        if len(arg_names) > 0:
            op_output += "  return op.redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::{0}), {1});\n".format(
                MICROBENCH_DISPATCH_KEY, ", ".join(arg_names))
        else:
            op_output += "  return op.redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::{0}));\n".format(
                MICROBENCH_DISPATCH_KEY)
        op_output += '}\n\n'
        output += op_output
        wrapper_metas.append([name, operator_name, overload_name,
                              wrapper_name, arg_types, arg_names, op_type_signature])
    output += "}\n\nTORCH_LIBRARY_IMPL(aten, " + \
        MICROBENCH_DISPATCH_KEY + ", m)\n{\n"
    for item in wrapper_metas:
        name, operator_name, overload_name, wrapper_name, _, _, _ = item
        aten_name = ".".join([operator_name, overload_name]) if len(
            overload_name) > 0 else name
        aten_name = 'aten::' + aten_name
        microbench_impl = 'microbench::' + wrapper_name
        output += "  m.impl(\"{0}\", c10::DispatchKey::{1}, TORCH_FN({2}));\n".format(
            aten_name, MICROBENCH_DISPATCH_KEY, microbench_impl)
    output += "}\n\n"
    output += "#define MICRO_BENCH_REGISTER \\\n"
    output += "py::class_<at::Scalar>(m, \"Scalar\").def(py::init<>()); \\\n"
    output += "m.def(\"bench_scalar_slow\", [](py::handle t) { return scalar_slow(t.ptr());}); \\\n"
    for item in wrapper_metas:
        name, operator_name, overload_name, wrapper_name, arg_types, arg_names, op_type_signature = item
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
