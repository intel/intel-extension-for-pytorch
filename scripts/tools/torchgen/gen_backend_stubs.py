import argparse
import os
import pathlib
import re
from collections import Counter, defaultdict, namedtuple
from typing import Any, Dict, List, Iterable, Optional, Sequence, Set, TypeVar, Union

import yaml

import torchgen.api.dispatcher as dispatcher
import torchgen.dest as dest
from torchgen.api.types import DispatcherSignature
from torchgen.code_template import CodeTemplate
from torchgen.context import native_function_manager
from torchgen.gen import get_grouped_native_functions, parse_native_yaml
from torchgen.model import (
    BackendIndex,
    BackendMetadata,
    DispatchKey,
    NativeFunction,
    NativeFunctionsGroup,
    OperatorName,
)
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import concatMap, context, FileManager, NamespaceHelper, Target
from torchgen.yaml_utils import YamlLoader

T = TypeVar("T")
S = TypeVar("S")

# Parses the external backend's yaml, and adds a new BackendIndex for the backend's dispatch key.
# Returns a Tuple of (backend_key, autograd_key, cpp_namespace, updated BackendIndex mapping)
ParsedExternalYaml = namedtuple(
    "ParsedExternalYaml",
    ["backend_key", "autograd_key", "class_name", "cpp_namespace", "backend_indices"],
)


def parse_backend_yaml(
    backend_yaml_path: str,
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    backend_indices: Dict[DispatchKey, BackendIndex],
) -> ParsedExternalYaml:
    native_functions_map: Dict[OperatorName, NativeFunction] = {
        f.func.name: f
        for f in concatMap(
            lambda f: [f] if isinstance(f, NativeFunction) else list(f.functions()),
            grouped_native_functions,
        )
    }

    with open(backend_yaml_path) as f:
        yaml_values = yaml.load(f, Loader=YamlLoader)
    assert isinstance(yaml_values, dict)

    valid_keys = [
        "backend",
        "class_name",
        "cpp_namespace",
        "extra_headers",
        "supported",
        "autograd",
        "full_codegen",
        "non_native",
        "ir_gen",
        "symint",
    ]

    backend = yaml_values.pop("backend", None)
    assert backend is not None, 'You must provide a value for "backend"'

    class_name = yaml_values.pop("class_name", None)

    cpp_namespace = yaml_values.pop("cpp_namespace", None)
    assert cpp_namespace is not None, 'You must provide a value for "cpp_namespace"'

    # Mostly just defaulting to false to stick with LazyTensor convention.
    use_out_as_primary = yaml_values.pop("use_out_as_primary", False)
    assert isinstance(
        use_out_as_primary, bool
    ), f"You must provide either True or False for use_out_as_primary. Provided: {use_out_as_primary}"

    use_device_guard = yaml_values.pop("device_guard", False)
    assert isinstance(
        use_device_guard, bool
    ), f"You must provide either True or False for device_guard. Provided: {use_device_guard}"

    supported = yaml_values.pop("supported", [])
    if supported is None:
        supported = []  # Allow an empty list of supported ops
    assert isinstance(
        supported, list
    ), f'expected "supported" to be a list, but got: {supported} (of type {type(supported)})'

    symint = yaml_values.pop("symint", [])
    if symint is None:
        symint = []  # Allow an empty list of symint ops
    assert isinstance(
        symint, list
    ), f'expected "symint" to be a list, but got: {supported} (of type {type(supported)})'
    symint_set = set(symint)

    supported_autograd = yaml_values.pop("autograd", [])
    assert isinstance(
        supported_autograd, list
    ), f'expected "autograd" to be a list, but got: {supported_autograd}'

    # full_codegen is ignored by parse_backend_yaml, and re-parsed in gen_lazy_tensor.py
    full_codegen = yaml_values.pop("full_codegen", [])
    supported.extend(full_codegen)

    # non_native is ignored by parse_backend_yaml, and re-parsed in gen_lazy_tensor.py
    non_native = yaml_values.pop("non_native", {})

    # ir_gen is ignored by parse_backend_yaml, and re-parsed in gen_lazy_tensor.py
    _ = yaml_values.pop("ir_gen", {})

    assert (
        len(yaml_values.keys()) == 0
    ), f'{backend_yaml_path} contains unexpected keys: {", ".join(yaml_values.keys())}. \
Only the following keys are supported: {", ".join(valid_keys)}'

    def create_backend_index(
        backend_ops: List[str],
        symint_ops: Set[str],
        dispatch_key: DispatchKey,
        *,
        use_out_as_primary: bool,
        use_device_guard: bool,
    ) -> BackendIndex:
        metadata: Dict[OperatorName, BackendMetadata] = {}
        for op in backend_ops:
            op_name = OperatorName.parse(op)
            assert (
                op_name in native_functions_map
            ), f"Found an invalid operator name: {op_name}"
            # See Note [External Backends Follow Dispatcher API]
            kernel_name = dispatcher.name(native_functions_map[op_name].func)
            if op in symint_ops:
                kernel_name += "_symint"
            # TODO: allow structured external backends later.
            m = BackendMetadata(
                kernel=kernel_name, structured=False, cpp_namespace=cpp_namespace
            )
            metadata[op_name] = m
        return BackendIndex(
            dispatch_key=dispatch_key,
            use_out_as_primary=use_out_as_primary,
            external=True,
            device_guard=use_device_guard,
            index=metadata,
        )

    backend_key: Optional[DispatchKey] = None
    if len(supported) > 0:
        with context(
            lambda: f'The provided value for "backend" must be a valid DispatchKey, but got {backend}.'
        ):
            backend_key = DispatchKey.parse(backend)

        backend_idx = create_backend_index(
            supported,
            symint_set,
            backend_key,
            use_out_as_primary=use_out_as_primary,
            use_device_guard=use_device_guard,
        )
        assert backend_key not in backend_indices
        backend_indices[backend_key] = backend_idx

    autograd_key: Optional[DispatchKey] = None
    if len(supported_autograd) > 0:
        with context(
            lambda: f'The "autograd" key was specified, which indicates that you would like to override \
the behavior of autograd for some operators on your backend. However "Autograd{backend}" is not a valid DispatchKey.'
        ):
            autograd_key = DispatchKey.parse(f"Autograd{backend}")

        autograd_idx = create_backend_index(
            supported_autograd,
            symint_set,
            autograd_key,
            use_out_as_primary=use_out_as_primary,
            use_device_guard=use_device_guard,
        )
        assert autograd_key not in backend_indices
        backend_indices[autograd_key] = autograd_idx

    for g in grouped_native_functions:
        if isinstance(g, NativeFunction):
            forward_kernels = (
                []
                if backend_key is None
                else [
                    m
                    for m in [backend_indices[backend_key].get_kernel(g)]
                    if m is not None
                ]
            )
            backward_kernels = (
                []
                if autograd_key is None
                else [
                    m
                    for m in [backend_indices[autograd_key].get_kernel(g)]
                    if m is not None
                ]
            )
        else:
            forward_kernels = (
                []
                if backend_key is None
                else [
                    m
                    for m in [
                        backend_indices[backend_key].get_kernel(f)
                        for f in g.functions()
                    ]
                    if m is not None
                ]
            )
            backward_kernels = (
                []
                if autograd_key is None
                else [
                    m
                    for m in [
                        backend_indices[autograd_key].get_kernel(f)
                        for f in g.functions()
                    ]
                    if m is not None
                ]
            )

        forward_kernels = [f for f in forward_kernels if f is not None]
        backward_kernels = [f for f in backward_kernels if f is not None]
        assert (
            len(forward_kernels) == 0 or len(backward_kernels) == 0
        ), f'Currently, all variants of an op must either be registered to a backend key, or to a backend\'s \
autograd key. They cannot be mix and matched. If this is something you need, feel free to create an issue! \
{forward_kernels[0].kernel} is listed under "supported", but {backward_kernels[0].kernel} is listed under "autograd".'

    return ParsedExternalYaml(
        backend_key, autograd_key, class_name, cpp_namespace, backend_indices
    )


def error_on_missing_kernels(
    native_functions: Sequence[NativeFunction],
    backend_indices: Dict[DispatchKey, BackendIndex],
    backend_key: DispatchKey,
    autograd_key: Optional[DispatchKey],
    class_name: str,
    kernel_defn_file_path: str,
    full_codegen: Optional[List[OperatorName]] = None,
) -> None:
    try:
        with open(kernel_defn_file_path) as f:
            backend_defns = f.read()
    except OSError as e:
        raise AssertionError(
            f"Unable to read from the specified impl_path file: {kernel_defn_file_path}"
        ) from e

    if full_codegen is None:
        full_codegen = []

    indices = [backend_indices[backend_key].index] + (
        [] if autograd_key is None else [backend_indices[autograd_key].index]
    )
    # Quick mapping from each OperatorName used by the external backend
    # to its backend kernel name
    expected_backend_op_names: Dict[OperatorName, str] = dict(
        list(
            concatMap(
                lambda index: [
                    (op_name, metadata.kernel) for op_name, metadata in index.items()
                ],
                indices,
            )
        )
    )
    expected_backend_native_funcs: List[NativeFunction] = [
        f
        for f in native_functions
        if f.func.name in expected_backend_op_names.keys()
        and f.func.name not in full_codegen
    ]
    expected_backend_kernel_name_counts: Dict[str, List[NativeFunction]] = defaultdict(
        list
    )
    for native_f in expected_backend_native_funcs:
        expected_backend_kernel_name_counts[
            expected_backend_op_names[native_f.func.name]
        ].append(native_f)

    # This just looks for lines containing "foo(", and assumes that the kernel foo has been implemented.
    # It might cause false negatives (we won't catch all cases), but that's ok - if we catch a missing kernel
    # here, then we get a nicer error message. If we miss it, you get a linker error.
    kernel_defn_regex = rf"(.*){class_name}::\s*([\w\d]*)\("
    actual_backend_kernel_name_counts = Counter(
        # A bit unwieldy (this could probably be moved into regex),
        # but we don't want to include kernel names that come from function calls,
        # like "return torch_xla::XLANativeFunctions::empty_strided_symint(...)".
        # Easy check is to ignore any lines with colons before the class name.
        [
            y
            for (x, y) in re.findall(kernel_defn_regex, backend_defns)
            if not x.endswith(":")
        ]
    )

    missing_kernels_err_msg = ""
    for expected_name, funcs in expected_backend_kernel_name_counts.items():
        expected_overload_count = len(funcs)
        actual_overload_count = actual_backend_kernel_name_counts[expected_name]
        if expected_overload_count != actual_overload_count:

            def create_decl(f: NativeFunction) -> str:
                with native_function_manager(f):
                    return DispatcherSignature.from_schema(f.func).decl()

            expected_schemas_str = "\n".join([create_decl(f) for f in funcs])
            missing_kernels_err_msg += f"""
{class_name} is missing a kernel definition for {expected_name}. We found {actual_overload_count} kernel(s) with that name,
but expected {expected_overload_count} kernel(s). The expected function schemas for the missing operator are:
{expected_schemas_str}

"""
    assert missing_kernels_err_msg == "", missing_kernels_err_msg


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate backend stub files")
    parser.add_argument(
        "-s",
        "--source-yaml",
        "--source_yaml",
        help="path to source yaml file containing operator external definitions",
    )
    parser.add_argument("-o", "--output-dir", "--output_dir", help="output directory")
    parser.add_argument(
        "--dry-run", "--dry_run", type=bool, default=False, help="output directory"
    )
    parser.add_argument(
        "--impl-path",
        "--impl_path",
        type=str,
        default=None,
        help="path to the source C++ file containing kernel definitions",
    )
    parser.add_argument(
        "--generate-impl",
        "--generate_impl",
        action='store_true',
        default=False,
        help="generate the kernel implementation of directly calling at::native::func",
    )
    parser.add_argument(
        "--header_file",
        "--header-file",
        default=None,
        help="the header file for the output kernel implementation",
    )
    parser.add_argument(
        "--output-impl-file",
        "--output_impl_file",
        default=None,
        help="the output file for the kernel implementation",
    )
    parser.add_argument(
        '--simple_trace',
        action='store_true',
        default=False,
        help='simple trace the entry to ipex',
    )
    options = parser.parse_args()

    run(
        options.source_yaml,
        options.output_dir,
        options.dry_run,
        options.simple_trace,
        options.impl_path,
        options.generate_impl,
        options.header_file,
        options.output_impl_file
    )


def gen_dispatchkey_nativefunc_headers(
    fm: FileManager,
    class_name: str,
    cpp_namespace: str,
    backend_indices: Dict[DispatchKey, BackendIndex],
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    backend_dispatch_key: DispatchKey,
    autograd_dispatch_key: Optional[DispatchKey],
    backend_name: str = "",
) -> None:
    assert class_name is not None
    generated_comment = (
        "Autogenerated file by gen_backend_stubs.py. Do not edit directly!"
    )

    # Convert to a set first to remove duplicate kernel names.
    # Backends are allowed to repeat kernel names; only generate the declaration once!
    # Sort for deterministic output.
    backend_declarations = sorted(
        set(
            concatMap(
                lambda f: dest.compute_native_function_declaration(
                    f, backend_indices[backend_dispatch_key]
                ),
                grouped_native_functions,
            )
        )
    )
    autograd_declarations = sorted(
        set(
            concatMap(
                lambda f: []
                if autograd_dispatch_key is None
                else dest.compute_native_function_declaration(
                    f, backend_indices[autograd_dispatch_key]
                ),
                grouped_native_functions,
            )
        )
    )

    ns_helper = NamespaceHelper(cpp_namespace)
    fm.write_with_template(
        f"{backend_dispatch_key}NativeFunctions.h",
        "DispatchKeyNativeFunctions.h",
        lambda: {
            "generated_comment": generated_comment,
            "namespace_prologue": ns_helper.prologue,
            "class_name": class_name,
            "namespace_epilogue": ns_helper.epilogue,
            "dispatch_declarations": backend_declarations + autograd_declarations,
            "BackendName": backend_name,
            "DispatchKey": backend_dispatch_key,
        },
    )


def gen_implementation(
    output_impl_file: str,
    header_file: str,
    impl_ns: str,
    native_yaml_file: str,
    search_dispatch_key: str,
    class_name: str,
    cpp_namespace: str,
    backend_indices: Dict[DispatchKey, BackendIndex],
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    backend_dispatch_key: DispatchKey,
) -> None:
    # impl_file is the needed generated file
    impl_file = open(output_impl_file, "w")
    yaml_file = open(native_yaml_file, "r").read()

    # for namespace
    ns_helper = NamespaceHelper(cpp_namespace)
    class_prologue = 'namespace ' + class_name + ' {\n'
    class_epilogue = '} // namespace ' + class_name + '\n'

    def collect_schema_args(
        g: Union[NativeFunctionsGroup, NativeFunction], backend_index: BackendIndex
    ) -> List[Any]:
        from .dest.native_functions import gen_unstructured

        def collect_arg_func_mapping(r: str, x: NativeFunction) -> Dict[str, Dict[str, List[str]]]:
            # {schema name : {schema: [arguments]}}
            collect_all: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
            schema_name = x.func.name.name.base

            # check schema overload
            if x.func.name.name.inplace:
                schema_name += '_'
            if len(x.func.name.overload_name):
                schema_name += '.'
                schema_name += x.func.name.overload_name

            # set the schema name, for example, div.Tensor
            collect_all[schema_name] = {r : []}

            # collect arg, pre + arg + post + out
            args = x.func.arguments
            if hasattr(args, 'pre_self_positional'):
                for kwarg in args.pre_self_positional:
                    collect_all[schema_name][r].append(kwarg.name)
            if hasattr(args, 'self_arg') and args.self_arg is not None:
                collect_all[schema_name][r].append(args.self_arg.argument.name)
            if hasattr(args, 'post_self_positional'):
                for other_arg in args.post_self_positional:
                    collect_all[schema_name][r].append(other_arg.name)
            if hasattr(args, 'pre_tensor_options_kwarg_only'):
                for kwarg in args.pre_tensor_options_kwarg_only:
                    collect_all[schema_name][r].append(kwarg.name)
            if hasattr(args, 'tensor_options'):
                tensor_args = args.tensor_options
                for key in ['dtype', 'layout', 'device', 'pin_memory']:
                    if hasattr(tensor_args, key):
                        collect_all[schema_name][r].append(getattr(tensor_args, key).name)
            if hasattr(args, 'post_tensor_options_kwarg_only'):
                for kwarg in args.post_tensor_options_kwarg_only:
                    collect_all[schema_name][r].append(kwarg.name)
            if hasattr(args, 'out'):
                for arg in args.out:
                    collect_all[schema_name][r].append(arg.name)
            return collect_all

        def map_collect(groups_func: Iterable[T]):
            collect_all = []
            for func in groups_func:
                r = gen_unstructured(func, backend_index)
                if r is not None:
                    collect_all.append(collect_arg_func_mapping(r, func))
            return collect_all

        # for functions group, it should be iterated for each function
        if isinstance(g, NativeFunctionsGroup):
            return map_collect(g.functions())
        else:
            r = gen_unstructured(g, backend_index)
            if r is not None:
                return [collect_arg_func_mapping(r, g)]

    # {schema_name : {schema; : [arguments(str)]}}
    collected_signature_args: List[Dict[str, Dict[str, List[str]]]] = []
    for g in grouped_native_functions:
        func_signature_args = collect_schema_args(g, backend_indices[backend_dispatch_key])
        if func_signature_args is not None and len(func_signature_args) > 0:
            for elem in func_signature_args:
                collected_signature_args.append(elem)

    # write the header file to the output file
    header_native_function = '#include <ATen/NativeFunctions.h>'
    impl_file.write(header_native_function + '\n')
    if header_file is not None:
        header_file = '#include "' + header_file + '"'
        impl_file.write(header_file + '\n')
    impl_file.write(ns_helper.prologue + '\n')
    impl_file.write(class_prologue)

    # find the target kernel func name to make it as callee
    def generate_impl_func_name(func: str):
        func_index = yaml_file.find('- func: ' + func + '(')
        assert func_index != -1, 'Cannot find the func {}'.format(func)
        target = yaml_file[func_index:]
        target_index = target.find(search_dispatch_key)
        target = target[target_index:target_index + target[target_index:].find('\n')]
        return target.replace(search_dispatch_key + ': ', '')

    def generate_func(elem: Dict[str, Dict[str, List[str]]]):
        for func, shcema_args in elem.items():
            for schema, args in shcema_args.items():
                func_head = schema.lstrip().replace(';', ' {\n')

                # generate func args
                func_arg = ''
                for arg in args:
                    arg += ', '
                    func_arg += arg

                # remove the tail arg ','
                if len(func_arg) > 0:
                    func_arg = func_arg[:-2]
                func_name = generate_impl_func_name(func) + '('
                func_body = '    return ' + cpp_namespace + '::' + impl_ns + '::' + func_name + func_arg + ');\n}\n\n'
                full_func = func_head + func_body
                return full_func

    for elem in collected_signature_args:
        impl_file.write(generate_func(elem))

    # write the epilogue
    impl_file.write(class_epilogue)
    impl_file.write(ns_helper.epilogue + '\n')


def gen_dispatcher_registrations(
    fm: FileManager,
    output_dir: str,
    class_name: str,
    backend_indices: Dict[DispatchKey, BackendIndex],
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    backend_dispatch_key: DispatchKey,
    simple_trace: bool,
    dispatch_key: DispatchKey,
    selector: "SelectiveBuilder",
    # build_in_tree is true for lazy TS backend and affects include paths, not used for external backends
    build_in_tree: bool = False,
    per_operator_headers: bool = False,
    backend_name: str = "",
    eager_registration: bool = True,
) -> None:
    headers = [
        f"{output_dir}/{backend_dispatch_key}NativeFunctions.h",
    ]
    if build_in_tree:
        external_backend_headers_str = "\n".join(f"#include <{h}>" for h in headers)
    else:
        external_backend_headers_str = "\n".join(f'#include "{h}"' for h in headers)

    assert class_name is not None
    backend_index = backend_indices[dispatch_key]

    dispatch_registrations_body = list(
        concatMap(
            dest.RegisterDispatchKey(
                backend_index,
                Target.REGISTRATION,
                selector,
                rocm=False,
                symint=True,
                simple_trace=simple_trace,
                class_method_name=f"{class_name}",
                skip_dispatcher_op_registration=False,
            ),
            grouped_native_functions,
        )
    )
    newline = "\n"
    ns_helper = NamespaceHelper(namespace_str="at")
    deferred_dispatch_registrations = ""
    static_init_dispatch_registrations = ""
    if eager_registration:
        static_template = CodeTemplate(
            """\
TORCH_LIBRARY_IMPL(aten, $dispatch_key, m) {
    $dispatch_registrations_body
};"""
        )
        static_init_dispatch_registrations = static_template.substitute(
            dispatch_key=dispatch_key,
            dispatch_registrations_body=dispatch_registrations_body,
        )
    else:
        deferred_template = CodeTemplate(
            """\
TORCH_API void Register${backend_name}${dispatch_key}NativeFunctions();
TORCH_API void Register${backend_name}${dispatch_key}NativeFunctions() {
    static auto m = MAKE_TORCH_LIBRARY_IMPL(aten, $dispatch_key);
    $dispatch_registrations_body
}"""
        )
        deferred_dispatch_registrations = deferred_template.substitute(
            backend_name=backend_name,
            dispatch_key=dispatch_key,
            dispatch_registrations_body=dispatch_registrations_body,
        )

    fm.write_with_template(
        f"Register{dispatch_key}.cpp",
        "RegisterDispatchKey.cpp",
        lambda: {
            "extra_cuda_headers": "",
            "external_backend_headers": external_backend_headers_str,
            "ops_headers": "#include <ATen/Functions.h>"
            if not per_operator_headers
            else "",
            "DispatchKey": dispatch_key,
            "dispatch_namespace": dispatch_key.lower(),
            "dispatch_headers": dest.gen_registration_headers(
                backend_index, per_operator_headers=per_operator_headers, rocm=False
            ),
            "dispatch_definitions": fm.substitute_with_template(
                "RegisterDispatchDefinitions.ini",
                lambda: {
                    "ns_prologue": ns_helper.prologue,
                    "ns_epilogue": ns_helper.epilogue,
                    "static_init_dispatch_registrations": static_init_dispatch_registrations,
                    "deferred_dispatch_registrations": deferred_dispatch_registrations,
                    "dispatch_helpers": dest.gen_registration_helpers(backend_index),
                    "dispatch_namespace": dispatch_key.lower(),
                    "dispatch_namespaced_definitions": "",
                    "dispatch_anonymous_definitions": list(
                        concatMap(
                            dest.RegisterDispatchKey(
                                backend_index,
                                Target.ANONYMOUS_DEFINITION,
                                selector,
                                rocm=False,
                                symint=True,
                                simple_trace=simple_trace,
                                class_method_name=f"{class_name}",
                                skip_dispatcher_op_registration=False,
                            ),
                            grouped_native_functions,
                        )
                    ),
                },
            ).split(newline),
        },
    )


def run(
    source_yaml: str,
    output_dir: str,
    dry_run: bool,
    simple_trace: bool,
    impl_path: Optional[str] = None,
    generate_impl: Optional[bool] = False,
    header_file: Optional[str] = None,
    output_impl_file: Optional[str] = None,
) -> None:
    # Assumes that this file lives at PYTORCH_ROOT/torchgen/gen_backend_stubs.py
    pytorch_root = pathlib.Path(__file__).parent.parent.absolute()
    template_dir = os.path.join(pytorch_root, "torchgen/packaged/ATen/templates")

    def make_file_manager(install_dir: str) -> FileManager:
        return FileManager(
            install_dir=install_dir, template_dir=template_dir, dry_run=dry_run
        )

    fm = make_file_manager(output_dir)

    native_yaml_path = os.path.join(
        pytorch_root, "torchgen/packaged/ATen/native/native_functions.yaml"
    )
    tags_yaml_path = os.path.join(pytorch_root, "torchgen/packaged/ATen/native/tags.yaml")
    parsed_yaml = parse_native_yaml(native_yaml_path, tags_yaml_path)
    native_functions, backend_indices = (
        parsed_yaml.native_functions,
        parsed_yaml.backend_indices,
    )
    grouped_native_functions = get_grouped_native_functions(native_functions)
    parsed_backend_yaml = parse_backend_yaml(
        source_yaml, grouped_native_functions, backend_indices
    )
    backend_key = parsed_backend_yaml.backend_key
    autograd_key = parsed_backend_yaml.autograd_key
    cpp_namespace = parsed_backend_yaml.cpp_namespace
    class_name = parsed_backend_yaml.class_name
    backend_indices = parsed_backend_yaml.backend_indices

    selector = SelectiveBuilder.get_nop_selector()

    if backend_key is None:
        # This could be useful if a backend wants to quickly set up a noop yaml file but doesn't have any kernels ready yet.
        return

    if class_name is None:
        # class_name is an optional argument to backend yaml file.
        # if specified it allows an external backend to override
        # the name of the class that all generated kernel definitions live under.
        # if not specified, its value is given as native_function_class_name.
        class_name = backend_indices[backend_key].native_function_class_name()
    assert class_name is not None

    if impl_path is not None:
        error_on_missing_kernels(
            native_functions,
            backend_indices,
            backend_key,
            autograd_key,
            class_name,
            impl_path,
        )

    if generate_impl:
        assert output_impl_file is not None, "No file path is passed for implementation"
        # for native implementation, so the search key is CPU + CUDA
        # for example, NestedTensorCPU, NestedTensorCUDA for NestedTensorXPU
        search_key = backend_key.name.replace('XPU', 'CPU') + ', ' + backend_key.name.replace('XPU', 'CUDA')
        impl_ns = 'native'
        native_yaml_file = './torchgen/packaged/ATen/native/native_functions.yaml'
        assert os.path.isfile(native_yaml_file), "Can not find the native_functions.yaml in {}".format(native_yaml_file)
        gen_implementation(output_impl_file,
                           header_file,
                           impl_ns,
                           native_yaml_file,
                           search_key,
                           class_name,
                           cpp_namespace,
                           backend_indices,
                           grouped_native_functions,
                           backend_key)
    else:
        gen_dispatchkey_nativefunc_headers(
            fm,
            class_name,
            cpp_namespace,
            backend_indices,
            grouped_native_functions,
            backend_key,
            autograd_key,
        )

        for dispatch_key in (
            [backend_key] if autograd_key is None else [backend_key, autograd_key]
        ):
            gen_dispatcher_registrations(
                fm,
                output_dir,
                class_name,
                backend_indices,
                grouped_native_functions,
                backend_key,
                simple_trace,
                dispatch_key,
                selector,
            )


if __name__ == "__main__":
    main()
