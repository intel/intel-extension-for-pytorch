import ast


def _add_xpu_imports(f_ast):
    """To add necessary xpu imports at the head of test file"""
    # 1. Get the first lineno of classdef or functiondef,
    #    we assume that this line is where the code body starts.
    #    We will add xpu imports right above this body start line.
    body_idx = -1
    for node in f_ast.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            body_idx = f_ast.body.index(node)
            break
    # 2. Format the xpu imports according to original pytorch imports
    xpu_import_node = ast.Import(names=[ast.alias(name="intel_extension_for_pytorch")])
    xpu_import_nodes = [xpu_import_node]
    # 3. add xpu import right above the code body
    f_ast.body[body_idx:body_idx] = xpu_import_nodes
