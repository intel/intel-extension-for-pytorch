import ast
import os

from .transformer import NodeTransformer
from .import_helper import _add_xpu_imports


def run(full_path, execute=False, export=False):
    f_ast = None
    new_data = None

    with open(full_path, "r") as in_f:
        # read source codes
        data = in_f.read()
        # parse source codes to ast object
        f_ast = ast.parse(data, type_comments=True)

    # ----- 1. replace cuda with xpu -----
    # instantiate node transformer
    node_trans = NodeTransformer()
    # visit nodes within ast object
    node_trans.visit(f_ast)
    # resolve linenos
    ast.fix_missing_locations(f_ast)

    # ----- 2. add necessary imports -----
    _add_xpu_imports(f_ast)
    ast.fix_missing_locations(f_ast)

    # ----- 3. unparse new ast obj   -----
    new_data = ast.unparse(f_ast)

    dirname, fname = os.path.split(full_path)
    split_fname = list(os.path.splitext(fname))
    new_fname = split_fname[0] + ".xpu" + split_fname[1]
    new_full_path = os.path.join(dirname, new_fname)
    export_file_path = full_path
    if export:
        export_file_path = new_full_path
    with open(export_file_path, "w") as out_f:
        out_f.write(new_data)

    if execute is True:
        exec(new_data)
