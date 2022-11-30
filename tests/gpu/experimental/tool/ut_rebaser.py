""" Author: Xunsong, Huang <xunsong.huang@intel.com> """

import ast
import argparse
import copy
import glob
import importlib
import os
import re
import shutil
import sys
from multiprocessing import Pool
from file_utils import save_to_json, load_from_json, read_file, write_file, \
                       copy_dir_or_file, touch_init_py


class NodeTransformer(ast.NodeTransformer):
    """ main transformer for replacing string in ast nodes """

    make_restore: bool
    cuda_to_xpu_map: dict

    def __init__(self, make_restore=False):
        super().__init__()
        self.make_restore = make_restore
        self.cuda_to_xpu_map = {
            'cuda': 'xpu',
            'CUDA': 'XPU',
            'Cuda': 'XPU',
            b'cuda': b'xpu',
            b'CUDA': b'XPU',
            b'Cuda': b'XPU',
        }

    def _replace(self, obj):
        """ implementation of replacement and restorer """
        for key, val in self.cuda_to_xpu_map.items():
            (key, val) = (val, key) if self.make_restore else (key, val)
            if isinstance(obj, type(key)):
                obj = obj.replace(key, val)
        return obj

    def visit_Constant(self, node):
        """ replace `cuda` to `xpu` for ast.Constant """
        self.generic_visit(node)
        node.value = self._replace(node.value)
        return node

    def visit_Attribute(self, node):
        """ replace `cuda` to `xpu` for ast.Attribute """
        self.generic_visit(node)
        node.attr = self._replace(node.attr)
        return node

    def visit_FunctionDef(self, node):
        """ replace `cuda` to `xpu` for ast.FunctionDef """
        self.generic_visit(node)
        node.name = self._replace(node.name)
        return node

    def visit_ClassDef(self, node):
        """ replace `cuda` to `xpu` for ast.ClassDef """
        self.generic_visit(node)
        node.name = self._replace(node.name)
        return node

    def visit_Name(self, node):
        """ replace `cuda` to `xpu` for ast.Name """
        self.generic_visit(node)
        node.id = self._replace(node.id)
        return node

    def visit_arg(self, node):
        """ replace `cuda` to `xpu` for ast.arg """
        self.generic_visit(node)
        node.arg = self._replace(node.arg)
        return node

    def visit_keyword(self, node):
        """ replace `cuda` to `xpu` for ast.keyword """
        self.generic_visit(node)
        node.arg = self._replace(node.arg)
        return node
# class NodeTransformer end.


class DecoTransformer(NodeTransformer):
    """ This transformer only for flatting given decorators """

    def visit_IfExp(self, node):
        """ For decorator dtypesIfXPU, there is no need
            to select different versions for mapping dtypes """
        self.generic_visit(node)
        return node.body
# class DecoTransformer end.


class TestFileModifier():
    """ This class focuses on adjust the given test file """

    def replace_cuda_with_xpu_(self, f_ast):
        """ To replace all `cuda` with `xpu` except for decorators """
        print("------ Start to modify cuda to xpu ... ", end="")
        transformer = NodeTransformer()
        # replace all `cuda` with `xpu`
        transformer.visit(f_ast)

        # restore all decorators to `cuda`
        def _restore_xpu_to_cuda(node):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                restorer = NodeTransformer(make_restore=True)
                node.decorator_list = [
                    restorer.visit(deco) for deco in node.decorator_list]
            return node

        for node in ast.walk(f_ast):
            node = _restore_xpu_to_cuda(node)
        print("SUCCESS")

    def add_xpu_imports_(self, f_ast):
        """ To add necessary xpu imports at the head of test file """
        print("------ Start to add necessary xpu imports ... ", end="")
        # 1. Get the first lineno of classdef or functiondef,
        #    we assume that this line is where the code body starts.
        #    We will add xpu imports right above this body start line.
        body_idx = -1
        for node in f_ast.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                body_idx = f_ast.body.index(node)
                break
        # 2. Format the xpu imports according to original pytorch imports
        #    They are:
        #    i. no matter what was imported by pytorch,
        #       we will:
        #       from common.pytorch_test_base import (TestCase, dtypesIfXPU,
        #                                             TEST_XPU, TEST_MULTIGPU,
        #                                             largeTensorTest)
        #    ii. if from torch.testing._internal.common_nn import somthing,
        #        we will import all the same modules from common.common_nn
        #    iii. if from torch.testing._internal.common_jit import something,
        #         we will import all the same modules from common.common_jit
        #    iv. if from torch.testing._internal.jit_utils import something,
        #        we will import all the same modules from common.jit_utils
        # import utils
        pytorch_test_base_node = ast.ImportFrom(
            module='common.pytorch_test_base',
            names=[
                ast.alias(name='TestCase'),
                ast.alias(name='dtypesIfXPU'),
                ast.alias(name='TEST_XPU'),
                ast.alias(name='TEST_MULTIGPU'),
                ast.alias(name='largeTensorTest'),
                ],
            level=0)
        # import common_nn
        old_common_nn_nodes = [
            node for node in f_ast.body
            if isinstance(node, ast.ImportFrom)
            and node.module == 'torch.testing._internal.common_nn']
        common_nn_node_names = [
            name for node in old_common_nn_nodes for name in node.names
            ]
        common_nn_node = ast.ImportFrom(
            module='common.common_nn',
            names=common_nn_node_names,
            level=0) if common_nn_node_names else None
        # import common_jit
        old_common_jit_nodes = [
            node for node in f_ast.body
            if isinstance(node, ast.ImportFrom)
            and node.module == 'torch.testing._internal.common_jit']
        common_jit_node_names = [
            name for node in old_common_jit_nodes for name in node.names
            ]
        common_jit_node = ast.ImportFrom(
            module='common.common_jit',
            names=common_jit_node_names,
            level=0) if common_jit_node_names else None
        # import jit_utils
        old_jit_utils_nodes = [
            node for node in f_ast.body
            if isinstance(node, ast.ImportFrom)
            and node.module == 'torch.testing._internal.jit_utils']
        jit_utils_node_names = [
            name for node in old_jit_utils_nodes for name in node.names
            ]
        jit_utils_node = ast.ImportFrom(
            module='common.jit_utils',
            names=jit_utils_node_names,
            level=0) if jit_utils_node_names else None
        imp_nodes = [pytorch_test_base_node,
                     common_nn_node,
                     common_jit_node,
                     jit_utils_node]
        while None in imp_nodes:
            imp_nodes.remove(None)
        # 3. add xpu import right above the code body
        f_ast.body[body_idx:body_idx] = imp_nodes
        print("SUCCESS")

    def add_xpu_decorators_(self, f_ast):
        """ To add xpu style decorators to necessary functions """

        print("------ Start to add necessary xpu decorators ... ", end="")

        # 1. add xpu decos to each needed decorator list
        def _add_xpu_decos_to_node(node):
            if not hasattr(node, "decorator_list"):
                return []

            deco_list = node.decorator_list
            new_decos = []

            # i. find dtypesIfCUDA and add dtypesIfXPU right above it
            def _find_cuda_dtypes_deco(deco):
                if isinstance(deco, ast.Call) \
                        and isinstance(deco.func, ast.Name) \
                        and deco.func.id == 'dtypesIfCUDA':
                    return True
                return False

            cuda_dtypes_deco_idx = list(
                map(lambda deco:
                    deco_list.index(deco)
                    if _find_cuda_dtypes_deco(deco)
                    else -1,
                    deco_list))
            carry = 0
            for idx in cuda_dtypes_deco_idx:
                if idx == -1:
                    continue
                xpu_dtypes_deco_node = copy.deepcopy(deco_list[idx + carry])
                xpu_dtypes_deco_node.func.id = "dtypesIfXPU"
                transformer = DecoTransformer()
                transformer.visit(xpu_dtypes_deco_node)
                deco_list.insert(idx + carry, xpu_dtypes_deco_node)
                new_decos.append(xpu_dtypes_deco_node)
                carry += 1
            # add more decos further if necessary
            return new_decos

        new_decos = [
            ast.unparse(deco) for deco in
            [_add_xpu_decos_to_node(node) for node in ast.walk(f_ast)]
            if deco]

        print("SUCCESS")
        for deco in new_decos:
            print(f"-------- Added xpu deco: {deco}")

    def run(self, f_ast):
        """ main entrance of class TestFileModifier """
        self.replace_cuda_with_xpu_(f_ast)
        self.add_xpu_imports_(f_ast)
        self.add_xpu_decorators_(f_ast)
        ast.fix_missing_locations(f_ast)
# class TestFileModifier end.


class ImportHelper():
    """ This helper focuses on handling dependancy issues """

    src_dir: str
    tgt_dir: str

    def __init__(self, s_dir, t_dir):
        super().__init__()
        self.src_dir = s_dir
        self.tgt_dir = t_dir

    def try_to_import_module(self, module_name):
        """ return (True, module_name) if succeed,
            else return (False, missed_module_name) """
        try:
            importlib.import_module(module_name)
            return True, module_name
        except ModuleNotFoundError as err:
            print("\n------ ", err)
            res = re.match(r"No module named \'(.*)\'", str(err),
                           re.M | re.S | re.I)
            assert res is not None, \
                f"ModuleNotFoundError mismatch in re.match," \
                f" the error msg is {err}"
            return False, res.group(1)

    def run(self, rel_path):
        """ main entrance for class ImportHelper """
        module_name = os.path.splitext(rel_path)[0].replace('/', '.')
        # add work dir to sys.path for searching modules
        if self.tgt_dir not in sys.path:
            sys.path.append(self.tgt_dir)
        print(f"---- Tring to import copied module {module_name} ... ", end="")
        status, missed_module = self.try_to_import_module(module_name)
        while status is False:
            print("FAILED")
            rel_path = missed_module.replace('.', '/')
            maybe_src_path = os.path.join(self.src_dir, rel_path)
            maybe_src_file = maybe_src_path + ".py"
            tgt_path = os.path.join(self.tgt_dir, rel_path)
            tgt_file = tgt_path + ".py"
            copy_dir_or_file(maybe_src_path, tgt_path)
            # this is the work-around for solving inf-loop issue
            if os.path.exists(tgt_file):
                os.remove(tgt_file)
            copy_dir_or_file(maybe_src_file, tgt_file)
            print(f"---- Retry to import copied module {module_name}... ",
                  end="")
            status, missed_module = self.try_to_import_module(module_name)
        print("SUCCESS")
# class ImportHelper end.


class SelectHelper():
    """ This helper fork sub-processes for each file and makes control """

    src_dir: str
    tgt_dir: str
    rel_path: str
    test_map: dict

    def __init__(self, s_dir, t_dir):
        super().__init__()
        self.src_dir = s_dir
        self.tgt_dir = t_dir
        self.test_map = {}

    def check_instantiate_call(self, top_node):
        """ check whether `instantiate_device_type_tests` in test file"""
        ret_list = []
        call_nodes = [node for node in ast.walk(top_node)
                      if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name)
                      and node.func.id == 'instantiate_device_type_tests']
        # select all instantiate_device_type_tests(cls, globals()) call nodes,
        # except for 'only_for'
        for node in call_nodes:
            args = node.args
            clsname = args[0].id
            if len(args) > 1 and args[1].func.id != "globals":
                continue
            keywords = node.keywords
            has_only_for = any(map(lambda k: k.arg == 'only_for', keywords))
            if not has_only_for and \
               not re.search(r'cuda', clsname, re.M | re.S | re.I):
                ret_list.append(clsname)
        return ret_list

    def select_file(self, src_full_path):
        """ From here each process handle a single file """
        self.rel_path = os.path.relpath(src_full_path, self.src_dir)
        print(f"-- Checking file {self.rel_path} ... ", end="")
        f_ast = ast.parse(read_file(src_full_path))
        classes = self.check_instantiate_call(f_ast)
        if classes:
            tgt_full_path = os.path.join(self.tgt_dir, self.rel_path)
            print("SELECT")
            copy_dir_or_file(src_full_path, tgt_full_path)
            print(f"---- Copied file to: {tgt_full_path}")
            import_helper = ImportHelper(self.src_dir, self.tgt_dir)
            import_helper.run(self.rel_path)
            test_file_modifier = TestFileModifier()
            test_file_modifier.run(f_ast)
            write_file(tgt_full_path, ast.unparse(f_ast))
            print(f"-- Modified file {tgt_full_path}")
            return self.rel_path, classes
        print("DISCARD")
        return "", []

    def run(self, file_list):
        """ main entrance for class SelectHelper, processes fork from here """
        results = []
        with Pool(os.cpu_count()) as pool:
            results = pool.map(self.select_file, file_list)
        pool.close()
        pool.join()

        for result in results:
            if result[1]:
                self.test_map[result[0]] = result[1]
        # generate test map (and json file)
        test_map_json_file = os.path.join(self.tgt_dir,
                                          "../config/test_map.json")
        save_to_json(self.test_map, test_map_json_file)
        print(f"Generated {{test_file: [test_classes]}} map:"
              f" {test_map_json_file}")
# class SelectHelper end.


class Worker():
    """ main worker for rebasement, only be instantiated by parent process """

    temp_work_dir: str = ""

    def __init__(self):
        super().__init__()

        self.temp_work_dir = "/tmp/ut_rebase_work"
        # clean old temp directory if exists
        if os.path.exists(self.temp_work_dir):
            shutil.rmtree(self.temp_work_dir)
        os.makedirs(self.temp_work_dir)
        for folder in ["test", "tool", "common", "config", "env", "log"]:
            temp_path = os.path.join(self.temp_work_dir, folder)
            os.makedirs(temp_path)
            print(f"Created temp work directory: {temp_path}")

    def __del__(self):
        if os.path.exists(self.temp_work_dir):
            shutil.rmtree(self.temp_work_dir)
            print(f"Deleted temp work directory: {self.temp_work_dir}")

    def keep_old_files(self, old_test_dir, keep_file_list):
        """ select and copy those files that should be kept during rebasing """
        print("Collecting old files those should be kept ...")
        for dir_name in keep_file_list:
            file_list = glob.glob(os.path.join(old_test_dir, dir_name))
            for f_name in file_list:
                temp_file = os.path.join(self.temp_work_dir,
                                         os.path.relpath(f_name, old_test_dir))
                shutil.copyfile(f_name, temp_file)
                print(f"-- Copied old file to temp dir: {temp_file}")

    def select_files_and_copy(self, src_dir):
        """ To select which test file contains device test should be copied """
        print("Collecting test files for device ...")
        file_list = glob.glob(os.path.join(src_dir, "**/*.py"), recursive=True)
        temp_test_dir = os.path.join(self.temp_work_dir, 'test')
        select_helper = SelectHelper(src_dir, temp_test_dir)
        select_helper.run(file_list)

    def copy_temp_to_target_dir(self, tgt_dir):
        """ finally copy files from temp work dir to target dir """

        def clean_target_dir(tgt_dir):
            """ clear target directory preparing for copy files in"""
            if os.path.exists(tgt_dir):
                shutil.rmtree(tgt_dir)
                print(f"Cleaned target directory: {tgt_dir}")

        clean_target_dir(tgt_dir)
        shutil.copytree(self.temp_work_dir, tgt_dir)
        print(f"Copied temp work directory to: {tgt_dir}")
# class Worker end.


def backup_old_files(ipex_test_dir):
    """ backup old experimental folder to experimental.old """
    backup_path = ipex_test_dir + '.old'
    os.rename(ipex_test_dir, backup_path)
    print(f"Backup-ed old test files to: {backup_path}")


def main():
    """ the main entrance of ut rebaser """
    # parse args
    parser = argparse.ArgumentParser(
        description="Rebase tool for PyTorch ported UTs")
    parser.add_argument(
        '-p', '--pytorch-dir',
        metavar='PYTORCH-DIR',
        dest='pytorch_root_dir',
        default="",
        type=str,
        required=True,
        help='Root directory of the source codes of PyTorch,'
        ' and this path should contain \'test\' folder under it.')
    parser.add_argument(
        '-x', '--ipex-dir',
        metavar='IPEX-DIR',
        dest='ipex_root_dir',
        default="",
        type=str,
        required=True,
        help='Root directory of Intel® Extension for PyTorch*,'
        ' and this path should contain \'tests/gpu/experimental\''
        ' folder under it.')
    parser.add_argument(
        '-o', '--target-dir',
        metavar='TARGET-DIR',
        dest='target_dir',
        default=None,
        type=str,
        help='Target directory to store output files.'
        ' As default, output files will be saved under'
        ' <ipex_root>/tests/gpu/experimental/')
    parser.add_argument(
        '-i', '--no-backup',
        action='store_true',
        help='If this flag is set True,'
        ' this tool won\'t backup old experimental/'
        ' and will replace all things under experimental/ with new files.')

    args = parser.parse_args()
    pytorch_root_dir = os.path.realpath(
        os.path.expanduser(args.pytorch_root_dir))
    assert os.path.exists(pytorch_root_dir), \
        f"Source directory of PyTorch does not exist." \
        f" Please re-check --pytorch-dir flag.\nError directory: " \
        f"{pytorch_root_dir}"
    pytorch_test_dir = os.path.join(pytorch_root_dir, "test")
    assert os.path.exists(pytorch_root_dir), \
        f"Source directory of PyTorch's tests does not exist." \
        f" Please re-check --pytorch-dir flag.\nError directory: " \
        f"{pytorch_test_dir}"
    ipex_root_dir = os.path.realpath(
        os.path.expanduser(args.ipex_root_dir))
    assert os.path.exists(ipex_root_dir), \
        f"Source directory of Intel® Extension for PyTorch* does not exist." \
        f" Please re-check --ipex-dir flag.\nError directory: " \
        f"{ipex_root_dir}"
    ipex_test_dir = os.path.join(ipex_root_dir, "tests/gpu/experimental")
    assert os.path.exists(ipex_test_dir), \
        f"Source directory of Intel® Extension for PyTorch*'s" \
        f" experimental tests does not exist." \
        f" Please re-check --ipex-dir flag.\nError directory: " \
        f"{ipex_root_dir}"
    target_test_dir = \
        args.target_dir if args.target_dir is not None else ipex_test_dir
    script_dir = os.path.dirname(os.path.realpath(__file__))

    keep_files_list = load_from_json(
        os.path.join(script_dir, "../config/keep_files_list.json"))

    print("========= Rebase work START =========")
    worker = Worker()
    worker.keep_old_files(ipex_test_dir, keep_files_list)
    worker.select_files_and_copy(pytorch_test_dir)
    if not args.no_backup:
        backup_old_files(ipex_test_dir)
    worker.copy_temp_to_target_dir(target_test_dir)
    touch_init_py(target_test_dir)

    print("========= Rebase work DONE! =========")


if __name__ == '__main__':
    main()
