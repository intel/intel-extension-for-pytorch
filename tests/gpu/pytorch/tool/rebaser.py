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
from collections import OrderedDict
from ruamel import yaml

from file_utils import (
    load_from_json,
    load_from_yaml,
    save_to_yaml,
    read_file,
    insert_header_lines,
    insert_main_lines,
    copy_dir_or_file,
    remove_files,
)


class ImportHelper:
    """This helper focuses on handling dependancy issues"""

    src_dir: str
    tgt_dir: str

    def __init__(self, s_dir, t_dir):
        super().__init__()
        self.src_dir = s_dir
        self.tgt_dir = t_dir

    def try_dry_import(self, rel_path):
        if os.path.exists(os.path.join(self.tgt_dir, rel_path)):
            return [], ""
        msg = ""
        related_files = [rel_path]
        copy_dir_or_file(os.path.join(self.src_dir, rel_path),
                         os.path.join(self.tgt_dir, rel_path))
        msg += f"+------ Copied related file {rel_path}\n"
        maybe_init_py = os.path.join(os.path.dirname(os.path.join(self.src_dir, rel_path)), "__init__.py")
        if os.path.exists(maybe_init_py):
            child_ret, child_msg = self.try_dry_import(os.path.relpath(maybe_init_py, self.src_dir))
            related_files.extend(child_ret)
            msg += child_msg
        f_ast = ast.parse(read_file(os.path.join(self.src_dir, rel_path)))
        import_nodes = [node for node in ast.walk(f_ast) if isinstance(node, (ast.Import, ast.ImportFrom))]
        modules_set = set()
        for node in import_nodes:
            for alias_node in node.names:
                full_module_name = alias_node.name
                if isinstance(node, ast.ImportFrom) and node.module:
                    full_module_name = node.module + "." + full_module_name
                meta_module_list = full_module_name.split(".")
                for i in range(1, len(meta_module_list) + 1):
                    modules_set.add('/'.join(meta_module_list[:i]))
        for sub_module in modules_set:
            sub_src_path = os.path.join(self.src_dir, sub_module) + ".py"
            if os.path.exists(sub_src_path):
                child_ret, child_msg = self.try_dry_import(sub_module + ".py")
                related_files.extend(child_ret)
                msg += child_msg
        return related_files, msg

    def run(self, rel_path):
        """main entrance for class ImportHelper"""
        related_files, msg = self.try_dry_import(rel_path)
        if related_files:
            return True, msg
        return False, msg
# class ImportHelper end.


class SelectHelper:
    """This helper fork sub-processes for each file and makes control"""

    src_dir: str
    tgt_dir: str
    rel_path: str

    def __init__(self, s_dir, t_dir):
        super().__init__()
        self.src_dir = s_dir
        self.tgt_dir = t_dir
        self.test_map_entries = []

    def check_instantiate_call(self, top_node):
        """check whether `instantiate_device_type_tests` in test file"""
        ret_list = []
        call_nodes = [
            node
            for node in ast.walk(top_node)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "instantiate_device_type_tests"
        ]
        # select all instantiate_device_type_tests call nodes,
        return len(call_nodes)

    def select_file(self, src_full_path):
        """From here each process handle a single file"""
        test_name = ""
        msg = ""
        self.rel_path = os.path.relpath(src_full_path, self.src_dir)
        msg += f"-- Checking file {self.rel_path} ... "
        f_ast = ast.parse(read_file(src_full_path))
        if self.check_instantiate_call(f_ast):
            tgt_full_path = os.path.join(self.tgt_dir, self.rel_path)
            msg += "TRY TO IMPORT\n"
            import_helper = ImportHelper(self.src_dir, self.tgt_dir)
            status, import_msg = import_helper.run(self.rel_path)
            if status:
                test_name = self.rel_path[:-len(".py")]
                insert_header_lines(f"""
import os
import sys
test_root = os.path.abspath(os.path.join(os.path.dirname(__file__), \'{'../' * (self.rel_path.count('/') + 1)}\'))
sys.path.append(test_root)

import common.xpu_test_base
""", tgt_full_path)
                insert_main_lines("common.xpu_test_base.customized_skipper()\n", tgt_full_path)
                msg += import_msg
            else:
                msg += "DISCARD\n"
        else:
            msg += "DISCARD\n"
        return test_name, msg

    def run(self, file_list):
        """main entrance for class SelectHelper, processes fork from here"""
        results = []
        with Pool(os.cpu_count()) as pool:
            results = pool.map(self.select_file, file_list)
        pool.close()
        pool.join()

        self.ported_tests = []
        for test_name, msg in results:
            if len(test_name) > 0:
                self.ported_tests.append(test_name)
            print(msg, end="")

# class SelectHelper end.


class Worker:
    """main worker for rebasement, only be instantiated by parent process"""

    temp_work_dir: str = ""

    def __init__(self):
        super().__init__()

        self.temp_work_dir = os.path.realpath("/tmp/ut_rebase_work")
        # clean old temp directory if exists
        if os.path.exists(self.temp_work_dir):
            shutil.rmtree(self.temp_work_dir)
        os.makedirs(self.temp_work_dir)
        for folder in ["test", "tool", "common", "config"]:
            temp_path = os.path.join(self.temp_work_dir, folder)
            os.makedirs(temp_path)
            print(f"Created temp work directory: {temp_path}")

    def __del__(self):
        if os.path.exists(self.temp_work_dir):
            shutil.rmtree(self.temp_work_dir)
            print(f"Deleted temp work directory: {self.temp_work_dir}")

    def keep_old_files(self, old_test_dir, keep_file_list):
        """select and copy those files that should be kept during rebasing"""
        print("Collecting old files those should be kept ...")
        for dir_name in keep_file_list:
            file_list = glob.glob(os.path.join(old_test_dir, dir_name))
            for f_name in file_list:
                temp_file = os.path.join(
                    self.temp_work_dir, os.path.relpath(f_name, old_test_dir)
                )
                shutil.copyfile(f_name, temp_file)
                print(f"-- Copied old file to temp dir: {temp_file}")

    def select_files_and_copy(self, src_dir):
        """To select which test file contains device test should be copied"""
        print("Collecting test files for device ...")
        file_list = glob.glob(os.path.join(src_dir, "**/*.py"), recursive=True)
        temp_test_dir = os.path.join(self.temp_work_dir, "test")
        self.select_helper = SelectHelper(src_dir, temp_test_dir)
        self.select_helper.run(file_list)

    def clean_empty_dirs(self):
        for root, dirs, files in os.walk(self.temp_work_dir, topdown=False):
            if root.endswith('__pycache__'):
                shutil.rmtree(root)
            elif not os.listdir(root):
                os.rmdir(root)
        print("Cleaned empty temp directories.")

    def copy_temp_to_target_dir(self, tgt_dir):
        """finally copy files from temp work dir to target dir"""

        def clean_target_dir(tgt_dir):
            """clear target directory preparing for copy files in"""
            if os.path.exists(tgt_dir):
                shutil.rmtree(tgt_dir)
                print(f"Cleaned target directory: {tgt_dir}")

        clean_target_dir(tgt_dir)
        shutil.copytree(self.temp_work_dir, tgt_dir)
        print(f"Copied temp work directory to: {tgt_dir}")
# class Worker end.


def backup_old_files(ipex_test_dir):
    """backup old pytorch folder to pytorch.old"""
    backup_path = ipex_test_dir + ".old"
    if os.path.exists(backup_path):
        print(f"Cleaned old backups: {backup_path}")
        shutil.rmtree(backup_path)
    os.rename(ipex_test_dir, backup_path)
    print(f"Backup-ed old test files to: {backup_path}")


def main():
    """the main entrance of ut rebaser"""
    # parse args
    parser = argparse.ArgumentParser(description="Rebase tool for PyTorch ported UTs")
    parser.add_argument(
        "-p",
        "--pytorch-dir",
        metavar="PYTORCH-DIR",
        dest="pytorch_root_dir",
        default="",
        type=str,
        required=True,
        help="Root directory of the source codes of PyTorch,"
        " and this path should contain 'test' folder under it.",
    )
    parser.add_argument(
        "-x",
        "--ipex-dir",
        metavar="IPEX-DIR",
        dest="ipex_root_dir",
        default="",
        type=str,
        required=True,
        help="Root directory of Intel® Extension for PyTorch*,"
        " and this path should contain 'tests/gpu/pytorch'"
        " folder under it.",
    )
    parser.add_argument(
        "-o",
        "--target-dir",
        metavar="TARGET-DIR",
        dest="target_dir",
        default=None,
        type=str,
        help="Target directory to store output files."
        " As default, output files will be saved under"
        " <ipex_root>/tests/gpu/pytorch/",
    )
    parser.add_argument(
        "-i",
        "--no-backup",
        action="store_true",
        help="If this flag is set True,"
        " this tool won't backup old pytorch/"
        " and will replace all things under pytorch/ with new files.",
    )

    args = parser.parse_args()
    pytorch_root_dir = os.path.realpath(os.path.expanduser(args.pytorch_root_dir))
    assert os.path.exists(pytorch_root_dir), (
        f"Source directory of PyTorch does not exist."
        f" Please re-check --pytorch-dir flag.\nError directory: "
        f"{pytorch_root_dir}"
    )
    pytorch_test_dir = os.path.join(pytorch_root_dir, "test")
    assert os.path.exists(pytorch_root_dir), (
        f"Source directory of PyTorch's tests does not exist."
        f" Please re-check --pytorch-dir flag.\nError directory: "
        f"{pytorch_test_dir}"
    )
    ipex_root_dir = os.path.realpath(os.path.expanduser(args.ipex_root_dir))
    assert os.path.exists(ipex_root_dir), (
        f"Source directory of Intel® Extension for PyTorch* does not exist."
        f" Please re-check --ipex-dir flag.\nError directory: "
        f"{ipex_root_dir}"
    )
    ipex_test_dir = os.path.join(ipex_root_dir, "tests/gpu/pytorch")
    assert os.path.exists(ipex_test_dir), (
        f"Source directory of Intel® Extension for PyTorch*'s"
        f" pytorch tests does not exist."
        f" Please re-check --ipex-dir flag.\nError directory: "
        f"{ipex_root_dir}"
    )
    target_test_dir = args.target_dir if args.target_dir is not None else ipex_test_dir

    keep_files_list = load_from_json(
        os.path.join(ipex_test_dir, "./config/keep_files_list.json")
    )

    print("========= Rebase work START =========")
    worker = Worker()
    worker.keep_old_files(ipex_test_dir, keep_files_list)
    worker.select_files_and_copy(pytorch_test_dir)
    worker.clean_empty_dirs()
    if not args.no_backup:
        backup_old_files(ipex_test_dir)
    worker.copy_temp_to_target_dir(target_test_dir)

    print("========= Rebase work DONE! =========")


if __name__ == "__main__":
    main()
