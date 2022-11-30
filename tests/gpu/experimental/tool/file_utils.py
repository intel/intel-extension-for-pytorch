import json
import os
import shutil
import sys

def load_from_json(json_file):
    with open(json_file, "r") as load_f:
        load_dict = json.load(load_f)
    return load_dict

def save_to_json(src, target_file):
    data = json.dumps(src, indent=2)
    with open(target_file, "w", newline='\n') as save_f:
        save_f.write(data)

def read_file(file_name):
    f = open(file_name, "r")
    data = f.read()
    f.close()
    return data 

def write_file(file_name, data):
    f = open(file_name, "w")
    f.write(data)
    f.close()

def copy_dir_or_file(src_path, tgt_path):
    src_full_path = os.path.realpath(src_path)
    tgt_full_path = os.path.realpath(tgt_path)
    # copy directory (NOT contain files/dirs under this directory)
    if os.path.isdir(src_full_path) and not os.path.exists(tgt_full_path):
        os.makedirs(tgt_full_path)
    # copy file
    elif os.path.isfile(src_full_path):
        tgt_base_dir = os.path.dirname(tgt_full_path)
        if not os.path.exists(tgt_base_dir):
            os.makedirs(tgt_base_dir)
        shutil.copyfile(src_full_path, tgt_full_path)

def touch_init_py(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for d in dirs:
            init_py = os.path.join(root, d, '__init__.py')
            if not os.path.exists(init_py):
                os.system(f"touch {init_py}")
            assert os.path.isfile(init_py), f"Touch {init_py} FAILED."
