import os
from collections import OrderedDict

from .file_utils import load_from_yaml, save_to_yaml

tool_path = os.path.dirname(os.path.abspath(__file__))
stt_skip_file = os.path.join(tool_path, "../config/static_skipped_cases_list.yaml")
dyn_skip_file = os.path.join(tool_path, "../config/dynamic_skipped_cases_list.yaml")

def add_dynamic_skipped_cases(should_skip_cases, reason):
    dyn_skipped_list = load_from_yaml(dyn_skip_file)
    idx = None
    for skip_dict in dyn_skipped_list:
        if skip_dict['reason'] == reason:
            idx = dyn_skipped_list.index(skip_dict) 
            break
    if idx is None:
        target_skip_dict = {"reason": reason, "cases": should_skip_cases}
        dyn_skipped_list.append(target_skip_dict)
        idx = -1
    else:
        dyn_skipped_list[idx]["cases"].extend(should_skip_cases)
    dyn_skipped_list[idx]["cases"] = sorted(dyn_skipped_list[idx]["cases"])
    save_to_yaml(dyn_skipped_list, dyn_skip_file)
