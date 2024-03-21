import torch

import re

def match_cases(data, pattern_str, *select_ids):
    pattern = re.compile(pattern_str, re.M | re.S)
    result = pattern.findall(data)
    ret = set()
    for res in result:
        elem = [res[i] for i in select_ids]
        ret.add(tuple(elem))
    return ret

def match_name(name, name_list):
    for should_skip in name_list:
        if name in should_skip:
            return True
        #if re.search(should_skip, name, re.M):
        #    return True
        #if re.search(name, should_skip, re.M):
        #    return True
    return False

def match_dtype(name, dtypes):
    name_set = set(name.split("_"))
    for dtype in dtypes:
        if isinstance(dtype, str) and dtype in name_set:
            return True
        if isinstance(dtype, torch.dtype) and str(dtype).split(".")[-1] in name_set:
            return True
    return False
