import re

def match_cases(data, pattern_str, *select_ids):
    pattern = re.compile(pattern_str, re.M | re.S)
    result = pattern.findall(data)
    ret = set()
    for res in result:
        elem = [res[i] for i in select_ids]
        ret.add(tuple(elem))
    return ret
