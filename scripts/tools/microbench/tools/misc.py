import torch
import collections


def identify_call_pipeline(lines: list) -> list:
    while len(lines) > 0:
        line = lines[0]
        if 'CPU total' in line and 'CPU time avg' in line and '# of Calls' in line:
            count_of__ = lines[1].count('-')
            lines = lines[2:]
            ret = []
            while len(lines) > 0:
                item = lines[0]
                if item.count('-') == count_of__:
                    return ret
                item = item.strip().split(' ')
                item = [t for t in item if len(t) > 0]
                if item[10] != '1':
                    break
                else:
                    ret.append(item)
                    lines = lines[1:]
        else:
            lines = lines[1:]
    return []


def get_op_model_time(call_pipeline: list):
    output = collections.defaultdict(str)
    for i, t in enumerate(call_pipeline):
        name = t[0].strip()
        if name.startswith('mb_'):
            output[name] = call_pipeline[i - 1][9]
    return output
