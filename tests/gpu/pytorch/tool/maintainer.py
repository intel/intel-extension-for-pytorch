import os
from collections import OrderedDict
from .file_utils import load_from_yaml, save_to_yaml

tool_path = os.path.dirname(os.path.abspath(__file__))
ref_file = os.path.join(tool_path, "../config/reference_list.yaml")

def check_reference(cosim_cases):
    def _lookup_reference_list(cur_case):
        for tag in ref_dict.keys():
            if cur_case in ref_dict[tag]:
                return tag
        return None

    global ref_file
    ref_dict = load_from_yaml(ref_file)
    diffs_cases = {}
    for tag, cases in cosim_cases.items():
        for case in cases:
            ref_tag = _lookup_reference_list(case)
            diff_tag = None
            if ref_tag and tag != ref_tag:
                diff_tag = ref_tag + " => " + tag
            elif not ref_tag:
                diff_tag = "New " + tag
            if diff_tag:
                if diff_tag not in diffs_cases:
                    diffs_cases[diff_tag] = []
                diffs_cases[diff_tag].append(case)
    return diffs_cases

def update_reference(total_results):
    global ref_file
    collected_cases = OrderedDict({"PASSED": [], "FAILED": [], "ERROR": [], "SKIPPED": [], "XFAIL": [], "XPASS": [], "NO_RESULT": []})
    for test_name, records in total_results.items():
        for dur, cases_result in records:
            for tag, cases in cases_result.items():
                collected_cases[tag].extend(cases)
    sets_list = []
    sets_list.append(set(collected_cases["PASSED"]))
    sets_list.append(set(collected_cases["FAILED"]))
    sets_list.append(set(collected_cases["ERROR"]))
    sets_list.append(set(collected_cases["SKIPPED"]))
    sets_list.append(set(collected_cases["XFAIL"]))
    sets_list.append(set(collected_cases["XPASS"]))
    sets_list.append(set(collected_cases["NO_RESULT"]))
    for i in range(7):  # we have 7 fields
        for j in range(i + 1, 7):
            sets_list[i] = sets_list[i] - sets_list[j]   # exclude cases which was recorded in following fields
    collected_cases["PASSED"]    = sorted(list(sets_list[0]))
    collected_cases["FAILED"]    = sorted(list(sets_list[1]))
    collected_cases["ERROR"]     = sorted(list(sets_list[2]))
    collected_cases["SKIPPED"]   = sorted(list(sets_list[3]))
    collected_cases["XFAIL"]     = sorted(list(sets_list[4]))
    collected_cases["XPASS"]     = sorted(list(sets_list[5]))
    collected_cases["NO_RESULT"] = sorted(list(sets_list[6]))
    save_to_yaml(collected_cases, ref_file)
