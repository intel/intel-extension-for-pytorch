import os
from collections import OrderedDict
from .file_utils import load_from_yaml, save_to_yaml
from .collector import collect_detailed_issues, collect_fatal_error

tool_path = os.path.dirname(os.path.abspath(__file__))
ref_file = os.path.join(tool_path, "../config/reference_list.yaml")
weekly_ref_file = os.path.join(tool_path, "../config/weekly_reference_list.yaml")
ci_file = os.path.join(tool_path, "../config/ci_list.yaml")

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

def check_ci_pass(cases_result, logfile):
    global ref_file
    ref_dict = load_from_yaml(ref_file)
    global ci_file
    ci_dict = load_from_yaml(ci_file)

    issued_cases = []
    short_details = []
    details = []
    for case in cases_result["NO_RESULT"]:
        if case not in ref_dict["NO_RESULT"]:
            issued_cases.append(case)
            detail = collect_fatal_error(case, logfile)
            if 'timed out' in detail:
                short_details.append("Timed Out\n" + detail)
            else:
                short_details.append("Fatal Error\n" + detail)
            details.append(detail)
            return issued_cases, short_details, details
    for tag, cases in cases_result.items():
        if tag == "PASSED" or tag == "XFAIL":
            continue
        for case in cases:
            if tag == "SKIPPED":
                if case in ci_dict["SKIPPED"]:
                    continue
            elif case in ref_dict["PASSED"]:
                issued_cases.append(case)
            #elif case not in ref_dict["FAILED"] and case not in ref_dict["ERROR"] and case not in ref_dict["XPASS"]:
            #    issued_cases.append(case)

    if issued_cases:
        short_details, details = collect_detailed_issues(issued_cases, logfile, with_short=True)
    return issued_cases, short_details, details

def update_reference(total_results, mode):
    global ref_file
    global weekly_ref_file
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
    if mode == "weekly":
        save_to_yaml(collected_cases, weekly_ref_file)
    elif mode == "maintain":
        save_to_yaml(collected_cases, ref_file)
