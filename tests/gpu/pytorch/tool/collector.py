import re
from .file_utils import read_file

re_flags = re.M | re.S

def collect_pytest_details(issued_cases, data):
    global re_flags
    issued_casenames = [case.split("::", 1)[-1] for case in issued_cases]
    details = []
    pattern = r'^_+[^\n]*\s(\S*test_\S*)\s_+$'
    lines = data.split('\n')
    started_linenos = []
    for line in lines:
        lineno = lines.index(line)
        res = re.search(pattern, line, re_flags)
        if res:
            casename = res.group(1).replace('.', '::')
            started_linenos.append((lineno, casename))
        if re.match(r'^=+.+?=+$', line, re_flags):
            started_linenos.append((lineno, "SPLITOR"))
    sorted_started_linenos = sorted(started_linenos)
    for i in range(len(sorted_started_linenos) - 1):
        start_lineno = sorted_started_linenos[i][0]
        casename = sorted_started_linenos[i][1]
        next_lineno = sorted_started_linenos[i + 1][0]
        if casename == "SPLITOR" or casename not in issued_casenames:
            continue
        detail = "\n".join(lines[start_lineno:next_lineno])
        details.append(detail)
    return details

def collect_pytorch_details(issued_cases, data):
    global re_flags
    issued_casenames = [case.split("::", 1)[-1] for case in issued_cases]
    details = []
    pattern = r'(^=+\n[A-Za-z]+:\s*(test_\S*)\s*\([^\.]*\.(\S*)\)\n-+.*?(?=\n\n))'
    results = re.findall(pattern, data, re_flags)
    for detail, casename, clsname in results:
        if clsname + "::" + casename in issued_casenames:
            details.append(detail)
    return details

def collect_detailed_issues(issued_cases, logfile):
    global re_flags
    test_name = logfile.split('.')[0].replace('-', '/')
    data = read_file(logfile)
    if re.search(r"^=*\s*test session starts\s*=*$", data, re_flags):
        return collect_pytest_details(issued_cases, data)
    else:
        test_file_name = test_name + ".py"
        return collect_pytorch_details(issued_cases, data)

def pytest_adjust_tag(failed_cases, data, test_file_name):
    global re_flags
    should_adjust_cases = []
    details = collect_pytest_details(failed_cases, data)
    for detail in details:
        lines = detail.split("\n")
        casename = re.search(r'^_+[^\n]*\s(\S*test_\S*)\s_+$', lines[0], re_flags).group(1).replace('.', "::")
        for line in lines:
            res = re.search(r'[^:]+:[0-9]+:\s*(\S+Error)$', line, re_flags)
            if res and "AssertionError" not in res.group(1):
                should_adjust_cases.append(test_file_name + "::" + casename)
                break
    return should_adjust_cases

def collect_pytest_cases(data, test_file_name):
    global re_flags
    pattern_full_case = r"^(\S+::test_\S+)\s+([A-Z]+)\s+\[\s*([0-9]+)\%\]$" 
    pattern_no_res_case = r"^(\S+::test_\S+)$"
    collected_cases = {"PASSED": [], "FAILED": [], "ERROR": [], "SKIPPED": [], "XFAIL": [], "XPASS": [], "NO_RESULT": []}
    lines = data.split('\n')
    prev_has_result = False
    for line in lines:
        res = re.match(pattern_full_case, line, re_flags)
        not_finished_res = re.match(pattern_no_res_case, line, re_flags)
        if res:
            full_case_name = "::".join([test_file_name, res.group(1).split("::")[-2], res.group(1).split("::")[-1]])
            result_string = res.group(2)
            # rate_string = res.group(3)
            collected_cases[result_string].append(full_case_name)
            prev_has_result = True
        elif not_finished_res and prev_has_result:
            full_case_name = "::".join([test_file_name, not_finished_res.group(1).split("::")[-2], not_finished_res.group(1).split("::")[-1]])
            collected_cases["NO_RESULT"].append(full_case_name)
            prev_has_result = False
    should_adjust_cases = pytest_adjust_tag(collected_cases["FAILED"], data, test_file_name)
    for case in should_adjust_cases:
        collected_cases["FAILED"].remove(case)
        collected_cases["ERROR"].append(case)
    return collected_cases

def collect_pytorch_cases(data, test_file_name):
    global re_flags
    collected_cases = {"PASSED": [], "FAILED": [], "ERROR": [], "SKIPPED": [], "XFAIL": [], "XPASS": [], "NO_RESULT": []}

    pattern_ok_case = r"^(test_\S+)\s\([^\.]*\.([^\.]*)\)\s+\.\.\..+?(ok)$"
    pattern_failed_case = r"^(test_\S+)\s\([^\.]*\.([^\.]*)\)\s+\.\.\..+?(FAIL)$"
    pattern_error_case = r"^(test_\S+)\s\([^\.]*\.([^\.]*)\)\s+\.\.\..+?(ERROR)$"
    pattern_skipped_case = r"^(test_\S+)\s\([^\.]*\.([^\.]*)\)\s+\.\.\..+?(skipped)\s"
    pattern_xfail_case = r"^(test_\S+)\s\([^\.]*\.([^\.]*)\)\s+\.\.\..+?(expected failure)$"
    # pattern_xpass_case = # haven't found related cases while using pytorch test run.

    ok_cases = re.findall(pattern_ok_case, data, re_flags)
    failed_cases = re.findall(pattern_failed_case, data, re_flags)
    error_cases = re.findall(pattern_error_case, data, re_flags)
    skipped_cases = re.findall(pattern_skipped_case, data, re_flags)
    xfail_cases = re.findall(pattern_xfail_case, data, re_flags)
    # xpass_cases = re.findall(pattern_xpass_case, data, re_flags)

    for case in ok_cases:
        full_case_name = test_file_name + "::" + case[1] + "::" + case[0]
        collected_cases["PASSED"].append(full_case_name)
    for case in failed_cases:
        full_case_name = test_file_name + "::" + case[1] + "::" + case[0]
        collected_cases["FAILED"].append(full_case_name)
    for case in error_cases:
        full_case_name = test_file_name + "::" + case[1] + "::" + case[0]
        collected_cases["ERROR"].append(full_case_name)
    for case in skipped_cases:
        full_case_name = test_file_name + "::" + case[1] + "::" + case[0]
        collected_cases["SKIPPED"].append(full_case_name)
    for case in xfail_cases:
        full_case_name = test_file_name + "::" + case[1] + "::" + case[0]
        collected_cases["XFAIL"].append(full_case_name)
    # for case in xpass_cases:
    #     full_case_name = test_file_name + "::" + case[1] + "::" + case[0]
    #     collected_cases["XPASS"].append(full_case_name)
    return collected_cases

def collect_cases_from_logfile(test, logfile):
    global re_flags
    data = read_file(logfile)
    duration = re.search(r"\[LOGGING\] Time cost: (.+?)s", data, re_flags).group(1)
    test_file_name = test + ".py"
    if re.search(r"^=*\s*test session starts\s*=*$", data, re_flags):
        return float(duration), collect_pytest_cases(data, test_file_name)
    else:
        return float(duration), collect_pytorch_cases(data, test_file_name)

