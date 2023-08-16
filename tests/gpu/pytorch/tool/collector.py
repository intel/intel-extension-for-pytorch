import re
from .file_utils import read_file

re_flags = re.M | re.S

def collect_fatal_error(issued_cases, logfile):
    global re_flags
    data = read_file(logfile)
    pytorch_result = re.search("Command.+?died with.+?(?=\n)", data, re_flags)
    pytest_result = re.search("fatal python error.+?(?=\n\n)", data, re_flags | re.I)
    timout_result = re.search("Command.+?timed out.+?(?=\n)", data, re_flags)
    if pytorch_result:
        return pytorch_result.group(0)
    if pytest_result:
        return pytest_result.group(0)
    if timeout_result:
        return timeout_result.group(0)
    return "Unknown fatal error. Please refer to the raw log directly"

def collect_pytest_details(issued_cases, data, with_short=False):
    global re_flags
    issued_casenames = [case.split("::", 1)[-1] for case in issued_cases]
    details = []
    pattern = r'^_+[^\n]*\s(\S*test_\S*)\s_+$'
    lines = data.split('\n')
    started_linenos = []
    for lineno, line in enumerate(lines):
        res = re.search(pattern, line, re_flags)
        if res:
            casename = res.group(1).replace('.', '::')
            started_linenos.append((lineno, casename))
        if re.match(r'^=+.+?=+$', line, re_flags):
            started_linenos.append((lineno, "SPLITOR"))
    sorted_started_linenos = sorted(started_linenos)
    for i in range(len(sorted_started_linenos) - 1):
        start_lineno, casename = sorted_started_linenos[i]
        next_lineno = sorted_started_linenos[i + 1][0]
        if casename == "SPLITOR" or casename not in issued_casenames:
            continue
        detail = "\n".join(lines[start_lineno:next_lineno])
        details.append(detail)
    if with_short:
        collected_short_details = re.findall(r"(\S+\s\S+?::(\S+)\s-\s?([^\n]*))$", data, re_flags)
        short_details = []
        for line, casename, short_detail in collected_short_details:
            if casename in issued_casenames:
                short_details.append(line)
        return short_details, details
    return details

def collect_pytorch_details(issued_cases, data, with_short=False):
    global re_flags
    issued_casenames = [case.split("::", 1)[-1] for case in issued_cases]
    details = []
    pattern = r'(^=+\n[A-Za-z]+:\s*(test_\S*)\s*\([^\.]*\.(\S*)\)\n-+.*?(?=\n\n))'
    results = re.findall(pattern, data, re_flags)
    for detail, casename, clsname in results:
        if clsname + "::" + casename in issued_casenames:
            details.append(detail + "\n")
    if with_short:
        short_details = []
        for detail in details:
            short_detail = re.search(r'^\S+Error:.+?$', detail, re_flags).group(0)
            short_details.append(short_detail)
        return short_details, details
    return details

def collect_detailed_issues(issued_cases, logfile, with_short=False):
    global re_flags
    test_name = logfile.split('.')[0].replace('-', '/')
    data = read_file(logfile)
    if re.search(r"^=*\s*test session starts\s*=*$", data, re_flags):
        return collect_pytest_details(issued_cases, data, with_short)
    else:
        test_file_name = test_name + ".py"
        return collect_pytorch_details(issued_cases, data, with_short)

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

    pattern_case = r"^(test_\S+)\s\([^\.]*\.([^\.]*)\)\s+\.\.\..+?(ok|FAIL|ERROR|skipped|expected failure|Command)\s"
    # pattern_xpass_case = # haven't found related cases while using pytorch test run.

    matched_cases = re.findall(pattern_case, data, re_flags)
    # xpass_cases = re.findall(pattern_xpass_case, data, re_flags)

    for case in matched_cases:
        full_case_name = test_file_name + "::" + case[1] + "::" + case[0]
        if case[2] == 'ok':
            collected_cases["PASSED"].append(full_case_name)
        elif case[2] == 'FAIL':
            collected_cases["FAILED"].append(full_case_name)
        elif case[2] == 'ERROR':
            collected_cases["ERROR"].append(full_case_name)
        elif case[2] == 'skipped':
            collected_cases["SKIPPED"].append(full_case_name)
        elif case[2] == 'expected failure':
            collected_cases["XFAIL"].append(full_case_name)
        elif case[2] == 'Command':
            collected_cases["NO_RESULT"].append(full_case_name)

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

