import os
import sys
import shutil
import uuid
import subprocess
import argparse
import warnings
import time
from math import ceil

from tool.file_utils import load_from_yaml, save_to_yaml, read_file, write_file
from tool.collector import collect_cases_from_logfile, collect_detailed_issues
from tool.skipper import add_dynamic_skipped_cases
from tool.maintainer import update_reference, check_reference, check_ci_pass
from tool.reporter import report_configurations, report_details, report_diffs, report_summary, report_ci_failure
from common import xpu_test_base

test_suite_root = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Main script to run all or specific tests"
    ) 
    parser.add_argument(
        "-o",
        "--log-dir",
        "--log_dir",
        metavar="LOG-DIR",
        nargs=1,
        default=os.getcwd(),
        type=str,
        help="the path to save log files, it should be a directory."
    )
    parser.add_argument(
        "-u",
        "--update-timeout",
        "--update_timeout",
        metavar="TIMEOUT",
        type=int,
        help="if set this flag, the timeout threshold (microsecond) will be updated into config. " \
             "When 0 was passed, the tool will update timeout threshold automatically according to actual processing time. " \
             "Otherwise, the configure will update each timeout threshold according to given TIMEOUT. "
             "In case you want to specify a temporary timeout threshold, please use flag '--set-timeout' instead."
    )
    parser.add_argument(
        "-t",
        "--set-timeout",
        "--set_timeout",
        metavar="TIMEOUT",
        type=int,
        help="set this flag to override the timeout threshold (microsecond) for current run. It won't change saved config." \
             "To change the timeout for the test permanently, please use option '--update-timeout'." \
             "To disable timeout check, please use option '--set-timeout=0'"
    )
    parser.add_argument(
        "-iter",
        "--iterations",
        "--iternum",
        nargs=1,
        default=1,
        type=int,
        help="to set number of iterations to run current test"
    )
    parser.add_argument(
        "-pt",
        "--use-pytest",
        "--use_pytest",
        action="store_true",
        help="try to use pytest to run test if could."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="output pytest or PyTorch verbose info to the screen."
    )
    parser.add_argument(
        "--ipex-verbose",
        nargs="?",
        type=str,
        help="(Experimental) output specified verbose info to the screen." \
             " Supports only onednn, onemkl and ipex verbose. When enabled two or more, use comma without space to join each other."
    )
    parser.add_argument(
        "-m",
        "--mode",
        nargs=1,
        choices=['normal', 'debug', 'ci', 'weekly', 'maintain'],
        default='normal',
        type=str,
        help="mode for run. Supports one of (normal, debug, ci, weekly, maintain). Default is set to normal." \
             " For details about mode, please refer to README.md."
    )
    parser.add_argument(
        '-q',
        '--quiet',
        action="store_true",
        help="quiet mode to run which won't output detail info into console until an unexpected error occurs."
    )
    parser.add_argument(
        '-k',
        '--keyword',
        nargs=1,
        type=str,
        help="tool will run cases which match the given keyword in its case name."
    )
    parser.add_argument(
        '-s',
        '--autoskip',
        action="store_true",
        help="autoskip those fatal errored cases to make sure each test run to an end."
    )
    parser.add_argument(
        "test",
        nargs="*",
        type=str,
        help="test to run. Supports multi-formats for specify test. Such as:\n" \
             "test_name[.py]\tmatch the name of the test file\n" \
             "test_name[.py]::TestClass\trun all cases in the exact class of TestClass in given test file\n" \
             "test_name[.py]::TestClass\t::test_case\tmatch the prefix name of case in given TestClass and given test file\n"
    )
    parser.add_argument(
        "--dry-run",
        "--dry_run",
        action="store_true",
        help="if set this flag, tool will only print out the command for running but won't execute anything."
    )

    return parser.parse_args()

def select_tests_from(include_tests, options):
    global test_suite_root
    tests_list = load_from_yaml(os.path.join(test_suite_root, "config/tests_list.yaml"))
    selected_tests = []
    if options.mode[0] == 'ci':
        selected_tests = tests_list['CI_TESTS'] + include_tests
        print("selected_tests:", selected_tests)
    elif options.mode[0] in ['weekly', 'maintain'] or len(include_tests) <= 0:
        selected_tests = tests_list['TESTS'] + include_tests
    else:
        selected_tests = include_tests if include_tests else tests_list['TESTS']
    # check whether there are tests not ported, skip them
    should_remove_tests = []
    for test in selected_tests:
        split_test_names = test.split("::")
        assert len(split_test_names) >= 1, "[ERROR] Empty test name found. Each test must have a non-empty name"
        test_name = split_test_names[0]
        if test_name.endswith(".py"):
            test_name = test_name[: -len(".py")]
        file_path = os.path.join(test_suite_root, 'test/' + test_name) + ".py"
        if not os.path.exists(file_path):
            should_remove_tests.append(test)
    for test in should_remove_tests:
        selected_tests.remove(test)
    return selected_tests

def run_test(cmd, timeout, quiet, test, logfile):
    global test_suite_root
    returncode = 0
    outmsg = ""
    errmsg = ""
    duration = 0.
    start_time = time.perf_counter()
    try:
        proc = subprocess.run(cmd,
                              capture_output=True,
                              shell=False,
                              timeout=timeout,
                              env=os.environ,
                              cwd=os.path.join(test_suite_root, 'test'),
                              check=True)
        returncode = proc.returncode
        outmsg = proc.stdout.decode('utf-8')
        errmsg = proc.stderr.decode('utf-8')
        end_time = time.perf_counter()
        duration = end_time - start_time
    except subprocess.TimeoutExpired as e_time:
        outmsg = e_time.stdout.decode('utf-8') if e_time.stdout else ""
        errmsg = e_time.stderr.decode('utf-8') + str(e_time) if e_time.stderr else str(e_time)
        end_time = time.perf_counter()
        duration = end_time - start_time
    except subprocess.CalledProcessError as e_called:
        outmsg = e_called.stdout.decode('utf-8') if e_called.stdout else ""
        errmsg = e_called.stderr.decode('utf-8') + str(e_called) if e_called.stderr else str(e_called)
        end_time = time.perf_counter()
        duration = end_time - start_time
    # finally:
    tail = f"[LOGGING] Time cost: {duration}s"
    if not quiet:
        write_file(outmsg, sys.stdout, end_char="\n")
        write_file(errmsg, sys.stderr, end_char="\n")
        write_file(tail, sys.stdout, end_char="\n")
    write_file(outmsg, logfile, mode="w", end_char="\n")
    write_file(errmsg, logfile, mode="a", end_char="\n")
    write_file(tail,   logfile, mode="a", end_char="\n")
    return collect_cases_from_logfile(test, logfile)

def run_tests(selected_tests, options):
    global test_suite_root
    test_root = os.path.join(test_suite_root, 'test/')
    output_path = os.path.join(os.path.abspath(options.log_dir), "test-reports")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        warnings.warn(f"[INFO] Clean old output directory for log files: {output_path}")
    os.makedirs(output_path)
    anls_path = os.path.join(os.path.abspath(options.log_dir), "analysis-reports")
    if os.path.exists(anls_path):
        shutil.rmtree(anls_path)
        warnings.warn(f"[INFO] Clean old anls directory for log files: {anls_path}")
    os.makedirs(anls_path)
    configure_log_file_path = os.path.join(anls_path, "configurations.log")
    ci_error_log_path = os.path.join(anls_path, "ci_error.log")
    maintaince_result_log_path = os.path.join(anls_path, "maintaince_result.log")
    diffs_log_path = os.path.join(anls_path, "diffs_result.log")
    weekly_report_path = os.path.join(anls_path, "weekly_summary.log")
    weekly_diffs_path = os.path.join(anls_path, "weekly_diffs.log")
    weekly_details_path = os.path.join(anls_path, "weekly_details.log")
    details_msg_list = []
    checked_cases = {}
    total_results = {}
    warnings.warn(f"[INFO] Created output directory for log files: {output_path}")
    timeout_list = load_from_yaml(os.path.join(test_suite_root, "config/timeout_config.yaml"))
    for test in selected_tests:
        split_test_names = test.split("::")
        assert len(split_test_names) >= 1, "[ERROR] Empty test name found. Each test must have a non-empty name"
        test_name = split_test_names[0]
        if test_name.endswith(".py"):
            test_name = test_name[: -len(".py")]
        file_name = test_name + ".py"
        full_file_path = os.path.join(test_root, file_name)
        assert os.path.exists(full_file_path), f"[ERROR] test file not found, please re-check the path {full_file_path}"
        class_name = None
        if len(split_test_names) > 1:
            class_name = split_test_names[1]
        case_prefix = None
        if len(split_test_names) > 2:
            case_prefix = split_test_names[2]
        keyword = options.keyword
        if case_prefix and keyword:
            warnings.warn(f"[WARNING] Keyword won't take effect as case name has been mentioned as {case_prefix}")
        
        cmd = [sys.executable]
        if options.use_pytest:
            cmd.extend(['-m', 'pytest', '-ra'])
        cmd.append(os.path.relpath(full_file_path, test_root))
        if options.verbose or options.mode[0] in ['ci', 'maintain', 'weekly']:
            cmd.append("-v")
        if class_name:
            cmd.append(class_name)
        if case_prefix:
            cmd.extend(["-k", case_prefix])
        elif keyword:
            cmd.extend(["-k", keyword])
        
        if options.dry_run:
            print("Prepare to execute command: ", " ".join(cmd))
        else:
            if test_name not in timeout_list.keys():
                timeout_list[test_name] = 0
            timeout = options.set_timeout if options.set_timeout else timeout_list[test_name]
            if options.update_timeout is not None:
                timeout = options.update_timeout
                assert timeout >= 0, f"[ERROR] time(ms) passed to --update-timeout must be non-negative integer, but got {options.update_timeout}"
                timeout_list[test_name] = timeout
            for i in range(1, options.iterations + 1):
                log_file_name = test_name.replace('/', '-') + ".iter_" + str(i) + ".uuid_" + str(uuid.uuid1()) + ".log"
                full_log_path = os.path.join(output_path, log_file_name)
                if not options.quiet:
                    report_configurations(iter=i, log_path=full_log_path, options=options, timeout=timeout,
                                          cmd=" ".join(cmd), file=sys.stdout)
                report_configurations(iter=i, log_path=full_log_path, options=options, timeout=timeout,
                                      cmd=" ".join(cmd), file=configure_log_file_path)
                autoskip = options.autoskip # whether to automatially skip fatal errors (seg fault, core dumped, timeout)
                if options.mode[0] in ['weekly', 'maintain']:
                    autoskip = True
                print(f"[INFO] Begin to run '{test_name}'")
                duration, cases_result = run_test(
                    cmd,
                    timeout if timeout > 0 else None,
                    options.quiet,
                    test_name,
                    full_log_path)
                while autoskip and cases_result["NO_RESULT"]:
                    add_dynamic_skipped_cases(cases_result["NO_RESULT"], "Auto skipped due to Fatal Error or Timed Out")
                    print(f"[INFO] Re-run '{test_name}' triggered by autoskip")
                    duration, cases_result = run_test(
                        cmd,
                        timeout if timeout > 0 else None,
                        options.quiet,
                        test_name,
                        full_log_path)
                print(f"[INFO] Finished '{test_name}' with time cost : {duration}s")

                if options.mode[0] == 'ci':
                    issued_cases, short_details, details = check_ci_pass(cases_result, full_log_path)
                    if issued_cases:
                        header = "============================= CI FAILED IN PORTED UT =============================\n"
                        tail = f"[CI FAILED] Please check cases with COMMAND: {' '.join(cmd)}\n"
                        report_ci_failure(issued_cases, short_details, header, tail, sys.stdout)
                        report_ci_failure(issued_cases, details, header, tail, ci_error_log_path)
                        warnings.warn(f"[INFO] CI failure detailed report written to {ci_error_log_path}", UserWarning)
                        exit(1)

                if test_name not in total_results:
                    total_results[test_name] = []
                total_results[test_name].append((duration, cases_result))
                for tag, cases in check_reference(cases_result).items():
                    if tag not in checked_cases:
                        checked_cases[tag] = []
                    checked_cases[tag].extend(cases)
                for tag, cases in cases_result.items():
                    details_msg_list.extend(collect_detailed_issues(cases, full_log_path))
                if options.update_timeout is not None:
                    if timeout > 0 and duration >= timeout:
                        timeout_list[test_name] = timeout
                    else:
                        timeout_list[test_name] = max(timeout, ceil(2 * duration))
    if options.mode[0] == 'ci':
        header = "=============================== CI SUCCEED IN PORTED UT ===========-====================\n"
        write_file(header, sys.stdout)
    if options.mode[0] == 'maintain':
        update_reference(total_results, "maintain")
    if options.mode[0] == 'weekly':
        update_reference(total_results, "weekly")
    if options.update_timeout is not None or options.mode[0] == 'maintain':
        save_to_yaml(timeout_list, os.path.join(test_suite_root, "config/timeout_config.yaml"))
    if options.mode[0] == 'maintain':
        header = "============================= MAINTAINCE DONE OF PORTED UT =============================\n"
        tail =   "==================================== SUMMARY FINISH ====================================\n"
        if not options.quiet:
            report_summary(total_results, header, "", sys.stdout)
        report_summary(total_results, header, tail, maintaince_result_log_path)
        header = "========================== STATUS CHANGED CASES IN MAINTAINCE ==========================\n"
        tail =   "================================== DIFFERENCE FINISH ===================================\n"
        if not options.quiet:
            report_diffs(checked_cases, header, tail, sys.stdout)
        report_diffs(checked_cases, header, tail, diffs_log_path)
    elif options.mode[0] == 'weekly':
        header = "========================= WEEKLY VALIDATION DONE OF PORTED UT ==========================\n"
        tail =   "==================================== SUMMARY FINISH ====================================\n"
        if not options.quiet:
            report_summary(total_results, header, "", sys.stdout)
        report_summary(total_results, header, tail, weekly_report_path)
        header = "============================ STATUS CHANGED CASES IN WEEKLY ============================\n"
        tail =   "================================== DIFFERENCE FINISH ===================================\n"
        if not options.quiet:
            report_diffs(checked_cases, header, tail, sys.stdout)
        report_diffs(checked_cases, header, tail, weekly_diffs_path)
        header = "=========================== DETAILED FAILED CASES IN WEEKLY ============================\n"
        tail =   "=================================== DETAILS FINISH =====================================\n"
        # if not options.quiet:
        #     report_details(details_msg_list, header, tail, sys.stdout)
        report_details(details_msg_list, header, tail, weekly_details_path)

def set_tool_env():
    global test_suite_root
    # os.environ['TORCH_TEST_DEVICES'] = os.path.join(test_suite_root, "common/xpu_test_base.py")
    sys.path.append(test_suite_root)

if __name__ == "__main__":
    options = parse_args()
    assert isinstance(options.iterations, int) and options.iterations > 0, \
        f"[ERROR] Iterations should be a positive iteger, but got {options.iterations}"
    if options.mode[0] == 'ci':
        assert options.update_timeout == None, \
            "[ERROR] Bad usage. In CI mode it should not update anything but only do check."
    assert not (options.update_timeout is not None and options.set_timeout is not None), \
            "[ERROR] Conflict in setting two timeout together. Please use one of it if you want to change the timeout threshold for current test."
    set_tool_env()
    selected_tests = select_tests_from(options.test, options)
    run_tests(selected_tests, options)
