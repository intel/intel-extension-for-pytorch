import os
import sys
from multiprocessing import Process, Queue
import json
import argparse
import re
import subprocess
from collections import OrderedDict
import copy

from case_utils import match_cases
from file_utils import save_to_json, load_from_json, read_file, write_file

script_path: str = os.path.dirname(os.path.realpath(__file__))
root_path: str = os.path.realpath(os.path.join(script_path, ".."))
ALL_PATHS: dict = {
    "script_path": script_path,
    "root_path": root_path,
    "common_path": os.path.join(root_path, "common"),
    "test_path"  : os.path.join(root_path, "test"),
    "config_path": os.path.join(root_path, "config"),
    "env_path"   : os.path.join(root_path, "env"),
    "tool_path"  : os.path.join(root_path, "tool"),
    "log_path"   : os.path.join(root_path, "log"),
}
raw_log_path: str = ""
anls_log_path: str = ""
saveref = False
need_compare = False
need_classify = False
final_result = OrderedDict()


class AnlsLogBase():
    def __init__(self, logfile):
        self.logfile = logfile
        self.data_valid = True
        self.time_consume = 0.  # this param saves time consuming for each test class
        self.total_set = set()
        self.pass_set = set()
        self.expected_failure_set = set()
        self.error_set = set()
        self.fail_set = set()
        self.skip_set = set()
        self.not_impl_set = set()
        self.not_ready_set = set()
        self.not_dtype_set = set()
        self.error_detailed_set = set()
        self.fail_detailed_set = set()
        self.DEBUG_MSG = ""

    def analysis(self):
        hasResult = False   # to mark if result line is found.
        hasRet = False      # to mark if reture line is found.

        data = read_file(self.logfile)
        # search for result line like "Ran 1234 tests in 678.90s"
        resultLine = re.search(r'^Ran (\d+) test[s]? in ([\d\.]+)s', data, re.M)
        returnLine = re.search(r'^OK|^FAILED', data, re.M)
        if returnLine is not None:
            hasRet = True
        if resultLine is not None:
            hasResult = True
            self.time_consume = float(resultLine.group(2))
            
            self.total_set = match_cases(
                data,
                r'(^test_\S*?) \([^()]*?\)((?!^test_|\n\n).)*',
                *(0,))
            self.pass_set = match_cases(
                data,
                r'(^test_\w+) \([^()]+\)((?!^test_|\n\n).)*?ok$',
                *(0,))
            self.expected_failure_set = match_cases(
                data,
                r'(^test_\w+) \([^()]+\)((?!^test_|\n\n).)*?expected failure$',
                *(0,))
            self.error_set = match_cases(
                data,
                r'(^test_\w+) \([^()]+\)((?!^test_|\n\n).)*?ERROR$',
                *(0,))
            self.fail_set = match_cases(
                data,
                r'(^test_\w+) \([^()]+\)((?!^test_|\n\n).)*?FAIL$',
                *(0,))
            self.skip_set = match_cases(
                data,
                r'(^test_\w+) \([^()]+\)((?!^test_|\n\n).)*?skipped (((?!^test_|\n\n).)*)$',
                *(0, 2))
            for case, msg in self.skip_set:
                if re.search(r'not ready on XPU', msg, re.M) is not None:
                    self.not_ready_set.add((case, msg))
                elif re.search(r'.not.implemented.', msg, re.M) is not None:
                    self.not_impl_set.add((case, msg))
                elif re.search(r'dtype not support on XPU', msg, re.M) is not None:
                    self.not_dtype_set.add((case, msg))
            self.error_detailed_set = match_cases(
                data,
r"""======================================================================
ERROR: (test_\w+) \([^()]+\).*?
----------------------------------------------------------------------
Traceback .+?
(^\S+Error):?(.*?)(?<=\n\n)""",
                *(0, 1, 2))
            self.fail_detailed_set = match_cases(
                data,
r"""======================================================================
FAIL: (test_\w+) \([^()]+\).*?
----------------------------------------------------------------------
Traceback .+?
(^\S+Error):?(.*?)(?<=\n\n)""",
                *(0, 1, 2))

            try:
                assert len(self.error_set) == len(self.error_detailed_set)
            except AssertionError:
                error_cases = set()
                error_dt_cases = set()
                for e in self.error_set:
                    error_cases.add(e[0])
                for e in self.error_detailed_set:
                    error_dt_cases.add(e[0])
                if error_cases - error_dt_cases:
                    self.DEBUG_MSG += f"DEBUGGING: Error cases not has detailed info in file {self.logfile}. They are:\n\t{error_cases - error_dt_cases}.\n"
                    self.data_valid = False
                if error_dt_cases - error_cases:
                    self.DEBUG_MSG += f"DEBUGGING: Error cases has detailed info but not logged in file {self.logfile}. They are:\n\t{error_dt_cases - error_cases}.\n"
                    self.data_valid = False
            try:
                assert len(self.fail_set) == len(self.fail_detailed_set)
            except AssertionError:
                fail_cases = set()
                fail_dt_cases = set()
                for f in self.fail_set:
                    fail_cases.add(f[0])
                for f in self.fail_detailed_set:
                    fail_dt_cases.add(f[0])
                if fail_cases - fail_dt_cases:
                    self.DEBUG_MSG += f"DEBUGGING: Fail cases not has detailed info in file {self.logfile}. They are:\n\t{fail_cases - fail_dt_cases}.\n"
                    self.data_valid = False
                if fail_dt_cases - fail_cases:
                    self.DEBUG_MSG += f"DEBUGGING: Fail cases has detailed info but not logged in file {self.logfile}. They are:\n\t{fail_dt_cases - fail_cases}.\n"
                    self.data_valid = False

        if not hasResult:
            self.DEBUG_MSG += f"DEBUGGING: Tests not ran to an end. Please re-check the raw log and re-run it. {self.logfile}\n"
            self.data_valid = False
        elif not hasRet:
            self.DEBUG_MSG = f"DEBUGGING: Tests got no result as OK or FAILED. Please re-check the raw log and re-run it. {self.logfile}\n"
            self.data_valid = False


def anls_reduce(anls_list):
    anls_ret = copy.deepcopy(anls_list[0])
    for i in range(1, len(anls_list)):
        anls_log = anls_list[i]
        anls_ret.time_consume += anls_log.time_consume
        anls_ret.pass_set.update(anls_log.pass_set)
        anls_ret.expected_failure_set.update(anls_log.expected_failure_set)
        anls_ret.error_set.update(anls_log.error_set)
        anls_ret.fail_set.update(anls_log.fail_set)
        anls_ret.skip_set.update(anls_log.skip_set)
        anls_ret.not_impl_set.update(anls_log.not_impl_set)
        anls_ret.not_ready_set.update(anls_log.not_ready_set)
        anls_ret.not_dtype_set.update(anls_log.not_dtype_set)
        anls_ret.error_detailed_set.update(anls_log.error_detailed_set)
        anls_ret.fail_detailed_set.update(anls_log.fail_detailed_set)
        anls_ret.data_valid &= anls_log.data_valid
        anls_ret.DEBUG_MSG += anls_log.DEBUG_MSG
    anls_ret.time_consume /= len(anls_list)
    # discard those random pass/expected/error/skip cases from fail set
    anls_ret.fail_set.difference_update(anls_ret.pass_set)
    anls_ret.fail_set.difference_update(anls_ret.expected_failure_set)
    anls_ret.fail_set.difference_update(anls_ret.error_set)
    anls_ret.fail_set.difference_update(anls_ret.skip_set)
    # discard those random skip cases from pass/expected/error set
    anls_ret.error_set.difference_update(anls_ret.pass_set)
    anls_ret.error_set.difference_update(anls_ret.expected_failure_set)
    anls_ret.error_set.difference_update(anls_ret.skip_set)
    # discard those random not_impl/not_ready cases from not_dtype set
    anls_ret.not_dtype_set.difference_update(anls_ret.not_ready_set)
    anls_ret.not_dtype_set.difference_update(anls_ret.not_impl_set)
    # discard those random not_ready cases from not_impl set
    anls_ret.not_impl_set.difference_update(anls_ret.not_ready_set)

    return anls_ret

def analysis_final():
    global final_result, saveref, need_compare, need_classify, anls_log_path

    total_time = 0.
    total_ran = 0
    total_pass = 0
    total_fail = 0
    total_error = 0
    total_not_ready = 0
    total_not_impl = 0
    total_not_dtype = 0
    DEBUG_MSG = ""

    pass_set = set()
    error_set = set()
    fail_set = set()
    error_detailed_set = set()
    fail_detailed_set = set()
    for full_name, anls_list in final_result.items():
        if len(anls_list) > 0:
            final_result[full_name] = anls_reduce(anls_list)
        else:
            raise f"{full_name} has 0 logs related, which is invalid"
        anls_log = final_result[full_name]
        
        if anls_log.data_valid is True:
            total_time += anls_log.time_consume
            total_ran += len(anls_log.total_set) - len(anls_log.skip_set)
            total_pass += len(anls_log.pass_set) + len(anls_log.expected_failure_set)
            total_error += len(anls_log.error_set)
            total_fail += len(anls_log.fail_set)
            total_not_ready += len(anls_log.not_ready_set)
            total_not_impl += len(anls_log.not_impl_set)
            total_not_dtype += len(anls_log.not_dtype_set)

            # update total set for saving or compararison
            pass_set.update(anls_log.pass_set)
            error_set.update(anls_log.error_set)
            fail_set.update(anls_log.fail_set)
    # print sets result to file
    write_file(os.path.join(anls_log_path, "pass_list.log"),
               "\n".join([x[0] for x in pass_set]))
    write_file(os.path.join(anls_log_path, "fail_list.log"),
               "\n".join([x[0] for x in fail_set]))
    write_file(os.path.join(anls_log_path, "error_list.log"),
               "\n".join([x[0] for x in error_set]))
    if saveref:
        write_file(os.path.join(anls_log_path, "pass_list_ref.log"),
                   "\n".join([x[0] for x in pass_set]))

    # print result to screen
    formatStr = "{:64}{:>16}{:>16}{:>16}{:>16}{:>16}{:>16}|{:>16}{:>16}"
    title = formatStr.format(
            "TestClass",
            "Time(s)",
            "Total Ran",
            "Passed",
            "Failures",
            "Errors",
            "Skipped",
            "XPU not impl",
            "XPU not dtype")
    titlewidth = len(title)
    print("=" * titlewidth)
    print(title)
    print("-" * titlewidth)
    total_devisor = max(total_ran, 1)
    print(str(total_pass) + f"({format(total_pass / total_devisor * 100, '.2f')}%)")
    print(formatStr.format(
        "Total",
        format(total_time, '.3f'),
        total_ran,
        str(total_pass) + f"({format(total_pass / total_devisor * 100, '.2f')}%)",
        str(total_fail) + f"({format(total_fail / total_devisor * 100, '.2f')}%)",
        str(total_error) + f"({format(total_error / total_devisor * 100, '.2f')}%)",
        total_not_ready,
        total_not_impl,
        total_not_dtype))
    print("-" * titlewidth)
    for full_name, anls_log in final_result.items():
        if anls_log.data_valid is True:
            ran = len(anls_log.total_set)
            devisor = max(ran, 1)
            passed = len(anls_log.pass_set)
            failed = len(anls_log.fail_set)
            errored = len(anls_log.error_set)
            not_ready = len(anls_log.not_ready_set)
            not_impl = len(anls_log.not_impl_set)
            not_dtype = len(anls_log.not_dtype_set)
            print(formatStr.format(
                full_name,
                format(anls_log.time_consume, '.3f'),
                ran,
                str(passed) + f"({format(passed / devisor * 100, '.2f')}%)",
                str(failed) + f"({format(failed / devisor * 100, '.2f')}%)",
                str(errored) + f"({format(errored / devisor * 100, '.2f')}%)",
                not_ready,
                not_impl,
                not_dtype))
        else:
            print("{:64}{}".format(full_name, "DATA INVALID, see DEBUGGING messages"))
            DEBUG_MSG += anls_log.DEBUG_MSG
    print("=" * titlewidth)
    print(DEBUG_MSG)

    # compare to reference
    if need_compare:
        pass_ref_set = set()
        data = read_file(os.path.join(anls_log_path, "pass_list_ref.log"))
        lines = data.split('\n')
        map(lambda l: pass_ref_set.add(l.strip()), lines)
        pass_ref_set.discard("")
        regression_set = pass_ref_set - pass_set
        if len(regression_set) > 0:
            print("Regression detected: (passed in reference but not in current run)")
            for case in regression_set:
                print(case)
            print("=" * titlewidth)
            os._exit(1)
        else:
            print("Regression NOT detected")
            print("=" * titlewidth)

    if need_classify:
        data_for_write = ""
        for full_name, anls_log in final_result.items():
            for case, err_type, err_msg in anls_log.error_detailed_set:
                data_for_write += '\t'.join([full_name + "::" + case,
                                             err_type,
                                             err_msg.replace('\n', ' ')])
                data_for_write += '\n'
        write_file(os.path.join(anls_log_path, 'for_xlsx/error_list.for_xlsx.txt'),
                   data_for_write)
        data_for_write = ""
        for full_name, anls_log in final_result.items():
            for case, fail_type, fail_msg in anls_log.fail_detailed_set:
                data_for_write += '\t'.join([full_name + "::" + case,
                                             fail_type,
                                             fail_msg.replace('\n', ' ')])
                data_for_write += '\n'
        write_file(os.path.join(anls_log_path, 'for_xlsx/fail_list.for_xlsx.txt'),
                   data_for_write)

def analysis(fname, clsname, logfile, Q):
    global final_result

    anls_instance = AnlsLogBase(logfile)
    anls_instance.analysis()
    full_name = fname + "::" + clsname
    Q.put((full_name, anls_instance))

def run_analysis(test_map):
    global final_result
    process_list = []

    for path, dir_list, file_list in os.walk(raw_log_path):
        for logfile in file_list:
            namelist = logfile.split('.')
            if len(namelist) != 4:
                continue    # we won't analysis logs for spec case test
            fname = namelist[0] + ".py"
            clsname = namelist[1]
            epoch = namelist[2]
            extname = namelist[3]
            if extname != 'log':
                continue    # we ingnore invalid logfiles which not end with ".log"
            if (fname not in test_map) or (clsname[:-3] not in test_map[fname]):
                continue    # we only analysis logs for recognized fname::clsname
            full_log_path = os.path.join(path, logfile)
            Q = Queue()
            p = Process(target=analysis, args=(fname, clsname, full_log_path, Q))
            p.deamon=True
            p.start()
            process_list.append(p)
            full_name, anls_instance = Q.get()
            if full_name not in final_result:
                final_result[full_name] = []
            final_result[full_name].append(anls_instance)

    for p in process_list:
        p.join()

    analysis_final()

def main():
    global script_path, root_path, ALL_PATHS, raw_log_path, anls_log_path, need_compare, need_classify

    # parse args
    parser = argparse.ArgumentParser(description="Auto script for analysing raw logs")
    parser.add_argument(
            '--logdir',
            metavar='logdir',
            dest='logdir',
            default=ALL_PATHS["log_path"],
            type=str,
            help="the path of logfiles stored, it should be a directory and must have 'raw_logs' under this path")
    parser.add_argument(
            '--saveref',
            action="store_true",
            help="save reference pass list if no break tests")
    parser.add_argument(
            '--compare',
            action="store_true",
            help="compare current pass list against reference to see if regression occurred")
    parser.add_argument(
            '--classify',
            action="store_true",
            help="output classified error/failure list for xlsx.")
    parser.add_argument(
            '--clean',
            action="store_true",
            help="clean analysis logs")

    args = parser.parse_args()
    raw_log_path = os.path.join(args.logdir, "raw_logs")
    anls_log_path = os.path.join(args.logdir, "anls_logs")
    saveref = args.saveref
    need_compare = args.compare
    need_classify = args.classify
    need_clean = args.clean

    if need_clean is True and os.path.exists(anls_log_path):
        for path, dir_list, file_list in os.walk(anls_log_path):
            for filename in file_list:
                if os.path.splitext(filename)[-1] == ".log":
                    full_path = os.path.join(path, filename)
                    os.remove(full_path)
                    print("Removed old logfile: {}".format(full_path))
        os._exit(0)
    elif need_clean is True:    # anls_log_path not exist
        print("Task skipped: clean path '{}' not exist".format(anls_log_path))
        os._exit(0)

    if not os.path.exists(anls_log_path):
        os.makedirs(anls_log_path)
        print("Created log directory: {}".format(anls_log_path))

    # get the test map
    test_map = load_from_json(os.path.join(ALL_PATHS["config_path"], "test_map.json"))

    # run analysis with multi-process
    run_analysis(test_map)

if __name__ == '__main__':
    main()
