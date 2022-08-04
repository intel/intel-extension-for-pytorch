import os
import sys
from multiprocessing import Process, Queue
import json
import argparse
import re
import subprocess
from collections import OrderedDict
import copy

script_path = ""
common_path = ""
log_path = ""
raw_log_path = ""
anls_log_path = ""
saveref = False
need_compare = False
final_result = OrderedDict()

class AnlsLogBase():
    def __init__(self, fname, clsname, epoch, logfile):
        self.fname = fname
        self.clsname = clsname
        self.epoch = epoch
        self.logfile = logfile
        self.time_consume = 0.  # this param saves time consuming for each test class
        self.total_ran = 0      # this param only contains those valid tests and except for invalid skips
        self.expected = 0       # this param contains all ok cases and expected failures
        self.fail = 0           # this param calculates those AssertionError(s)
        self.error = 0          # this param contains all errors, such as RuntimeError, IndexError, TypeError, etc.
        self.total_skip = 0     # this param contains all type of skipped cases, even though part of them must not calculates into total_ran
        self.valid_skip = 0     # this param contains those skipped cases should be fixed in Q3 or planned
        self.xpu_not_impl = 0   # this param calculates those not implemented issues raised by PyTorch with XPU related backends
        self.xpu_not_ready = 0  # this param calculates those skipped manually by ourselves in skip_list.json
        self.xpu_not_dtype = 0  # this param calculates those skipped cases due to dtype not support by XPU yet
        self.pass_set = set()
        self.fail_set = set()
        self.error_set = set()
        self.not_impl_set = set()
        self.not_ready_set = set()
        self.not_dtype_set = set()
        self.errMsg:str = None

    def analysis(self):
        hasResult = False   # to mark if result line is found.
        hasRet = False      # to mark if reture line is found.
        hasDot = False      # to mark if pattern '...' is found
        cur_case = ""       # to mark the current case not archived
        with open(self.logfile, "r") as ifile:
            lines = ifile.readlines()
            for line in lines:
                # search for result line like "Ran 1234 tests in 678.90s"
                resultLine = re.search(r'^Ran (\d+) test[s]? in ([\d\.]+)s', line, re.M)
                if resultLine is not None:
                    hasResult = True
                    self.total_ran = int(resultLine.group(1))
                    self.time_consume = float(resultLine.group(2))
                # search for return line like "FAILED (failures=3, errors=4, skipped=5, expected failures=6)"
                retLine = re.search(r'^OK.*|^FAILED \(.*?\)', line, re.M)
                if retLine is not None:
                    hasRet = True
                    failures = re.search(r'(?<!expected\s)failure[s]?=(\d+)', retLine.group(0), re.M)
                    errors = re.search(r'(?<!expected\s)error[s]?=(\d+)', retLine.group(0), re.M)
                    skips = re.search(r'(?<!expected\s)skipped=(\d+)', retLine.group(0), re.M)
                    if failures is not None:
                        self.fail = int(failures.group(1))
                    if errors is not None:
                        self.error = int(errors.group(1))
                    if skips is not None:
                        self.total_skip = int(skips.group(1))
                caseLine = re.search(r'^test_\w+', line, re.M)
                if caseLine is not None:
                    cur_case = caseLine.group()
                dotLine = re.search(r' \.\.\. ', line, re.M)
                if dotLine is not None:
                    hasDot = True
                okLine = re.search(r'\bok\b|\bexpected failure\b', line, re.M)
                if okLine is not None and hasDot:
                    self.pass_set.add(cur_case)
                    hasDot = False      # matched dots should return to false
                failLine = re.search(r'\bFAIL\b', line, re.M)
                if failLine is not None and hasDot:
                    self.fail_set.add(cur_case)
                    hasDot = False
                errorLine = re.search(r'\bERROR\b', line, re.M)
                if errorLine is not None and hasDot:
                    self.error_set.add(cur_case)
                    hasDot = False
                skipLine = re.search(r'\bskipped\b', line, re.M)
                if skipLine is not None and hasDot:
                    if re.search(r'not ready on XPU', line, re.M) is not None:
                        self.not_ready_set.add(cur_case)
                    elif re.search(r'.not.implemented.', line, re.M) is not None:
                        self.not_impl_set.add(cur_case)
                    elif re.search(r'dtype not support on XPU', line, re.M) is not None:
                        self.not_dtype_set.add(cur_case)
                    hasDot = False
            self.xpu_not_ready = len(self.not_ready_set)
            self.xpu_not_impl = len(self.not_impl_set)
            self.xpu_not_dtype = len(self.not_dtype_set)

            self.expected = self.total_ran - self.fail - self.error - self.total_skip
            assert self.expected == len(self.pass_set), "{}::{} numeric pass is {} but pass set has {} cases".format(self.fname, self.clsname, self.expected, len(self.pass_set))
            
            assert self.fail == len(self.fail_set), "{}::{} numeric failure is {} but fail set has {} cases".format(self.fname, self.clsname, self.fail, len(self.fail_set))
            assert self.error == len(self.error_set), "{}::{} numeric error is {} but error set has {} cases".format(self.fname, self.clsname, self.error, len(self.error_set))

            self.valid_skip = len(self.not_ready_set)
            self.total_ran = self.total_ran - self.total_skip + self.valid_skip
            assert self.total_ran == len(self.pass_set) + len(self.fail_set) + len(self.error_set) + len(self.not_ready_set), "{}::{} numeric total ran is {} but case set has {} cases as total".format(self.fname, self.clsname, self.total_ran, len(self.pass_set) + len(self.fail_set) + len(self.error_set) + len(self.not_ready_set))

            if not hasResult:
                self.errMsg = "This test not ran to an end. Please re-check the raw log and re-run it."
            elif not hasRet:
                self.errMsg = "This test got no result as OK or FAILED. Please re-check the raw log and re-run it."

def load_from_json(json_file):
    with open(json_file, "r") as load_f:
        load_dict = json.load(load_f)
    return load_dict

def save_to_json(src, target_file):
    data = json.dumps(src, indent=2)
    with open(target_file, "w", newline='\n') as save_f:
        save_f.write(data)

def analysis(fname, clsname, epoch, logfile, Q):
    global final_result

    anls_instance = AnlsLogBase(fname, clsname, epoch, logfile)
    anls_instance.analysis()
    full_name = fname + "::" + clsname
    # if full_name not in final_result:
    #     final_result[full_name] = []
    # final_result[full_name].append(anls_instance)
    Q.put((full_name, anls_instance))

def anls_reduce(anls_list):
    errMsg_set = set()

    anls_ret = copy.deepcopy(anls_list[0])
    anls_ret.time_consume = 0.
    for anls_log in anls_list:
        if anls_log.errMsg is not None:
            errMsg_set.add(anls_log.errMsg)
        anls_ret.time_consume += anls_log.time_consume
        anls_ret.pass_set = anls_ret.pass_set & anls_log.pass_set
        anls_ret.fail_set = anls_ret.fail_set | anls_log.fail_set
        anls_ret.error_set = anls_ret.error_set | anls_log.error_set
        anls_ret.not_impl_set = anls_ret.not_impl_set | anls_log.not_impl_set
        anls_ret.not_ready_set = anls_ret.not_ready_set | anls_log.not_ready_set
        anls_ret.not_dtype_set = anls_ret.not_dtype_set | anls_log.not_dtype_set
    temp_set = copy.deepcopy(anls_ret.error_set)
    anls_ret.fail_set = anls_ret.fail_set - temp_set
    temp_set = temp_set | anls_ret.fail_set
    anls_ret.not_ready_set = anls_ret.not_ready_set - temp_set
    temp_set = temp_set | anls_ret.not_ready_set
    anls_ret.not_impl_set = anls_ret.not_impl_set - temp_set
    temp_set = temp_set | anls_ret.not_impl_set
    anls_ret.not_dtype_set = anls_ret.not_dtype_set - temp_set

    anls_ret.expected = len(anls_ret.pass_set)
    anls_ret.fail = len(anls_ret.fail_set)
    anls_ret.error = len(anls_ret.error_set)
    anls_ret.xpu_not_ready = len(anls_ret.not_ready_set)
    anls_ret.xpu_not_impl = len(anls_ret.not_impl_set)
    anls_ret.xpu_not_dtype = len(anls_ret.not_dtype_set)

    # This is an old version, we include xpu_not_ready(skipped) into total
    # anls_ret.total_ran = anls_ret.expected + anls_ret.fail + anls_ret.error + anls_ret.xpu_not_ready

    # This is an temp version, we exclude xpu_not_ready(skipped) out of total
    anls_ret.total_ran = anls_ret.expected + anls_ret.fail + anls_ret.error

    anls_ret.time_consume /= len(anls_list)
    if len(errMsg_set) == 1:
        anls_ret.errMsg = errMsg_set.pop()
    elif len(errMsg_set) > 1:
        anls_ret.errMsg = "Multiple error occurred. Maybe not ran to an end or has random issue. Please re-check raw logs and re-run corresponding test."
    else:
        anls_ret.errMsg = None

    return anls_ret

def analysis_final():
    global final_result, saveref, need_compare, anls_log_path

    total_time = 0.
    total_ran = 0
    total_pass = 0
    total_fail = 0
    total_error = 0
    total_not_ready = 0
    total_not_impl = 0
    total_not_dtype = 0
    total_pass_set = set()
    total_fail_set = set()
    total_error_set = set()
    total_not_impl_set = set()
    total_not_dtype_set = set()
    total_not_ready_set = set()
    for full_name, anls_list in final_result.items():
        if len(anls_list) == 1:
            final_result[full_name] = anls_list[0]
        elif len(anls_list) > 1:
            final_result[full_name] = anls_reduce(anls_list)
        else:
            raise "{} has 0 logs related, which is invalid".format(full_name)
        anls_log = final_result[full_name]
        
        if anls_log.errMsg is None:
            total_time += anls_log.time_consume
            total_ran += anls_log.total_ran
            total_pass += anls_log.expected
            total_fail += anls_log.fail
            total_error += anls_log.error
            total_not_ready += anls_log.xpu_not_ready
            total_not_impl += anls_log.xpu_not_impl
            total_not_dtype += anls_log.xpu_not_dtype

            # update total set for saving or compararison
            for case in anls_log.pass_set:
                total_pass_set.add("{}::{}::{}".format(anls_log.fname, anls_log.clsname, case))
            for case in anls_log.fail_set:
                total_fail_set.add("{}::{}::{}".format(anls_log.fname, anls_log.clsname, case))
            for case in anls_log.error_set:
                total_error_set.add("{}::{}::{}".format(anls_log.fname, anls_log.clsname, case))
            for case in anls_log.not_impl_set:
                total_not_impl_set.add("{}::{}::{}".format(anls_log.fname, anls_log.clsname, case))
            for case in anls_log.not_dtype_set:
                total_not_dtype_set.add("{}::{}::{}".format(anls_log.fname, anls_log.clsname, case))
            for case in anls_log.not_ready_set:
                total_not_ready_set.add("{}::{}::{}".format(anls_log.fname, anls_log.clsname, case))

    # print sets result to file
    with open(os.path.join(anls_log_path, "pass_list.log"), "w") as ofile:
        ofile.write("\n".join(list(total_pass_set)))
    with open(os.path.join(anls_log_path, "fail_list.log"), "w") as ofile:
        ofile.write("\n".join(list(total_fail_set)))
    with open(os.path.join(anls_log_path, "error_list.log"), "w") as ofile:
        ofile.write("\n".join(list(total_error_set)))
    with open(os.path.join(anls_log_path, "not_impl_list.log"), "w") as ofile:
        ofile.write("\n".join(list(total_not_impl_set)))
    with open(os.path.join(anls_log_path, "not_dtype_list.log"), "w") as ofile:
        ofile.write("\n".join(list(total_not_dtype_set)))
    with open(os.path.join(anls_log_path, "not_ready_list.log"), "w") as ofile:
        ofile.write("\n".join(list(total_not_ready_set)))
    if saveref:
        with open(os.path.join(anls_log_path, "pass_list_ref.log"), "w") as ofile:
            ofile.write("\n".join(list(total_pass_set)))

    # print result to screen
    formatStr = "{:60}{:>16}{:>16}{:>16}{:>16}{:>16}{:>16}|{:>16}{:>16}"
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
    print(formatStr.format(
        "Total",
        format(total_time, '.3f'),
        total_ran,
        str(total_pass) + "({}%)".format(format(total_pass / total_ran * 100, '.2f')),
        str(total_fail) + "({}%)".format(format(total_fail / total_ran * 100, '.2f')),
        str(total_error) + "({}%)".format(format(total_error / total_ran * 100, '.2f')),
        # str(total_not_ready) + "({}%)".format(format(total_not_ready / total_ran * 100, '.2f')),
        str(total_not_ready),
        total_not_impl,
        total_not_dtype))
    print("-" * titlewidth)
    for full_name, anls_log in final_result.items():
        if anls_log.errMsg is None:
            if anls_log.total_ran != 0:
                print(formatStr.format(
                    full_name,
                    format(anls_log.time_consume, '.3f'),
                    anls_log.total_ran,
                    str(anls_log.expected) + "({}%)".format(format(anls_log.expected / anls_log.total_ran * 100, '.2f')),
                    str(anls_log.fail) + "({}%)".format(format(anls_log.fail / anls_log.total_ran * 100, '.2f')),
                    str(anls_log.error) + "({}%)".format(format(anls_log.error / anls_log.total_ran * 100, '.2f')),
                    # str(anls_log.xpu_not_ready) + "({}%)".format(format(anls_log.xpu_not_ready / anls_log.total_ran * 100, '.2f')),
                    str(anls_log.xpu_not_ready),
                    anls_log.xpu_not_impl,
                    anls_log.xpu_not_dtype))
            else:
                print(formatStr.format(
                    full_name,
                    format(anls_log.time_consume, '.3f'),
                    anls_log.total_ran,
                    anls_log.expected,
                    anls_log.fail,
                    anls_log.error,
                    anls_log.xpu_not_ready,
                    anls_log.xpu_not_impl,
                    anls_log.xpu_not_dtype))
        else:
            print("{:60}{}".format(full_name, anls_log.errMsg))
    print("=" * titlewidth)

    # compare to reference
    if need_compare:
        pass_ref_set = set()
        with open(os.path.join(anls_log_path, "pass_list_ref.log"), "r") as ifile:
            for line in ifile.readlines():
                pass_ref_set.add(line.strip())
        new_pass_set = total_pass_set - pass_ref_set    # new pass is not necessary for preCI
        regression_set = pass_ref_set - total_pass_set
        if len(regression_set) > 0:
            print("Regression detected: (passed in reference but not in current run)")
            for case in regression_set:
                print(case)
            print("=" * titlewidth)
            os._exit(1)
        else:
            print("Regression NOT detected")
            print("=" * titlewidth)

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
            if (fname not in test_map) or (clsname[:-3] not in test_map[fname]):
                continue    # we only analysis logs for recognized fname::clsname
            full_log_path = os.path.join(path, logfile)
            Q = Queue()
            p = Process(target=analysis, args=(fname, clsname, epoch, full_log_path, Q))
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
    global script_path, common_path, log_path, raw_log_path, anls_log_path, saveref, need_compare

    # get paths for scripts
    script_path = os.path.split(os.path.realpath(__file__))[0]
    common_path = script_path
    log_path = os.path.join(script_path, "../logs")

    # parse args
    parser = argparse.ArgumentParser(description="Auto script for analysing raw logs")
    parser.add_argument(
            '--logdir',
            metavar='logdir',
            dest='logdir',
            default=log_path,
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
            '--clean',
            action="store_true",
            help="clean analysis logs")

    args = parser.parse_args()
    raw_log_path = os.path.join(args.logdir, "raw_logs")
    anls_log_path = os.path.join(args.logdir, "anls_logs")
    saveref = args.saveref
    need_compare = args.compare
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
    test_map = load_from_json(os.path.join(common_path, "test_map.json"))

    # run analysis with multi-process
    run_analysis(test_map)

if __name__ == '__main__':
    main()
