import os
from multiprocessing import Process
import argparse
import re
import subprocess
from tool.file_utils import save_to_json, load_from_json, write_file

# global environments
timeout: int = 0
count: int = 1
script_path: str = os.path.dirname(os.path.realpath(__file__))
ALL_PATHS: dict = {
    "script_path": script_path,
    "common_path": os.path.join(script_path, "common"),
    "test_path"  : os.path.join(script_path, "test"),
    "config_path": os.path.join(script_path, "config"),
    "env_path"   : os.path.join(script_path, "env"),
    "tool_path"  : os.path.join(script_path, "tool"),
    "log_path"   : os.path.join(script_path, "log"),
}
quiet: bool = False
autoskip: bool = False
run_in_single: bool = True
ignore: bool = False


def auto_add_skip(fname, clsname, case, exitcode=0):
    reason = ""
    if exitcode == 11:
        reason = "Segmentation fault"
    elif exitcode == 124:
        reason = "extremely slow"
    else:
        reason = "Exitcode is {}".format(exitcode)
    assert case is not None
    skip_dict = load_from_json(os.path.join(ALL_PATHS['config_path'], "skip_list.json"))
    if clsname not in skip_dict:
        skip_dict[clsname] = {reason: list()}
    if reason not in skip_dict[clsname]:
        skip_dict[clsname][reason] = list()
    skip_dict[clsname][reason].append(case)
    save_to_json(skip_dict, os.path.join(ALL_PATHS['config_path'], "skip_list.json"))
    print(f"Skipped issue case: {case},"
          " and will re-run the test suite {fname}::{clsname}")


def handle_case_skip(fname, clsname, case, logfile, status):
    should_skip_case = True
    skip_case: str = None
    with open(logfile, "r") as infile:
        lines = infile.readlines()
        for i in range(len(lines)):
            if re.match(r"Ran \d+ tests in \d+\.?\d*s", lines[-1 - i], re.M):
                should_skip_case = False
                break
            else:
                skip_case = lines[-1 - i].split(' ')[0]
                if re.match(r'test_\w*', skip_case, re.M) is not None:
                    should_skip_case = True
                    auto_add_skip(fname, clsname, skip_case, -status)
                    break
                skip_case = None

    if should_skip_case is True:
        if skip_case is not None:
            print(f"{fname}::{clsname}::{skip_case} ... Case Auto Skipped.")
            return True     # marks need to re-run
        else:
            print("Exited: Failed to find case should skip,"
                  f" please check logfile {logfile} manually.")
            os._exit(1)     # marks fail and abort the process
    else:
        print("Test ran to the end but no case can be skipped."
              " Will continue to next test.")
        return False    # marks no need to re-run


def run_test(fname, clsname, case=None, epoch=1):
    global script_path, ALL_PATHS

    real_fname = os.path.splitext(fname)[0]
    if case is None or case == "":
        logfile = os.path.join(ALL_PATHS['log_path'], f"{real_fname}.{clsname}.{epoch}.log")
    else:
        logfile = os.path.join(ALL_PATHS['log_path'],
                               f"{real_fname}.{clsname}.{case}.{epoch}.log")
    logdir = os.path.dirname(logfile)
    if not os.path.exists(logdir):
         os.makedirs(logdir)
		
    full_file_path = os.path.join(ALL_PATHS['test_path'], fname)
    case_title = fname + "::" + clsname
    if case is not None:
        case_title = case_title + "::" + case

    status = 0
    print("{} ... Start.".format(case_title))
    cmd = ["python3", full_file_path, "-v", clsname]
    if case is not None and case != "":
        cmd.append("-k")
        cmd.append(case)
    retmsg = ""
    errmsg = ""
    try:
        proc = None
        if timeout > 0:
            proc = subprocess.run(cmd, capture_output=True, check=True,
                                  cwd=script_path, env=os.environ, timeout=timeout)
        else:
            proc = subprocess.run(cmd, capture_output=True, check=True,
                                  cwd=script_path, env=os.environ)
        status = proc.returncode
        outmsg = proc.stdout.decode('utf-8')
        errmsg = proc.stderr.decode('utf-8')
        # proc.check_returncode()
    except subprocess.TimeoutExpired as e_time:
        # this is the shell timeout's return code, python don't have one
        status = -124
        outmsg = e_time.stderr.decode('utf-8')
        errmsg = f"Raised TimeoutExpired: timeout at {e_time.timeout}s."
    except subprocess.CalledProcessError as e_called:
        status = e_called.returncode
        outmsg = e_called.stderr.decode('utf-8')
        errmsg = "Raised CalledProcessError: return code {}.".format(status)
    finally:
        write_file(logfile, outmsg + errmsg)

    if not quiet:
        subprocess.run("cat {}".format(logfile), shell=True)
        print()

    print(f"{case_title} ... Finish (status {status}),"
          f" see results in logfile: {logfile}")

    if status == 0:
        return status, ""
    elif autoskip and handle_case_skip(fname, clsname, case, logfile, status):
        print("{} ... ReRun.".format(case_title))
        run_test(fname, clsname, case, epoch)
    elif autoskip is False and ignore is False:
        os._exit(-status)

    return status, retmsg


def run_tests(test_map):
    global script_path, ALL_PATHS

    process_list = []
    for epoch in range(count):
        print("Epoch #{}:".format(epoch+1))
        for fname in test_map.keys():
            if fname == "_comments":
                continue
            full_file_path = os.path.join(ALL_PATHS['test_path'], fname)
            if not os.path.exists(full_file_path):
                print(f"Skipped: test file {full_file_path} not exist.")
                continue
            if isinstance(test_map[fname], list):
                for clsname in test_map[fname]:
                    matchLine = re.match("#(.*)/(.*)", clsname, re.M | re.I)
                    if matchLine is not None:
                        clsname = matchLine.group(1) + "XPU"
                        reason = matchLine.group(2)
                        print(f"Skipped: won't run {fname}::{clsname},"
                              f" reason=\"{reason}\"")
                    elif not run_in_single:
                        p = Process(target=run_test,
                                    args=(fname,
                                          clsname + "XPU",
                                          None,
                                          epoch + 1))
                        p.deamon = True
                        p.start()
                        process_list.append(p)
                    else:   # run in single process
                        status, errmsg = run_test(
                            fname, clsname + "XPU", None, epoch + 1)

        if not run_in_single:
            for p in process_list:
                p.join()

    print("Full run finished. See log files in {}".format(ALL_PATHS['log_path']))


def main():
    global script_path, ALL_PATHS
    global timeout, count, autoskip, quiet, run_in_single, ignore

    # parse args
    parser = argparse.ArgumentParser(
        description="Main script to run all or specific tests")
    parser.add_argument(
        '--logdir',
        metavar='LOG-DIR',
        dest='logdir',
        default=ALL_PATHS['log_path'],
        type=str,
        help="the path of logfile to store, it should be a directory")
    parser.add_argument(
        '--spec',
        metavar='CASE',
        dest='spectest',
        default="",
        type=str,
        help="the full name of a specific test case."
        " It should be in format: 'filename::classname::casename'."
        " In some cases, the case name is optional like 'filename::classname'"
        " and all cases in the test class will be triggered.")
    parser.add_argument(
        '-c', '--count',
        metavar='COUNT',
        dest='times',
        default=1,
        type=int,
        help="loop times of each test class."
        " Each round of run will be logged into different log files.")
    parser.add_argument(
        '-t', '--timeout',
        metavar='TIMEOUT',
        dest='timelimit',
        default=0,
        type=int,
        help="time limit for each test class in seconds."
        " A zero stand for non-limitation.")
    parser.add_argument(
        '--parallel',
        action="store_false",
        dest="single",
        help="run whole test in single process if set")
    parser.add_argument(
        '--autoskip',
        action="store_true",
        dest='autoskip',
        help="auto skip core dumped cases and hang cases"
        " a re-run corresponding test class if set")
    parser.add_argument(
        '-q', '--quiet',
        action="store_true",
        dest='quiet',
        help="don't print out detailed results to screen if set")
    parser.add_argument(
        '--ignore',
        action="store_true",
        dest='ignore',
        help="ignore common failures and errors,"
        " and continue to run next test class if set")
    parser.add_argument(
        '--clean',
        action="store_true",
        help="clean raw logs")

    args = parser.parse_args()
    ALL_PATHS['log_path']= os.path.join(args.logdir, "raw_logs")
    spec_test = args.spectest
    need_clean = args.clean
    run_in_single = args.single
    timeout = args.timelimit
    count = args.times
    autoskip = args.autoskip
    quiet = args.quiet
    ignore = args.ignore

    # clean log directory if `--clean` flag is set
    if need_clean is True and os.path.exists(ALL_PATHS['log_path']):
        for path, dir_list, file_list in os.walk(ALL_PATHS['log_path']):
            for filename in file_list:
                if os.path.splitext(filename)[-1] == ".log":
                    full_path = os.path.join(path, filename)
                    os.remove(full_path)
                    print("Removed old logfile: {}".format(full_path))
        os._exit(0)
    elif need_clean is True:     # log_path not exist
        print("Task skipped: clean path '{}' not exist".format(ALL_PATHS['log_path']))
        os._exit(0)

    if not os.path.exists(ALL_PATHS['log_path']):
        os.makedirs(ALL_PATHS['log_path'])
        print("Created log directory: {}".format(ALL_PATHS['log_path']))

    # set local environment
    if 'PYTHONPATH' in os.environ.keys():
        os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ":" + ":".join(ALL_PATHS.values())
    else:
        os.environ['PYTHONPATH'] = ":".join(ALL_PATHS.values())
    os.environ['TORCH_TEST_DEVICES'] = os.path.join(ALL_PATHS['common_path'], 'pytorch_test_base.py')
    os.environ['PYTORCH_TEST_WITH_SLOW'] = "1"
    # os.environ['CPU_NUM_DEVICES'] = "4"

    # run specific case if `--spec` flag is set
    if spec_test != "":
        arglist = spec_test.split("::")
        filename = arglist[0]
        classname = arglist[1]
        casename = arglist[2] if len(arglist) == 3 else None
        run_test(fname=filename, clsname=classname, case=casename)
        os._exit(0)

    # get the test map
    test_map = load_from_json(os.path.join(ALL_PATHS['config_path'], "test_map.json"))

    # run tests with multi-process
    run_tests(test_map)


if __name__ == '__main__':
    main()
