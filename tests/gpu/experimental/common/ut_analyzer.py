import argparse
import re
import os
import sys


def _findall_and_append(src, tgt, key, pattern, flags):
    tgt[key] += len(re.findall(pattern, src, flags))


def summary(logfile, basefile):
    matchFlags = re.M | re.I
    totalDict = {
        # for summary
        'tests': 0,
        'seconds': 0.,
        'failures': 0,
        'errors': 0,
        'skipped': 0,
        # for skipped
        'dtype_not_support': 0,
        'not_implemented': 0,
        'only_runs_on': 0,
        'should_skip': 0,
        'not_ready_on_xpu': 0,
        'insufficient_memory': 0,
        'pytorch_issue': 0,
        # for assertion errors
        'assertion_errors': 0,
        # for runtime errors
        'runtime_errors': 0,
        'miss_dtype': 0,
        'mismatch_dtype': 0,
        'internal_assert_failed': 0,
        # foe detailed 'hang' and 'core dumped' issue
        'too_slow': 0,
        'core_dumped': 0,
        'free_issue': 0,
        'floating_issue': 0,
        'munmap_issue': 0,
        'segment_fault': 0,
    }

    with open(logfile) as f:
        for line in f.readlines():
            # the result line for each file is like:
            # --------------------------------------------------
            # Ran 362 tests in 700.043s
            #
            # FAILED (failures=59, errors=61, skipped=129, expected failures=1)
            #
            # We will catch these lines for the summary
            ranResultLine = re.search(r'Ran (\d+) tests in (\d+\.?\d*)s', line, matchFlags)
            failuresLine = re.search(r'\(failures=(\d+)', line, matchFlags)
            errorsLine = re.search(r'errors=(\d+)', line, matchFlags)
            skippedLine = re.search(r'skipped=(\d+)', line, matchFlags)

            if ranResultLine is not None:
                totalDict['tests'] += int(ranResultLine.group(1))
                totalDict['seconds'] += float(ranResultLine.group(2))

            if failuresLine is not None:
                totalDict['failures'] += int(failuresLine.group(1))

            if errorsLine is not None:
                totalDict['errors'] += int(errorsLine.group(1))

            if skippedLine is not None:
                totalDict['skipped'] += int(skippedLine.group(1))

            # analyze the skipped tests
            # we classfied skipped tests into 8 classes (include those 'hang' and 'core dumped' issues)
            # 1. 'XPU does not support dtype xxx' -> these will be fixed at Q3 and not included into total summary
            # 2. 'not_implemented: ...'
            #    "not implemented: ..." -> these will be fixed at Q3 and not included into total summary
            # 3. 'Only runs on cpu/cuda' -> these were cpu/cuda spec cases and not included into total summary
            # 4. "Skipped! xxxxx" -> these were cases what pytorch defined to skip. Lots of them are related to the ops who against the case template and should not be tested. They should not be included into total summary
            # 5. 'not ready on XPU' -> these were cases which were skipped manually by IPEX developer. We will keep these cases into sammary and try to fix all of them
            # 6. 'Insufficient xpu:0 memory' -> these cases related to large tensor test and have no environment for run yet. We will keep these cases into sammary and try to run all of them in future
            # 7. 'See https://github.com/pytorch/...' -> these cases raised issues on PyTorch and will skip by PyTorch. We will not include them into total summary
            # 8. others -> no clear reason of skipping and will be kept in total summary until we slassify them
            _findall_and_append(line, totalDict, 'dtype_not_support', r'skipped .XPU does not support dtype', matchFlags)
            _findall_and_append(line, totalDict, 'not_implemented', r'skipped .not.implemented:', matchFlags)
            _findall_and_append(line, totalDict, 'only_runs_on', r'skipped .Only runs on', matchFlags)
            _findall_and_append(line, totalDict, 'should_skip', r'skipped .Skipped!', matchFlags)
            _findall_and_append(line, totalDict, 'not_ready_on_xpu', r'skipped .not ready on XPU', matchFlags)
            _findall_and_append(line, totalDict, 'insufficient_memory', r'skipped .Insufficient', matchFlags)
            _findall_and_append(line, totalDict, 'pytorch_issue', r'skipped .See ', matchFlags)

            # analyze the assertion error tests
            _findall_and_append(line, totalDict, 'assertion_errors', r'AssertionError:', matchFlags)

            # analyze the runtime error tests
            # These cases were classified into 4 classes
            # 1. RuntimeError: "angle" not implemented for 'BFloat16' -> some ops missed dtype support
            # 2. RuntimeError: Found dtype Float but expected BFloat16 -> some ops got different dtype from expected while running
            # 3. RuntimeError: <op_name>INTERNAL ASSERT FAILED at "xxx" -> cases which triggerred PyTorch internal assert
            # 4. others -> including a lot of different cases of RuntimeError, such as what raised by oneDNN, etc.
            _findall_and_append(line, totalDict, 'runtime_errors', r'RuntimeError', matchFlags)
            _findall_and_append(line, totalDict, 'miss_dtype', r'RuntimeError: .* not implemented for .*', matchFlags)
            _findall_and_append(line, totalDict, 'mismatch_dtype', r'RuntimeError: Found dtype .* but expected .*', matchFlags)
            _findall_and_append(line, totalDict, 'internal_assert_failed', r'INTERNAL ASSERT FAILED', matchFlags)

    # analyze what we skip manually due to 'hang' or 'core dumped'
    # these skipped cases were recorded in common/pytorch_test_base.py
    with open(basefile) as bf:
        for line in bf.readlines():
            # do not need to handle line which was commented out
            if len(line.lstrip()) == 0 or line.lstrip()[0] == '#':
                continue
            # analyze 'hang' issue
            _findall_and_append(line, totalDict, 'too_slow', r'# too slow', matchFlags)

            # analyze 'core dumped' issue
            # 'core dumped' issue were classified into 5 types, they are:
            # 1. free(): invalid size
            # 2. munmap_chunk(): invalid pointer
            # 3. Floating point exception
            # 4. Segmentation fault
            # 5. others (not classified)
            _findall_and_append(line, totalDict, 'core_dumped', r'# core dumped', matchFlags)
            _findall_and_append(line, totalDict, 'free_issue', r'\.\.\. free', matchFlags)
            _findall_and_append(line, totalDict, 'munmap_issue', r'\.\.\. munmap_chunk', matchFlags)
            _findall_and_append(line, totalDict, 'floating_issue', r'\.\.\. Floating', matchFlags)
            _findall_and_append(line, totalDict, 'segment_fault', r'\.\.\. Segmentation fault', matchFlags)

    # print the results
    print('=================== Total Results =====================')
    # here calcualte those skipped should be deleted from total summary
    need_delete = totalDict['dtype_not_support'] + totalDict['not_implemented'] + totalDict['only_runs_on'] + totalDict['should_skip'] + totalDict['pytorch_issue']
    # here update valid total tests and valid skipped tests
    valid_total = totalDict['tests'] - need_delete
    valid_skip = totalDict['skipped'] - need_delete
    passed = totalDict['tests'] - totalDict['failures'] - totalDict['errors'] - totalDict['skipped']
    print("-- total tests ran: ", valid_total, ' in ', totalDict['seconds'], 's')
    print("-- passed: ", passed)
    print("-- failures:", totalDict['failures'])
    print("-- errors: ", totalDict['errors'])
    print("-- skipped: ", valid_skip)
    print()
    print("** pass rate: ", passed / valid_total * 100, "%")
    print("** fail rate: ", (totalDict['failures'] + totalDict['errors']) / valid_total * 100, "%")
    print("** skip rate: ", valid_skip / valid_total * 100, "%")
    print()
    print('------------------ Detail Results ---------------------')
    print("-- core dumped issues (within skipped): ", totalDict['core_dumped'])
    print("\t-- free() issues: ", totalDict['free_issue'])
    print("\t-- munmap_chunk() issues: ", totalDict['munmap_issue'])
    print("\t-- Floating point exception issues: ", totalDict['floating_issue'])
    print("\t-- Segmentation fault issues: ", totalDict['segment_fault'])
    print("\t-- others: ", totalDict['core_dumped'] - totalDict['free_issue'] - totalDict['munmap_issue'] - totalDict['floating_issue'] - totalDict['segment_fault'])
    print("-- hang issues: ", totalDict['too_slow'])
    print("-- AssertionErrors (within failures): ", totalDict['assertion_errors'])
    print("-- RuntimeErrors (within errors): ", totalDict['runtime_errors'])
    print("\t-- miss required dtype issues: ", totalDict['miss_dtype'])
    print("\t-- got unexpected dtype issues: ", totalDict['mismatch_dtype'])
    print("\t-- INTERNAL ASSERT FAILED issues: ", totalDict['internal_assert_failed'])
    print("\t-- others: ", totalDict['runtime_errors'] - totalDict['miss_dtype'] - totalDict['mismatch_dtype'] - totalDict['internal_assert_failed'])
    print('--------------  Not Implemented List ------------------')
    print("-- Op not implemented: ", totalDict['not_implemented'])
    print("-- Dtype not support: ", totalDict['dtype_not_support'])


def details(logfile, basefile):
    matchFlags = re.M
    
    _filePool = {
        'failures_handler': open(sys.path[0] + "/../logs/failures.log", "a"),
        'errors_dtype_handler': open(sys.path[0] + "/../logs/errors_dtype.log", "a"),
        'errors_other_handler': open(sys.path[0] + "/../logs/errors_other.log", "a"),
        'not_support_handler': open(sys.path[0] + "/../logs/skipped_not_support_dtype.log", "a"),
        'not_impl_handler': open(sys.path[0] + "/../logs/skipped_not_impl.log", "a"),
        'pass_handler': open(sys.path[0] + "/../logs/pass_list.log", "a"),
    }

    class caseEntry():
        def __init__(self, csn, cln, fln, stm="", em="", sm=""):
            self.caseName = csn
            self.className = cln
            self.fileName = fln
            self.statusMsg = stm
            self.skipMsg = sm
            self.errorMsg = em

        def _write(self, handler, writeLine):
            _filePool[handler].write(writeLine + "\n")

        def write(self):
            writeLine = "{}::{}::{}".format(self.fileName, self.className, self.caseName)
            if self.statusMsg == "ok":
                self._write('pass_handler', writeLine)
            elif self.statusMsg == "skipped":
                if re.match(r'\S*not.impl', self.skipMsg, re.M | re.I):
                    self._write('not_impl_handler', writeLine + " ... " + self.skipMsg)
                elif re.match(r'\S*XPU does not support', self.skipMsg, re.M | re.I):
                    self._write('not_support_handler', writeLine + " ... " + self.skipMsg)
                else:
                    # if need, add more handler here
                    pass
            elif self.statusMsg == "FAIL or ERROR":
                errorType = re.match(r'\S*Error', self.errorMsg, re.M).group()
                if errorType == "AssertionError":
                    self._write('failures_handler', writeLine + " ... " + self.errorMsg)
                elif errorType == "RuntimeError":
                    errorDetail = re.search(r'RuntimeError: .* not implemented for .*', self.errorMsg, re.M)
                    classified = False
                    if errorDetail is not None:
                        classified = True
                    errorDetail = re.search(r'RuntimeError: Found dtype .* but expected .*', self.errorMsg, re.M)
                    if errorDetail is not None:
                        classified = True
                    errorDetail = re.search(r'INTERNAL ASSERT FAILED', self.errorMsg, re.M)
                    if errorDetail is not None:
                        classified = True
                    if classified is True:
                        self._write('errors_dtype_handler', writeLine + " ... " + self.errorMsg)
                    else:
                        self._write('errors_other_handler', writeLine + " ... " + self.errorMsg)
                else:
                    self._write('errors_other_handler', writeLine + " ... " + self.errorMsg)

    cases: dict[(str, str), caseEntry] = {}

    with open(logfile) as f:
        curFile = ""
        curClass = ""
        curCase = ""
        for line in f.readlines():
            titleLine = re.match(r'\+ timeout \d* python3 \S*/(\w+\.py) -v (Test\w+)', line, matchFlags)
            if titleLine is not None:
                curFile = titleLine.group(1)
                curClass = titleLine.group(2)
            caseLine = re.match(r'test_\w*', line, matchFlags)
            if caseLine is not None:
                curCase = caseLine.group()
                cases[(curCase, curClass)] = caseEntry(curCase, curClass, curFile)

            # try to find "test_xxx (__main__.TestXXXXPU) ... ok"
            okLine = re.match(r'.* \.\.\. ok', line, matchFlags)
            if okLine is not None and (curCase, curClass) in cases:
                cases[(curCase, curClass)].statusMsg = "ok"
            # try to find single line of "ok"
            okLine = re.match(r'^ok$', line, matchFlags)
            if okLine is not None and (curCase, curClass) in cases:
                cases[(curCase, curClass)].statusMsg = "ok"
            # try to find "expected failure"
            okLine = re.match(r'.* \.\.\. expected failure', line, matchFlags)
            if okLine is not None and (curCase, curClass) in cases:
                cases[(curCase, curClass)].statusMsg = "ok"

            # try to find "test_xxx (__main__.TestXXXXPU) ... skipped 'reasons...'"
            skipLine = re.match(r'.* \.\.\. skipped (.*)', line, matchFlags)
            if skipLine is not None and (curCase, curClass) in cases:
                skipMsg = skipLine.group(1)
                cases[(curCase, curClass)].statusMsg = "skipped"
                cases[(curCase, curClass)].skipMsg = skipMsg
            # try to find single line of "skipped 'reasons...'"
            skipLine = re.match(r'^skipped (.*)', line, matchFlags)
            if skipLine is not None and (curCase, curClass) in cases:
                skipMsg = skipLine.group(1)
                cases[(curCase, curClass)].statusMsg = "skipped"
                cases[(curCase, curClass)].skipMsg = skipMsg

            # try to find detailed FAIL or ERROR line beneath "============="
            detailedCaseLine = re.match(r'[^ ]*: (test_\w*) .*\.(Test\w*)', line, matchFlags)
            if detailedCaseLine is not None:
                curCase = detailedCaseLine.group(1)
                curClass = detailedCaseLine.group(2)
            errorMsgLine = re.match(r'(\S*Error):.*', line, matchFlags)
            if errorMsgLine is not None and (curCase, curClass) in cases:
                cases[(curCase, curClass)].statusMsg = "FAIL or ERROR"
                cases[(curCase, curClass)].errorMsg = errorMsgLine.group()

    for case in cases.values():
        case.write()
    for handler, f in _filePool.items():
        f.close()

def compare(curpasslog, refpasslog):
    with open(curpasslog, 'r') as cup:
        cuplist = [each.strip() for each in cup.readlines() if ((each.strip() is not None) and (each.strip() != ""))]
    with open(refpasslog, 'r') as rfp:
        rfplist = [each.strip() for each in rfp.readlines() if ((each.strip() is not None) and (each.strip() != ""))]
    newfix = list(set(cuplist) - set(rfplist))
    if len(newfix) > 0:
        print("[ INFO ] new fix found:")
        print("\n".join(newfix))
    regression = list(set(rfplist) - set(cuplist))
    if len(regression) > 0:
        print("[ Error ] new regressions found:")
        print("\n".join(regression))
        sys.exit(1)


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="Analysis the UT report automatically")
    parser.add_argument(
        '--logfile',
        metavar='filename',
        dest='filename',
        default=sys.path[0] + '/../logs/ipex_pytorch_ut_test.log',
        type=str,
        help="the path of logfile to analysis")
    parser.add_argument(
        '--clean',
        action="store_true",
        help="clean analysis logs")

    args = parser.parse_args()

    script_path = sys.path[0]
    file_list = [
        script_path + '/../logs/failures.log',
        script_path + '/../logs/errors_dtype.log',
        script_path + '/../logs/errors_other.log',
        script_path + '/../logs/skipped_not_support_dtype.log',
        script_path + '/../logs/skipped_not_impl.log',
        script_path + '/../logs/pass_list.log',
    ]
    if args.clean is True:
        for f in file_list:
            if os.path.exists(f):
                os.remove(f)
                print("Cleaned old log file: {}".format(f))
    else:
        if os.path.exists(args.filename):
            # clean
            for f in file_list:
                if os.path.exists(f):
                    os.remove(f)
                    print("Cleaned old log file: {}".format(f))
            # analysis
            summary(args.filename, script_path + "/./pytorch_test_base.py")
            details(args.filename, script_path + "/./pytorch_test_base.py")
            compare(script_path + "/../logs/pass_list.log",
                    script_path + "/../logs/pass_list_ref.log")
        else:
            print("ERROR: raw log file ({}) not exist.".format(args.filename))
