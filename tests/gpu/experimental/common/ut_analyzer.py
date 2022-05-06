import argparse
import re
import os
import sys

def _findall_and_append(src, tgt, key, pattern, flags):
    tgt[key] += len(re.findall(pattern, src, flags))

def summary(logfile, basefile):
    matchFlags = re.M | re.I
    totalDict = {
        'tests': 0,
        'seconds': 0.,
        'failures': 0,
        'errors': 0,
        'skipped': 0,
        'not_implemented': 0,
        'miss_dtype': 0,
        'mismatch_dtype': 0,
        'internal_assert_failed': 0,
        'assertion_errors': 0,
        'runtime_errors': 0,
        'too_slow': 0,
        'core_dumped': 0,
        'free_issue': 0,
        'floating_issue': 0,
        'munmap_issue': 0,
        'segment_fault': 0,
    }

    with open(logfile) as f:
        for line in f.readlines():
            ranResultLine = re.search(r'Ran (\d+) tests in (\d+\.?\d*)s', line, matchFlags)
            failuresLine = re.search(r'failures=(\d+)', line, matchFlags)
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

            _findall_and_append(line, totalDict, 'not_implemented', r'skipped "not implemented:', matchFlags)
            _findall_and_append(line, totalDict, 'assertion_errors', r'AssertionError:', matchFlags)
            _findall_and_append(line, totalDict, 'runtime_errors', r'RuntimeError', matchFlags)
            _findall_and_append(line, totalDict, 'miss_dtype', r'RuntimeError: .* not implemented for .*', matchFlags)
            _findall_and_append(line, totalDict, 'mismatch_dtype', r'RuntimeError: Found dtype .* but expected .*', matchFlags)
            _findall_and_append(line, totalDict, 'internal_assert_failed', r'INTERNAL ASSERT FAILED', matchFlags)

    with open(basefile) as bf:
        for line in bf.readlines():
            if len(line.lstrip()) == 0 or line.lstrip()[0] == '#':
                continue
            _findall_and_append(line, totalDict, 'too_slow', r'# too slow', matchFlags)
            _findall_and_append(line, totalDict, 'core_dumped', r'# core dumped', matchFlags)
            _findall_and_append(line, totalDict, 'free_issue', r'\.\.\. free', matchFlags)
            _findall_and_append(line, totalDict, 'munmap_issue', r'\.\.\. munmap_chunk', matchFlags)
            _findall_and_append(line, totalDict, 'floating_issue', r'\.\.\. Floating', matchFlags)
            _findall_and_append(line, totalDict, 'segment_fault', r'\.\.\. Segmentation fault', matchFlags)

    # print the results
    print('=================== Total Results =====================')
    print("-- total tests ran: ", totalDict['tests'], ' in ', totalDict['seconds'], 's')
    passed = totalDict['tests'] - totalDict['failures'] - totalDict['errors'] - totalDict['skipped']
    print("-- passed: ", passed)
    print("-- failures:", totalDict['failures'])
    print("-- errors: ", totalDict['errors'])
    print("-- skipped: ", totalDict['skipped'])
    print()
    print("** pass rate: ", passed / totalDict['tests'] * 100, "%")
    print("** fail rate: ", (totalDict['failures'] + totalDict['errors']) / totalDict['tests'] * 100, "%")
    print("** skip rate: ", totalDict['skipped'] / totalDict['tests'] * 100, "%")
    print()
    print('------------------ Detail Results ---------------------')
    print("-- not implemented issues (within skipped): ", totalDict['not_implemented'])
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

def details(logfile, basefile):
    matchFlags = re.M

    class caseEntry():
        _filePool = {
            'failures_handler': open(sys.path[0] + "/../logs/failures.log", "a"),
            'errors_dtype_handler': open(sys.path[0] + "/../logs/errors_dtype.log", "a"),
            'errors_other_handler': open(sys.path[0] + "/../logs/errors_other.log", "a"),
        }

        def __init__(self, csn, cln, fln, em=""):
            self.caseName = csn
            self.className = cln
            self.fileName = fln
            self.errorMsg = em
        
        def write(self):
            writeLine = "{}::{}::{} ... {}\n".format(self.fileName, self.className, self.caseName, self.errorMsg)

            if self.errorMsg == "":
                return

            errorType = re.match(r'\S*Error', self.errorMsg, re.M).group()
            if errorType == "AssertionError":
                self._filePool['failures_handler'].write(writeLine)
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
                    self._filePool['errors_dtype_handler'].write(writeLine)
                else:
                    self._filePool['errors_other_handler'].write(writeLine)
            else:
                self._filePool['errors_other_handler'].write(writeLine)
                

    cases: dict[(str, str), caseEntry] = {}

    with open(logfile) as f:
        curFile = ""
        curClass = ""
        curCase = ""
        for line in f.readlines():
            titleLine = re.match(r'\+ python3 \S*/(\w+\.py) -v (Test\w+)', line, matchFlags)
            if titleLine is not None:
                curFile = titleLine.group(1)
                curClass = titleLine.group(2)
            caseLine = re.match(r'test_\w*', line, matchFlags)
            if caseLine is not None:
                curCase = caseLine.group()
                cases[(curCase, curClass)] = caseEntry(curCase, curClass, curFile)
            # rematch caseLine for details
            caseLine = re.match(r'[^ ]*: (test_\w*) .*\.(Test\w*)', line, matchFlags)
            if caseLine is not None:
                curCase = caseLine.group(1)
                curClass = caseLine.group(2)
            msgLine = re.match(r'(\S*Error):.*', line, matchFlags)
            if msgLine is not None and (curCase, curClass) in cases:
                cases[(curCase, curClass)].errorMsg = msgLine.group()

    for case in cases.values():
        case.write()

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="Analysis the UT report automatically")
    parser.add_argument('--logfile', metavar='filename', dest='filename', default='/tmp/ipex_pytorch_ut_test.log', type=str, help="the path of logfile to analysis")
    parser.add_argument('--clean', action="store_true", help="clean analysis logs")

    args = parser.parse_args()

    script_path = sys.path[0]
    if args.clean is True:
        os.remove(script_path + '/../logs/failures.log')
        os.remove(script_path + '/../logs/errors_dtype.log')
        os.remove(script_path + '/../logs/errors_other.log')
    else:
        # analysis
        summary(args.filename, script_path + "/./pytorch_test_base.py")
        details(args.filename, script_path + "/./pytorch_test_base.py")
