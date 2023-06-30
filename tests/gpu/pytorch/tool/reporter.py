from .file_utils import write_file

class ReportInfo():
    def __init__(self, passed=0, failed=0, error=0, skipped=0, xfail=0, xpass=0, nores=0, duration=0.):
        super(ReportInfo, self).__init__()
        self._passed: int = passed
        self._failed: int = failed
        self._error: int = error
        self._skipped: int = skipped
        self._xfail: int = xfail
        self._xpass: int = xpass
        self._nores: int = nores
        self._duration: float = duration
        self._count: int = 1
    
    def add(self, other):
        self._passed = other._passed
        self._failed = other._failed
        self._error = other._error
        self._skipped = other._skipped
        self._xfail = other._xfail
        self._xpass = other._xpass
        self._nores = other._nores
        self._duration = other._duration
        self._count = other._count
    
    def get_avg_passed(self):
        return (self._passed + self._xfail) // self._count

    def get_avg_failed(self):
        return (self._failed + self._xpass) // self._count

    def get_avg_error(self):
        return (self._error + self._nores) // self._count

    def get_avg_skipped(self):
        return self._skipped // self._count

    def get_avg_time_cost(self):
        return self._duration / self._count
    
    def get_avg_total_without_skipped(self):
        return (self._passed + self._failed + self._error + self._xfail + self._xpass + self._nores) // self._count

def report_summary(total_cases, header, tail, file):
    def _reduce_info(collected_cases):
        res = ReportInfo()
        for duration, case_info in collected_cases:
            res.add(ReportInfo(len(case_info["PASSED"]),
                               len(case_info["FAILED"]),
                               len(case_info["ERROR"]),
                               len(case_info["SKIPPED"]),
                               len(case_info["XFAIL"]),
                               len(case_info["XPASS"]),
                               len(case_info["NO_RESULT"]),
                               duration))
        return res

    outmsg = header
    reduced_results = {}
    for test_name in total_cases.keys():
        reduced_results[test_name] = _reduce_info(total_cases[test_name])
    total_result = ReportInfo(
        sum([info._passed   // info._count for info in reduced_results.values()]),
        sum([info._failed   // info._count for info in reduced_results.values()]),
        sum([info._error    // info._count for info in reduced_results.values()]),
        sum([info._skipped  // info._count for info in reduced_results.values()]),
        sum([info._xfail    // info._count for info in reduced_results.values()]),
        sum([info._xpass    // info._count for info in reduced_results.values()]),
        sum([info._nores    // info._count for info in reduced_results.values()]),
        sum([info._duration /  info._count for info in reduced_results.values()]),
        )
    
    width_list = []
    width_list.append(max([len(test_name) for test_name in total_cases.keys()]) + 2)  # Test Name
    width_list.append(10)   # Time cost
    width_list.append(4)    # '|'
    width_list.append(10)   # Total Ran
    width_list.append(4)    # '|'
    width_list.append(16)   # Passed(rate)
    width_list.append(16)   # Failed(rate)
    width_list.append(16)   # Error(rate)
    width_list.append(4)    # '|'
    width_list.append(10)   # Skipped

    def _format_string(fields):
        return "".join([f"{fields[i]:<{width_list[i]}}" for i in range(len(width_list))])

    def _format_line(test_name, info):
        total_ran = info.get_avg_total_without_skipped()
        if total_ran == 0:
            return _format_string([
                test_name, 
                f"{info.get_avg_time_cost():.2f}s", "|",
                f"{total_ran}", "|",
                f"{info.get_avg_passed()}",
                f"{info.get_avg_failed()}",
                f"{info.get_avg_error()}", "|",
                f"{info.get_avg_skipped()}"])
        else:
            return _format_string([
                test_name, 
                f"{info.get_avg_time_cost():.2f}s", "|",
                f"{total_ran}",  "|",
                f"{info.get_avg_passed()}({info.get_avg_passed() / total_ran:.2%})",
                f"{info.get_avg_failed()}({info.get_avg_failed() / total_ran:.2%})",
                f"{info.get_avg_error()}({info.get_avg_error() / total_ran:.2%})", "|",
                f"{info.get_avg_skipped()}"])

    max_width = sum(width_list)
    outmsg += "=" * max_width + "\n"
    outmsg += _format_string(["Test Name", "Time Cost", "|", "Total Ran", "|", "Passed(rate)", "Failed(rate)", "Error(rate)", "|", "Skipped"])  + "\n"
    outmsg += "-" * max_width + "\n"
    outmsg += _format_line("Total", total_result) + "\n"
    outmsg += "-" * max_width + "\n"
    for test_name in sorted(total_cases.keys()):
        cur_info = reduced_results[test_name]
        outmsg += _format_line(test_name, cur_info) + "\n"
    outmsg += "=" * max_width + "\n"
    outmsg += tail
    write_file(outmsg, file, "w", end_char="\n")

def report_diffs(diffs, header, tail, file):
    outmsg = header
    def _get_sorted_key(sx):
        prior = ["PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL", "XPASS", "NO_RESULT"]
        if " => " in sx:
            return prior.index(sx.split(" => ")[0]) * 10 + prior.index(sx.split(" => ")[1])
        else:
            assert "New " in sx, f"Invalid Diff Tag: {sx}"
            return prior.index(sx[len("New "):]) * 100

    sorted_keys = sorted(diffs.keys(), key=lambda sx: _get_sorted_key(sx))
    for key in sorted_keys:
        outmsg += "_________________  Diffs: " + key + "______________________\n"
        for casename in diffs[key]:
            outmsg += casename + "\n"
    outmsg += tail
    write_file(outmsg, file, "w", end_char="\n")

def report_details(details, header, tail, file):
    outmsg = header
    for detail in details:
        outmsg += detail
    outmsg += tail
    write_file(outmsg, file, "w", endchar="\n")


def report_configurations(iter, log_path, options, timeout, cmd, file):
    outmsg = ""
    outmsg += f"================================Iteration #{iter}================================\n"
    outmsg += f"---------------------------   Configurations   -----------------------------\n"
    outmsg += f"+ Mode:\t'{options.mode}'\n"
    outmsg += f"+ Output log file: {log_path}\n"
    outmsg += f"+ Use PyTest:\t\t\t\t{options.use_pytest}\n"
    outmsg += f"+ Verbose :\t\t\t\t{'ON' if options.verbose or (options.mode[0] in ['ci', 'maintain', 'weekly']) else 'OFF'}\n"
    if options.ipex_verbose:
        outmsg += f"+ IPEX Verbose:\t\t\t\tON\n"
        outmsg += f"+---- oneDNN Verbose:\t\t\t\t{'ON' if 'onednn' in options.ipex_verbose else 'OFF'}\n"
        outmsg += f"+---- oneMKL Verbose:\t\t\t\t{'ON' if 'onemkl' in options.ipex_verbose else 'OFF'}\n"
        outmsg += f"+---- IPEX FWK Verbose:\t\t\t\t{'ON' if 'ipex' in options.ipex_verbose else 'OFF'}\n"
    else:
        outmsg += f"+ IPEX Verbose:\t\t\t\tOFF\n"
    outmsg += f"+ Timeout Configurations:\n"
    outmsg += f"+---- Check Timeout:\t\t\t{'ON' if timeout > 0 else 'OFF'}\n"
    outmsg += f"+---- Set Current Timeout:\t\t{timeout if timeout > 0 else 'INFINITE'}\n"
    outmsg += f"+---- Update Timeout After Run:\t\t{'ON' if options.update_timeout else 'OFF'}\n"
    outmsg += f"+ Implicit configurations:\n"
    outmsg += f"+---- Check Reference Passed List:\t{'YES' if options.mode[0] == 'ci' else 'NO'}\n"
    outmsg += f"+---- Update Reference Passed List:\t{'YES' if options.mode[0] == 'maintain' else 'NO'}\n"
    outmsg += f"+---- Update Skipped Cases List:\t{'YES' if options.mode[0] in ['weekly', 'maintain'] else 'NO'}\n"
    outmsg += f"-------------------------   Configurations End  ----------------------------\n"
    outmsg += f"Executing command: {cmd}\n"
    write_file(outmsg, file, "a", end_char="\n")
