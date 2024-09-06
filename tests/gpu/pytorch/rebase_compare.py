import json
import requests
import os
from numpy import array
import sys
import argparse
from pathlib import Path
import shutil
from tool.file_utils import load_from_yaml
from tool.collector import collect_cases_from_logfile, collect_detailed_issues
#from tool.skipper import add_dynamic_skipped_cases
from tool.maintainer import update_reference, check_reference, check_ci_pass
from tool.reporter import report_configurations, report_details, report_diffs, report_summary, report_ci_failure

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=str, help="specify the path of raw logs folder of ref")
    parser.add_argument("--ref_yaml", type=str, help="specify the path of yaml format of ref")
    parser.add_argument("--cur", type=str, help="specify the path of cur")
    return parser.parse_args()

args = parse_arguments()

if args.ref:
    finished_tests_ref = os.listdir(args.ref)
else:
    finished_tests_ref = []
finished_tests_cur = os.listdir(args.cur)
total_results_ref = {}
total_results_cur = {}
checked_cases = {}
#details_msg_list = []
details_total = {}
casenames_total = {}
if os.path.exists("details"):
    shutil.rmtree("details")

if args.ref_yaml:
    total_results_ref = load_from_yaml(args.ref_yaml)
else:
    for test in finished_tests_ref:
        testcase = test.split(".")[0].replace("-", "/")
        print(f"Loading {testcase} result from reference log file: {test}")
        duration, cases_result = collect_cases_from_logfile(testcase, os.path.join(args.ref, test))
        if testcase not in total_results_ref:
            total_results_ref[testcase] = []
        total_results_ref[testcase].append((duration, cases_result))

    update_reference(total_results_ref, "maintain")

for test in finished_tests_cur:
    testcase = test.split(".")[0].replace("-", "/")
    print(f"Loading {testcase} result from rebase log file: {test}")
    duration, cases_result = collect_cases_from_logfile(testcase, os.path.join(args.cur, test))
    if testcase not in total_results_cur:
        total_results_cur[testcase] = []
    total_results_cur[testcase].append((duration, cases_result))

    #print(cases_result)

    for tag, cases in check_reference(cases_result).items():
        if tag not in checked_cases:
            checked_cases[tag] = []
        checked_cases[tag].extend(cases)
        print(tag)
        #print(cases)
        print("----------------------------------")
        if "=> PASSED" not in tag and tag != "New PASSED" and tag != "New SKIPPED" and "=> XFAIL" not in tag:
            #details_msg_list.extend(collect_detailed_issues(cases, os.path.join(args.cur, test), True))
            subdetails, subcasenames = collect_detailed_issues(cases, os.path.join(args.cur, test), False, True)
            #print(subdetails)
            #print(subcasenames)
            #print("+++++++++++++++++++++++++++++++++")
            #if "New ERROR" in tag or "=> ERROR" in tag:
            if "PASSED => ERROR" in tag or "XFAIL => ERROR" in tag:
                tagy = "Error"
            elif "New ERROR" in tag or "New XPASS" in tag:
                tagy = "Errornew"
            elif "New FAILED" in tag:
                tagy = "FAILEDnew"
            elif "PASSED => FAILED" in tag or "XFAIL => FAILED" in tag:
                tagy = "Failed"
            elif "=> ERROR" in tag:
                tagy = "error_type_change"
            elif "=> FAILED" in tag:
                tagy = "fail_type_change"
            else:
                tagy = "exception"
            for tag1, detail in subdetails.items():
                tagx = tag1 + "_" + tagy
                if tagx not in details_total:
                    details_total[tagx] = []
                details_total[tagx].extend(detail)
            for tag1, subcase in subcasenames.items():
                #print(subcase)
                #print(tag1)
                tagx = tag1 + "_" + tagy
                if tagx not in casenames_total:
                    casenames_total[tagx] = []
                casenames_total[tagx].extend(subcase)



header = "========================= COMPARE DONE OF PORTED UT ==========================\n"
tail =   "==================================== SUMMARY FINISH ====================================\n"
report_summary(total_results_cur, header, tail, "cur.compare.summary")
if args.ref:
    report_summary(total_results_ref, header, tail, "ref.compare.summary")
header = "============================ STATUS CHANGED CASES IN WEEKLY ============================\n"
tail =   "================================== DIFFERENCE FINISH ===================================\n"
report_diffs(checked_cases, header, tail, "compare.diffs")
header = "=========================== DETAILED FAILED CASES IN WEEKLY ============================\n"
tail =   "=================================== DETAILS FINISH =====================================\n"
#report_details(details_msg_list, header, tail, "compare.details")
for tag, detail in details_total.items():
    header = f'=========================== DETAILED FAILED on {tag} CASES IN WEEKLY ============================\n'
    tail =   "=================================== DETAILS FINISH =====================================\n"
    report_details(detail, header, tail, "details/" + tag + ".log")
    #print(detail)
msg = ''

for tag, cases in casenames_total.items():
    num = len(cases)
    msg += f'=========================== FAILED on {tag} CASES total {num} ============================\n'
    for case in cases:
        case = "  - " + case
        msg += case
report_details(msg, header, tail, "casekind.log")

from openpyxl import Workbook
wb = Workbook()
sheet = wb.active

sheet.append(("casename", "classname", "subcasename", "wholecasename", "kind", "totalnum", "details"))
i = 0
tot = 0
for tag, cases in casenames_total.items():
    num=len(cases)
    tot = tot + num
    for eachcase in cases:
        eachcase = eachcase.rstrip('\n')
        casename = eachcase.split("::")[0]
        clsname = eachcase.split("::")[1]
        subname = eachcase.split("::")[2]
        mtag = 0
        for detail in details_total[tag]:
            detail = detail.replace("=", "_", 2)
            if clsname in detail and subname in detail:
                i = i + 1
                mtag = 1
                #print(f'{casename}, {clsname}, {subname}, {eachcase}, {tag}, {num}, {detail}')
                sheet.append((casename, clsname, subname, eachcase, tag, num, detail))
                break
        if mtag == 0:
            print(f'{casename}, {clsname}, {subname}, {eachcase}, {tag}, {num}')
            print(f'{details_total[tag]}')




print(f'tot = {tot}, i={i}')
wb.save("ptcase.xlsx")

