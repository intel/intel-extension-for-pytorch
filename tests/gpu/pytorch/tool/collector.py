import re
from .file_utils import read_file

re_flags = re.M | re.S

def collect_fatal_error(issued_cases, logfile):
    global re_flags
    data = read_file(logfile)
    pytorch_result = re.search("Command.+?died with.+?(?=\n)", data, re_flags)
    pytest_result = re.search("fatal python error.+?(?=\n\n)", data, re_flags | re.I)
    timeout_result = re.search("Command.+?timed out.+?(?=\n)", data, re_flags)
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
            # this code should be open when the assertion issue is fixed.
            detail_search = re.search(r'^\S+Error:.+?$', detail, re_flags)
            if detail_search:
                short_detail = detail_search.group(0)
            else:
                short_detail = "Assertion error or Runtime error with no details"
                # print(f'the detail is {detail}')
            #short_detail = re.search(r'^\S+Error:.+?$', detail, re_flags).group(0)
            short_details.append(short_detail)
        return short_details, details
    return details

def collect_pytorch_rebase_details(testname, issued_cases, data):
    #print(f'{testname} {issued_cases}')
    global re_flags
    issued_casenames = [case.split("::", 1)[-1] for case in issued_cases]
    #print(issued_cases)
    details = {}
    casenames = {}
    #pattern = r'(^=+\n[A-Za-z]+:\s*(test_\S*)\s*\([^\.]*\.(\S*)\)\n-+.*?(?====\n))'
    #pattern = r'(^=+\n[A-Za-z]+:\s*(test_\S*)\s*\([^\.]*\.(\S*)\)\n.*?-+.*?(?====))'
    pattern = r'(^=+\n[A-Za-z]+:\s*(test_\S*)\s*\([^\.]*\.([^\.\)]*[^\)]*)\)\n.*?-+.*?(?====))'
    p1 = r'AssertionError:\sThe\ssupported\sdtypes\sfor.*on\sdevice\stype\scuda\sare\sincorrect!'
    p1_1 = r'[A-Za-z0-9]*\.[[A-Za-z0-9]*\s-\s\"[A-Za-z0-9_]*\"\snot\simplemented\sfor\s\'[A-Za-z0-9]*\''
    p1_2 = r'[A-Za-z0-9]*\.[[A-Za-z0-9]*\s-\sCould\snot\srun\s\'[A-Za-z0-9_:\.]*\'\swith\sarguments\sfrom\sthe\s\'[A-Za-z]*\'\sbackend\.'
    p1_3 = r'[A-Za-z0-9]*\.[[A-Za-z0-9]*\s-\sUnsupported\sdtype\s[A-Za-z0-9_:\.]*'
    p1_4 = r'[A-Za-z0-9]*\.[[A-Za-z0-9]*\s-\sMKL\sFFT\sdoesn\'t\ssupport\stensor\sof\stype'
    p1_5 = r'[A-Za-z0-9]*\.[[A-Za-z0-9]*\s-\selement\s\d\sof\stensors\sdoes\snot\srequire\s[A-Za-z0-9_:\.]*\sand\sdoes\snot\shave\sa\s[A-Za-z0-9_:\.]*'
    p1_6 = r'[A-Za-z0-9]*\.[[A-Za-z0-9]*\s-\svalue\scannot\sbe\sconverted\sto\stype\s[A-Za-z0-9_:\.]*\swithout\soverflow'
    p1_7 = r'[A-Za-z0-9]*\.[[A-Za-z0-9]*\s-\sinput\smust\sbe\s2-d\smatrix\.'
    p1_8 = r'[A-Za-z0-9]*\.[[A-Za-z0-9]*\s-\sinconsistent\stensors\ssizes\sinput\:'
    p1_9 = r'[A-Za-z0-9]*\.[[A-Za-z0-9]*\s-\smodule\s\'[A-Za-z0-9_:\.]*\'\shas\sno\sattribute\s\'[A-Za-z0-9_:\.]*\''
    p1_10 = r'[A-Za-z0-9]*\.[[A-Za-z0-9]*\s-\s[A-Za-z0-9_:\.]*\sdoes\snot\ssupport\s[A-Za-z0-9_:\.]*\sinputs'
    p1_11 = r'functions\swith\s[A-Za-z0-9_:\.]*=\.\.\.\sarguments\sdon\'t\ssupport\sautomatic\sdifferentiation,\sbut\sone\sof\sthe\sarguments\srequires'
    p2 = r'AssertionError:\sRuntimeError\snot\sraised'
    p2_1 = r'Expected\sRuntimeError\swhen\sdoing\san\sunsafe\scast\sfrom\sa\sresult\sof\sdtype'
    p2_2 = r'Expected\sRuntimeError\swhen\sdoing\san\sunsafe\scast!'
    p2_3 = r'self\.assertFalse\(isinstance\(out,\stype\(NotImplemented\)\)\)'
    p2_4 = r'not\simplemented\sfor'
    p3 = r'NotImplementedError:\s'
    p3_1 = r'Could\snot\srun\s\'[A-Za-z0-9_:\.]*\'\swith\sarguments\sfrom\sthe\s\'[A-Za-z0-9_:\.]*\'\sbackend\.'
    p3_2 = r'Cannot\scopy\sout\sof\smeta\stensor;\sno\sdata!'
    p3_3 = r'could\snot\sfind\skernel\sfor'
    p3_4 = r'is\snot\scurrently\simplemented\sfor\sthe'
    p4 = r'AssertionError: False is not true'
    p4_1 = r'self\.assertTrue\(result\.device\s==\scuda_device\)'
    p4_2 = r'self\.assertTrue\(all\(t\.device\s==\scuda_device\sfor\st\sin\sresult\)\)'
    p4_3 = r'self\.assertTrue\(all\(p\.device\s==\sdevice_'
    p4_4 = r'self\.assertTrue\(all\(p\.device\.type\s==\sdevice_'
    p5 = r'AssertionError:\sTensor-likes\sare\snot\sclose!'
    p6 = r'RuntimeError:\s'
    p6_1 = r'RuntimeError:\sFFT_INVALID_DESCRIPTOR'
    p6_2 = r'RuntimeError:\sNative\sAPI\sfailed'
    p6_3 = r'currently\sas\soneDNN\sdon\'t\ssupport\sthis\salgorithm!'
    p6_4 = r'RuntimeError:\s1D\starget\stensor\sexpected,\smulti-target\snot\ssupported'
    p6_5 = r'RuntimeError:\sNULL\spointer\sargument\sin\smemory\scopy\soperation\.'
    p6_6 = r'RuntimeError:\s\"[A-Za-z0-9_:\.]*\"\snot\simplemented\sfor\s\'[A-Za-z0-9_:\.]*\''
    p6_7 = r'RuntimeError:\sFFT_UNIMPLEMENTED'
    p6_8 = r'RuntimeError:\scould\snot\screate\sa\sprimitive'
    p6_9 = r'RuntimeError:\s[A-Za-z0-9_:\.]*:\sinvalid\sargument:\sIllegal\svalue\ssupplied\sfor\sparameter'
    p6_10 = r'derivative\sfor\s[A-Za-z0-9_:\.]*\sis\snot\simplemented'
    p6_11 = r'element\s\d\sof\stensors\sdoes\snot\srequire\s[A-Za-z0-9_:\.]*\sand\sdoes\snot\shave\sa\s[A-Za-z0-9_:\.]*'
    p6_12 = r'RuntimeError:\s[A-Za-z0-9_:\.]*\soperator\sdoes\snot\ssupport\s[A-Za-z0-9_:\.]*'
    p6_13 = r'RuntimeError:\scould\snot\screate\sa\sdescriptor\sfor\sa'
    p6_14 = r'RuntimeError:\sWhen\scomparing\sthe\soutput'
    p6_15 = r'provided\sdtype\smust\smatch\sdtype\sof\sresult'
    p6_16 = r'RuntimeError:\sCan\sonly\scalculate\sthe\smean\sof\sfloating\stypes'
    p6_17 = r'RuntimeError:\scould\snot\sconstruct\sa\smemory\sdescriptor\susing\sa\sformat\stag'
    p6_18 = r'expected\sinput\sto\shave\snon-empty\sspatial\sdimensions,\sbut\sinput\shas\ssizes'
    p6_19 = r'RuntimeError:\sfound\snegative\sindex\s-\d+\sfor\sdim\s\d+'
    p6_20 = r'RuntimeError:\stensor\sdoes\snot\shave\sa\sdevice'
    p6_21 = r'RuntimeError:\sno\s_refs\ssupport\sfor\sNone'
    p6_22 = r'can\'t\sbe\scast\sto\sthe\sdesired\soutput\stype'
    p6_23 = r'RuntimeError:\sMKL\sFFT\sdoesn\'t\ssupport\stensor\sof\stype'
    p6_24 = r'is\snot\ssupported\sin\soneDNN!'
    p6_25 = r'RuntimeError:\sunsupported\soperation:\smore\sthan\sone\selement\sof\sthe\swritten-to\stensor\srefers\sto\sa\ssingle\smemory\slocation\.'
    p6_26 = r'RuntimeError:\sThe\ssize\sof\stensor\sa\s\(\d\)\smust\smatch\sthe\ssize\sof\stensor\sb\s\(\d\)\sat\snon-singleton\sdimension\s\d'
    p6_27 = r'RuntimeError:\sFound\sdtype\s[A-Za-z0-9_:\.]*\sbut\sexpected\s[A-Za-z0-9_:\.]*'
    p6_28 = r'RuntimeError:\sUnsupported\sdtype'
    p6_29 = r'Output size is too small'
    p6_30 = r'RuntimeError:\sbad\soptional\saccess'
    p6_31 = r'non-empty\s\dD\sor\s\dD\s\(batch\smode\)\stensor\sexpected\sfor\sinput,\sbut\sgot:'
    p6_32 = r'RuntimeError:\sExpected\sa\sproper\sTensor\sbut\sgot\sNone'
    p6_33 = r'arguments\sdon\'t\ssupport\sautomatic\sdifferentiation,\sbut\sone\sof\sthe\sarguments\srequires\sgrad'
    p6_34 = r'RuntimeError:\soutput\s\d:\smeta\sdisagrees\swith\sreal\simpl:'
    p6_35 = r'RuntimeError:\sTensor\son\sdevice\s[a-z:0-9]*\sis\snot\son\sthe\sexpected\sdevice\s[a-z:0-9]*!'
    p6_36 = r'RuntimeError:\sfailed\sto\srun:'
    p6_37 = r'RuntimeError:\s[A-Za-z0-9_:\.]*:\smask\sis\suncoalesced'
    p6_40 = r'Allocation\sis\sout\sof\sdevice\smemory\son\scurrent\splatform'
    p6_39 = r'unsupported\sdevice'
    p6_38 = r'does\snot\shave\sa\sdeterministic\simplementation,\sbut\syou\sset'
    p6_41 = r'INTERNAL\sASSERT\sFAILED\sat'
    p6_42 = r'Storage\sdevice\snot\srecognized:'
    p6_43 = r'resulted\sin\sthe\sunexpected\sexception\sabove\.'
    p6_44 = r'without\spivoting\sis\snot\simplemented\son\sthe\s'
    p6_45 = r'should\sbe\sthe\ssame\sor\sinput\sshould\sbe\sa\sMKLDNN\stensor\sand\sweight\sis\sa\sdense\stensor'
    p6_46 = r'indices\sexpected\ssparse\scoordinate\stensor\slayout\sbut\sgot\sSparse'
    p6_47 = r'I\sgot\sthis\soutput\sfor'
    p6_48 = r'values\sexpected\ssparse\stensor\slayout\sbut\sgot\sSparse'
    p6_49 = r'Expected\sall\stensors\sto\sbe\son\sthe\ssame\sdevice'
    p6_50 = r'returned\san\sinvalid\sgradient\sat'
    p6_51 = r'Non-uniform\swork-groups\sare\snot\ssupported\sby\sthe\starget\sdevice'
    p6_52 = r'mode\sonly\ssupports\sCPU\sAND\sCUDA\sdevice\stype'
    p6_53 = r'requiring\sa\sstorage\ssize\sof'
    p6_54 = r'inconsistent\stensors\ssizes\sinput'
    p6_55 = r'Difference\sfrom'
    p6_56 = r'Unsupported\sdevice\stype\sfor'
    p6_57 = r'device\stype\sof\svalues\s\(xpu\)\smust\sbe\sCPU\sor\sCUDA'
    p6_58 = r'Expected\sboth\sself\sand\sother\sto\sbe\snested'
    p6_59 = r'Expected\sdata\.numel\(\)\s>\s0\sto\sbe\strue,\sbut\sgot\sfalse.'
    p6_60 = r'input\smust\sbe\s2-d\smatrix'
    p6_61 = r'value\scannot\sbe\sconverted\sto\stype'
    p6_62 = r'Cannot\sset\spreferred\sbackend\sto'
    p6_63 = r'different\selements'
    p6_64 = r'doesn\'t\smatch\sthe\sbroadcast\sshape'
    p6_65 = r'Expected\seigenvectors\sto\sbe\ssafely\scastable\sfrom'
    p6_66 = r'Expected\s[A-Za-z:0-9]*\s!=\s[A-Za-z:0-9]*\sto\sbe\strue,\sbut\sgot\sfalse'
    p6_67 = r'Expected\sscalars\sto\sbe\son\sCPU,\sgot'
    p6_68 = r'Original\sOpInfo\sis\sbroken'
    p6_69 = r'self\smust\sbe\sa\smatrix'
    p6_70 = r'expect\sweight\sto\sbe\skInt'
    p6_71 = r'Double\sand\scomplex\sdatatype\smatmul\sis\snot\ssupported\sin\soneDNN'
    p7 = r'TypeError:\smodule,\sclass,\smethod,\sfunction,\straceback,\sframe,\sor\scode\sobject\swas\sexpected,\sgot\sbuiltin_function_or_method'
    p8 = r'AssertionError:\sUserWarning\snot\striggered\s:'
    p8_1 = r'Resized\sa\snon-empty\stensor\sbut\sdid\snot\swarn\sabout\sit\.'
    p9 = r'AssertionError:\stensor\(False,\sdevice=\'xpu:0\'\)\sis\snot\strue\s:\sReference\sresult\swas\sfarther'
    p10 = r'AssertionError:\sDevices\s[a-z:0-9]*\sand\s[a-z:0-9]*\sare\snot\sequal!'
    p11 = r'AssertionError:\sWhen\scomparing\sthe\soutput'
    p12 = r'provided\sdtype\smust\smatch\sdtype\sof\sresult'
    p13 = r'AttributeError:\s'
    p13_1 = r'AttributeError:\smodule\s\'[A-Za-z0-9_:\.]*\'\shas\sno\sattribute'
    p13_2 = r'AttributeError:\s\'NoneType\'\sobject\shas\sno\sattribute'
    p13_3 = r'AttributeError:\s\'function\'\sobject\shas\sno\sattribute'
    p14 = r'AssertionError:\sThe\svalues\sfor\sattribute\s\'[A-Za-z0-9_:\.]*\'\sdo\snot\smatch:'
    p15 = r'IndexError:\sDimension\sout\sof\srange'
    p16 = r'AssertionError:\sTensor-likes\sare\snot\sequal!'
    p17 = r'AssertionError:\sScalars\sare\snot\sequal!'
    p18 = r'AssertionError:\sShapes\s.*?\sand\s.*?are\snot\sequal!'
    p19 = r'AssertionError:\sScalars\sare\snot\sclose!'
    p20 = r'AssertionError:\sWarning:\sthis\storch\sapi\sxpu\sdoes\snot\ssupport'
    p20_1 = r'torch\.sparse_compressed_tensor'
    p20_2 = r'torch\.sparse_bsc_tensor'
    p20_3 = r'torch\.sparse_bsr_tensor'
    p20_4 = r'torch\.sparse_csr_tensor'
    p20_5 = r'torch\.sparse_csc_tensor'
    p20_6 = r'test_sparse_compressed_constructor'
    p21 = r'AssertionError:\sObject\scomparison\sfailed:\sdevice'
    p22 = r'The\sabove\squantities\srelating\sthe\snumerical\sand\sanalytical\sjacobians\sare\scomputed'
    p23 = r'AssertionError\n'
    p24 = r'AssertionError:\sObject\scomparison\sfailed:\stensor'
    p25 = r'torch\._dynamo'
    p25_1 = r'isinstance\(\)\sarg\s2\smust\sbe\sa\stype\sor\stuple\sof\stypes'
    p26 = r'AssertionError:\s[A-Za-z0-9\._:]*\sOperation:'
    p27 = r'ValueError:\sAttempted\sto\suse\san\suninitialized\sparameter'
    p28 = r'AssertionError:\sError:\sthe\sapi\s[A-Za-z:0-9\._]*\sis\snot\ssupported\sby'
    p29 = r'AssertionError:\sObject\scomparison\sfailed:\s'
    p30 = r'AssertionError:\sThe\slength\sof\sthe\ssequences\smismatch'
    p31 = r'when\scomputing\selementwise\stype\spromotion'
    p1s = [p1_1, p1_2, p1_3, p1_4, p1_5, p1_6, p1_7, p1_8, p1_9, p1_10, p1_11]
    p2s = [p2_1, p2_2, p2_3, p2_4]
    p3s = [p3_1, p3_2, p3_3, p3_4]
    p4s = [p4_1, p4_2, p4_3, p4_4]
    p6s = [p6_1, p6_2, p6_3, p6_4, p6_5, p6_6, p6_7, p6_8, p6_9, p6_10, p6_11, p6_12, p6_13, p6_14, p6_15, p6_16, p6_17, p6_18, p6_19, p6_20, p6_21, p6_22, p6_23, p6_24, p6_25, p6_26, p6_27, p6_28, p6_29, p6_30, p6_31, p6_32, p6_33, p6_34, p6_35, p6_36, p6_37, p6_38, p6_39, p6_40, p6_41, p6_42, p6_43, p6_44, p6_45, p6_46, p6_47, p6_48, p6_49, p6_50, p6_51, p6_52, p6_53, p6_54, p6_55, p6_56, p6_57, p6_58, p6_59, p6_60, p6_61, p6_62, p6_63, p6_64, p6_65, p6_66, p6_67, p6_68, p6_69, p6_70, p6_71]
    p8s = [p8_1]
    p13s = [p13_1, p13_2, p13_3]
    p20s = [p20_1, p20_2, p20_3, p20_4, p20_5, p20_6]
    p25s = [p25_1]
    pats = [(p1, p1s), (p2, p2s), (p3, p3s), (p4, p4s), (p5, ), (p6, p6s), (p7, ), (p8, p8s), (p9, ), (p10, ), (p11, ), (p12, ), (p13, p13s), (p14, ), (p15, ), (p16, ), (p17, ), (p18, ), (p19, ), (p20, p20s), (p21, ), (p22, ), (p23, ), (p24, ), (p25, p25s), (p26, ), (p27, ), (p28, ), (p29, ), (p30, ), (p31, )]

    results = re.findall(pattern, data, re_flags)
    #print(results)
    for detail, casename, clsname in results:
        print(clsname + "::" + casename)
        if not clsname + "::" + casename in issued_casenames:
            #print(clsname + "::" + casename)
            #details.append(detail + "\n")
            continue
        #if casename == "test_errors_amin_xpu":
        #    print(detail)
        #    break
        print(casename)
        gcount = 0
        for gvalue in pats:
            gtag = 0
            gcount = gcount + 1
            gpat = "p" + str(gcount)
            gresult = re.search(gvalue[0], detail, re_flags)
            if gresult:
                tag = 0
                gtag = 1
                if len(gvalue) <= 1:
                    if gpat not in details:
                        details[gpat] = []
                        casenames[gpat] = []
                    details[gpat].append(detail + "\n")
                    casenames[gpat].append(testname + "::" + clsname + "::" + casename + "\n")
                    break
                count = 0
                for pat in gvalue[1]:
                    count += 1
                    kind = gpat + "_"  + str(count)
                    result = re.search(pat, detail, re_flags)
                    if result:
                        if kind not in details:
                            details[kind] = []
                            casenames[kind] = []
                        details[kind].append(detail + "\n")
                        casenames[kind].append(testname + "::" + clsname + "::" + casename + "\n")
                        tag = 1
                        break
                if tag != 1:
                    if gpat + "_other" not in details:
                        details[gpat + "_other"] = []
                        casenames[gpat + "_other"] = []
                    details[gpat + "_other"].append(detail + "\n")
                    casenames[gpat + "_other"].append(testname + "::" + clsname + "::" + casename + "\n")
                break
        if gtag != 1:
            if "other" not in details:
                details["other"] = []
                casenames["other"] = []
            details["other"].append(detail + "\n")
            casenames["other"].append(testname + "::" + clsname + "::" + casename + "\n")
            #if "other" not in details:
            #    details["other"] = []
            #    casenames["other"] = []
            #details["other"].append(detail + "\n")
            #casenames["other"].append(clsname + "::" + casename + "\n")
    return details, casenames

def collect_detailed_issues(issued_cases, logfile, with_short=False, rebase=False):
    global re_flags
    print(logfile)
    #test_name = logfile.split('.')[0].split('/')[-1].replace('-', '/')
    test_name = logfile.split('/')[-1].split('.')[0].replace('-', '/')
    data = read_file(logfile)
    if re.search(r"^=*\s*test session starts\s*=*$", data, re_flags):
        return collect_pytest_details(issued_cases, data, with_short)
    elif rebase:
        test_file_name = test_name + ".py"
        return collect_pytorch_rebase_details(test_name, issued_cases, data)
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
    pattern_case = r"^(?:  )?(test_\S+)\s\([^\.]*\.([^\.\)]*[^\)]*)\).*?\s+\.\.\..+?(ok|FAIL|ERROR|skipped|expected failure|Command)\s"
    #pattern_case = r"^(test_\S+)\s\([^\.]*\.([^\.\)]*[^\)]*)\).*?\s+\.\.\..+?(ok|FAIL|ERROR|skipped|expected failure|Command)\s"
    #pattern_case = r"^(test_\S+)\s\([^\.]*\.([^\.\)]*)[^\)]*\)\s+\.\.\..+?(ok|FAIL|ERROR|skipped|expected failure|Command)\s"
    # pattern_xpass_case = # haven't found related cases while using pytorch test run.

    matched_cases = re.findall(pattern_case, data, re_flags)
    # xpass_cases = re.findall(pattern_xpass_case, data, re_flags)
    case_num = len(matched_cases)
    count = 1
    for case in matched_cases:
        full_case_name = test_file_name + "::" + case[1] + "::" + case[0]
        #This is for those cases which have result but will trigger signal to break the test
        if count == case_num:
            if case[2] == 'skipped':
                break
            alldata = data.split("\n")
            idx = -1
            tag = 0
            for i in range(-1, -len(alldata), -1):
                if not alldata[i]:
                    idx = i - 1
                elif tag == 0 and "[LOGGING]" in alldata[i]:
                    idx = i - 1
                    tag = 1
                elif tag == 1:
                    break
            if "Command" in alldata[idx] and "died with" in alldata[idx]:
                collected_cases["NO_RESULT"].append(full_case_name)
                break
        count = count + 1
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

