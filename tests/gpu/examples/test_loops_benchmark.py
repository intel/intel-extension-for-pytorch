import torch
import threading
import numpy as np
from torch.testing._internal.common_utils import TestCase

try:
    import intel_extension_for_pytorch
except Exception:
    intel_extension_for_pytorch = None
try:
    import microbench as mb
except Exception:
    mb = None
mb = None
PRINT_TIME_INFO = False
VERBOSE_ONLY = False

import platform
import pytest


test_shapes = [
    # instruction bound nobc
    [[23, 72, 72], [5184, 72, 1], [23, 72, 72], [5184, 72, 1]],
    # instruction bound bc
    [[23, 16, 16], [23, 1, 16]],
    [[23, 16, 17], [23, 1, 17]],
    [[1, 72, 72], [23, 72, 72]],
    [[23, 72, 1], [23, 72, 72]],
    # memory bound nobc
    [[23000, 72, 72], [5184, 72, 1], [23000, 72, 72], [5184, 72, 1]],
    [[16, 16, 256, 256], [16, 16, 256, 256]],
    # memory bound bc
    [[16, 16, 512, 512], [16, 1, 1, 512]],
    [[4, 15000, 3], [105000, 1, 15000], [4, 1, 3], [3, 3, 1]],
    # memory bound bc no vectorized (launch_legancy_kernel)
    [[16, 16, 512, 513], [16, 1, 1, 513]],
    [[28, 4096, 9], [36864, 9, 1], [28, 4096, 1], [4096, 1, 1]],
]


def get_normalized_time(time_: str, base="us") -> float:
    norm = {"s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1.0}
    time_ = time_.strip()
    t = time_[:-2]
    b = time_[-2:]
    try:
        b_ = float(norm[b]) / norm[base]
    except Exception as e:
        b_ = float(1e9) / norm[base]  # s
    try:
        return float(t) * b_
    except Exception as e:
        return float("nan")


def run_binary_fn(info, fn, dtype, dynamiccast=False, runs=10):
    def identify_time(string):
        tokens = string.split()
        return tokens[-1]

    if len(info) == 4:
        tensor1 = torch.as_strided(
            torch.randn(info[0][0] * info[1][0]), info[0], info[1]
        )
        tensor2 = torch.as_strided(
            torch.randn(info[2][0] * info[3][0]), info[2], info[3]
        )
    elif len(info) == 2:
        tensor1 = torch.randn(info[0])
        tensor2 = torch.randn(info[1])
    tensor1 = eval("tensor1." + dtype.strip() + "()")
    if not dynamiccast:
        tensor2 = eval("tensor2." + dtype.strip() + "()")
    else:
        tensor2 = eval("tensor2.float()")
    durs = []
    maxdiffs = []
    for i in range(runs):
        ret_cpu = eval("tensor1.cpu() " + fn + " tensor2.cpu()")
        if intel_extension_for_pytorch:
            tensor1_, tensor2_ = tensor1.xpu(), tensor2.xpu()
            with torch.autograd.profiler_legacy.profile(True, use_xpu=True) as prof:
                if i == 0 and mb:
                    mb.enable_verbose()
                ret_gpu = eval("tensor1_ " + fn + " tensor2_")
                if i == 0 and mb:
                    mb.disable_verbose()
            timeinfo = str(prof.key_averages().table(sort_by="self_cpu_time_total"))
            maxdiff = float((ret_cpu.xpu() - ret_gpu).abs().max().item())
        else:
            tensor1_, tensor2_ = tensor1.cuda(), tensor2.cuda()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                if i == 0 and mb:
                    mb.enable_verbose()
                ret_gpu = eval("tensor1_ " + fn + " tensor2_")
                if i == 0 and mb:
                    mb.disable_verbose()
            timeinfo = str(prof.key_averages().table(sort_by="self_cpu_time_total"))
            maxdiff = float((ret_cpu.cuda() - ret_gpu).abs().max().item())
        if VERBOSE_ONLY:
            return
        if PRINT_TIME_INFO:
            print(timeinfo)
        dur = float(get_normalized_time(identify_time(timeinfo)))
        if i > 3:
            durs.append(dur)
            maxdiffs.append(maxdiff)
    durs, maxdiffs = sorted(durs), sorted(maxdiffs)
    dur_ = durs[len(durs) // 2]
    maxdiff_ = maxdiffs[len(maxdiffs) - 1]
    assert maxdiff_ < 1e-4
    print(
        "maxdiff: "
        + str(maxdiff_)
        + ", dur(us): "
        + str(dur_)
        + "  "
        + str(info)
        + " "
        + str(dtype)
    )
    return dur_


baseline_a100 = """
maxdiff: 0.0, dur(us): 6.0
maxdiff: 0.0, dur(us): 6.0
maxdiff: 0.0, dur(us): 6.0
maxdiff: 0.0, dur(us): 7.0
maxdiff: 0.0, dur(us): 7.0
maxdiff: 0.0, dur(us): 7.0
maxdiff: 0.0, dur(us): 7.0
maxdiff: 0.0, dur(us): 7.0
maxdiff: 0.0, dur(us): 7.0
maxdiff: 0.0, dur(us): 8.0
maxdiff: 0.0, dur(us): 8.0
maxdiff: 0.0, dur(us): 8.0
maxdiff: 0.0, dur(us): 8.0
maxdiff: 0.0, dur(us): 8.0
maxdiff: 0.0, dur(us): 8.0
maxdiff: 0.0, dur(us): 1058.0
maxdiff: 0.0, dur(us): 1059.0
maxdiff: 0.0, dur(us): 1059.0
maxdiff: 0.0, dur(us): 154.0
maxdiff: 0.0, dur(us): 154.0
maxdiff: 0.0, dur(us): 154.0
maxdiff: 0.0, dur(us): 442.0
maxdiff: 0.0, dur(us): 426.0
maxdiff: 0.0, dur(us): 413.0
maxdiff: 0.0, dur(us): 5.0
maxdiff: 0.0, dur(us): 5.0
maxdiff: 0.0, dur(us): 5.0
maxdiff: 0.0, dur(us): 410.0
maxdiff: 0.0, dur(us): 409.0
maxdiff: 0.0, dur(us): 409.0
maxdiff: 0.0, dur(us): 8.0
maxdiff: 0.0, dur(us): 8.0
maxdiff: 0.0, dur(us): 8.0
maxdiff: 0.0, dur(us): 3.0
maxdiff: 0.0, dur(us): 3.0
maxdiff: 0.0, dur(us): 3.0
maxdiff: 0.0, dur(us): 5.0
maxdiff: 0.0, dur(us): 5.0
maxdiff: 0.0, dur(us): 5.0
maxdiff: 0.0, dur(us): 5.0
maxdiff: 0.0, dur(us): 7.0
maxdiff: 0.0, dur(us): 5.0
maxdiff: 0.0, dur(us): 5.0
maxdiff: 0.0, dur(us): 5.0
maxdiff: 0.0, dur(us): 6.0
maxdiff: 0.0, dur(us): 5.0
maxdiff: 0.0, dur(us): 5.0
maxdiff: 0.0, dur(us): 5.0
maxdiff: 0.0, dur(us): 7.0
maxdiff: 0.0, dur(us): 5.0
maxdiff: 0.0, dur(us): 5.0
maxdiff: 0.0, dur(us): 5.0
maxdiff: 0.0, dur(us): 7.0
maxdiff: 0.0, dur(us): 520.0
maxdiff: 0.0, dur(us): 519.0
maxdiff: 0.0, dur(us): 519.0
maxdiff: 0.0, dur(us): 1057.0
maxdiff: 0.0, dur(us): 78.0
maxdiff: 0.0, dur(us): 80.0
maxdiff: 0.0, dur(us): 80.0
maxdiff: 0.0, dur(us): 183.0
maxdiff: 0.0, dur(us): 512.0
maxdiff: 0.0, dur(us): 512.0
maxdiff: 0.0, dur(us): 512.0
maxdiff: 0.0, dur(us): 769.0
maxdiff: 0.0, dur(us): 10.0
maxdiff: 0.0, dur(us): 10.0
maxdiff: 0.0, dur(us): 10.0
maxdiff: 0.0, dur(us): 12.0
maxdiff: 0.0, dur(us): 513.0
maxdiff: 0.0, dur(us): 513.0
maxdiff: 0.0, dur(us): 513.0
maxdiff: 0.0, dur(us): 777.0
maxdiff: 0.0, dur(us): 14.0
maxdiff: 0.0, dur(us): 14.0
maxdiff: 0.0, dur(us): 14.0
maxdiff: 0.0, dur(us): 20.0
maxdiff: 0.0, dur(us): 6.0
maxdiff: 0.0, dur(us): 6.0
maxdiff: 0.0, dur(us): 6.0
maxdiff: 0.0, dur(us): 10.0
maxdiff: 0.0, dur(us): 9.0
maxdiff: 0.0, dur(us): 9.0
maxdiff: 0.0, dur(us): 9.0
maxdiff: 0.0, dur(us): 12.0
maxdiff: 0.0, dur(us): 9.0
maxdiff: 0.0, dur(us): 9.0
maxdiff: 0.0, dur(us): 9.0
maxdiff: 0.0, dur(us): 12.0
maxdiff: 0.0, dur(us): 9.0
maxdiff: 0.0, dur(us): 9.0
maxdiff: 0.0, dur(us): 9.0
maxdiff: 0.0, dur(us): 13.0
maxdiff: 0.0, dur(us): 9.0
maxdiff: 0.0, dur(us): 9.0
maxdiff: 0.0, dur(us): 9.0
maxdiff: 0.0, dur(us): 12.0
maxdiff: 0.0, dur(us): 524.0
maxdiff: 0.0, dur(us): 523.0
maxdiff: 0.0, dur(us): 523.0
maxdiff: 0.0, dur(us): 1090.0
maxdiff: 0.0, dur(us): 80.0
maxdiff: 0.0, dur(us): 80.0
maxdiff: 0.0, dur(us): 80.0
maxdiff: 0.0, dur(us): 184.0
maxdiff: 0.0, dur(us): 545.0
maxdiff: 0.0, dur(us): 545.0
maxdiff: 0.0, dur(us): 545.0
maxdiff: 0.0, dur(us): 779.0
maxdiff: 0.0, dur(us): 10.0
maxdiff: 0.0, dur(us): 10.0
maxdiff: 0.0, dur(us): 10.0
maxdiff: 0.0, dur(us): 13.0
maxdiff: 0.0, dur(us): 539.0
maxdiff: 0.0, dur(us): 505.0
maxdiff: 0.0, dur(us): 505.0
maxdiff: 0.0, dur(us): 786.0
maxdiff: 0.0, dur(us): 15.0
maxdiff: 0.0, dur(us): 15.0
maxdiff: 0.0, dur(us): 15.0
maxdiff: 0.0, dur(us): 20.0
"""


baseline_pvcb4 = """
maxdiff: 0.0, dur(us): 7.2
maxdiff: 0.0, dur(us): 7.2
maxdiff: 0.0, dur(us): 7.2
maxdiff: 0.0, dur(us): 16.16
maxdiff: 0.0, dur(us): 16.16
maxdiff: 0.0, dur(us): 14.88
maxdiff: 0.0, dur(us): 14.08
maxdiff: 0.0, dur(us): 20.0
maxdiff: 0.0, dur(us): 19.2
maxdiff: 0.0, dur(us): 12.96
maxdiff: 0.0, dur(us): 12.96
maxdiff: 0.0, dur(us): 12.64
maxdiff: 0.0, dur(us): 17.28
maxdiff: 0.0, dur(us): 16.48
maxdiff: 0.0, dur(us): 17.12
maxdiff: 0.0, dur(us): 1388.0
maxdiff: 0.0, dur(us): 1378.0
maxdiff: 0.0, dur(us): 1393.0
maxdiff: 0.0, dur(us): 55.04
maxdiff: 0.0, dur(us): 54.88
maxdiff: 0.0, dur(us): 55.68
maxdiff: 0.0, dur(us): 436.0
maxdiff: 0.0, dur(us): 418.56
maxdiff: 0.0, dur(us): 421.28
maxdiff: 0.0, dur(us): 21.12
maxdiff: 0.0, dur(us): 21.28
maxdiff: 0.0, dur(us): 10.88
maxdiff: 0.0, dur(us): 662.72
maxdiff: 0.0, dur(us): 643.36
maxdiff: 0.0, dur(us): 635.68
maxdiff: 0.0, dur(us): 19.68
maxdiff: 0.0, dur(us): 19.68
maxdiff: 0.0, dur(us): 19.52
maxdiff: 0.0, dur(us): 12.8
maxdiff: 0.0, dur(us): 12.8
maxdiff: 0.0, dur(us): 11.52
maxdiff: 0.0, dur(us): 13.28
maxdiff: 0.0, dur(us): 19.2
maxdiff: 0.0, dur(us): 19.36
maxdiff: 0.0, dur(us): 19.52
maxdiff: 0.0, dur(us): 14.88
maxdiff: 0.0, dur(us): 12.96
maxdiff: 0.0, dur(us): 12.96
maxdiff: 0.0, dur(us): 12.8
maxdiff: 0.0, dur(us): 14.88
maxdiff: 0.0, dur(us): 17.28
maxdiff: 0.0, dur(us): 17.28
maxdiff: 0.0, dur(us): 17.28
maxdiff: 0.0, dur(us): 15.52
maxdiff: 0.0, dur(us): 16.32
maxdiff: 0.0, dur(us): 16.16
maxdiff: 0.0, dur(us): 16.48
maxdiff: 0.0, dur(us): 15.68
maxdiff: 0.0, dur(us): 664.96
maxdiff: 0.0, dur(us): 663.68
maxdiff: 0.0, dur(us): 661.6
maxdiff: 0.0, dur(us): 1352.0
maxdiff: 0.0, dur(us): 31.52
maxdiff: 0.0, dur(us): 31.52
maxdiff: 0.0, dur(us): 31.68
maxdiff: 0.0, dur(us): 161.92
maxdiff: 0.0, dur(us): 226.08
maxdiff: 0.0, dur(us): 224.8
maxdiff: 0.0, dur(us): 223.84
maxdiff: 0.0, dur(us): 913.6
maxdiff: 0.0, dur(us): 16.48
maxdiff: 0.0, dur(us): 16.64
maxdiff: 0.0, dur(us): 16.48
maxdiff: 0.0, dur(us): 16.16
maxdiff: 0.0, dur(us): 589.28
maxdiff: 0.0, dur(us): 591.04
maxdiff: 0.0, dur(us): 583.84
maxdiff: 0.0, dur(us): 916.96
maxdiff: 0.0, dur(us): 18.56
maxdiff: 0.0, dur(us): 18.56
maxdiff: 0.0, dur(us): 18.56
maxdiff: 0.0, dur(us): 24.64
maxdiff: 0.0, dur(us): 13.12
maxdiff: 0.0, dur(us): 12.96
maxdiff: 0.0, dur(us): 13.92
maxdiff: 0.0, dur(us): 13.6
maxdiff: 0.0, dur(us): 24.64
maxdiff: 0.0, dur(us): 24.32
maxdiff: 0.0, dur(us): 21.44
maxdiff: 0.0, dur(us): 15.2
maxdiff: 0.0, dur(us): 12.96
maxdiff: 0.0, dur(us): 12.96
maxdiff: 0.0, dur(us): 12.96
maxdiff: 0.0, dur(us): 15.2
maxdiff: 0.0, dur(us): 20.96
maxdiff: 0.0, dur(us): 20.96
maxdiff: 0.0, dur(us): 17.44
maxdiff: 0.0, dur(us): 15.68
maxdiff: 0.0, dur(us): 16.0
maxdiff: 0.0, dur(us): 16.16
maxdiff: 0.0, dur(us): 16.32
maxdiff: 0.0, dur(us): 16.0
maxdiff: 0.0, dur(us): 665.44
maxdiff: 0.0, dur(us): 672.16
maxdiff: 0.0, dur(us): 660.16
maxdiff: 0.0, dur(us): 1350.0
maxdiff: 0.0, dur(us): 32.16
maxdiff: 0.0, dur(us): 32.32
maxdiff: 0.0, dur(us): 32.32
maxdiff: 0.0, dur(us): 162.08
maxdiff: 0.0, dur(us): 225.92
maxdiff: 0.0, dur(us): 228.48
maxdiff: 0.0, dur(us): 229.76
maxdiff: 0.0, dur(us): 915.36
maxdiff: 0.0, dur(us): 16.64
maxdiff: 0.0, dur(us): 16.64
maxdiff: 0.0, dur(us): 16.32
maxdiff: 0.0, dur(us): 16.32
maxdiff: 0.0, dur(us): 602.4
maxdiff: 0.0, dur(us): 600.32
maxdiff: 0.0, dur(us): 587.84
maxdiff: 0.0, dur(us): 916.48
maxdiff: 0.0, dur(us): 18.72
maxdiff: 0.0, dur(us): 18.72
maxdiff: 0.0, dur(us): 18.56
maxdiff: 0.0, dur(us): 24.8
"""


def perf_gap(index, ref, runtime, enabled):
    perfreg = ref / runtime
    print(perfreg)
    if enabled:
        if perfreg < 0.5:  # not stable here
            raise
    return index + 1


def get_binary_bandwidth():
    print("get_binary_bandwidth...")
    if mb:
        mb.disable_verbose()
    shape0 = [16, 16, 512, 512]
    timeus = run_binary_fn([shape0, shape0], "+", "float")
    total_bytes = np.prod(shape0) * 3 * 4
    bw = total_bytes / (timeus / 1000000.0) / (1024**3)
    print("bw: ", bw)
    if mb:
        mb.enable_verbose()
    return bw


def run_tests():
    bw = get_binary_bandwidth()
    enable_perf_test = False
    if bw > 99999:
        print("enable perf test")
        enable_perf_test = True
    test_types = ["float", "half", "bfloat16"]
    if len(baseline_pvcb4) > 100:
        baselines = baseline_pvcb4.strip().split("\n")
        baselines = [float(line.split()[-1]) for line in baselines]
        index = 0
        for dtype in test_types:
            for item in test_shapes:
                index = perf_gap(
                    index,
                    baselines[index],
                    run_binary_fn(item, "+", dtype),
                    enable_perf_test,
                )
                index = perf_gap(
                    index,
                    baselines[index],
                    run_binary_fn(item, "-", dtype),
                    enable_perf_test,
                )
                index = perf_gap(
                    index,
                    baselines[index],
                    run_binary_fn(item, "*", dtype),
                    enable_perf_test,
                )
                if dtype != "float":  # dynamic cast
                    index = perf_gap(
                        index,
                        baselines[index],
                        run_binary_fn(item, "*", dtype, True),
                        enable_perf_test,
                    )
    else:
        for dtype in test_types:
            for item in test_shapes:
                run_binary_fn(item, "+", dtype)
                run_binary_fn(item, "-", dtype)
                run_binary_fn(item, "*", dtype)
                if dtype != "float":  # dynamic cast
                    run_binary_fn(item, "*", dtype, True)


class TestTensorMethod(TestCase):
    @pytest.mark.skipif(platform.system() == 'Windows', 
                        reason="torch.autograd.profiler_legacy not available on Windows.")
    def test_loops_backbone(self):
        run_tests()


if __name__ == "__main__":
    mutex = threading.Lock()
    with mutex:
        run_tests()
