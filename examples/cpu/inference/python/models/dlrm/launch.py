import sys

args = sys.argv
sockets = int(args[1])
numas = int(args[2])
numa_per_socket = int(numas / sockets)
cores_per_socket = int(args[3])
enable_2nd_process = int(args[4])
aot_inductor = int(args[5])

import os

os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
os.environ["MALLOC_CONF"] = (
    "oversize_threshold:1,background_thread:true,metadata_thp:auto"
)
assert "LD_PRELOAD" in os.environ.keys()
assert "iomp" in os.environ["LD_PRELOAD"]
assert "jemalloc" in os.environ["LD_PRELOAD"]


avgcore_per_numa = int(cores_per_socket // numa_per_socket)
coreidx_per_numa = list(range(0, cores_per_socket * sockets + 1, avgcore_per_numa))
for s in range(sockets):
    current_n = s * numa_per_socket
    print(coreidx_per_numa)
    print((s + 1) * numa_per_socket)
    while coreidx_per_numa[(s + 1) * numa_per_socket] != cores_per_socket * (s + 1):
        for n in range(current_n + 1, numas + 1):
            coreidx_per_numa[n] += 1
        current_n += 1

if aot_inductor:
    model_dir = str(args[5])
    BS = int(os.environ["BATCH_SIZE"])
    EVAL_BATCH = int(os.environ["EVAL_BATCH"])
    from dlrm_main import aoti_benchmark_compile

    composed_cmd = ""
    for i in range(numas):
        target_dir = f"./aoti-dir-{i}"
        if os.path.exists(target_dir):
            os.system(f"rm -r {target_dir}")
        os.system(f"mkdir {target_dir}")
        start = coreidx_per_numa[i]
        end = coreidx_per_numa[i + 1] - 1
        bench_bin = aoti_benchmark_compile(
            end - start + 1, EVAL_BATCH, BS, model_dir, target_dir
        )
        composed_cmd += f"taskset -c {start}-{end} {bench_bin} "
        if i != (numas - 1):
            composed_cmd += " & "

    print(composed_cmd)
    os.system(composed_cmd)
else:
    cmd = str(args[6])
    composed_cmd = ""
    for i in range(numas):
        start = coreidx_per_numa[i]
        end = coreidx_per_numa[i + 1] - 1
        if enable_2nd_process:
            mid = (start + end) // 2
            composed_cmd += (
                f"taskset -c {start}-{mid} {cmd} --share-weight-instance={mid-start+1} "
            )
            composed_cmd += " & "
            composed_cmd += (
                f"taskset -c {mid + 1}-{end} {cmd} --share-weight-instance={end-mid} "
            )
        else:
            composed_cmd += (
                f"taskset -c {start}-{end} {cmd} --share-weight-instance={end-start+1} "
            )
        if i != (numas - 1):
            composed_cmd += " & "
    print(composed_cmd)
    os.system(composed_cmd)
