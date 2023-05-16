import subprocess
import re
import os
import torch
import intel_extension_for_pytorch

# Call the function below periodically, for example, every 100 iterations, pass in the xpu with something like ‘xpu:0’
def info_snapshot(epoch, iteration, xpu): 
    pid = os.getpid()
 
    # Dump the memory mapping content
    cmd = 'cp /proc/{}/maps {}.{}-{}.maps -f'.format(pid, pid, epoch, iteration)
    os.system(cmd)
 
    # Dump the file descriptor usage (so far not seeing any issues)
    cmd = 'ls -l /proc/{}/fd > {}.{}-{}.fd'.format(pid, pid, epoch, iteration)
    os.system(cmd)
 
    # Dump the memory usages
    display_mem(xpu)
 
def display_mem(xpu):
    get_xpu_allocated_mem(xpu)
    if xpu == 'xpu:0':
        get_gpu_device_mem()

def get_xpu_allocated_mem(xpu):
    print('Rank {}: memory allocated {}MB, reserved {}MB'.format(xpu, int(torch.xpu.memory_allocated(xpu) / (1024 * 1024) + 0.5), int(torch.xpu.memory_reserved() / (1024 * 1024)+ 0.5)))

def get_gpu_device_mem():
    # Need export SYSMON_PATH before running, for example:
    # export SYSMON_PATH=${HOME}/pti-gpu/tools/sysmon/build/sysmon
    assert os.getenv("SYSMON_PATH"), "please export SYSMON_PATH before running"
    sysmon_path = os.getenv("SYSMON_PATH")

    lines = subprocess.check_output(sysmon_path).decode('utf8').split("\n")
    total = len(lines)
    i = 0
    gpu = 0
    device_memory = 0
    shared_memory = 0
    while i < total:
        line = lines[i]
        if re.search("GPU\s+\d+", line):
            tmp = re.match("GPU\s+\d+", line).group(0)
            gpu = int(re.sub("GPU\s+", "", tmp))
            device_memory = 0
            shared_memory = 0
        if re.search("Device Memory", line):
            i += 1
            while i < total:
                line = lines[i]
                if re.search("====", line):
                    device_memory = int(device_memory * 10 + 0.5) / 10
                    shared_memory = int(shared_memory * 10 + 0.5) / 10
                    print('GPU {}: device memory {}MB, shared Memory {}MB'.format(gpu, device_memory, shared_memory))
                    gpu = -1
                    break
                numbers = re.split(",\s*", line)
                if len(numbers) > 4: # and re.search("python", numbers[4]):
                    device_memory += float(numbers[1])
                    shared_memory += float(numbers[2])
                i += 1
        i += 1
    if gpu != -1:
        device_memory = int(device_memory * 10 + 0.5) / 10
        shared_memory = int(shared_memory * 10 + 0.5) / 10
        print('GPU {}: device memory {}MB, shared Memory {}MB'.format(gpu, device_memory, shared_memory))

