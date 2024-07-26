#!/bin/bash

MASTER_NET_IF=eth0
model_id=meta-llama/Llama-2-7b-hf
data_type=float32
batch_size=1
output=32
input=32
num_iter=10
warmup=2
ONECCL_NUM_WORKERS=4  ## You could tune the worker number for your workload

NODEFILE=hostfile.txt
if ! [ -f $NODEFILE ]; then
  echo "File does not exist."
  exit 0
fi

WORKDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}}" )" &> /dev/null && pwd )

function get_hw_info()
{
    number_threads=`nproc --all`
    number_cores=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`
    number_sockets=`grep physical.id /proc/cpuinfo | sort -u | wc -l`
    cpu_per_socket=$((number_cores/number_sockets))
}

# Config OneCCL 
function nw_config()
{
    worker=$1
    omp_worker=$2
    get_hw_info
    if [ $number_sockets != 1 ]; then
        ccl_cpu_list=$(seq -s, $((cpu_per_socket - worker/number_sockets)) $((cpu_per_socket - 1))),$(seq -s, $((number_cores - worker/number_sockets)) $((number_cores - 1)))
        omp_cpu_list=$(seq -s, $((0)) $((omp_worker - 1))),$(seq -s, $((cpu_per_socket)) $((cpu_per_socket + omp_worker -1)))
    else
        ccl_cpu_list=$(seq -s, $((cpu_per_socket - worker)) $((cpu_per_socket - 1)))
        omp_cpu_list=$(seq -s, $((0)) $((omp_worker - 1)))
    fi
    export CCL_WORKER_AFFINITY=$ccl_cpu_list
    export CCL_WORKER_COUNT=$((worker/number_sockets))

    export CCL_ALLREDUCE=rabenseifner # Other algorithms inlcude nreduce, ring and recursive_doubling. Rabenseifner algorithm is more friendly for latency sensitive workload
    export CCL_ATL_TRANSPORT=ofi #Other option is mpi
}

# Create mpi argments
# Assume your ibv_devices output has 4 nics: irdma-cvl01tf2,irdma-cvl02tf2,irdma-cvl11tf2,irdma-cvl12tf2
function build_launch_args_fi_tcp(){
    PKG_PATH=$1
    margs="--genv CCL_WORKER_COUNT=${CCL_WORKER_COUNT}"
    #margs="$margs --genv CCL_MNIC=global"                  # Select all NICs local for the NUMA node that corresponds to process pinning
    margs="$margs --genv CCL_MNIC_COUNT=1"                # The maximum number of NICs that should be selected for oneCCL workers. 
    margs="$margs --genv CCL_MNIC_NAME=${MASTER_NET_IF}"  # to control multi-NIC selection by NIC names
    margs="$margs --genv CCL_WORKER_AFFINITY=${CCL_WORKER_AFFINITY}"
    margs="$margs --genv CCL_ATL_TRANSPORT=$CCL_ATL_TRANSPORT"   # Select the transport for inter-process communications
    margs="$margs --genv I_MPI_PIN=0"
    #margs="$margs --genv FI_LOG_LEVEL=debug"
    margs="$margs --genv FI_PROVIDER=tcp"
    margs="$margs --genv FI_TCP_IFACE=${MASTER_NET_IF}"
    margs="$margs --genv I_MPI_OFI_PROVIDER=tcp"
    margs="$margs --genv I_MPI_FABRICS=ofi"
    margs="$margs --genv I_MPI_HYDRA_IFACE=${MASTER_NET_IF}"
    margs="$margs --genv CCL_KVS_IFACE=${MASTER_NET_IF}"
    margs="$margs --genv PDSH_RCMD_TYPE=ssh"
}

function build_launch_args_fi_psm3(){
    PKG_PATH=$1
    margs="--genv CCL_WORKER_COUNT=${CCL_WORKER_COUNT}"
    margs="$margs --genv CCL_ALLREDUCE=${CCL_ALLREDUCE}"
    margs="$margs --genv CCL_MNIC=global"
    margs="$margs --genv CCL_LOG_LEVEL=debug"
    margs="$margs --genv CCL_MNIC_COUNT=1"
    margs="$margs --genv CCL_MNIC_NAME=${MASTER_NET_IF}"
    margs="$margs --genv CCL_WORKER_AFFINITY=${CCL_WORKER_AFFINITY}"
    margs="$margs --genv CCL_ATL_TRANSPORT=ofi"
    margs="$margs --genv PSM3_ALLOW_ROUTERS=1"
    margs="$margs --genv PSM3_RDMA=1"
    margs="$margs --genv PSM3_IDENTIFY=1"
    margs="$margs --genv PSM3_RV_MR_CACHE_SIZE=8192"
    margs="$margs --genv FI_PROVIDER_PATH=${PKG_PATH}/oneccl_bindings_for_pytorch/lib/"  # Specify the location of the installed PSM3 provider, when use torch-ccl the version in torch-ccl enviroment will be used
    margs="$margs --genv PSM3_NIC_SPEED=100000"
    margs="$margs --genv PSM3_KASSIST_MODE=none"
    margs="$margs --genv PSM3_NIC=${MASTER_NET_IF}"
    margs="$margs --genv I_MPI_PIN=0"
    margs="$margs --genv I_MPI_PIN_PROCESSOR_LIST=1,33"
    #margs="$margs --genv FI_LOG_LEVEL=debug"
    margs="$margs --genv FI_PROVIDER=psm3"
    margs="$margs --genv I_MPI_OFI_PROVIDER=psm3"
    margs="$margs --genv FI_TCP_IFACE=${MASTER_NET_IF}"
    margs="$margs --genv I_MPI_FABRICS=ofi"
    margs="$margs --genv I_MPI_HYDRA_IFACE=${MASTER_NET_IF}"
    #margs="$margs --genv PSM3_DEVICES=\'self,nic\'"
}

# Run 
PKG_PATH=$(python -m pip show intel-extension-for-pytorch | grep "Location" | cut -d " " -f 2)
source ${PKG_PATH}/intel_extension_for_pytorch/env/setvars.sh
export MASTER_ADDR=$(ifconfig $MASTER_NET_IF | grep 'inet ' | awk '{print $2}')
export MASTER_PORT=29500
echo $MASTER_ADDR

get_hw_info
OMP_NUM_THREADS=$((cpu_per_socket - ONECCL_NUM_WORKERS/number_sockets)) # Leave some cores for CCL worker and leave some cores idle to reduce stragger effect
nw_config $ONECCL_NUM_WORKERS $OMP_NUM_THREADS
build_launch_args_fi_tcp ${PKG_PATH}

# For example to run bf16
deepspeed --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} --no_ssh_check --hostfile=${NODEFILE} --bind_cores_to_rank --bind_core_list ${omp_cpu_list} --launcher impi --launcher_args " --genv LD_LIBRARY_PATH=$LD_LIBRARY_PATH $margs" distributed/run_generation_with_deepspeed.py  --model-id $model_id --dtype $data_type --ipex  --batch-size $batch_size --benchmark --max-new-tokens ${output} --input-tokens ${input} --token-latency --num-iter ${num_iter} --num-warmup ${warmup}
