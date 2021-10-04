export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export LD_PRELOAD="${CONDA_PREFIX}/lib/libjemalloc.so:${CONDA_PREFIX}/lib/libiomp5.so"
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_SETTINGS=1

CORES=`lscpu | grep Core | awk '{print $4}'`
numa_cmd="numactl -C 0-$((CORES-1)) -m 0"
OMP_NUM_THREADS=1 $numa_cmd python -u interaction.py --num-instance=$CORES --inference
DNNL_MAX_CPU_ISA=AVX512_CORE_AMX OMP_NUM_THREADS=1 $numa_cmd python -u interaction.py --num-instance=$CORES --inference --bf16
OMP_NUM_THREADS=$CORES $numa_cmd python -u interaction.py
DNNL_MAX_CPU_ISA=AVX512_CORE_AMX OMP_NUM_THREADS=$CORES $numa_cmd python -u interaction.py --bf16
