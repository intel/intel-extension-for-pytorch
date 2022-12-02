# Running benchmarks for Intel Extension for PyTorch Custom OPs
Evaluate performance for custom operator with [launcher](../../../../tutorials/intro_launch.md).
## Prepare envrioment
Follow [performance_tuning_guide](../../../../tutorials/Performance_Tuning.md) to install Memory_Allocator(you can choose Tcmalloc or Jemalloc).
Install intel-openmp:

```
conda install intel-openmp
```

## Evaluate [Interaction](../../../../intel_extension_for_pytorch/nn/functional/interaction.py)

1.Inference: 1 instance per core in real world scenario

```
export OMP_NUM_THREADS=1
export CORES=`lscpu | grep Core | awk '{print $4}'`
python -m intel_extension_for_pytorch.cpu.launch --node_id 0 interaction.py --num-instance=$CORES --inference # for fp32
python -m intel_extension_for_pytorch.cpu.launch --node_id 0 interaction.py --num-instance=$CORES --inference --bf16 # for bf16
```
unset OMP_NUM_THREADS
2.Training: 1 instance on 1 socket in real world scenario

```
python -m intel_extension_for_pytorch.cpu.launch --node_id 0 interaction.py # for fp32
python -m intel_extension_for_pytorch.cpu.launch --node_id 0 interaction.py --bf16 # for bf16
```

## Evaluate IPEX fused optimizer
```
python -m intel_extension_for_pytorch.cpu.launch --node_id 0 optimizer.py --optimizer sgd # for sgd
python -m intel_extension_for_pytorch.cpu.launch --node_id 0 optimizer.py --optimizer lamb # for lamb
python -m intel_extension_for_pytorch.cpu.launch --node_id 0 optimizer.py --optimizer adagrad # for adagrad
python -m intel_extension_for_pytorch.cpu.launch --node_id 0 optimizer.py --optimizer adam # for adam
```

## Evaluate IPEX [MergedEmbeddingBag](../../../../intel_extension_for_pytorch/nn/module/merged_embeddingbag.py)
```
export CORES=`lscpu | grep Core | awk '{print $4}'`
export BATCHSIZE=$((128*CORES))
# Data distribution will not impact inference performance
python -m intel_extension_for_pytorch.cpu.launch --node_id 0 merged_embeddingbag.py --inference --data-distribution=balance --batch-size=${BATCHSIZE}

# For training, data distribution will have big impact while update weight. Under the "unbalance" arg, we will use generate datas with half of indice update same raw (which is similiar with real world dataset as DLRM mlperf dataset)
python -m intel_extension_for_pytorch.cpu.launch --node_id 0 merged_embeddingbag.py --data-distribution=balance --batch-size=${BATCHSIZE}
python -m intel_extension_for_pytorch.cpu.launch --node_id 0 merged_embeddingbag.py --data-distribution=unbalance --batch-size=${BATCHSIZE}
```
