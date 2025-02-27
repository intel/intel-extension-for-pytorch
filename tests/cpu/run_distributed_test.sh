oneccl_bindings_for_pytorch_path=$(python -c "import torch; import oneccl_bindings_for_pytorch; import os; print('path:', os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))")
oneccl_bindings_for_pytorch_path=$(echo $oneccl_bindings_for_pytorch_path | awk -F'path:' '{print $2}')
source $oneccl_bindings_for_pytorch_path/env/setvars.sh
DIR=$(dirname $(readlink -f "$0"))
DISTRIBUTED_EMB=${DIR}/test_distributed_merged_emb.py
for w_size in 2 4 8 16
do
    export W_SIZE=$w_size
    python -m intel_extension_for_pytorch.cpu.launch --ccl-worker-count=1 --nprocs-per-node=$W_SIZE --nnodes 1 $DISTRIBUTED_EMB
done
