oneccl_bindings_for_pytorch_path=$(python -c "import torch; import oneccl_bindings_for_pytorch; import os;  print(os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh
DIR=$(dirname $(readlink -f "$0"))
DISTRIBUTED_EMB=${DIR}/test_distributed_merged_emb.py
for w_size in 2 4 8 16
do
    export W_SIZE=$w_size
    python -m intel_extension_for_pytorch.cpu.launch --ccl_worker_count=1 --nproc_per_node=$W_SIZE --distributed --nnodes 1 $DISTRIBUTED_EMB
done
