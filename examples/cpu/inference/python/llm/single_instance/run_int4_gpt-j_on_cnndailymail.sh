# Unset a few env variables because they slow down GPTQ calibration
unset KMP_BLOCKTIME \
      KMP_TPAUSE \
      KMP_SETTINGS \
      KMP_AFFINITY \
      KMP_FORKJOIN_BARRIER_PATTERN \
      KMP_PLAIN_BARRIER_PATTERN \
      KMP_REDUCTION_BARRIER_PATTERN

# Download finetuned GPT-J model
echo "`date +%Y-%m-%d\ %T` - INFO - Download finetuned GPT-J model..."
wget https://cloud.mlcommons.org/index.php/s/QAZ2oM94MkFtbQx/download -O gpt-j-checkpoint.zip
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "`date +%Y-%m-%d\ %T` - ERROR - Downloading finetuned GPT-J model failed. Exit."
    exit $retVal
fi
echo "`date +%Y-%m-%d\ %T` - INFO - Finetuned GPT-J model downloaded"
echo "`date +%Y-%m-%d\ %T` - INFO - Extract GPT-J model..."
unzip -q gpt-j-checkpoint.zip
model_path=$(pwd)/gpt-j/checkpoint-final/
echo "`date +%Y-%m-%d\ %T` - INFO - GPT-J model extracted to  ${model_path}"

# Run GPTQ calibration
python single_instance/run_int4_gpt-j_on_cnndailymail.py \
    --model ${model_path}

retVal=$?
if [ $retVal -ne 0 ]; then
    echo "`date +%Y-%m-%d\ %T` - ERROR - Exit."
    exit $retVal
fi

# Set a few env variables to get best performance
export KMP_BLOCKTIME=INF
export KMP_TPAUSE=0
export KMP_SETTINGS=1
export KMP_FORKJOIN_BARRIER_PATTERN=dist,dist
export KMP_PLAIN_BARRIER_PATTERN=dist,dist
export KMP_REDUCTION_BARRIER_PATTERN=dist,dist
env | grep CONDA_PREFIX > /dev/null
if [ $? -eq 0 ]; then
    export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so.6
    export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so # Intel OpenMP
    export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
else
    echo "Conda environment is not available. You need to set environment variable LD_PRELOAD to dynamic libraries of Intel OpenMP and TcMalloc manually."
fi

# Run benchmark
python single_instance/run_int4_gpt-j_on_cnndailymail.py \
    --dataset-path ./saved_results/cnn_dailymail_validation.json \
    --model ${model_path} \
    --low-precision-checkpoint ./saved_results/gptq_checkpoint_g128.pt
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "`date +%Y-%m-%d\ %T` - ERROR - Exit."
    exit $retVal
fi
echo "`date +%Y-%m-%d\ %T` - INFO - Finished successfully."
