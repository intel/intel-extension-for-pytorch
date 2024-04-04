SUITE=${1:-huggingface}
DT=${2:-amp_bf16}
MODE=${3:-inference}
SCENARIO=${4:-accuracy}
expected_pass_num=${5:-46}

echo -e "========================================================================="
echo -e "Dependancy Install"
echo -e "========================================================================="
source activate e2e_ci
cp .github/scripts/inductor_xpu_test.sh ../pytorch
cd ../pytorch
if [[ ${SUITE} == "timm_MODEls" ]]; then
    pip install --no-deps "git+https://github.com/rwightman/pytorch-image-MODEls@b9d43c7dcac1fe05e851dd7be7187b108af593d2"
elif [[ ${SUITE} == "torchbench" ]]; then
    pip install transformers==4.38.1 --no-deps
    pip install timm==0.9.7 --no-deps
    apt-get update -y
    apt install libgl1-mesa-glx -y
    conda install -y git-lfs pyyaml pandas scipy psutil
    pip install tqdm pandas pyre-extensions torchrec tensorboardX dalle2_pytorch torch_geometric scikit-image matplotlib  gym fastNLP doctr matplotlib opacus python-doctr higher opacus dominate kaldi-io librosa effdet pycocotools diffusers
    pip uninstall -y pyarrow pandas
    pip install pyarrow pandas
    
    cd ..
    git clone https://github.com/facebookresearch/detectron2.git
    python -m pip install -e detectron2

    git clone --recursive https://github.com/facebookresearch/multimodal.git multimodal
    pushd multimodal
    pip install -e .
    popd
fi

#TRANSFORMERS_COMMIT=$(cat .ci/docker/ci_commit_pins/huggingface.txt)
#pip install --force-reinstall git+https://github.com/huggingface/transformers@${TRANSFORMERS_COMMIT}
echo -e "========================================================================="
echo -e "E2E Test"
echo -e "========================================================================="
source /opt/intel/oneapi/setvars.sh
export HUGGING_FACE_HUB_TOKEN=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
#export PYTORCH_ENABLE_XPU_FALLBACK=1
rm -rf inductor_log
bash inductor_xpu_test.sh ${SUITE} ${DT} ${MODE} ${SCENARIO} xpu 3

echo -e "========================================================================="
echo -e "Test Results Summary"
echo -e "========================================================================="
cd ../pytorch/inductor_log/${SUITE}
cd ${DT}
echo -e "============ Summary for ${SUITE} ${DT} ${MODE} ${SCENARIO} ============" | tee -a ./${SUITE}_${DT}_${MODE}_${SCENARIO}_e2e_summary.log
awk -i inplace '!seen[$0]++' inductor_${SUITE}_${DT}_${MODE}_xpu_${SCENARIO}.csv
csv_lines=$(cat inductor_${SUITE}_${DT}_${MODE}_xpu_${SCENARIO}.csv | wc -l)
let num_total=csv_lines-1
num_passed=$(grep -c "pass" inductor_${SUITE}_${DT}_${MODE}_xpu_${SCENARIO}.csv)
let num_failed=num_total-num_passed
#pass_rate=`awk 'BEGIN{printf "%.2f%%\n",('$num_passed'/'$num_total')*100}'`
echo "num_total: $num_total" | tee -a ./${SUITE}_${DT}_${MODE}_${SCENARIO}_e2e_summary.log
echo "num_passed: $num_passed" | tee -a ./${SUITE}_${DT}_${MODE}_${SCENARIO}_e2e_summary.log
echo "num_failed: $num_failed" | tee -a ./${SUITE}_${DT}_${MODE}_${SCENARIO}_e2e_summary.log
#echo "pass_rate: $pass_rate" | tee -a ./${SUITE}_${DT}_${MODE}_${SCENARIO}_e2e_summary.log
