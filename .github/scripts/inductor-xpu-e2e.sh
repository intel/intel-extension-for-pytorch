suite=${1:-huggingface}
dt=${2:-amp_bf16}
mode=${3:-inference}
scenario=${4:-accuracy}
expected_pass_num=${5:-46}

echo -e "========================================================================="
echo -e "Dependancy Install"
echo -e "========================================================================="
source activate e2e_ci
cp .github/scripts/inductor_xpu_test.sh ../pytorch
cd ../pytorch
if [[ ${suite} == "timm_models" ]]; then
    pip install --no-deps "git+https://github.com/rwightman/pytorch-image-models@b9d43c7dcac1fe05e851dd7be7187b108af593d2"
elif [[ ${suite} == "torchbench" ]]; then
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
#export PYTORCH_ENABLE_XPU_FALLBACK=1
rm -rf inductor_log
bash inductor_xpu_test.sh ${suite} ${dt} ${mode} ${scenario} xpu 3

echo -e "========================================================================="
echo -e "Test Results Summary"
echo -e "========================================================================="
cd ../pytorch/inductor_log/${suite}
cd ${dt}
echo -e "============ Summary for ${suite} ${dt} ${mode} ${scenario} ============" | tee -a ./${suite}_${dt}_${mode}_${scenario}_e2e_summary.log
awk -i inplace '!seen[$0]++' inductor_${suite}_${dt}_${mode}_xpu_${scenario}.csv
csv_lines=$(cat inductor_${suite}_${dt}_${mode}_xpu_${scenario}.csv | wc -l)
let num_total=csv_lines-1
num_passed=$(grep -c "pass" inductor_${suite}_${dt}_${mode}_xpu_${scenario}.csv)
let num_failed=num_total-num_passed
pass_rate=`awk 'BEGIN{printf "%.2f%%\n",('$num_passed'/'$num_total')*100}'`
echo "num_total: $num_total" | tee -a ./${suite}_${dt}_${mode}_${scenario}_e2e_summary.log
echo "num_passed: $num_passed" | tee -a ./${suite}_${dt}_${mode}_${scenario}_e2e_summary.log
echo "num_failed: $num_failed" | tee -a ./${suite}_${dt}_${mode}_${scenario}_e2e_summary.log
echo "pass_rate: $pass_rate" | tee -a ./${suite}_${dt}_${mode}_${scenario}_e2e_summary.log
