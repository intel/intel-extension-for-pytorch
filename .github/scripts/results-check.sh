suite=${1:-huggingface}
dt=${2:-amp_bf16}
mode=${3:-inference}
scenario=${4:-accuracy}
expected_pass_num=${5:-46}

echo -e "========================================================================="
echo -e "Results Check"
echo -e "========================================================================="

cd ../pytorch/inductor_log/${suite}
cd ${dt}
num_passed=$(grep "num_passed:" ${suite}_${dt}_${mode}_${scenario}_e2e_summary.log | sed -e 's/.*://;s/[^0-9.]//')
if [ $num_passed -lt ${{ inputs.expected_pass_num }} ]; then
  echo -e "[ERROR] Inductor E2E CI test for ${suite} ${dt} ${mode} passed_num < ${{ inputs.expected_pass_num }}"
  exit 1
fi
