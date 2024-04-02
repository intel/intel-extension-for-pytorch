SUITE=${1:-huggingface}
DT=${2:-amp_bf16}
MODE=${3:-inference}
SCENARIO=${4:-accuracy}
expected_pass_num=${5:-46}

echo -e "========================================================================="
echo -e "Results Check"
echo -e "========================================================================="

cd ../pytorch/inductor_log/${SUITE}
cd ${DT}
num_passed=$(grep "num_passed:" ${SUITE}_${DT}_${MODE}_${SCENARIO}_e2e_summary.log | sed -e 's/.*://;s/[^0-9.]//')
if [ $num_passed -lt ${expected_pass_num} ]; then
  echo -e "[ERROR] Inductor E2E CI test for ${SUITE} ${DT} ${MODE} passed_num < ${expected_pass_num}"
  exit 1
fi
