PYTORCH_DIR=${HOME}/xunsongh/pytorch
IPEX_DIR=${HOME}/xunsongh/ipex-gpu
TARGET_DIR=${HOME}/xunsongh/ipex-gpu/tests/gpu/experimental

python ${IPEX_DIR}/tests/gpu/experimental/tool/ut_rebaser.py \
  -p ${PYTORCH_DIR} \
  -x ${IPEX_DIR} \
  -o ${TARGET_DIR}
