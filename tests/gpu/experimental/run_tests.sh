#!/bin/bash
# set -exo pipefail
set -x
CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
LOGFILE=${CDIR}/logs/ipex_pytorch_ut_test.log
SPECTEST=

while getopts 'L:SF:V:K:' OPTION
do
  case $OPTION in
    L)
      LOGFILE=$OPTARG
      ;;
    S)
      SPECTEST=1
      ;;
    F)
      FILENAME=$OPTARG
      ;;
    V)
      CLASSNAME=$OPTARG
      ;;
    K)
      TESTNAME=$OPTARG
      ;;
  esac
done
shift $(($OPTIND - 1))

export TORCH_TEST_DEVICES="$CDIR/common/pytorch_test_base.py"
export PYTORCH_TEST_WITH_SLOW=1
export CPU_NUM_DEVICES=4

declare -A TEST_MAP

TEST_MAP["TestShapeOps"]="test_shape_ops.py"
TEST_MAP["TestBinaryUfuncs"]="test_binary_ufuncs.py"
TEST_MAP["TestModule"]="test_modules.py"
TEST_MAP["TestFFT"]="test_spectral_ops.py"
# TEST_MAP["TestFFTDocExamples"]="test_spectral_ops.py"     # only for cpu
TEST_MAP["TestIndexing"]="test_indexing.py"
TEST_MAP["NumpyTests"]="test_indexing.py"
# TEST_MAP["TestBufferProtocol"]="test_buffer_protocol.py"  # only for cpu
TEST_MAP["TestForeach"]="test_foreach.py"
TEST_MAP["TestOperatorSignatures"]="test_fx.py"   # No module named 'fx'
TEST_MAP["TestTensorCreation"]="test_tensor_creation_ops.py"
TEST_MAP["TestRandomTensorCreation"]="test_tensor_creation_ops.py"
TEST_MAP["TestLikeTensorCreation"]="test_tensor_creation_ops.py"
TEST_MAP["TestSortAndSelect"]="test_sort_and_select.py"
TEST_MAP["TestAutogradDeviceType"]="test_autograd.py"
# TEST_MAP["TestViewOps"]="test_torch.py"                   # it's empty
# TEST_MAP["TestVitalSignsCuda"]="test_torch.py"              # only for cuda
# TEST_MAP["TestTensorDeviceOps"]="test_torch.py"           # it's empty
TEST_MAP["TestTorchDeviceType"]="test_torch.py"
TEST_MAP["TestDevicePrecision"]="test_torch.py"
TEST_MAP["TestReductions"]="test_reductions.py"
TEST_MAP["TestBothSerialization"]="test_serialization.py"
# TEST_MAP["TestThatContainsCUDAAssertFailure"]="test_testing.py"   # only for cuda
TEST_MAP["TestTesting"]="test_testing.py"
# TEST_MAP["TestAssertCloseMultiDevice"]="test_testing.py"  # only for cuda
TEST_MAP["TestTestParametrizationDeviceType"]="test_testing.py"
TEST_MAP["TestLinalg"]="test_linalg.py"
TEST_MAP["TestUnaryUfuncs"]="test_unary_ufuncs.py"
# TEST_MAP["TestNormalizeOperators"]="test_fx_experimental.py"    # only for cpu
TEST_MAP["TestModuleInit"]="test_module_init.py"
TEST_MAP["TestTypePromotion"]="test_type_promotion.py"
TEST_MAP["TestViewOps"]="test_view_ops.py"
TEST_MAP["TestOldViewOps"]="test_view_ops.py"
# TEST_MAP["TestNNCOpInfo"]="test_jit_fuser_te.py"          # only for cpu and cuda 
TEST_MAP["TestSparseUnaryUfuncs"]="test_sparse.py"
# TEST_MAP["TestSparse"]="test_sparse.py"  # all cases core dumped ... Segmentation fault
TEST_MAP["TestCommon"]="test_ops.py"
TEST_MAP["TestGradients"]="test_ops.py"
TEST_MAP["TestJit"]="test_ops.py"
TEST_MAP["TestMathBits"]="test_ops.py"
TEST_MAP["TestVmapBatchedGradient"]="test_vmap.py"
# TEST_MAP["TestComplexTensor"]="test_complex.py"           # complex not support yet
TEST_MAP["TestSparseCSR"]="test_sparse_csr.py"
TEST_MAP["TestNumPyInterop"]="test_numpy_interop.py"
TEST_MAP["TestSegmentReductions"]="test_segment_reductions.py"
TEST_MAP["TestNNDeviceType"]="test_nn.py"


function run_test {
  python3 "$CDIR/$1" -v "$2XPU"
}

function run_spec_test {
  python3 "$CDIR/$1" -v "$2" -k "$3"
}

function run_all_tests {
  for TEST_CASE in $(echo ${!TEST_MAP[*]}); do
    run_test "${TEST_MAP[$TEST_CASE]}" "$TEST_CASE"
  done
}

mkdir ${CDIR}/logs
if [ "$SPECTEST" != "" ]; then
  if [ "$LOGFILE" != "" ]; then
    run_spec_test "$FILENAME" "$CLASSNAME" "$TESTNAME" 2>&1 | tee $LOGFILE
  else
    run_spec_test "$FILENAME" "$CLASSNAME" "$TESTNAME"
  fi
else
  if [ "$LOGFILE" != "" ]; then
    run_all_tests 2>&1 | tee $LOGFILE
  else
    run_all_tests
  fi
fi
