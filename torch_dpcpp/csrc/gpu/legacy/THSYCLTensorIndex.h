#ifndef TH_SYCL_TENSOR_INDEX_INC
#define TH_SYCL_TENSOR_INDEX_INC

#include <THDP/THSYCLTensor.h>
#include <THDP/THSYCLGeneral.h>
#include <THDP/generic/THSYCLTensorIndex.h>
#include <THDP/THSYCLGenerateAllTypes.h>

#include <THDP/generic/THSYCLTensorIndex.h>
#include <THDP/THSYCLGenerateBoolType.h>

DP_DEF_K1(index_select_ker);
DP_DEF_K1(index_fill_ker);
DP_DEF_K1(index_add_ker);

#endif
