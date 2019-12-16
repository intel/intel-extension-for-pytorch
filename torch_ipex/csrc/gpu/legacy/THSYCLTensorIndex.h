#ifndef TH_SYCL_TENSOR_INDEX_INC
#define TH_SYCL_TENSOR_INDEX_INC

#include <legacy/THSYCLTensor.h>
#include <legacy/THSYCLGeneral.h>
#include <legacy/generic/THSYCLTensorIndex.h>
#include <legacy/THSYCLGenerateAllTypes.h>

#include <legacy/generic/THSYCLTensorIndex.h>
#include <legacy/THSYCLGenerateBoolType.h>

DP_DEF_K1(index_select_ker);
DP_DEF_K1(index_fill_ker);
DP_DEF_K1(index_add_ker);

#endif
