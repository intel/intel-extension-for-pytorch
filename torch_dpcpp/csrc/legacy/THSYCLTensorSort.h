#ifndef THSYCL_TENSORSORT_H
#define THSYCL_TENSORSORT_H

#include <c10/dpcpp/SYCLMemory.h>
#include <c10/dpcpp/SYCLUtils.h>
#include <c10/dpcpp/SYCLStream.h>
#include <THDP/THSYCLTensor.h>
#include <THDP/THSYCLGeneral.h>
#include <THDP/THSYCLNumerics.h>
#include <THDP/generic/THSYCLTensorSort.h>
#include <THDP/THSYCLGenerateAllTypes.h>

template <typename IndexType, int Dim>
class fill_slice_sycl_ker {};

template <typename IndexType, int Dim>
void fillSliceWithIndex(TensorInfo<int64_t, IndexType> out,
                   IndexType totalSlices,
                   IndexType sliceSize,
                   IndexType sliceStride);

void THSyclLongTensor_fillSliceWithIndex(THSYCLState* state,
                                         THSyclLongTensor* t,
                                         int dim);

#endif
