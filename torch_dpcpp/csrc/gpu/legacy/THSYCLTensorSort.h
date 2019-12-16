#ifndef THSYCL_TENSORSORT_H
#define THSYCL_TENSORSORT_H

#include <core/SYCLMemory.h>
#include <core/SYCLUtils.h>
#include <core/SYCLStream.h>
#include <legacy/THSYCLTensor.h>
#include <legacy/THSYCLGeneral.h>
#include <legacy/THSYCLNumerics.h>
#include <legacy/generic/THSYCLTensorSort.h>
#include <legacy/THSYCLGenerateAllTypes.h>

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
