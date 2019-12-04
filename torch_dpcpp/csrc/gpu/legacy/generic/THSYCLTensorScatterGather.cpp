#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "THDP/generic/THSYCLTensorScatterGather.cpp"
#else

#define RUN(TYPE, DIMS, REAL)                                           \
  THSYCLTensor_gatherKernel<TYPE, REAL, DIMS>                           \
  (tensorInfo, srcInfo, indexInfo, dim, (TYPE)totalElements);

void THSYCLTensor_(gather)(THSYCLState* state, THSYCLTensor* tensor,
    THSYCLTensor* src, int dim, THSyclLongTensor* index) {

  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, tensor, src));
  THSYCLAssertSameGPU(THSyclLongTensor_checkGPU(state, 1, index));

  THArgCheck(THSyclLongTensor_nDimensionLegacyNoScalars(state, index) == THSYCLTensor_(nDimensionLegacyNoScalars)(state, src), 4,
             "Index tensor must have same dimensions as input tensor");
  THArgCheck(tensor->sizes().equals(index->sizes()), 4,
             "Index tensor must have the same size as output tensor.");
  THArgCheck(dim >= 0 && dim < THSYCLTensor_(nDimensionLegacyNoScalars)(state, tensor), 3,
             "Index dimension is out of bounds");
  THArgCheck(THSYCLTensor_(nDimensionLegacyNoScalars)(state, src) == THSYCLTensor_(nDimensionLegacyNoScalars)(state, tensor), 2,
             "Input tensor must have same dimensions as output tensor");

  for (int d = 0; d < THSYCLTensor_(nDimensionLegacyNoScalars)(state, tensor); d++) {
    if (d != dim) {
      THArgCheck(THSYCLTensor_(sizeLegacyNoScalars)(state, tensor, d) == THSYCLTensor_(sizeLegacyNoScalars)(state, src, d), 2,
                 "Input tensor must have same size as output tensor apart from the specified dimension");
    }
  }

  THArgCheck(THSYCLTensor_(nDimensionLegacyNoScalars)(state, tensor) <= MAX_SYCLTORCH_DIMS,
             1, SYCLTORCH_DIM_WARNING);

  const ptrdiff_t totalElements = THSyclLongTensor_nElement(state, index);

  THSYCLTensor* oldTensor = NULL;
  if (THSYCLTensor_maybeOverlappingIndices(state, tensor)) {
    oldTensor = tensor;
    tensor = THSYCLTensor_(newContiguous)(state, tensor);
  }

  if (totalElements > 0) {
    if (THSYCLTensor_canUse32BitIndexMath(state, tensor) &&
        THSYCLTensor_canUse32BitIndexMath(state, src) &&
        THSYCLTensor_canUse32BitIndexMath(state, index)) {
      TensorInfo<scalar_t, unsigned int> tensorInfo =
              getTensorInfo<scalar_t, THSYCLTensor, unsigned int>(state, tensor);
      TensorInfo<scalar_t, unsigned int> srcInfo =
              getTensorInfo<scalar_t, THSYCLTensor, unsigned int>(state, src);
      TensorInfo<int64_t, unsigned int> indexInfo =
              getTensorInfo<int64_t, THSyclLongTensor, unsigned int>(state, index);

      // Specialize for a small number of dimensions.
      switch (indexInfo.dims) {
        case 1:
          RUN(unsigned int, 1, scalar_t);
//          THSYCLCheck(cudaGetLastError());
          break;
        case 2:
          RUN(unsigned int, 2, scalar_t);
//          THSYCLCheck(cudaGetLastError());
          break;
        case 3:
          RUN(unsigned int, 3, scalar_t);
//          THSYCLCheck(cudaGetLastError());
          break;
        default:
          RUN(unsigned int, -1, scalar_t);
//          THSYCLCheck(cudaGetLastError());
          break;
      }
    } else {
      TensorInfo<scalar_t, uint64_t> tensorInfo =
              getTensorInfo<scalar_t, THSYCLTensor, uint64_t>(state, tensor);
      TensorInfo<scalar_t, uint64_t> srcInfo =
              getTensorInfo<scalar_t, THSYCLTensor, uint64_t>(state, src);
      TensorInfo<int64_t, uint64_t> indexInfo =
              getTensorInfo<int64_t, THSyclLongTensor, uint64_t>(state, index);
      RUN(uint64_t, -1, scalar_t);
//      THSYCLCheck(cudaGetLastError());
    }
  }

  if (oldTensor) {
    THSYCLTensor_(copyIgnoringOverlaps)(state, oldTensor, tensor);
    THSYCLTensor_(free)(state, tensor);
    tensor = oldTensor;
  }
//  THSYCLCheck(cudaGetLastError());
}

#define RUN(TYPE, DIMS, REAL)                                           \
  THSyclTensor_scatterKernel<TYPE, REAL, DIMS>(                               \
    tensorInfo, srcInfo, indexInfo, dim, (TYPE)totalElements);

void THSYCLTensor_(scatter)(THSYCLState* state, THSYCLTensor* tensor,
    int dim, THSyclLongTensor* index, THSYCLTensor* src) {
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, tensor, src));
  THSYCLAssertSameGPU(THSyclLongTensor_checkGPU(state, 1, index));

  int index_ndim_legacy_all = THSyclLongTensor_nDimensionLegacyAll(state, index);
  THArgCheck(dim >= 0 && dim < THSYCLTensor_(nDimensionLegacyNoScalars)(state, tensor), 2,
             "Index dimension is out of bounds");
  THArgCheck(index_ndim_legacy_all == 0
             || THSyclLongTensor_nDimensionLegacyNoScalars(state, index) == THSYCLTensor_(nDimensionLegacyNoScalars)(state, src), 3,
             "Index tensor must be either empty or have same dimensions as input tensor");
  THArgCheck(THSYCLTensor_(nDimensionLegacyNoScalars)(state, src) == THSYCLTensor_(nDimensionLegacyNoScalars)(state, tensor), 4,
             "Input tensor must have same dimensions as output tensor");

  // no-op if index is empty
  if (index_ndim_legacy_all == 0)
      return;

  for (int d = 0; d < THSYCLTensor_(nDimensionLegacyNoScalars)(state, tensor); d++) {
    int64_t indexSizeD = THSyclLongTensor_sizeLegacyNoScalars(state, index, d);
    if (d != dim) {
      THArgCheck(indexSizeD <= THSYCLTensor_(sizeLegacyNoScalars)(state, tensor, d), 3,
                 "Index tensor must not have larger size than output tensor apart from the specified dimension %d, but got index %s output %s",
                 dim, THSyclLongTensor_sizeDesc(state, index).str, THSYCLTensor_(sizeDesc)(state, tensor).str);
    }
    THArgCheck(indexSizeD <= THSYCLTensor_(sizeLegacyNoScalars)(state, src, d), 3,
               "Index tensor must not have larger size than input tensor, but got index %s input %s",
               THSyclLongTensor_sizeDesc(state, index).str, THSYCLTensor_(sizeDesc)(state, src).str);
  }
  
  THArgCheck(THSYCLTensor_(nDimensionLegacyNoScalars)(state, tensor) <= MAX_SYCLTORCH_DIMS,
             1, SYCLTORCH_DIM_WARNING);

  const ptrdiff_t totalElements = THSyclLongTensor_nElement(state, index);

  THSYCLTensor* oldTensor = NULL;
  if (THSYCLTensor_maybeOverlappingIndices(state, tensor)) {
    oldTensor = tensor;
    tensor = THSYCLTensor_(newContiguous)(state, tensor);
  }

  if (totalElements > 0) {
    if (THSYCLTensor_canUse32BitIndexMath(state, tensor) &&
        THSYCLTensor_canUse32BitIndexMath(state, src) &&
        THSYCLTensor_canUse32BitIndexMath(state, index)) {
      TensorInfo<scalar_t, unsigned int> tensorInfo =
        getTensorInfo<scalar_t, THSYCLTensor, unsigned int>(state, tensor);
      TensorInfo<scalar_t, unsigned int> srcInfo =
        getTensorInfo<scalar_t, THSYCLTensor, unsigned int>(state, src);
      TensorInfo<int64_t, unsigned int> indexInfo =
        getTensorInfo<int64_t, THSyclLongTensor, unsigned int>(state, index);
      // Specialize for a small number of dimensions.
      switch (indexInfo.dims) {
        case 1:
          RUN(unsigned int, 1, scalar_t);
          break;
        case 2:
          RUN(unsigned int, 2, scalar_t);
          break;
        case 3:
          RUN(unsigned int, 3, scalar_t);
          break;
        default:
          RUN(unsigned int, -1, scalar_t);
          break;
      }
    } else {
      TensorInfo<scalar_t, uint64_t> tensorInfo =
        getTensorInfo<scalar_t, THSYCLTensor, uint64_t>(state, tensor);
      TensorInfo<scalar_t, uint64_t> srcInfo =
        getTensorInfo<scalar_t, THSYCLTensor, uint64_t>(state, src);
      TensorInfo<int64_t, uint64_t> indexInfo =
        getTensorInfo<int64_t, THSyclLongTensor, uint64_t>(state, index);

      RUN(uint64_t, -1, scalar_t)
    }
  }

  if (oldTensor) {
    THSYCLTensor_copyIgnoringOverlaps<scalar_t>(state, oldTensor, tensor);
    THSYCLTensor_(free)(state, tensor);
    tensor = oldTensor;
  }
  
}

#undef RUN

#define RUN(TYPE, DIMS, REAL)                                           \
  THSyclTensor_scatterAddKernel<TYPE, REAL, DIMS>(                               \
      tensorInfo, srcInfo, indexInfo, dim, (TYPE)totalElements);

void THSYCLTensor_(scatterAdd)(THSYCLState* state, THSYCLTensor* tensor,
    int dim, THSyclLongTensor* index, THSYCLTensor* src) {

#if defined(THSYCL_REAL_IS_FLOAT) || defined(THSYCL_REAL_IS_INT)
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, tensor, src));
  THSYCLAssertSameGPU(THSyclLongTensor_checkGPU(state, 1, index));

  THArgCheck(dim >= 0 && dim < THSYCLTensor_(nDimensionLegacyNoScalars)(state, tensor), 2,
             "Index dimension is out of bounds");
  int index_ndim_legacy_all = THSyclLongTensor_nDimensionLegacyAll(state, index);

  THArgCheck(index_ndim_legacy_all == 0
             || THSyclLongTensor_nDimensionLegacyNoScalars(state, index) == THSYCLTensor_(nDimensionLegacyNoScalars)(state, src), 3,
             "Index tensor must either be empty or have same dimensions as input tensor");
  THArgCheck(THSYCLTensor_(nDimensionLegacyNoScalars)(state, src) == THSYCLTensor_(nDimensionLegacyNoScalars)(state, tensor), 4,
             "Input tensor must have same dimensions as output tensor");


  // no-op if index is empty
  if (index_ndim_legacy_all == 0)
      return;

  for (int d = 0; d < THSYCLTensor_(nDimensionLegacyNoScalars)(state, tensor); d++) {
    int64_t indexSizeD = THSyclLongTensor_sizeLegacyNoScalars(state, index, d);
    if (d != dim) {
      THArgCheck(indexSizeD <= THSYCLTensor_(sizeLegacyNoScalars)(state, tensor, d), 3,
                 "Index tensor must not have larger size than output tensor apart from the specified dimension %d, but got index %s output %s",
                 dim, THSyclLongTensor_sizeDesc(state, index).str, THSYCLTensor_(sizeDesc)(state, tensor).str);
    }
    THArgCheck(indexSizeD <= THSYCLTensor_(sizeLegacyNoScalars)(state, src, d), 3,
               "Index tensor must not have larger size than input tensor, but got index %s input %s",
               THSyclLongTensor_sizeDesc(state, index).str, THSYCLTensor_(sizeDesc)(state, src).str);
  }

  THArgCheck(THSYCLTensor_(nDimensionLegacyNoScalars)(state, tensor) <= MAX_SYCLTORCH_DIMS,
             1, SYCLTORCH_DIM_WARNING);  

  const ptrdiff_t totalElements = THSyclLongTensor_nElement(state, index);

  THSYCLTensor* oldTensor = NULL;
  if (THSYCLTensor_maybeOverlappingIndices(state, tensor)) {
    oldTensor = tensor;
    tensor = THSYCLTensor_(newContiguous)(state, tensor);
  }

  if (totalElements > 0) {
    if (THSYCLTensor_canUse32BitIndexMath(state, tensor) &&
        THSYCLTensor_canUse32BitIndexMath(state, src) &&
        THSYCLTensor_canUse32BitIndexMath(state, index)) {
      TensorInfo<scalar_t, unsigned int> tensorInfo =
        getTensorInfo<scalar_t, THSYCLTensor, unsigned int>(state, tensor);
      TensorInfo<scalar_t, unsigned int> srcInfo =
        getTensorInfo<scalar_t, THSYCLTensor, unsigned int>(state, src);
      TensorInfo<int64_t, unsigned int> indexInfo =
        getTensorInfo<int64_t, THSyclLongTensor, unsigned int>(state, index);

      // Specialize for a small number of dimensions.
      switch (indexInfo.dims) {
        case 1:
          RUN(unsigned int, 1, scalar_t);
          break;
        case 2:
          RUN(unsigned int, 2, scalar_t);
          break;
        case 3:
          RUN(unsigned int, 3, scalar_t);
          break;
        default:
          RUN(unsigned int, -1, scalar_t);
          break;
      }
    } else {
      TensorInfo<scalar_t, uint64_t> tensorInfo =
        getTensorInfo<scalar_t, THSYCLTensor, uint64_t>(state, tensor);
      TensorInfo<scalar_t, uint64_t> srcInfo =
        getTensorInfo<scalar_t, THSYCLTensor, uint64_t>(state, src);
      TensorInfo<int64_t, uint64_t> indexInfo =
        getTensorInfo<int64_t, THSyclLongTensor, uint64_t>(state, index);

      RUN(uint64_t, -1, scalar_t)
    }
  }

  if (oldTensor) {
    THSYCLTensor_copyIgnoringOverlaps<scalar_t>(state, oldTensor, tensor);
    THSYCLTensor_(free)(state, tensor);
    tensor = oldTensor;
  }  
#else
  AT_ERROR("scatter_add only supported for float and integer type\n");
#endif
}

#undef RUN

void THSYCLTensor_(scatterFill)(THSYCLState* state, THSYCLTensor* tensor,
    int dim, THSyclLongTensor* index, scalar_t value) {
  AT_ERROR("not implemented THSYCLTensor_scatterFill\n");
}
#endif
