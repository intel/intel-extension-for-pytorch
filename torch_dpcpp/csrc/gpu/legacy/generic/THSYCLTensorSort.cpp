#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "THDP/generic/THSYCLTensorSort.cpp"
#else

// In alignment with default sort on a c++ map, this function
// will permute key and value tensors identically, and
// in such a way that the 'key' tensor is ordered numerically
void THSYCLTensor_(sortKeyValueInplace)(THSYCLState* state,
                                     THSYCLTensor* key,
                                     THSyclLongTensor* value,
                                     int dim, bool dir) {
  THArgCheck(key->sizes().equals(value->sizes()), 2,
             "Key tensor must have same size as value tensor");
  int dims = THSyclLongTensor_nDimensionLegacyNoScalars(state, value);
  THArgCheck(dims <= MAX_SYCLTORCH_DIMS, 3, SYCLTORCH_DIM_WARNING);
  dims = THSYCLTensor_(nDimensionLegacyNoScalars)(state, key);
  THArgCheck(dims <= MAX_SYCLTORCH_DIMS, 2, SYCLTORCH_DIM_WARNING);

  ptrdiff_t inElements = THSYCLTensor_(nElement)(state, key);

  if (inElements == 0) {
    return;
  }

  int64_t keySliceSize = THSYCLTensor_(sizeLegacyNoScalars)(state, key, dim);
  ptrdiff_t keySlices = inElements / keySliceSize;

  // The amount of shared memory and block size is based on
  // 2^ceil(lg(n)); we choose that sorting implementation for a given
  // size.
  int64_t ceilPowerOf2 = nextHighestPowerOf2(keySliceSize);

  // FIXME: We'd have to find some other trick with Thrust to perform a
  // vectorized (key, value) sort by slice segment
  if (ceilPowerOf2 > 2048) {
    THError("sortKeyValueInplace only works for sizes <= 2048 at present");
  }

#define HANDLE_CASE(TYPE, A, SIZE)                                      \
  do {                                                                  \
    int blockSize = SIZE / 2;                                           \
    if (blockSize < 1) {                                                \
      blockSize = 1;                                                    \
    }                                                                   \
                                                                        \
                                                                        \
    if (dir) {                                                          \
      bitonicSortKVInPlace<scalar_t, int64_t, A, -1, GTComp<scalar_t>, TYPE, SIZE>(  \
          keyInfo,                                                      \
          keySlices,                                                    \
          (TYPE) keySliceSize,                                          \
          (TYPE) keyInfo.strides[collapseKeyDim],                       \
          valueInfo,                                                    \
          (TYPE) valueInfo.strides[collapseValueDim],                   \
          GTComp<scalar_t>());                                              \
    } else {                                                            \
      bitonicSortKVInPlace<scalar_t, int64_t, A, -1, LTComp<scalar_t>, TYPE, SIZE> ( \
          keyInfo,                                                      \
          keySlices,                                                    \
          (TYPE) keySliceSize,                                          \
          (TYPE) keyInfo.strides[collapseKeyDim],                       \
          valueInfo,                                                    \
          (TYPE) valueInfo.strides[collapseValueDim],                   \
          LTComp<scalar_t>());                                              \
    }                                                                   \
  } while (0)

#define HANDLE_SORT_CASE(TYPE, A)                       \
  {                                                     \
    switch (ceilPowerOf2) {                             \
      case 2048:                                        \
      case 1024:                                        \
      case 512:                                         \
      case 256:                                         \
      HANDLE_CASE(TYPE, A, 512);                       \
      break;                                            \
      case 128:                                         \
      case 64:                                          \
      HANDLE_CASE(TYPE, A, 128);                        \
      break;                                            \
      case 32:                                          \
      case 16:                                          \
      case 8:                                           \
      case 4:                                           \
      case 2:                                           \
      HANDLE_CASE(TYPE, A, 32);                         \
      break;                                            \
      case 1:                                           \
      /* Nothing to do, data already sorted */          \
      break;                                            \
      default:                                          \
      assert(false);                                    \
    }                                                   \
  }

  // The constructed key/value tensor info is used to select the slice
  // we are sorting on a per-block basis
  if (THSYCLTensor_canUse32BitIndexMath(state, key)) {
    TensorInfo<scalar_t, unsigned int> keyInfo =
      getTensorInfo<scalar_t, THSYCLTensor, unsigned int>(state, key);
    keyInfo.reduceDim(dim);
    int collapseKeyDim = keyInfo.collapseDims(dim);

    TensorInfo<int64_t, unsigned int> valueInfo =
      getTensorInfo<int64_t, THSyclLongTensor, unsigned int>(state, value);
    valueInfo.reduceDim(dim);
    int collapseValueDim = valueInfo.collapseDims(dim);

    if (keyInfo.isContiguous()) {
      HANDLE_SORT_CASE(unsigned int, -2);
    } else {
      switch (keyInfo.dims) {
        case 2:
          HANDLE_SORT_CASE(unsigned int, 2);
          break;
        default:
          HANDLE_SORT_CASE(unsigned int, -1);
          break;
      }
    }
  } else {
    TensorInfo<scalar_t, uint64_t> keyInfo =
      getTensorInfo<scalar_t, THSYCLTensor, uint64_t>(state, key);
    keyInfo.reduceDim(dim);
    int collapseKeyDim = keyInfo.collapseDims(dim);

    TensorInfo<int64_t, uint64_t> valueInfo =
      getTensorInfo<int64_t, THSyclLongTensor, uint64_t>(state, value);
    valueInfo.reduceDim(dim);
    int collapseValueDim = valueInfo.collapseDims(dim);

    // int64_t case is rare, just instantiate the generic version
    HANDLE_SORT_CASE(uint64_t, -1);
  }
#undef HANDLE_CASE
#undef HANDLE_SORT_CASE
#undef HANDLE_A_CASE

}


void THSYCLTensor_(sort)(THSYCLState *state,
                         THSYCLTensor *sorted,
                         THSyclLongTensor *indices,
                         THSYCLTensor *input,
                         int dim, int order) {
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 2, sorted, input));
  THSYCLAssertSameGPU(THSyclLongTensor_checkGPU(state, 1, indices));
  int64_t dims = THSYCLTensor_(nDimensionLegacyNoScalars)(state, sorted);
  THArgCheck(dims <= MAX_SYCLTORCH_DIMS, 2, SYCLTORCH_DIM_WARNING);
  dims = THSYCLTensor_(nDimensionLegacyNoScalars)(state, input);
  THArgCheck(dims <= MAX_SYCLTORCH_DIMS, 4, SYCLTORCH_DIM_WARNING);
  dims = THSyclLongTensor_nDimensionLegacyNoScalars(state, indices);
  THArgCheck(dims <= MAX_SYCLTORCH_DIMS, 3, SYCLTORCH_DIM_WARNING);

  // Make sure sufficient output space is allocated
  THSYCLTensor_(resizeAs)(state, sorted, input);
  THSyclLongTensor_resize(state, indices, input->sizes(), {});

  // How large are the slices that we are sorting?
  int64_t sliceSize = THSYCLTensor_(sizeLegacyNoScalars)(state, input, dim);
  int maxSliceSize = 2048;
  if (sliceSize <= maxSliceSize) { // inplace sort
    // Fill `indices` (the values) with the
    // slice-relative index.
    THSyclLongTensor_fillSliceWithIndex(state, indices, dim);

    // We sort k/v pairs in-place; copy unsorted input to output
    THSYCLTensor_(copy)(state, sorted, input);

    // Sort using our in-place k/v kernel that supports arbitrary
    // layout
    THSYCLTensor_(sortKeyValueInplace)(state, sorted, indices, dim, order);

  } else {
  }

}


#endif
