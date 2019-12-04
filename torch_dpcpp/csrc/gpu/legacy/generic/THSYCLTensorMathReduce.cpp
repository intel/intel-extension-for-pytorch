#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "THDP/generic/THSYCLTensorMathReduce.cpp"
#else

scalar_t THSYCLTensor_(maxall)(THSYCLState *state, THSYCLTensor *self) {
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 1, self));
  accreal val;
  if (!THSYCL_reduceAll<scalar_t>(state, self,
                                  sycl_identity<accreal>{},
                                  ReduceMax<accreal>{},
                                  THSYCLNumerics<accreal>::lower_bound(), &val, 0)) {
    THArgCheck(false, 1, SYCLTORCH_DIM_WARNING);
  }
  return val;
}

scalar_t THSYCLTensor_(minall)(THSYCLState *state, THSYCLTensor *self) {
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 1, self));
  accreal val;
  if (!THSYCL_reduceAll<scalar_t>(state, self,
                                  sycl_identity<accreal>{},
                                  ReduceMin<accreal>{},
                                  THSYCLNumerics<accreal>::upper_bound(), &val, 0)) {
    THArgCheck(false, 1, SYCLTORCH_DIM_WARNING);
  }
  return val;
}

accreal THSYCLTensor_(sumall)(THSYCLState *state, THSYCLTensor *self) {
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 1, self));
  accreal val;
  if (!THSYCL_reduceAll<scalar_t>(state, self,
                                  sycl_identity<accreal>{},
                                  ReduceAdd<accreal>{},
                                  scalar_cast<accreal>(0), &val, 0)) {
    THArgCheck(false, 1, SYCLTORCH_DIM_WARNING);
  }
  return val;
}

void THSYCLTensor_(max)(THSYCLState *state,
                     THSYCLTensor *values,
                     THSyclLongTensor *indices,
                     THSYCLTensor *src,
                     int dimension,
                     int keepdim) {
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 3, values, indices, src));
  std::pair<scalar_t, int64_t>
    init =
    std::make_pair<scalar_t, int64_t>(
      THSYCLNumerics<scalar_t>::lower_bound(), 0);

  return THSYCL_reduceDimIndex<scalar_t, int64_t>(
    state, values, indices, src, dimension, keepdim, init,
    MaxValuePair<scalar_t, int64_t>());
}

void THSYCLTensor_(min)(THSYCLState *state,
                     THSYCLTensor *values,
                     THSyclLongTensor *indices,
                     THSYCLTensor *src,
                     int dimension,
                     int keepdim) {
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 3, values, indices, src));

  std::pair<scalar_t, int64_t>
    init =
    std::make_pair<scalar_t, int64_t>(
      THSYCLNumerics<scalar_t>::upper_bound(), 0);

  return THSYCL_reduceDimIndex<scalar_t, int64_t>(
    state, values, indices, src, dimension, keepdim, init,
    MinValuePair<scalar_t, int64_t>());
}

accreal THSYCLTensor_(dist)(THSYCLState* state, THSYCLTensor* self,
                            THSYCLTensor* src, scalar_t _value) {
  AT_ERROR("not implemented THSYCLTensor_dist\n");
}

scalar_t THSYCLTensor_(medianall)(THSYCLState *state, THSYCLTensor *self) {
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 1, self));

  scalar_t val;
  ptrdiff_t nelem, k;

  nelem = THSYCLTensor_(nElement)(state, self);
  k = (nelem-1) >> 1;

  THSYCLTensor *view = THSYCLTensor_(newView)(state, self, {nelem});

  THSYCLTensor *sorted = THSYCLTensor_(new)(state);
  THSyclLongTensor *indices = THSyclLongTensor_new(state);

  THSYCLTensor_(sort)(state, sorted, indices, view, 0, 0);

  val = THSYCLTensor_(get1d)(state, sorted, k);

  THSYCLTensor_(free)(state, view);
  THSYCLTensor_(free)(state, sorted);
  THSyclLongTensor_free(state, indices);

  return val;
}

void THSYCLTensor_(median)(THSYCLState *state,
                        THSYCLTensor *values,
                        THSyclLongTensor *indices,
                        THSYCLTensor *self,
                        int dimension,
                        int keepdim) {
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 1, self));

  int64_t t_size_dim, k;

  t_size_dim = THSYCLTensor_(size)(state, self, dimension);

  k = (t_size_dim-1) >> 1;

  THSYCLTensor *sorted = THSYCLTensor_(new)(state);
  THSyclLongTensor *sorted_indices = THSyclLongTensor_new(state);

  THSYCLTensor_(sort)(state, sorted, sorted_indices, self, dimension, 0);

  THSYCLTensor *newValues = THSYCLTensor_(newNarrow)(state, sorted, dimension, k, 1);
  THSyclLongTensor *newIndices = THSyclLongTensor_newNarrow(state, sorted_indices, dimension, k, 1);

  THSYCLTensor_(free)(state, sorted);
  THSyclLongTensor_free(state, sorted_indices);

  if (!keepdim) {
    THSYCLTensor_(squeeze1d)(state, newValues, newValues, dimension);
    THSyclLongTensor_squeeze1d(state, newIndices, newIndices, dimension);
  }

  THSYCLTensor_(resizeAs)(state, values, newValues);
  THSyclLongTensor_resizeAs(state, indices, newIndices);
  THSYCLTensor_(copy)(state, values, newValues);
  THSyclLongTensor_copy(state, indices, newIndices);

  THSYCLTensor_(free)(state, newValues);
  THSyclLongTensor_free(state, newIndices);
}


#endif
