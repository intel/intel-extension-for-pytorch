#ifndef __THSYCLTENSORMATHPAIRWISE_HPP__
#define __THSYCLTENSORMATHPAIRWISE_HPP__

#define IMPLEMENT_SYCL_TENSOR_BASIC_FUNC_(NAME, CFUNC, REAL)                             \
  struct Tensor_##NAME##_##REAL##_Op {                                                   \
    inline void operator()(scalar_t& out, scalar_t& in) const {                          \
      out = CFUNC(in);                                                                   \
    }                                                                                    \
                                                                                         \
    inline void operator()(scalar_t& v) const {                                          \
      v = CFUNC(v);                                                                      \
    }                                                                                    \
  };                                                                                     \
                                                                                         \
  void THSYCLTensor_(NAME)(THSYCLState* state, THSYCLTensor* self_, THSYCLTensor* src) { \
    if (self_ == src) {                                                                  \
      at::sycl::SYCL_tensor_apply1<scalar_t>(                                            \
          THTensor_wrap(self_), Tensor_##NAME##_##REAL##_Op());                          \
    } else {                                                                             \
      THSYCLTensor_(resizeAs)(state, self_, src);                                        \
      at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(                                  \
          THTensor_wrap(self_), THTensor_wrap(src), Tensor_##NAME##_##REAL##_Op());      \
    }                                                                                    \
  }

#define IMPLEMENT_SYCL_TENSOR_BASIC_FUNC(NAME, CFUNC, REAL) \
  IMPLEMENT_SYCL_TENSOR_BASIC_FUNC_(NAME, CFUNC, REAL)

#endif // __THSYCLTENSORMATHPAIRWISE_HPP__
