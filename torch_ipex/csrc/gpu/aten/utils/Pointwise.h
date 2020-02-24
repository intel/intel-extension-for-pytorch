#include <ATen/ATen.h>

#include <core/TensorImplUtils.h>
#include <core/SYCLApplyUtils.h>
#include <ATen/aten_ipex_type_dpcpp.h>


#define IMPLEMENT_POINTWISE_FUNC_(NAME, CFUNC, REAL)            \
  template <typename scalar_t>                                     \
  struct Tensor_##NAME##_##REAL##_Op {                             \
    inline void operator()(scalar_t& out, scalar_t& in) const {    \
      out = CFUNC(in);                                             \
    }                                                              \
                                                                   \
    inline void operator()(scalar_t& v) const {                    \
      v = CFUNC(v);                                                \
    }                                                              \
  };                                                               \
                                                                   \
  template <typename scalar_t>                                     \
  void NAME(Tensor& self_, const Tensor& src) {                    \
    if (TensorImpl_Unwrap(self_) == TensorImpl_Unwrap(src)) {      \
      at::sycl::SYCL_tensor_apply1<scalar_t>(                      \
          self_, Tensor_##NAME##_##REAL##_Op<scalar_t>());         \
    } else {                                                       \
      at::AtenIpexTypeDPCPP::resize_as_(self_, src, c10::nullopt); \
      at::sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(            \
          self_, src, Tensor_##NAME##_##REAL##_Op<scalar_t>());    \
    }                                                              \
  }

#define IMPLEMENT_POINTWISE_FUNC(NAME, CFUNC, REAL)                \
  IMPLEMENT_POINTWISE_FUNC_(NAME, CFUNC, REAL)
