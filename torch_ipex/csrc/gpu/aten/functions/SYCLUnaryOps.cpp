#include <ATen/ATen.h>
#include <ATen/LegacyTHFunctionsSYCL.h>

namespace at { namespace native {

#define IMPLEMENT_UNARY_OP_PREQUEL(op)                           \
  Tensor& _##op##__sycl(Tensor& self) {                          \
    return legacy::sycl::_th_##op##_out(self, self);             \
  }                                                              \
  Tensor& _##op##_out_sycl(Tensor& result, const Tensor& self) { \
    return legacy::sycl::_th_##op##_out(result, self);           \
  }

IMPLEMENT_UNARY_OP_PREQUEL(atan)
IMPLEMENT_UNARY_OP_PREQUEL(cos)
IMPLEMENT_UNARY_OP_PREQUEL(cosh)
IMPLEMENT_UNARY_OP_PREQUEL(erf)
IMPLEMENT_UNARY_OP_PREQUEL(erfc)
IMPLEMENT_UNARY_OP_PREQUEL(exp)
IMPLEMENT_UNARY_OP_PREQUEL(log)
IMPLEMENT_UNARY_OP_PREQUEL(log10)
IMPLEMENT_UNARY_OP_PREQUEL(log1p)
IMPLEMENT_UNARY_OP_PREQUEL(log2)
IMPLEMENT_UNARY_OP_PREQUEL(rsqrt)
IMPLEMENT_UNARY_OP_PREQUEL(sin)
IMPLEMENT_UNARY_OP_PREQUEL(sinh)
IMPLEMENT_UNARY_OP_PREQUEL(sqrt)
IMPLEMENT_UNARY_OP_PREQUEL(tan)
IMPLEMENT_UNARY_OP_PREQUEL(tanh)

}}
