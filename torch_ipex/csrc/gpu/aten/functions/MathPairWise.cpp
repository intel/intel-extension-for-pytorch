#include <core/SYCL.h>
#include <utils/Pairwise.h>


using namespace at::native;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {
} // impl

IPEX_OUT_INT_CALLABLE_1_UNARY_OPS(__and___out, TensorBitAndConstantOp);

Tensor __and__(const Tensor & self, Scalar other) {
  auto result = at::empty_like(self);
  return at::AtenIpexTypeDPCPP::__and___out(result, self, other);
}

Tensor & __iand__(Tensor & self, Scalar other) {
  return at::AtenIpexTypeDPCPP::__and___out(self, self, other);
}

IPEX_OUT_INT_CALLABLE_1_UNARY_OPS(__or___out, TensorBitOrConstantOp);

Tensor __or__(const Tensor & self, Scalar other) {
  auto result = at::empty_like(self);
  return at::AtenIpexTypeDPCPP::__or___out(result, self, other);
}

Tensor & __ior__(Tensor & self, Scalar other) {
  return at::AtenIpexTypeDPCPP::__or___out(self, self, other);
}

} // AtenIpexTypeDPCPP
} // at
