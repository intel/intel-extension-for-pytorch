#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

#include <utils/Numerics.h>
#include <utils/Pointwise.h>


using namespace at::native;

namespace at {
namespace AtenIpexTypeDPCPP {

#define DEFINE_IPEX_OUT_OPS(op, func, real)                                   \
  namespace impl {                                                            \
    IMPLEMENT_POINTWISE_FUNC(op, func, real)                                  \
  }                                                                           \
                                                                              \
  Tensor & op(Tensor & out, const Tensor & self) {                            \
    AT_DISPATCH_ALL_TYPES(self.scalar_type(), #op,                            \
        [&]() {                                                               \
          impl::op<scalar_t>(out, self);                                      \
        }                                                                     \
    );                                                                        \
    return out;                                                               \
  }

DEFINE_IPEX_OUT_OPS(abs_out, Numerics<scalar_t>::abs, Real);

} // namespace AtenIpexTypeDPCPP
} // namespace at
