#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

#include <core/SYCLContext.h>
#include <core/SYCLMemory.h>
#include <core/SYCLUtils.h>
#include <core/SYCLApplyUtils.h>

using namespace at::sycl;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

template <typename scalar_t, typename scalar1_t>
class where_functor {
 public:
  where_functor() {}
  void operator()(
      scalar_t& ret_val,
      const scalar1_t& cond_val,
      const scalar_t& self_val,
      const scalar_t& other_val) const {
    ret_val = cond_val ? self_val : other_val;
  }
};

template <typename scalar_t>
void _s_where(
    at::Tensor& ret,
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other) {
  if (condition.scalar_type() == at::ScalarType::Byte) {
    SYCL_tensor_apply4<scalar_t, uint8_t, scalar_t, scalar_t>(
        ret, condition, self, other, where_functor<scalar_t, uint8_t>());
  } else {
    SYCL_tensor_apply4<scalar_t, bool, scalar_t, scalar_t>(
        ret, condition, self, other, where_functor<scalar_t, bool>());
  }
}

} // namespace impl

Tensor _s_where(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other) {
  Tensor ret = at::empty(self.sizes(), self.options());
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::Bool, ret.scalar_type(), "where", [&] {
    impl::_s_where<scalar_t>(ret, condition, self, other);
  });
  return ret;
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
