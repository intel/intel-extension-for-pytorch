#include <core/TensorImplUtils.h>
#include <core/SYCLApplyUtils.h>
#include <utils/Numerics.h>
#include <utils/Pointwise.h>
#include <utils/General.h>


using namespace at::native;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

template <typename scalar_t>
typename std::enable_if<!IS_FLOAT(scalar_t), void>::type
pow(Tensor & self_, const Tensor & src, scalar_t value) {
  if (TensorImpl_Unwrap(self_) == TensorImpl_Unwrap(src)) {
    if (Numerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(1))) {
      sycl::SYCL_tensor_apply1<scalar_t>(self_, TensorPowOp<scalar_t, 1>(value));
    } else if (Numerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(2))) {
      sycl::SYCL_tensor_apply1<scalar_t>(self_, TensorPowOp<scalar_t, 2>(value));
    } else if (Numerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(3))) {
      sycl::SYCL_tensor_apply1<scalar_t>(self_, TensorPowOp<scalar_t, 3>(value));
    } else {
      sycl::SYCL_tensor_apply1<scalar_t>(self_, TensorPowOp<scalar_t, -3>(value));
    }
  } else {
    self_.resize_as_(src);
    if (Numerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(1))) {
      sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(self_, src, TensorPowOp<scalar_t, 1>(value));
    } else if (Numerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(2))) {
      sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(self_, src, TensorPowOp<scalar_t, 2>(value));
    } else if (Numerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(3))) {
      sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(self_, src, TensorPowOp<scalar_t, 3>(value));
    } else {
      sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(self_, src, TensorPowOp<scalar_t, -3>(value));
    }
  }
}

template <typename scalar_t>
typename std::enable_if<IS_FLOAT(scalar_t), void>::type
pow(Tensor & self_, const Tensor & src, scalar_t value) {
  if (TensorImpl_Unwrap(self_) == TensorImpl_Unwrap(src)) {
    if (Numerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(1))) {
      sycl::SYCL_tensor_apply1<scalar_t>(self_, TensorPowOp<scalar_t, 1>(value));
    } else if (Numerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(2))) {
      sycl::SYCL_tensor_apply1<scalar_t>(self_, TensorPowOp<scalar_t, 2>(value));
    } else if (Numerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(3))) {
      sycl::SYCL_tensor_apply1<scalar_t>(self_, TensorPowOp<scalar_t, 3>(value));
    } else if (Numerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(-1))) {
      sycl::SYCL_tensor_apply1<scalar_t>(self_, TensorPowOp<scalar_t, -1>(value));
    } else if (Numerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(-2))) {
      sycl::SYCL_tensor_apply1<scalar_t>(self_, TensorPowOp<scalar_t, -2>(value));
    } else {
      sycl::SYCL_tensor_apply1<scalar_t>(self_, TensorPowOp<scalar_t, -3>(value));
    }
  } else {
    self_.resize_as_(src);
    if (Numerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(1))) {
      sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(self_, src, TensorPowOp<scalar_t, 1>(value));
    } else if (Numerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(2))) {
      sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(self_, src, TensorPowOp<scalar_t, 2>(value));
    } else if (Numerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(3))) {
      sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(self_, src, TensorPowOp<scalar_t, 3>(value));
    } else if (Numerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(-1))) {
      sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(self_, src, TensorPowOp<scalar_t, -1>(value));
    } else if (Numerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(-2))) {
      sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(self_, src, TensorPowOp<scalar_t, -2>(value));
    } else {
        sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(self_, src, TensorPowOp<scalar_t, -3>(value));
    }
  }
}

template <typename scalar_t>
void cpow(Tensor & self_, const Tensor & src1, const Tensor & src2) {
  TORCH_CHECK(src1.numel() == src2.numel(), "sizes do not match");
  if (TensorImpl_Unwrap(self_) == TensorImpl_Unwrap(src1)) {
    sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(self_, src2, TensorCPowOp<scalar_t>());
  } else {
    self_.resize_as_(src1);
    sycl::SYCL_tensor_apply3<scalar_t, scalar_t, scalar_t>(self_, src1, src2, TensorCPowOp<scalar_t>());
  }
}

template <typename scalar_t>
void tpow(Tensor & self_, scalar_t value, const Tensor & src) {
  if (TensorImpl_Unwrap(self_) == TensorImpl_Unwrap(src)) {
    sycl::SYCL_tensor_apply1<scalar_t>(self_, TensorTPowOp<scalar_t>(value));
  } else {
    self_.resize_as_(src);
    sycl::SYCL_tensor_apply2<scalar_t, scalar_t>(self_, src, TensorTPowOp<scalar_t>(value));
  }
}
} // namespace impl

Tensor & pow_out(Tensor & out, const Tensor & self, Scalar exponent) {
  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "pow",
      [&]() {
        impl::pow<scalar_t>(out, self, exponent.to<scalar_t>());
      }
  );
  return out;
}

Tensor pow(const Tensor & self, Scalar exponent) {
  auto result = at::empty({0}, self.options());
  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "pow",
      [&]() {
        impl::pow<scalar_t>(result, self, exponent.to<scalar_t>());
      }
  );
  return result;
}

Tensor & pow_(Tensor & self, Scalar exponent) {
  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "pow",
      [&]() {
        impl::pow<scalar_t>(self, self, exponent.to<scalar_t>());
      }
  );
  return self;
}

Tensor & pow_(Tensor & self, const Tensor & exponent) {
  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "cpow",
      [&]() {
        impl::cpow<scalar_t>(self, self, exponent);
      }
  );
  return self;
}

Tensor & pow_out(Tensor & out, const Tensor & self, const Tensor & exponent) {
  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "cpow",
      [&]() {
        impl::cpow<scalar_t>(out, self, exponent);
      }
  );
  return out;
}

Tensor pow(const Tensor & self, const Tensor & exponent) {
  auto result = at::empty({0}, self.options());
  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "cpow",
      [&]() {
        impl::cpow<scalar_t>(result, self, exponent);
      }
  );
  return result;
}

Tensor & pow_out(Tensor & out, Scalar self, const Tensor & exponent) {
  AT_DISPATCH_ALL_TYPES(out.scalar_type(), "tpow",
      [&]() {
        impl::tpow<scalar_t>(out, self.to<scalar_t>(), exponent);
      }
  );
  return out;
}

Tensor pow(Scalar self, const Tensor & exponent) {
  auto result = at::empty({0}, exponent.options());
  AT_DISPATCH_ALL_TYPES(exponent.scalar_type(), "tpow",
      [&]() {
        impl::tpow<scalar_t>(result, self.to<scalar_t>(), exponent);
      }
  );
  return result;
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
