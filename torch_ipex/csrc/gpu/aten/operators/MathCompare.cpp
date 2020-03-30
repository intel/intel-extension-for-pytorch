#include <ATen/Functions.h>
#include <ATen/ScalarOps.h>

#include <ATen/aten_ipex_type_dpcpp.h>
#include <core/ApplyUtils.h>
#include <utils/Numerics.h>

using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

namespace {

template <typename T, typename TOut>
struct TensorLTOp {
  inline void operator()(TOut& out, T& a, T& b) const {
    out = ScalarConvert<bool, TOut>::to(Numerics<T>::lt(a, b));
  }
};

template <typename T, typename TOut>
struct TensorGTOp {
  inline void operator()(TOut& out, T& a, T& b) const {
    out = ScalarConvert<bool, TOut>::to(Numerics<T>::gt(a, b));
  }
};

template <typename T, typename TOut>
struct TensorLEOp {
  inline void operator()(TOut& out, T& a, T& b) const {
    out = ScalarConvert<bool, TOut>::to(Numerics<T>::le(a, b));
  }
};

template <typename T, typename TOut>
struct TensorGEOp {
  inline void operator()(TOut& out, T& a, T& b) const {
    out = ScalarConvert<bool, TOut>::to(Numerics<T>::ge(a, b));
  }
};

template <typename T, typename TOut>
struct TensorEQOp {
  void operator()(TOut& out, T& a, T& b) const {
    out = ScalarConvert<bool, TOut>::to(Numerics<T>::eq(a, b));
  }
};

template <typename T, typename TOut>
struct TensorNEOp {
  inline void operator()(TOut& out, T& a, T& b) const {
    out = ScalarConvert<bool, TOut>::to(Numerics<T>::ne(a, b));
  }
};

template <typename ScalarTypeOut, typename ScalarType, class Op>
void logicalTensor(
    Tensor& self_,
    const Tensor& src1,
    const Tensor& src2,
    Op op) {
  at::AtenIpexTypeDPCPP::resize_as_(self_, src1, c10::nullopt);

  TORCH_CHECK(src1.numel() == src2.numel(), "sizes do not match");
  at::dpcpp::DPCPP_tensor_apply3<ScalarTypeOut, ScalarType, ScalarType>(
      self_, src1, src2, op);
}

} // namespace

template <typename scalar_t>
void ltTensor(Tensor& self_, const Tensor& src1, const Tensor& src2) {
  logicalTensor<bool, scalar_t>(
      self_, src1, src2, TensorLTOp<scalar_t, bool>());
}

template <typename scalar_t>
void gtTensor(Tensor& self_, const Tensor& src1, const Tensor& src2) {
  logicalTensor<bool, scalar_t>(
      self_, src1, src2, TensorGTOp<scalar_t, bool>());
}

template <typename scalar_t>
void leTensor(Tensor& self_, const Tensor& src1, const Tensor& src2) {
  logicalTensor<bool, scalar_t>(
      self_, src1, src2, TensorLEOp<scalar_t, bool>());
}

template <typename scalar_t>
void geTensor(Tensor& self_, const Tensor& src1, const Tensor& src2) {
  logicalTensor<bool, scalar_t>(
      self_, src1, src2, TensorGEOp<scalar_t, bool>());
}

template <typename scalar_t>
void eqTensor(Tensor& self_, const Tensor& src1, const Tensor& src2) {
  logicalTensor<bool, scalar_t>(
      self_, src1, src2, TensorEQOp<scalar_t, bool>());
}

template <typename scalar_t>
void neTensor(Tensor& self_, const Tensor& src1, const Tensor& src2) {
  logicalTensor<bool, scalar_t>(
      self_, src1, src2, TensorNEOp<scalar_t, bool>());
}

#if COMPARE_PORTED
void THDPCPPTensor_(neTensor)(
    THDPCPPState* state,
    THSyclBoolTensor* self_,
    THDPCPPTensor* src1,
    THDPCPPTensor* src2) {
  THDPCPPAssertSameGPU(THDPCPPTensor_(checkGPU)(state, 3, self_, src1, src2));
  THDPCPP_logicalTensor<bool, scalar_t>(
      state, self_, src1, src2, TensorNEOp<scalar_t, bool>());
}

void THDPCPPTensor_(ltTensorT)(
    THDPCPPState* state,
    THDPCPPTensor* self_,
    THDPCPPTensor* src1,
    THDPCPPTensor* src2) {
  THDPCPPAssertSameGPU(THDPCPPTensor_(checkGPU)(state, 3, self_, src1, src2));
  THDPCPP_logicalTensor<scalar_t, scalar_t>(
      state, self_, src1, src2, TensorLTOp<scalar_t, scalar_t>());
}

void THDPCPPTensor_(gtTensorT)(
    THDPCPPState* state,
    THDPCPPTensor* self_,
    THDPCPPTensor* src1,
    THDPCPPTensor* src2) {
  THDPCPPAssertSameGPU(THDPCPPTensor_(checkGPU)(state, 3, self_, src1, src2));
  THDPCPP_logicalTensor<scalar_t, scalar_t>(
      state, self_, src1, src2, TensorGTOp<scalar_t, scalar_t>());
}

void THDPCPPTensor_(leTensorT)(
    THDPCPPState* state,
    THDPCPPTensor* self_,
    THDPCPPTensor* src1,
    THDPCPPTensor* src2) {
  THDPCPPAssertSameGPU(THDPCPPTensor_(checkGPU)(state, 3, self_, src1, src2));
  THDPCPP_logicalTensor<scalar_t, scalar_t>(
      state, self_, src1, src2, TensorLEOp<scalar_t, scalar_t>());
}

void THDPCPPTensor_(geTensorT)(
    THDPCPPState* state,
    THDPCPPTensor* self_,
    THDPCPPTensor* src1,
    THDPCPPTensor* src2) {
  THDPCPPAssertSameGPU(THDPCPPTensor_(checkGPU)(state, 3, self_, src1, src2));
  THDPCPP_logicalTensor<scalar_t, scalar_t>(
      state, self_, src1, src2, TensorGEOp<scalar_t, scalar_t>());
}
#endif

// void eqTensorT(Tensor *self_, THDPCPPTensor *src1, THDPCPPTensor *src2)
// {
//   THDPCPP_logicalTensor<scalar_t, scalar_t>(state, self_, src1, src2,
//                                 TensorEQOp<scalar_t,
//                                 scalar_t>());
// }

#if COMPARE_PORTED
void THDPCPPTensor_(neTensorT)(
    THDPCPPState* state,
    THDPCPPTensor* self_,
    THDPCPPTensor* src1,
    THDPCPPTensor* src2) {
  THDPCPPAssertSameGPU(THDPCPPTensor_(checkGPU)(state, 3, self_, src1, src2));
  THDPCPP_logicalTensor<scalar_t, scalar_t>(
      state, self_, src1, src2, TensorNEOp<scalar_t, scalar_t>());
}

void THDPCPPTensor_(ltTensorByte)(
    THDPCPPState* state,
    THSyclByteTensor* self_,
    THDPCPPTensor* src1,
    THDPCPPTensor* src2) {
  THDPCPPAssertSameGPU(THDPCPPTensor_(checkGPU)(state, 3, self_, src1, src2));
  THDPCPP_logicalTensor<unsigned char, scalar_t>(
      state, self_, src1, src2, TensorLTOp<scalar_t, unsigned char>());
}

void THDPCPPTensor_(gtTensorByte)(
    THDPCPPState* state,
    THSyclByteTensor* self_,
    THDPCPPTensor* src1,
    THDPCPPTensor* src2) {
  THDPCPPAssertSameGPU(THDPCPPTensor_(checkGPU)(state, 3, self_, src1, src2));
  THDPCPP_logicalTensor<unsigned char, scalar_t>(
      state, self_, src1, src2, TensorGTOp<scalar_t, unsigned char>());
}

void THDPCPPTensor_(leTensorByte)(
    THDPCPPState* state,
    THSyclByteTensor* self_,
    THDPCPPTensor* src1,
    THDPCPPTensor* src2) {
  THDPCPPAssertSameGPU(THDPCPPTensor_(checkGPU)(state, 3, self_, src1, src2));
  THDPCPP_logicalTensor<unsigned char, scalar_t>(
      state, self_, src1, src2, TensorLEOp<scalar_t, unsigned char>());
}

void THDPCPPTensor_(geTensorByte)(
    THDPCPPState* state,
    THSyclByteTensor* self_,
    THDPCPPTensor* src1,
    THDPCPPTensor* src2) {
  THDPCPPAssertSameGPU(THDPCPPTensor_(checkGPU)(state, 3, self_, src1, src2));
  THDPCPP_logicalTensor<unsigned char, scalar_t>(
      state, self_, src1, src2, TensorGEOp<scalar_t, unsigned char>());
}
#endif

// void THDPCPPTensor_(eqTensorByte)(THDPCPPState *state, THSyclByteTensor
// *self_, THDPCPPTensor *src1, THDPCPPTensor *src2)
// {
//   THDPCPPAssertSameGPU(THDPCPPTensor_(checkGPU)(state, 3, self_, src1,
//   src2));
//   THDPCPP_logicalTensor<unsigned char, scalar_t>(state, self_, src1, src2,
//                                              TensorEQOp<scalar_t,
//                                              unsigned char>());
// }

#if COMPARE_PORTED
void THDPCPPTensor_(neTensorByte)(
    THDPCPPState* state,
    THSyclByteTensor* self_,
    THDPCPPTensor* src1,
    THDPCPPTensor* src2) {
  THDPCPPAssertSameGPU(THDPCPPTensor_(checkGPU)(state, 3, self_, src1, src2));
  THDPCPP_logicalTensor<unsigned char, scalar_t>(
      state, self_, src1, src2, TensorNEOp<scalar_t, unsigned char>());
}
#endif
} // namespace impl

Tensor& lt_out(Tensor& out, const Tensor& self, Scalar other_) {
  auto other = c10::scalar_to_tensor(other_, kDPCPP);
  other.unsafeGetTensorImpl()->set_wrapped_number(true);
  // TODO: broadcast
  auto new_other =
      other.resize_as_(self).fill_(other_).toType(self.scalar_type());
  at::lt_out(out, self, new_other);
  return out;
}

Tensor lt(const Tensor& self, Scalar other_) {
  auto result = at::empty({0}, self.options().dtype(kBool));
  auto other = c10::scalar_to_tensor(other_, kDPCPP);
  other.unsafeGetTensorImpl()->set_wrapped_number(true);
  // TODO: broadcast
  auto new_other =
      other.resize_as_(self).fill_(other_).toType(self.scalar_type());
  return at::lt_out(result, self, new_other);
}

Tensor& lt_out(Tensor& out, const Tensor& self, const Tensor& other) {
  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "ltTensor", [&]() {
    impl::ltTensor<scalar_t>(out, self, other);
  });

  return out;
}

Tensor lt(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  return at::lt_out(result, self, other);
}

Tensor& gt_out(Tensor& out, const Tensor& self, Scalar other_) {
  auto other = c10::scalar_to_tensor(other_, kDPCPP);
  other.unsafeGetTensorImpl()->set_wrapped_number(true);
  // TODO: broadcast
  auto new_other =
      other.resize_as_(self).fill_(other_).toType(self.scalar_type());
  at::gt_out(out, self, new_other);
  return out;
}

Tensor gt(const Tensor& self, Scalar other_) {
  auto result = at::empty({0}, self.options().dtype(kBool));
  auto other = c10::scalar_to_tensor(other_, kDPCPP);
  other.unsafeGetTensorImpl()->set_wrapped_number(true);
  // TODO: broadcast
  auto new_other =
      other.resize_as_(self).fill_(other_).toType(self.scalar_type());
  at::gt_out(result, self, new_other);
  return result;
}

Tensor& gt_out(Tensor& out, const Tensor& self, const Tensor& other) {
  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "gtTensor", [&]() {
    impl::gtTensor<scalar_t>(out, self, other);
  });

  return out;
}

Tensor gt(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  return at::gt_out(result, self, other);
}

Tensor& ge_out(Tensor& out, const Tensor& self, Scalar other_) {
  auto other = c10::scalar_to_tensor(other_, kDPCPP);
  other.unsafeGetTensorImpl()->set_wrapped_number(true);
  // TODO: broadcast
  auto new_other =
      other.resize_as_(self).fill_(other_).toType(self.scalar_type());
  at::ge_out(out, self, new_other);
  return out;
}

Tensor ge(const Tensor& self, Scalar other_) {
  auto result = at::empty({0}, self.options().dtype(kBool));
  auto other = c10::scalar_to_tensor(other_, kDPCPP);
  other.unsafeGetTensorImpl()->set_wrapped_number(true);
  // TODO: broadcast
  auto new_other =
      other.resize_as_(self).fill_(other_).toType(self.scalar_type());
  return at::ge_out(result, self, new_other);
}

Tensor& ge_out(Tensor& out, const Tensor& self, const Tensor& other) {
  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "geTensor", [&]() {
    impl::geTensor<scalar_t>(out, self, other);
  });

  return out;
}

Tensor ge(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  return at::ge_out(result, self, other);
}

Tensor& le_out(Tensor& out, const Tensor& self, Scalar other_) {
  auto other = c10::scalar_to_tensor(other_, kDPCPP);
  other.unsafeGetTensorImpl()->set_wrapped_number(true);
  // TODO: broadcast
  auto new_other =
      other.resize_as_(self).fill_(other_).toType(self.scalar_type());
  at::le_out(out, self, new_other);
  return out;
}

Tensor le(const Tensor& self, Scalar other_) {
  auto result = at::empty({0}, self.options().dtype(kBool));
  auto other = c10::scalar_to_tensor(other_, kDPCPP);
  other.unsafeGetTensorImpl()->set_wrapped_number(true);
  // TODO: broadcast
  auto new_other =
      other.resize_as_(self).fill_(other_).toType(self.scalar_type());
  return at::le_out(result, self, new_other);
}

Tensor& le_out(Tensor& out, const Tensor& self, const Tensor& other) {
  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "leTensor", [&]() {
    impl::leTensor<scalar_t>(out, self, other);
  });

  return out;
}

Tensor le(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  return at::le_out(result, self, other);
}

Tensor& eq_out(Tensor& out, const Tensor& self, Scalar other_) {
  auto other = c10::scalar_to_tensor(other_, kDPCPP);
  other.unsafeGetTensorImpl()->set_wrapped_number(true);
  // TODO: broadcast
  auto new_other =
      other.resize_as_(self).fill_(other_).toType(self.scalar_type());
  return at::eq_out(out, self, new_other);
}

Tensor eq(const Tensor& self, Scalar other_) {
  auto result = at::empty({0}, self.options().dtype(kBool));
  auto other = c10::scalar_to_tensor(other_, kDPCPP);
  other.unsafeGetTensorImpl()->set_wrapped_number(true);
  // TODO: broadcast
  auto new_other =
      other.resize_as_(self).fill_(other_).toType(self.scalar_type());
  return at::eq_out(result, self, new_other);
}

Tensor& eq_out(Tensor& out, const Tensor& self, const Tensor& other) {
  AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Bool, self.scalar_type(), "eqTensor", [&]() {
        impl::eqTensor<scalar_t>(out, self, other);
      });

  return out;
}

Tensor eq(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  return at::eq_out(result, self, other);
}

Tensor& ne_out(Tensor& out, const Tensor& self, Scalar other_) {
  auto other = c10::scalar_to_tensor(other_, kDPCPP);
  other.unsafeGetTensorImpl()->set_wrapped_number(true);
  // TODO: broadcast
  auto new_other =
      other.resize_as_(self).fill_(other_).toType(self.scalar_type());
  at::ne_out(out, self, new_other);
  return out;
}

Tensor ne(const Tensor& self, Scalar other_) {
  auto result = at::empty({0}, self.options().dtype(kBool));
  auto other = c10::scalar_to_tensor(other_, kDPCPP);
  other.unsafeGetTensorImpl()->set_wrapped_number(true);
  // TODO: broadcast
  auto new_other =
      other.resize_as_(self).fill_(other_).toType(self.scalar_type());
  return at::ne_out(result, self, new_other);
}

Tensor& ne_out(Tensor& out, const Tensor& self, const Tensor& other) {
  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "neTensor", [&]() {
    impl::neTensor<scalar_t>(out, self, other);
  });

  return out;
}

Tensor ne(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  return at::ne_out(result, self, other);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
