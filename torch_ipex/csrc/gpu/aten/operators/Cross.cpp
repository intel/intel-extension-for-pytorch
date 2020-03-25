#include <ATen/ATen.h>

#include <utils/Pointwise.h>

using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

template <typename scalar_t>
void cross(Tensor& self_, const Tensor& x, const Tensor& y, int dimension) {
  int64_t sx = x.stride(dimension);
  int64_t sy = y.stride(dimension);
  int64_t so = self_.stride(dimension);

  auto nx = x.narrow(dimension, 0, 1);
  auto ny = y.narrow(dimension, 0, 1);
  auto nself = at::empty_like(self_);
  TensorImpl_set(TensorImpl_Unwrap(nself), TensorImpl_Unwrap(self_));
  nself = nself.narrow(dimension, 0, 1);

  DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
      nself, nx, ny, TensorCrossOp<scalar_t>(sx, sy, so));
}

} // namespace impl

Tensor cross(
    const Tensor& input,
    const Tensor& other,
    const c10::optional<int64_t> dimension) {
  Tensor out = at::empty_like(input);
  at::AtenIpexTypeDPCPP::cross_out(out, input, other, dimension);
  return out;
}

Tensor& cross_out(
    Tensor& out,
    const Tensor& input,
    const Tensor& other,
    const c10::optional<int64_t> dimension) {
  TORCH_CHECK(
      input.dim() == other.dim(),
      "inconsistent tensors dimensions input: ",
      input.dim(),
      " other: ",
      other.dim());
  TORCH_CHECK(
      input.sizes() == other.sizes(),
      "inconsistent tensors sizes input: ",
      input.sizes(),
      " other: ",
      other.sizes());

  int64_t dim = -1;
  if (!dimension.has_value()) {
    for (int64_t i = 0; i < input.dim(); i++) {
      if (input.size(i) == 3) {
        dim = i;
        break;
      }
    }
    TORCH_CHECK(dim >= 0, "no dimension of size 3 in input");
  } else {
    dim = maybe_wrap_dim(dimension.value(), input.dim());
    TORCH_CHECK(
        input.size(dim) == 3,
        "dimension ",
        dimension.value(),
        " does not have size 3");
  }

  if (out.sizes() != input.sizes()) {
    out.resize_as_(input);
  }

  AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Half, input.scalar_type(), "cross", [&]() {
        impl::cross<scalar_t>(out, input, other, dim);
      });

  return out;
}
}
} // namespace at::AtenIpexTypeDPCPP
