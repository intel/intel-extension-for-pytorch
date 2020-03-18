#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

#include <core/Context.h>
#include <core/DPCPP.h>
#include <core/DPCPPUtils.h>
#include <core/Memory.h>
#include <utils/Numerics.h>
#include <utils/Pointwise.h>

using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

template <typename scalar_t> class sigmoid_ker {};

template <typename scalar_t>
static inline void sigmoid(Tensor &output, const Tensor &self) {
  auto queue = dpcppGetCurrentQueue();
  int64_t rng, grng, tile_size, size;

  parallel_for_setup(self.numel(), tile_size, rng, grng);
  size = self.numel() * sizeof(scalar_t);
  output.resize_as_(self);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto in_acc =
        DPCPPAccessor<dpcpp_r_mode>(cgh, self.data_ptr<scalar_t>(), size);
    auto out_acc = DPCPPAccessor<dpcpp_discard_w_mode>(
        cgh, output.data_ptr<scalar_t>(), size);
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item) {
      size_t id = item.get_global_linear_id();
      auto in_ptr = in_acc.template get_pointer<scalar_t>();
      auto out_ptr = out_acc.template get_pointer<scalar_t>();
      if (id < size / sizeof(scalar_t))
        out_ptr[id] =
            1 /
            (1 + Numerics<scalar_t>::exp(-static_cast<scalar_t>(in_ptr[id])));
    };

    cgh.parallel_for<sigmoid_ker<scalar_t>>(
        DPCPP::nd_range<1>(DPCPP::range<1>(grng), DPCPP::range<1>(tile_size)),
        kfn);
  };

  DPCPP_Q_ASYNC_SUBMIT(queue, cgf);
}

Tensor &_sigmoid_out(Tensor &output, const Tensor &self) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      self.scalar_type(), "_sigmoid_out",
      [&]() { impl::sigmoid<scalar_t>(output, self); });
  return output;
}

} // namespace impl

Tensor &sigmoid_out(Tensor &out, const Tensor &self) {
  return impl::_sigmoid_out(out, self);
}
Tensor sigmoid(const Tensor &self) {
  Tensor result = at::empty({0}, self.options());
  return at::AtenIpexTypeDPCPP::sigmoid_out(result, self);
}
Tensor &sigmoid_(Tensor &self) {
  return at::AtenIpexTypeDPCPP::sigmoid_out(self, self);
}

Tensor &sigmoid_backward_out(Tensor &grad_input, const Tensor &grad_output,
                             const Tensor &output) {
  TORCH_CHECK(output.numel() == grad_output.numel(), "different elements ...");
  grad_input.resize_as_(output);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      output.scalar_type(), "sigmoid_backward_out", [&]() {
        at::dpcpp::DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
            grad_input, output, grad_output, TensorSigmoidGradOp<scalar_t>());
      });

  return grad_input;
}

Tensor sigmoid_backward(const Tensor &grad_output, const Tensor &output) {
  auto grad_input = at::empty({0}, grad_output.options());
  return at::AtenIpexTypeDPCPP::sigmoid_backward_out(grad_input, grad_output,
                                                     output);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
