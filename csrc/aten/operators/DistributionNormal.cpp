#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/ExpandUtils.h>

#include <core/Generator.h>
#include <utils/ATDispatch.h>
#include <utils/AccumulateType.h>

#include "Random.h"
#include "Distributions.h"


namespace at {
namespace AtenIpexTypeXPU {

bool resize_output_for_normal(at::Tensor& output, const at::Tensor& mean, const at::Tensor& std) {
  bool expandable = at::are_expandable(mean.sizes(), std.sizes());
  bool empty_output = output.numel() == 0;

  if (expandable) {
    auto shape = at::infer_size(mean.sizes(), std.sizes());
    TORCH_CHECK(
        empty_output || output.sizes().equals(shape),
        "inconsistent tensor, output size (", output.sizes(), ") is not the same as broadcasted mean and std size (", shape, ")");
    if (empty_output) {
      at::native::resize_(output, shape);
    }
    return false;
  }
  else {
    TORCH_CHECK(
        mean.numel() == std.numel(),
        "inconsistent tensor, std and mean are not broadcastable and have different number of elements, "
        "expected mean ", mean.sizes(), " and std ", std.sizes(), " to have same number of elements)");
    TORCH_CHECK(
        empty_output || output.sizes().equals(mean.sizes()),
        "inconsistent tensor, std and mean are not broadcastable, output size (", output.sizes(), ") is not the same as mean size (", mean.sizes(), ")");
    TORCH_WARN_ONCE(
        "std and mean have the same number of elements, but are not broadcastable. This was previously a "
        "supported mode of operation, but is now deprecated and the support will be removed in version 1.6 release. "
        "Note that the current implementation reshapes std to the shape of mean, which may be incur data copies. "
        "Please ensure that std and mean are broadcastable to avoid these issues.");
    if (empty_output) {
      at::native::resize_(output, mean.sizes());
    }
    return true;
  }
}


void normal_dpcpp(TensorIterator& iter, double mean_, double std_, c10::optional<Generator> gen_) {
  auto gen = get_generator_or_default<xpu::dpcpp::DPCPPGeneratorImpl>(
      gen_, xpu::dpcpp::detail::getDefaultDPCPPGenerator());
  IPEX_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "normal_dpcpp", [&] {
    using accscalar_t = dist_acctype<scalar_t>;
    auto mean = static_cast<accscalar_t>(mean_);
    auto std = static_cast<accscalar_t>(std_);
    // define lambda to multiply std and add mean
    auto normal_func = [mean, std] (accscalar_t rand) {
      auto ret = static_cast<scalar_t>(rand * std + mean);
      return ret;
    };
    AtenIpexTypeXPU::distribution_nullary_kernel<scalar_t, accscalar_t>(iter,
      gen,
      [] (RandomState<Philox4_32_10>* state) { return state->normal<scalar_t>(); },
      normal_func);
  });


}

Tensor& normal_(Tensor& self, double mean, double std, c10::optional<Generator> generator) {
  TORCH_CHECK(std > 0.0, "normal_ expects std > 0.0, but found std=", std);
  auto iter = TensorIterator::nullary_op(self);
  normal_dpcpp(iter, mean, std, generator);
  return self;
}

Tensor normal(const Tensor& mean, double std, c10::optional<Generator> generator) {
  Tensor ret = at::empty_like(mean, MemoryFormat::Contiguous);
  normal_out(ret, mean, std, generator);
  return ret;
}

Tensor normal(double mean, const Tensor& std, c10::optional<Generator> generator) {
  Tensor ret = at::empty_like(std, MemoryFormat::Contiguous);
  normal_out(ret, mean, std, generator);
  return ret;
}

Tensor normal(const Tensor& mean, const Tensor& std, c10::optional<Generator> generator) {
  Tensor ret = at::empty({0}, mean.options(), MemoryFormat::Contiguous);
  normal_out(ret, mean, std, generator);
  return ret;
}

Tensor& normal_out(Tensor& output, const Tensor& mean, double std, c10::optional<Generator> generator) {
  normal_(output, 0, std, generator);
  output.add_(mean);
  return output;
}

Tensor& normal_out(Tensor& output, double mean, const Tensor& std, c10::optional<Generator> generator) {
  normal_(output, 0, 1, generator);
  auto mean_tensor = at::full({}, mean, output.options());
  output.mul_(std).add_(mean_tensor);
  return output;
}

Tensor& normal_out(Tensor& output, const Tensor& mean, const Tensor& std, c10::optional<Generator> generator) {
  bool is_deprecated_th_impl = resize_output_for_normal(output, mean, std);
  normal_(output, 0, 1, generator);
  if (is_deprecated_th_impl) {
    output.mul_(std.reshape(mean.sizes())).add_(mean);
  }
  else {
    output.mul_(std).add_(mean);
  }
  return output;
}

} // namespace AtenIpexTypeXPU
} // namespace at
