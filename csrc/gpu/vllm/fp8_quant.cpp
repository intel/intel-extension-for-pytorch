#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/xpu/XPUContext.h>
#include "utils/CustomOperatorRegistration.h"

#include <sycl/sycl.hpp>

#include "fp8_quant.h"
#include "utils.h"

using namespace at::AtenIpexTypeXPU;

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t, typename fp8_type>
class scaled_fp8_quant_kernel {
 private:
  fp8_type* out;
  const scalar_t* input;
  const float* scale;
  int64_t num_elems;

 public:
  scaled_fp8_quant_kernel(
      fp8_type* out_,
      const scalar_t* input_,
      const float* scale_,
      int64_t num_elems_)
      : out(out_), input(input_), scale(scale_), num_elems(num_elems_) {}
  void operator()(sycl::nd_item<1> item) const {
    int tid = item.get_global_linear_id();

    // Invert the scale so that we can use multiplications to avoid expensive
    // division.
    const float inverted_scale = 1.0f / (*scale);
    scaled_fp8_conversion_vec<scalar_t, true>(
        out,
        input,
        inverted_scale,
        num_elems,
        tid,
        item.get_local_range(0) * item.get_group_range(0));
  }
};

void static_scaled_fp8_quant(
    torch::Tensor& out, // [..., d]
    torch::Tensor const& input, // [..., d]
    torch::Tensor const& scale) // [1]
{
  int64_t num_tokens = input.numel() / input.size(-1);
  int64_t num_elems = input.numel();
  sycl::range<1> grid(num_tokens);
  sycl::range<1> block(1024);
  at::Device curDevice = at::Device(at::kXPU, at::xpu::current_device());
  at::DeviceGuard device_guard(curDevice);

  auto stream = at::xpu::getCurrentXPUStream().queue();
  // TODO: change name?
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "scaled_fp8_quant_kernel_scalar_type", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "scaled_fp8_quant_kernel_fp8_type", [&] {
              // Launch the kernel
              stream.submit([&](sycl::handler& cgh) {
                auto kernel = scaled_fp8_quant_kernel<scalar_t, fp8_t>(
                    out.data_ptr<fp8_t>(),
                    input.data_ptr<scalar_t>(),
                    scale.data_ptr<float>(),
                    num_elems);
                cgh.parallel_for(
                    sycl::nd_range<1>(grid * block, block), kernel);
              });
            });
      });
}

void dynamic_scaled_fp8_quant(
    torch::Tensor& out, // [..., d]
    torch::Tensor const& input, // [..., d]
    torch::Tensor& scale) // [1]
{
  int64_t num_tokens = input.numel() / input.size(-1);
  int64_t num_elems = input.numel();
  sycl::range<1> grid(num_tokens);
  sycl::range<1> block(1024);
  at::Device curDevice = at::Device(at::kXPU, at::xpu::current_device());
  at::DeviceGuard device_guard(curDevice);

  auto stream = at::xpu::getCurrentXPUStream().queue();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "scaled_fp8_quant_kernel_scalar_type", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "scaled_fp8_quant_kernel_fp8_type", [&] {
              // Launch the kernel
              stream.submit([&](sycl::handler& cgh) {
                auto max_reduce_kernel =
                    segmented_max_reduction<scalar_t, fp8_t>(
                        scale.data_ptr<float>(),
                        input.data_ptr<scalar_t>(),
                        num_elems);
                cgh.parallel_for(
                    sycl::nd_range<1>(grid * block, block), max_reduce_kernel);
              });
              stream.submit([&](sycl::handler& cgh) {
                auto kernel = scaled_fp8_quant_kernel<scalar_t, fp8_t>(
                    out.data_ptr<fp8_t>(),
                    input.data_ptr<scalar_t>(),
                    scale.data_ptr<float>(),
                    num_elems);
                cgh.parallel_for(
                    sycl::nd_range<1>(grid * block, block), kernel);
              });
            });
      });
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "static_scaled_fp8_quant.xpu",
      at::AtenIpexTypeXPU::static_scaled_fp8_quant,
      c10::DispatchKey::XPU);
}

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "dynamic_scaled_fp8_quant.xpu",
      at::AtenIpexTypeXPU::dynamic_scaled_fp8_quant,
      c10::DispatchKey::XPU);
}

} // namespace
