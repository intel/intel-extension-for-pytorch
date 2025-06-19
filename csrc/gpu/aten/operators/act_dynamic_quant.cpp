#include <ATen/ATen.h>
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "runtime/Utils.h"
#include "utils/CustomOperatorRegistration.h"
#include "utils/DPCPP.h"

#include "act_dynamic_quant.h"

using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

std::tuple<Tensor, Tensor, Tensor> dynamic_per_token_quant(
    const Tensor& input,
    bool use_sym_quant) {
  TORCH_CHECK(
      input.is_contiguous(),
      "dynamic_per_token_quant only supports contiguous input tensor");
  // init out tensor, scales, zp
  int64_t hidden_size = input.size(-1);
  int64_t m = input.numel() / hidden_size;
  auto out_dtype = use_sym_quant ? at::kChar : at::kByte;
  auto qout = at::empty_like(input, out_dtype);
  // scales use same dtype as input, zp use int32
  auto scales = at::empty({m, 1}, input.options());
  auto zps = at::empty({m, 1}, input.options().dtype(at::kInt));
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);
  int64_t group_size = std::min(max_wg_size, hidden_size);

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "dynamic_per_token_quant",
      [=]() {
        auto cgf = DPCPP_Q_CGF(cgh) {
          if (use_sym_quant) {
            auto kernel = DynamicPerTokenQuantActFunctor<scalar_t, int8_t>{
                input.data_ptr<scalar_t>(),
                qout.data_ptr<int8_t>(),
                scales.data_ptr<scalar_t>(),
                zps.data_ptr<int32_t>(),
                hidden_size,
                use_sym_quant};
            cgh.parallel_for<decltype(kernel)>(
                sycl::nd_range<1>(
                    sycl::range<1>(m * group_size), sycl::range<1>(group_size)),
                kernel);
          } else {
            auto kernel = DynamicPerTokenQuantActFunctor<scalar_t, uint8_t>{
                input.data_ptr<scalar_t>(),
                qout.data_ptr<uint8_t>(),
                scales.data_ptr<scalar_t>(),
                zps.data_ptr<int32_t>(),
                hidden_size,
                use_sym_quant};
            cgh.parallel_for<decltype(kernel)>(
                sycl::nd_range<1>(
                    sycl::range<1>(m * group_size), sycl::range<1>(group_size)),
                kernel);
          }
        };
        DPCPP_Q_SUBMIT(dpcppGetCurrentQueue(), cgf);
      });

  return std::make_tuple<Tensor&&, Tensor&&, Tensor&&>(
      std::move(qout), std::move(scales), std::move(zps));
}

std::tuple<Tensor, Tensor, Tensor> dynamic_per_tensor_quant(
    const Tensor& input,
    bool use_sym_quant) {
  TORCH_CHECK(
      input.is_contiguous(),
      "dynamic_per_tensor_quant only supports contiguous input tensor");
  // init out tensor, scales, zp
  int64_t hidden_size = input.size(-1);
  int64_t num_elements = input.numel();
  int64_t num_tokens = num_elements / hidden_size;
  auto out_dtype = use_sym_quant ? at::kChar : at::kByte;
  auto qout = at::empty_like(input, out_dtype);
  // scales use same dtype as input, zp use int32
  // OPTIMIZE ME! currently use a torch op to find the min/max
  auto [input_min, input_max] = input.aminmax();
  auto scale = at::empty({1}, input.options());
  auto zp = at::empty({1}, input.options().dtype(at::kInt));

  switch (input.scalar_type()) {
    case at::kFloat: {
      GetPerTensorScaleZPFunctor<float> compute_scale_zp(
          input_min.data_ptr<float>(),
          input_max.data_ptr<float>(),
          scale.data_ptr<float>(),
          zp.data_ptr<int32_t>(),
          use_sym_quant);
      dpcppGetCurrentQueue().single_task<decltype(compute_scale_zp)>(
          compute_scale_zp);
      break;
    }
    case at::kHalf: {
      GetPerTensorScaleZPFunctor<at::Half> compute_scale_zp(
          input_min.data_ptr<at::Half>(),
          input_max.data_ptr<at::Half>(),
          scale.data_ptr<at::Half>(),
          zp.data_ptr<int32_t>(),
          use_sym_quant);
      dpcppGetCurrentQueue().single_task<decltype(compute_scale_zp)>(
          compute_scale_zp);
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported input type for dynamic_per_tensor_quant");
  }

  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);

  auto stream = at::xpu::getCurrentXPUStream().queue();
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "dynamic_per_tensor_quant",
      [=]() {
        auto cgf = DPCPP_Q_CGF(cgh) {
          if (use_sym_quant) {
            auto kernel = DynamicPerTensorQuantFunctor<scalar_t, int8_t>{
                input.data_ptr<scalar_t>(),
                qout.data_ptr<int8_t>(),
                scale.data_ptr<scalar_t>(),
                zp.data_ptr<int32_t>(),
                num_elements,
                use_sym_quant};
            cgh.parallel_for<decltype(kernel)>(
                sycl::nd_range<1>(
                    sycl::range<1>(num_tokens * max_wg_size),
                    sycl::range<1>(max_wg_size)),
                kernel);
          } else {
            auto kernel = DynamicPerTensorQuantFunctor<scalar_t, uint8_t>{
                input.data_ptr<scalar_t>(),
                qout.data_ptr<uint8_t>(),
                scale.data_ptr<scalar_t>(),
                zp.data_ptr<int32_t>(),
                num_elements,
                use_sym_quant};
            cgh.parallel_for<decltype(kernel)>(
                sycl::nd_range<1>(
                    sycl::range<1>(num_tokens * max_wg_size),
                    sycl::range<1>(max_wg_size)),
                kernel);
          }
        };
        DPCPP_Q_SUBMIT(dpcppGetCurrentQueue(), cgf);
      });

  return std::make_tuple<Tensor&&, Tensor&&, Tensor&&>(
      std::move(qout), std::move(scale), std::move(zp));
}

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "dynamic_per_token_quant",
      dynamic_per_token_quant,
      c10::DispatchKey::XPU);
  IPEX_OP_REGISTER_DISPATCH(
      "dynamic_per_tensor_quant",
      dynamic_per_tensor_quant,
      c10::DispatchKey::XPU);
}
} // namespace

} // namespace AtenIpexTypeXPU
} // namespace at