#include <ATen/ATen.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename input_t, typename output_t>
void convert_indices_from_coo_to_csr_kernel_dpcpp(
    const Tensor& result,
    const Tensor& input,
    const int64_t size) {
  int64_t numel = input.numel();
  const input_t* data_in = input.data_ptr<input_t>();
  output_t* data_out = result.data_ptr<output_t>();

  if (numel == 0) {
    result.zero_();
    return;
  }

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  const auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  const auto wgroup_size = dpcppMaxWorkGroupSize(dev_id);
  const auto ngroups = (numel + wgroup_size - 1) / wgroup_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(ngroups * wgroup_size, wgroup_size),
        [=](sycl::nd_item<1> itemId) {
          auto linear_id = itemId.get_global_linear_id();
          if (linear_id == 0) {
            for (int64_t i = 0; i <= data_in[0]; i++)
              data_out[i] = static_cast<output_t>(0);
          } else if (linear_id < numel) {
            for (int64_t i = data_in[linear_id - 1]; i < data_in[linear_id];
                 i++)
              data_out[i + 1] = static_cast<output_t>(linear_id);
          } else if (linear_id == numel) {
            for (int64_t i = data_in[numel - 1] + 1; i < size + 1; i++)
              data_out[i] = static_cast<output_t>(numel);
          }
        });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}
} // namespace impl

Tensor& _convert_indices_from_coo_to_csr_out(
    const Tensor& input,
    const int64_t size,
    const bool out_int32,
    Tensor& result) {
  if (out_int32) {
    IPEX_DISPATCH_INTEGRAL_TYPES(
        input.scalar_type(), "convert_indices_from_coo_to_csr_dpcpp", [&] {
          impl::convert_indices_from_coo_to_csr_kernel_dpcpp<scalar_t, int>(
              result, input, size);
        });
  } else {
    IPEX_DISPATCH_INTEGRAL_TYPES(
        input.scalar_type(), "convert_indices_from_coo_to_csr_dpcpp", [&] {
          impl::convert_indices_from_coo_to_csr_kernel_dpcpp<scalar_t, int64_t>(
              result, input, size);
        });
  }
  return result;
}

} // namespace AtenIpexTypeXPU
} // namespace at
