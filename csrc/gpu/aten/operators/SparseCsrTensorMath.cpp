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

template <typename input_t, typename output_t>
void convert_indices_from_csr_to_coo_kernel_dpcpp(
    const Tensor& indices,
    const Tensor& crow_indices,
    const Tensor& col_indices,
    const bool transpose = false) {
  int64_t nrows = crow_indices.numel() - 1;

  if (nrows == 0) {
    indices.zero_();
    return;
  }

  auto crow_indices_ = crow_indices.expect_contiguous();
  const input_t* crow_indices_data_in = crow_indices_->data_ptr<input_t>();
  TORCH_INTERNAL_ASSERT(indices.is_contiguous());
  auto row0 = indices.select(0, transpose ? 1 : 0);
  auto row1 = indices.select(0, transpose ? 0 : 1);
  output_t* data_out = row0.data_ptr<output_t>();
  row1.copy_(*col_indices.expect_contiguous());

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  const auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  const auto wgroup_size = dpcppMaxWorkGroupSize(dev_id);
  const auto ngroups = (nrows + wgroup_size - 1) / wgroup_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(ngroups * wgroup_size, wgroup_size),
        [=](sycl::nd_item<1> itemId) {
          int64_t linear_id = itemId.get_global_linear_id();
          if (linear_id < nrows) {
            for (int64_t i = crow_indices_data_in[linear_id];
                 i < crow_indices_data_in[linear_id + 1];
                 i++)
              data_out[i] = static_cast<output_t>(linear_id);
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

Tensor& _convert_indices_from_csr_to_coo_out(
    const Tensor& crow_indices,
    const Tensor& col_indices,
    const bool out_int32,
    const bool transpose,
    Tensor& result) {
  if (out_int32) {
    IPEX_DISPATCH_INTEGRAL_TYPES(
        crow_indices.scalar_type(),
        "convert_indices_from_csr_to_coo_dpcpp",
        [&] {
          impl::convert_indices_from_csr_to_coo_kernel_dpcpp<scalar_t, int>(
              result, crow_indices, col_indices, transpose);
        });
  } else {
    IPEX_DISPATCH_INTEGRAL_TYPES(
        crow_indices.scalar_type(),
        "convert_indices_from_coo_to_csr_dpcpp",
        [&] {
          impl::convert_indices_from_csr_to_coo_kernel_dpcpp<scalar_t, int64_t>(
              result, crow_indices, col_indices, transpose);
        });
  }
  return result;
}
} // namespace AtenIpexTypeXPU
} // namespace at
