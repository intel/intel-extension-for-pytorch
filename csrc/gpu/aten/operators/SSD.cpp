#include <ATen/ATen.h>

#include <ATen/record_function.h>
#include <core/Generator.h>
#include <runtime/Utils.h>
#include <utils/oneMKLUtils.h>
#include "DistributionTemplates.h"
#include "comm/ATDispatch.h"
#include "comm/ApplyUtils.h"
#include "comm/RegistrationDeclarations.h"

#include <aten/operators/MemoryAccess.h>
#include <torch/library.h>
#include "comm/AccumulateType.h"
using namespace at;
using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

// Fuse small ops in the end of SSD-MobileNetv1, python frontend code is:
//
//  def locations_to_boxes(locations, priors, center_variance, size_variance):
//     locations = torch.cat([
//         locations[..., :2] * center_variance * priors[..., 2:] +
//  priors[..., :2], torch.exp(locations[..., 2:] * size_variance) *
//  priors[..., 2:]], dim=locations.dim() - 1)
//
//      return torch.cat([locations[..., :2] - locations[..., 2:]/2,
// locations[..., :2] + locations[..., 2:]/2], locations.dim() - 1)
//
// This kernel is for location with small size in SSDMobilenet,
// location size=[3000, 4]. To improve the occupancy, we use each
// workitem to process each output, and the output can be written
// continuously while the input is read in-contiguously.
// For more general case, we may use each workitem to process 4 points,
// then both the input and output can be read and written contiguously.
// This general method can achieve good performance for large case.
// However, the EU occupancy will decrease and the performance may not
// good for small case (SSDMobilenet case for example).

template <typename scalar_t>
void locations_to_boxes_kernel_impl(
    scalar_t* locations_ptr,
    scalar_t* priors_ptr,
    scalar_t* ret_ptr,
    float center_variance,
    float size_variance,
    int64_t numel) {
  auto& sycl_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t group_size = std::min(numel, dpcppMaxWorkGroupSize(dev_id));
  int64_t num_group = (numel + group_size - 1) / group_size;
  sycl::range<1> global_range{num_group * group_size};
  sycl::range<1> local_range{group_size};
  using accscalar_t = acc_type<scalar_t>;
  auto cgf = DPCPP_Q_CGF(cgh) {
    dpcpp_local_acc_t<accscalar_t> local_boxes_buf(group_size, cgh);
    cgh.parallel_for(
        sycl::nd_range<1>{global_range, local_range},
        [=](sycl::nd_item<1> item_id) {
          auto index = item_id.get_global_linear_id();
          auto local_index = item_id.get_local_id(0);

          auto location_elm = static_cast<accscalar_t>(locations_ptr[index]);
          auto prior_elm_x = static_cast<accscalar_t>(priors_ptr[index + 2]);
          auto prior_elm_y = static_cast<accscalar_t>(priors_ptr[index]);
          if (local_index % 4 < 2) {
            local_boxes_buf[local_index] =
                location_elm * center_variance * prior_elm_x + prior_elm_y;
          } else {
            local_boxes_buf[local_index] =
                Numerics<accscalar_t>::exp(location_elm * size_variance) *
                prior_elm_y;
          }
          item_id.barrier(sycl::access::fence_space::local_space);
          if (local_index % 4 < 2) {
            ret_ptr[index] = static_cast<scalar_t>(
                local_boxes_buf[local_index] -
                local_boxes_buf[local_index + 2] / 2);
          } else {
            ret_ptr[index] = static_cast<scalar_t>(
                local_boxes_buf[local_index - 2] +
                local_boxes_buf[local_index] / 2);
          }
        });
  };
  DPCPP_Q_SUBMIT(sycl_queue, cgf);
}
} // namespace impl

// For location box conversion in SSD-MobileNetv1
Tensor locations_to_boxes(
    Tensor& locations,
    Tensor& priors,
    double center_variance,
    double size_variance) {
  RECORD_FUNCTION(
      "locations_to_boxes", std::vector<c10::IValue>({locations, priors}));

  TORCH_CHECK(locations.size(-1) == 4, "Incorrect locations shape.");
  TORCH_CHECK(priors.size(-1) == 4, "Incorrect priors shape.");

  Tensor ret = at::empty_like(locations, locations.suggest_memory_format());
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      locations.scalar_type(),
      "locations_to_boxes",
      [&] {
        impl::locations_to_boxes_kernel_impl<scalar_t>(
            locations.data_ptr<scalar_t>(),
            priors.data_ptr<scalar_t>(),
            ret.data_ptr<scalar_t>(),
            center_variance,
            size_variance,
            locations.numel());
      });
  return ret;
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "locations_to_boxes(Tensor locations, Tensor priors, float center_variance, float size_variance) -> Tensor");
  m.impl(
      "locations_to_boxes",
      c10::DispatchKey::XPU,
      at::AtenIpexTypeXPU::locations_to_boxes);
}
} // namespace
