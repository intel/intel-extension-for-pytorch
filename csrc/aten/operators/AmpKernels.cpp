#include <ATen/ATen.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

// _amp_update_scale_kernel is launched with a single work-item to compute the
// new scale. The scale factor is maintained and updated on the XPU
// asynchronously.
void _amp_update_scale_kernel(
    float* current_scale,
    int* growth_tracker,
    float* found_inf,
    double growth_factor,
    double backoff_factor,
    int growth_interval) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(cgf) {
    auto kfn = [=]() {
      if (*found_inf) {
        *current_scale *= backoff_factor;
        *growth_tracker = 0;
      } else {
        // Entering this branch means we just carried out a successful step,
        // so growth_tracker is incremented before comparing to growth_interval.
        auto successful = (*growth_tracker) + 1;
        if (successful == growth_interval) {
          *current_scale *= growth_factor;
          *growth_tracker = 0;
        } else {
          *growth_tracker = successful;
        }
      }
    };

    cgf.single_task(kfn);
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

// _amp_update_scale_ asynchronously updates the scale tensor in place.
//
// Args:
// current_scale:  A one-element xpu float tensor containing the scale value.
// growth_tracker:  A one-element torch.xpu.IntTensor containing the number of
// recent consecutive unskipped steps.
// found_inf:  A one-element xpu float tensor. If > 0, indicates that infs/nans
// were found by the relevant prior _amp_non_finite_check_and_unscale_ call,
// and 0 if no infs/nans were found.
// growth_factor:  Multiplier if no infs/NaNs were found (typically slightly
// >1). backoff_factor:  Multiplier if infs/NaNs were found (typically 0.5).
// growth_interval:  Number of consecutive unskipped steps that must occur for
// current_scale to be multiplied by growth_factor.
//
// Returns:
// current_scale:  updated in place.
Tensor& _amp_update_scale_(
    Tensor& current_scale,
    Tensor& growth_tracker,
    const Tensor& found_inf,
    double growth_factor,
    double backoff_factor,
    int64_t growth_interval) {
  TORCH_CHECK(growth_tracker.is_xpu(), "growth_tracker must be a XPU tensor.");
  TORCH_CHECK(current_scale.is_xpu(), "current_scale must be a XPU tensor.");
  TORCH_CHECK(found_inf.is_xpu(), "found_inf must be a XPU tensor.");
  TORCH_CHECK(
      growth_tracker.numel() == 1,
      "growth_tracker must be a 1-element tensor.");
  TORCH_CHECK(
      current_scale.numel() == 1, "current_scale must be a 1-element tensor.");
  TORCH_CHECK(found_inf.numel() == 1, "found_inf must be a 1-element tensor.");
  TORCH_CHECK(
      growth_tracker.scalar_type() == at::ScalarType::Int,
      "growth_tracker must be an int tensor.");
  TORCH_CHECK(
      current_scale.scalar_type() == at::ScalarType::Float,
      "current_scale must be a float tensor.");
  TORCH_CHECK(
      found_inf.scalar_type() == at::ScalarType::Float,
      "found_inf must be a float tensor.");

  _amp_update_scale_kernel(
      current_scale.data_ptr<float>(),
      growth_tracker.data_ptr<int>(),
      found_inf.data_ptr<float>(),
      growth_factor,
      backoff_factor,
      growth_interval);

  return current_scale;
}

} // namespace AtenIpexTypeXPU
} // namespace at
