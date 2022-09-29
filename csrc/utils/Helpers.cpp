#include <aten/operators/Utils.h>
#include <core/Device.h>
#include <core/Stream.h>
#include <runtime/Exception.h>
#include <runtime/Utils.h>

#include <utils/Helpers.h>
#include <utils/Settings.h>

#include <cmath>

namespace xpu {
namespace dpcpp {

cl::sycl::event queue_barrier(cl::sycl::queue& queue) {
  return at::AtenIpexTypeXPU::dpcpp_q_barrier(queue);
}

cl::sycl::event queue_barrier(
    cl::sycl::queue& queue,
    std::vector<cl::sycl::event>& events) {
  return at::AtenIpexTypeXPU::dpcpp_q_barrier(queue, events);
}

} // namespace dpcpp
} // namespace xpu
