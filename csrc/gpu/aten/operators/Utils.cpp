#include "Utils.h"
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "utils/CustomOperatorRegistration.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

sycl::event dpcpp_q_barrier(sycl::queue& q) {
#ifdef USE_QUEUE_BARRIER
  return q.ext_oneapi_submit_barrier();
#else
  auto cgf = [&](sycl::handler& cgh) { cgh.single_task([=]() {}); };
  return q.submit(cgf);
#endif
}

sycl::event dpcpp_q_barrier(sycl::queue& q, std::vector<sycl::event>& events) {
#ifdef USE_QUEUE_BARRIER
  return q.ext_oneapi_submit_barrier(events);
#else
  auto cgf = [&](sycl::handler& cgh) {
    cgh.depends_on(events);
    cgh.single_task([=]() {});
  };
  return q.submit(cgf);
#endif
}

bool check_onednn_layout(const at::Tensor& input) {
  return xpu::oneDNN::is_onednn_layout(input);
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER(
      "check_onednn_layout.xpu", at::AtenIpexTypeXPU::check_onednn_layout);
}
} // namespace
