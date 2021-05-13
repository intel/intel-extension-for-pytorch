#include <core/GuardImpl.h>

namespace xpu {
namespace dpcpp {
namespace impl {

constexpr DeviceType DPCPPGuardImpl::static_type;

C10_REGISTER_GUARD_IMPL(XPU, DPCPPGuardImpl);
} // namespace impl
} // namespace dpcpp
} // namespace xpu
