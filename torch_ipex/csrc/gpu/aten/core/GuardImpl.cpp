#include <core/GuardImpl.h>

namespace at {
namespace dpcpp {
namespace impl {

constexpr DeviceType DPCPPGuardImpl::static_type;

C10_REGISTER_GUARD_IMPL(DPCPP, DPCPPGuardImpl);

}}} // namespace at::dpcpp::impl
