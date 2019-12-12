#include <core/SYCLGuardImpl.h>

namespace c10 {
namespace sycl {
namespace impl {

constexpr DeviceType SYCLGuardImpl::static_type;

C10_REGISTER_GUARD_IMPL(DPCPP, SYCLGuardImpl);

}}} // namespace c10::sycl::detail
