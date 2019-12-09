#include <c10/dpcpp/impl/SYCLGuardImpl.h>

namespace c10 {
namespace sycl {
namespace impl {

constexpr DeviceType SYCLGuardImpl::static_type;

C10_REGISTER_GUARD_IMPL(SYCL, SYCLGuardImpl);

}}} // namespace c10::sycl::detail
