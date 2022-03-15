#include "isa_help.h"

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(get_current_isa_level_kernel_stub);

// get_current_isa_level_kernel_impl
std::string get_current_isa_level() {
#if defined(DYN_DISP_BUILD)
  return get_current_isa_level_kernel_stub(kCPU);
#else
  return get_current_isa_level_kernel_impl();
#endif
}

std::string get_highest_cpu_support_isa_level() {
  CPUCapability level = _get_highest_cpu_support_isa_level();

  return CPUCapabilityToString(level);
}

std::string get_highest_binary_support_isa_level() {
  CPUCapability level = _get_highest_binary_support_isa_level();

  return CPUCapabilityToString(level);
}

} // namespace cpu
} // namespace torch_ipex