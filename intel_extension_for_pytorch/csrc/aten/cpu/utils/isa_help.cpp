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

} // namespace cpu
} // namespace torch_ipex