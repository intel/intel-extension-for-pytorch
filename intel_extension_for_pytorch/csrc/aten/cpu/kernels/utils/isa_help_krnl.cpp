#include <csrc/aten/cpu/utils/isa_help.h>

#define MICRO_TO_STRING(x) #x
#define MICRO_VALUE_TO_STRING(x) MICRO_TO_STRING(x)

namespace torch_ipex {
namespace cpu {

#if defined(DYN_DISP_BUILD)
namespace {
#endif

std::string get_current_isa_level_kernel_impl() {
  return MICRO_VALUE_TO_STRING(CPU_CAPABILITY);
}

#if defined(DYN_DISP_BUILD)
} // anonymous namespace

REGISTER_DISPATCH(
    get_current_isa_level_kernel_stub,
    &get_current_isa_level_kernel_impl);

#endif

} // namespace cpu
} // namespace torch_ipex