#include <aten/utils/isa_help.h>

#define MACRO_TO_STRING(x) #x
#define MACRO_VALUE_TO_STRING(x) MACRO_TO_STRING(x)

namespace torch_ipex {
namespace cpu {

namespace {

std::string get_current_isa_level_kernel_impl() {
  return MACRO_VALUE_TO_STRING(CPU_CAPABILITY);
}

} // anonymous namespace

REGISTER_DISPATCH(
    get_current_isa_level_kernel_stub,
    &get_current_isa_level_kernel_impl);

} // namespace cpu
} // namespace torch_ipex