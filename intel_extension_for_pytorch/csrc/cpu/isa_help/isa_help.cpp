#include "isa_help.h"

#include "cpu_feature.hpp"
#include "embedded_function.h"

namespace isa_help {
bool check_isa_avx2() {
  return torch_ipex::cpu::CPUFeature::get_instance().os_avx2();
}

bool check_isa_avx512() {
  return torch_ipex::cpu::CPUFeature::get_instance().os_avx512();
}

} // namespace isa_help