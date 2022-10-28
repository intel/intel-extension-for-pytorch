#include "isa_help.h"

#include <dnnl.hpp>

namespace torch_ipex {
namespace cpu {

using namespace dnnl;

DEFINE_DISPATCH(get_current_isa_level_kernel_stub);

// get_current_isa_level_kernel_impl
std::string get_current_isa_level() {
  // pointer to get_current_isa_level_kernel_impl();
  return get_current_isa_level_kernel_stub(kCPU);
}

std::string get_highest_cpu_support_isa_level() {
  CPUCapability level = _get_highest_cpu_support_isa_level();

  return CPUCapabilityToString(level);
}

std::string get_highest_binary_support_isa_level() {
  CPUCapability level = _get_highest_binary_support_isa_level();

  return CPUCapabilityToString(level);
}

const char* OneDNNIsaLevelToString(cpu_isa isa) {
  // convert dnnl::cpu_isa to string
  switch (isa) {
    case cpu_isa::avx2:
      return "AVX2";
    case cpu_isa::avx2_vnni:
      return "AVX2_VNNI";
    case cpu_isa::avx512_core:
      return "AVX512";
    case cpu_isa::avx512_core_vnni:
      return "AVX512_VNNI";
    case cpu_isa::avx512_core_bf16:
      return "AVX512_BF16";
    case cpu_isa::avx512_core_amx:
      return "AMX";

    default:
      return "WrongLevel";
  }
}

std::string get_current_onednn_isa_level() {
  cpu_isa onednn_isa_level = get_effective_cpu_isa();
  return OneDNNIsaLevelToString(onednn_isa_level);
}

} // namespace cpu
} // namespace torch_ipex