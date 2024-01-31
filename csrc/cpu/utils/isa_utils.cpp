#include "isa_utils.h"
#include "../isa/cpu_feature.hpp"

namespace torch_ipex {
namespace utils {

bool isa_has_amx_fp16_support() {
  return cpu::CPUFeature::get_instance().isa_level_amx_fp16();
}

bool isa_has_avx512_fp16_support() {
  return cpu::CPUFeature::get_instance().isa_level_avx512_fp16();
}

bool isa_has_amx_support() {
  return cpu::CPUFeature::get_instance().isa_level_amx();
}

bool isa_has_avx512_bf16_support() {
  return cpu::CPUFeature::get_instance().isa_level_avx512_bf16();
}

bool isa_has_avx512_vnni_support() {
  return cpu::CPUFeature::get_instance().isa_level_avx512_vnni();
}

bool isa_has_avx512_support() {
  return cpu::CPUFeature::get_instance().isa_level_avx512();
}

bool isa_has_avx2_vnni_support() {
  return cpu::CPUFeature::get_instance().isa_level_avx2_vnni();
}

bool isa_has_avx2_support() {
  return cpu::CPUFeature::get_instance().isa_level_avx2();
}

} // namespace utils
} // namespace torch_ipex
