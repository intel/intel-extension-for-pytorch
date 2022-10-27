#include <stdio.h>
#include "cpu_feature.hpp"
#include "embedded_function.h"

using namespace torch_ipex::cpu;

#ifdef CPU_FEATURE_EXEC
int main() {
  uint32_t eax = 0;
  uint32_t ebx = 0;
  uint32_t ecx = 0;
  uint32_t edx = 0;

  read_cpuid(0, &eax, &ebx, &ecx, &edx);
  // printf("%08x-%08x-%08x-%08x\n", eax, ebx, ecx, edx);

  read_cpuidex(1, 0, &eax, &ebx, &ecx, &edx);
  // printf("%08x-%08x-%08x-%08x\n", eax, ebx, ecx, edx);

  uint64_t xcr_data = 0;
  read_xcr(0, &xcr_data);
  printf("XCR0: %016lx\n", xcr_data);

  printf(
      "os --> avx: %s\n",
      CPUFeature::get_instance().os_avx() ? "true" : "false");
  printf(
      "os --> avx2: %s\n",
      CPUFeature::get_instance().os_avx2() ? "true" : "false");
  printf(
      "os --> avx512: %s\n",
      CPUFeature::get_instance().os_avx512() ? "true" : "false");
  printf(
      "os --> amx: %s\n",
      CPUFeature::get_instance().os_amx() ? "true" : "false");
  CPUFeature::get_instance().show_features();

  return 0;
}
#endif