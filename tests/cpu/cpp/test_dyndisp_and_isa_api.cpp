#include <stdio.h>
#include "gtest/gtest.h"
#include "intel_extension_for_pytorch/csrc/cpu/isa/cpu_feature.hpp"
#include "intel_extension_for_pytorch/csrc/cpu/isa/embedded_function.h"
#include "intel_extension_for_pytorch/csrc/dyndisp/DispatchStub.h"

#define ASSERT_VARIABLE_EQ(a, b) ASSERT_TRUE(torch::allclose((a), (b)))
#define EXPECT_VARIABLE_EQ(a, b) EXPECT_TRUE(torch::allclose((a), (b)))
#define ASSERT_STRING_EQ(a, b) ASSERT_TRUE(strcmp((a), (b)) == 0)

using namespace torch_ipex::cpu;

TEST(TestDynDispAndIsaAPI, TestIsaRegs) {
  uint32_t eax = 0;
  uint32_t ebx = 0;
  uint32_t ecx = 0;
  uint32_t edx = 0;

  read_cpuid(0, &eax, &ebx, &ecx, &edx);
  read_cpuidex(1, 0, &eax, &ebx, &ecx, &edx);

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
}

TEST(TestDynDispAndIsaAPI, TestIsaLevels) {
  CPUFeature::get_instance().isa_level_amx();
  CPUFeature::get_instance().isa_level_avx2();
  CPUFeature::get_instance().isa_level_avx2_vnni();
  CPUFeature::get_instance().isa_level_avx512_core();
  CPUFeature::get_instance().isa_level_avx512_vnni();
  CPUFeature::get_instance().isa_level_avx512_bf16();
}

TEST(TestDynDispAndIsaAPI, TestDynDispFunc) {
  ASSERT_STRING_EQ(CPUCapabilityToString(CPUCapability::AMX), "AMX");
  ASSERT_STRING_EQ(CPUCapabilityToString(CPUCapability::AVX2), "AVX2");
  ASSERT_STRING_EQ(
      CPUCapabilityToString(CPUCapability::AVX2_VNNI), "AVX2_VNNI");
  ASSERT_STRING_EQ(CPUCapabilityToString(CPUCapability::AVX512), "AVX512");
  ASSERT_STRING_EQ(
      CPUCapabilityToString(CPUCapability::AVX512_BF16), "AVX512_BF16");
  ASSERT_STRING_EQ(
      CPUCapabilityToString(CPUCapability::AVX512_VNNI), "AVX512_VNNI");
  ASSERT_STRING_EQ(CPUCapabilityToString(CPUCapability::DEFAULT), "DEFAULT");

  ASSERT_STRING_EQ(
      CPUCapabilityToString(CPUCapability::NUM_OPTIONS), "OutOfBoundaryLevel");
  ASSERT_STRING_EQ(
      CPUCapabilityToString(
          (CPUCapability)((int)CPUCapability::NUM_OPTIONS + 1)),
      "WrongLevel");
}
