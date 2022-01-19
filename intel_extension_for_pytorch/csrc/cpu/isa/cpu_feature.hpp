#pragma once

#define MICRO_CLASS_MEMBER_DECL(feature_name) bool m_##feature_name = false
#define MICRO_CLASS_MEMBER(feature_name) m_##feature_name
#define MICRO_CLASS_CHECK_FUNC(feature_name) \
  bool cpuid_##feature_name() {              \
    return m_##feature_name;                 \
  }
#define MICRO_CLASS_PRINT_BOOL_STATUS(feature_name) \
  print_bool_status(#feature_name, m_##feature_name)

namespace torch_ipex {
namespace cpu {
class CPUFeature {
 private:
  CPUFeature();

  void detect_intel_cpu_feature();

 public:
  static CPUFeature& get_instance();
  void show_features();

 public:
  bool os_avx();
  bool os_avx2();
  bool os_avx512();
  bool os_amx();

 private:
  MICRO_CLASS_MEMBER_DECL(mmx);
  MICRO_CLASS_MEMBER_DECL(sse);
  MICRO_CLASS_MEMBER_DECL(sse2);
  MICRO_CLASS_MEMBER_DECL(sse3);
  MICRO_CLASS_MEMBER_DECL(ssse3);
  MICRO_CLASS_MEMBER_DECL(sse4_1);
  MICRO_CLASS_MEMBER_DECL(sse4_2);
  MICRO_CLASS_MEMBER_DECL(aes_ni);
  MICRO_CLASS_MEMBER_DECL(sha);

  MICRO_CLASS_MEMBER_DECL(xsave);

  MICRO_CLASS_MEMBER_DECL(avx);
  MICRO_CLASS_MEMBER_DECL(avx2);
  MICRO_CLASS_MEMBER_DECL(avx_vnni);

  MICRO_CLASS_MEMBER_DECL(fma);
  MICRO_CLASS_MEMBER_DECL(f16c);

 public:
  MICRO_CLASS_CHECK_FUNC(sse);
  MICRO_CLASS_CHECK_FUNC(sse2);
  MICRO_CLASS_CHECK_FUNC(sse3);
  MICRO_CLASS_CHECK_FUNC(ssse3);
  MICRO_CLASS_CHECK_FUNC(sse4_1);
  MICRO_CLASS_CHECK_FUNC(sse4_2);
  MICRO_CLASS_CHECK_FUNC(aes_ni);
  MICRO_CLASS_CHECK_FUNC(sha);

  MICRO_CLASS_CHECK_FUNC(xsave);

  MICRO_CLASS_CHECK_FUNC(avx);
  MICRO_CLASS_CHECK_FUNC(avx2);
  MICRO_CLASS_CHECK_FUNC(avx_vnni);

  MICRO_CLASS_CHECK_FUNC(fma);
  MICRO_CLASS_CHECK_FUNC(f16c);
  // AVX512
 private:
  MICRO_CLASS_MEMBER_DECL(avx512_f);
  MICRO_CLASS_MEMBER_DECL(avx512_cd);
  MICRO_CLASS_MEMBER_DECL(avx512_pf);
  MICRO_CLASS_MEMBER_DECL(avx512_er);
  MICRO_CLASS_MEMBER_DECL(avx512_vl);
  MICRO_CLASS_MEMBER_DECL(avx512_bw);
  MICRO_CLASS_MEMBER_DECL(avx512_dq);
  MICRO_CLASS_MEMBER_DECL(avx512_ifma);
  MICRO_CLASS_MEMBER_DECL(avx512_vbmi);
  MICRO_CLASS_MEMBER_DECL(avx512_vpopcntdq);
  MICRO_CLASS_MEMBER_DECL(avx512_4fmaps);
  MICRO_CLASS_MEMBER_DECL(avx512_4vnniw);
  MICRO_CLASS_MEMBER_DECL(avx512_vbmi2);
  MICRO_CLASS_MEMBER_DECL(avx512_vpclmul);
  MICRO_CLASS_MEMBER_DECL(avx512_vnni);
  MICRO_CLASS_MEMBER_DECL(avx512_bitalg);
  MICRO_CLASS_MEMBER_DECL(avx512_fp16);
  MICRO_CLASS_MEMBER_DECL(avx512_bf16);
  MICRO_CLASS_MEMBER_DECL(avx512_vp2intersect);

 public:
  MICRO_CLASS_CHECK_FUNC(avx512_f);
  MICRO_CLASS_CHECK_FUNC(avx512_cd);
  MICRO_CLASS_CHECK_FUNC(avx512_pf);
  MICRO_CLASS_CHECK_FUNC(avx512_er);
  MICRO_CLASS_CHECK_FUNC(avx512_vl);
  MICRO_CLASS_CHECK_FUNC(avx512_bw);
  MICRO_CLASS_CHECK_FUNC(avx512_dq);
  MICRO_CLASS_CHECK_FUNC(avx512_ifma);
  MICRO_CLASS_CHECK_FUNC(avx512_vbmi);
  MICRO_CLASS_CHECK_FUNC(avx512_vpopcntdq);
  MICRO_CLASS_CHECK_FUNC(avx512_4fmaps);
  MICRO_CLASS_CHECK_FUNC(avx512_4vnniw);
  MICRO_CLASS_CHECK_FUNC(avx512_vbmi2);
  MICRO_CLASS_CHECK_FUNC(avx512_vpclmul);
  MICRO_CLASS_CHECK_FUNC(avx512_vnni);
  MICRO_CLASS_CHECK_FUNC(avx512_bitalg);
  MICRO_CLASS_CHECK_FUNC(avx512_fp16);
  MICRO_CLASS_CHECK_FUNC(avx512_bf16);
  MICRO_CLASS_CHECK_FUNC(avx512_vp2intersect);

  // AMX
 private:
  MICRO_CLASS_MEMBER_DECL(amx_bf16);
  MICRO_CLASS_MEMBER_DECL(amx_tile);
  MICRO_CLASS_MEMBER_DECL(amx_int8);

 public:
  MICRO_CLASS_CHECK_FUNC(amx_bf16);
  MICRO_CLASS_CHECK_FUNC(amx_tile);
  MICRO_CLASS_CHECK_FUNC(amx_int8);

  // prefetch
 private:
  MICRO_CLASS_MEMBER_DECL(prefetchw);
  MICRO_CLASS_MEMBER_DECL(prefetchwt1);

 public:
  MICRO_CLASS_CHECK_FUNC(prefetchw);
  MICRO_CLASS_CHECK_FUNC(prefetchwt1);
};
} // namespace cpu
} // namespace torch_ipex