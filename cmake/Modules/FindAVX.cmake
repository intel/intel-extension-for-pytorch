INCLUDE(CheckCSourceCompiles)
INCLUDE(CheckCXXSourceCompiles)

SET(AVX2_CODE "
  #include <immintrin.h>

  int main()
  {
    __m256i a = {0};
    a = _mm256_abs_epi16(a);
    __m256i x;
    _mm256_extract_epi64(x, 0); // we rely on this in our AVX2 code
    return 0;
  }
")

SET(AVX2_VNNI_CODE "
  #include <stdint.h>
  #include <immintrin.h>

  int main()
  {
    char a1 = 1;
    char a2 = 2;
    char a3 = 0;
    __m256i src1 = _mm256_set1_epi8(a1);
    __m256i src2 = _mm256_set1_epi8(a2);
    __m256i src3 = _mm256_set1_epi8(a3);
    // detect avx2_vnni
    _mm256_dpbusds_epi32(src3, src1, src2);

    return 0;
  }
")

SET(AVX512_CODE "
  #include <immintrin.h>

  int main()
  {
    __m512i a = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0);
    __m512i b = a;
    __mmask64 equality_mask = _mm512_cmp_epi8_mask(a, b, _MM_CMPINT_EQ);
    return 0;
  }
")

SET(AVX512_VNNI_CODE "
  #include <stdint.h>
  #include <immintrin.h>

  int main() {
    char a1 = 1;
    char a2 = 2;
    char a3 = 0;
    __m512i src1 = _mm512_set1_epi8(a1);
    __m512i src2 = _mm512_set1_epi8(a2);
    __m512i src3 = _mm512_set1_epi8(a3);
    // detect avx512_vnni
    _mm512_dpbusds_epi32(src3, src1, src2);

    // detect avx512
    __m512i a = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0);
    __m512i b = a;
    __mmask64 equality_mask = _mm512_cmp_epi8_mask(a, b, _MM_CMPINT_EQ);
    return 0;
  }
")

SET(AVX512_BF16_CODE "
  #include <stdint.h>
  #include <immintrin.h>

  int main() {
    __m512 src;
    // detect avx512f and avx512_bf16
    _mm512_cvtneps_pbh(src);

    // Enhance check logical for Ubuntu 18.04 + gcc 11.1. Which compiler is not fully support BF16.
    __m128 a;
    __m128bh b;
    b = _mm_cvtneps_pbh(a);

    return 0;
  }
")

SET(AMX_CODE "
  #include <stdint.h>
  #include <immintrin.h>

  int main() {
    _tile_dpbusd (1, 2, 3);
    return 0;
  }
")

SET(AVX512_FP16_CODE "
  #include <stdint.h>
  #include <immintrin.h>

  int main() {
    __m512 src;
    // detect avx512f and avx512_bf16
    _mm512_cvtneps_pbh(src);
    // Enhance check logical for Ubuntu 18.04 + gcc 11.1. Which compiler is not fully support BF16.
    __m128 a;
    __m128bh b;
    b = _mm_cvtneps_pbh(a);

    // detect AMX
    _tile_dpbusd (1, 2, 3);

    // detect avx512f and avx512_fp16
    _mm512_cvtxps_ph(src);
    // Enhance check logical for Ubuntu 20.04 + gcc 12.1. Which compiler is not fully support FP16.
    __m128 a1;
    __m128h b1;
    b1 = _mm_cvtxps_ph(a1);
    return 0;
  }
")

MACRO(CHECK_SSE lang type flags)
  SET(__FLAG_I 1)
  SET(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
  FOREACH(__FLAG ${flags})
    IF(NOT ${lang}_${type}_FOUND)
      SET(CMAKE_REQUIRED_FLAGS ${__FLAG})
      IF(lang STREQUAL "CXX")
        CHECK_C_SOURCE_COMPILES("${${type}_CODE}" ${lang}_HAS_${type}_${__FLAG_I})
      ELSE()
        CHECK_C_SOURCE_COMPILES("${${type}_CODE}" ${lang}_HAS_${type}_${__FLAG_I})
      ENDIF()
      IF(${lang}_HAS_${type}_${__FLAG_I})
        SET(${lang}_${type}_FOUND TRUE CACHE BOOL "${lang} ${type} support")
        SET(${lang}_${type}_FLAGS "${__FLAG}" CACHE STRING "${lang} ${type} flags")
      ENDIF()
      MATH(EXPR __FLAG_I "${__FLAG_I}+1")
    ENDIF()
  ENDFOREACH()
  SET(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})

  IF(NOT ${lang}_${type}_FOUND)
    SET(${lang}_${type}_FOUND FALSE CACHE BOOL "${lang} ${type} support")
    SET(${lang}_${type}_FLAGS "" CACHE STRING "${lang} ${type} flags")
  ENDIF()

  MARK_AS_ADVANCED(${lang}_${type}_FOUND ${lang}_${type}_FLAGS)
ENDMACRO()

CHECK_SSE(C "AVX2" " ;-mavx2 -mfma -mf16c;/arch:AVX2")
CHECK_SSE(CXX "AVX2" " ;-mavx2 -mfma -mf16c;/arch:AVX2")

# gcc start to support avx2_vnni from version 11.2
# https://gcc.gnu.org/onlinedocs/gcc-11.2.0/gcc/x86-Options.html#x86-Options
CHECK_SSE(C "AVX2_VNNI" " ;-mavx2 -mavxvnni -mfma -mf16c;/arch:AVX2")
CHECK_SSE(CXX "AVX2_VNNI" " ;-mavx2 -mavxvnni -mfma -mf16c;/arch:AVX2")

CHECK_SSE(C "AVX512" " ;-mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma;/arch:AVX512")
CHECK_SSE(CXX "AVX512" " ;-mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma;/arch:AVX512")

# gcc version 9.2 can support this avx512_vnni.
# https://gcc.gnu.org/onlinedocs/gcc-9.2.0/gcc/x86-Options.html#x86-Options
CHECK_SSE(C "AVX512_VNNI" " ;-mavx512f -mavx512dq -mavx512vl -mavx512bw -mavx512vnni -mfma;/arch:AVX512")
CHECK_SSE(CXX "AVX512_VNNI" " ;-mavx512f -mavx512dq -mavx512vl -mavx512bw -mavx512vnni -mfma;/arch:AVX512")

# gcc start to support avx512bf16 from version 10.3
# https://gcc.gnu.org/onlinedocs/gcc-10.3.0/gcc/x86-Options.html#x86-Options
CHECK_SSE(C "AVX512_BF16" " ;-mavx512f -mavx512dq -mavx512vl -mavx512bw -mavx512bf16 -mfma;/arch:AVX512")
CHECK_SSE(CXX "AVX512_BF16" " ;-mavx512f -mavx512dq -mavx512vl -mavx512bw -mavx512bf16 -mfma;/arch:AVX512")

# gcc start to support amx from version 11.2
# https://gcc.gnu.org/onlinedocs/gcc-11.2.0/gcc/x86-Options.html#x86-Options
CHECK_SSE(C "AMX" " ;-mavx512f -mavx512dq -mavx512vl -mavx512bw -mavx512bf16 -mfma\
 -mamx-tile -mamx-int8 -mamx-bf16;/arch:AVX512")
CHECK_SSE(CXX "AMX" " ;-mavx512f -mavx512dq -mavx512vl -mavx512bw -mavx512bf16 -mfma\
 -mamx-tile -mamx-int8 -mamx-bf16;/arch:AVX512")

# gcc starts to support avx512fp16 from version 12.1
# https://gcc.gnu.org/onlinedocs/gcc-12.1.0/gcc/x86-Options.html#x86-Options
CHECK_SSE(C "AVX512_FP16" " ;-mavx512f -mavx512dq -mavx512vl -mavx512bw -mavx512bf16 -mfma\
 -mamx-tile -mamx-int8 -mamx-bf16 -mavx512fp16;/arch:AVX512")
CHECK_SSE(CXX "AVX512_FP16" " ;-mavx512f -mavx512dq -mavx512vl -mavx512bw -mavx512bf16 -mfma\
 -mamx-tile -mamx-int8 -mamx-bf16 -mavx512fp16;/arch:AVX512")