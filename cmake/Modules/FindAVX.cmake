INCLUDE(CheckCSourceCompiles)
INCLUDE(CheckCXXSourceCompiles)

SET(AVX512_CODE "
  #include <stdint.h>
  #include <immintrin.h>

  int main() {
    __m256i src;
    __mmask16 mask;
    int16_t addr[16];
    // detect avx512f, avx512bw and avx512vl.
    _mm512_cvtepi16_epi32(_mm256_mask_loadu_epi16(src, mask, (void *)addr));
    return 0;
  }
")

SET(AVX512_BF16_CODE "
  #include <stdint.h>
  #include <immintrin.h>

  int main() {
    __m512 src;
    // detect avx512f and avx512bf16
    _mm512_cvtneps_pbh(src);
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
        CHECK_CXX_SOURCE_COMPILES("${${type}_CODE}" ${lang}_HAS_${type}_${__FLAG_I})
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

CHECK_SSE(C "AVX512" " ;-mavx512f -mavx512bw -mavx512vl")
CHECK_SSE(CXX "AVX512" " ;-mavx512f -mavx512bw -mavx512vl")

CHECK_SSE(C "AVX512_BF16" " ;-mavx512f -mavx512bf16")
CHECK_SSE(CXX "AVX512_BF16" " ;-mavx512f -mavx512bf16")
