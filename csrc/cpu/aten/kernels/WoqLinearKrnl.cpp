#ifdef USE_LIBXSMM
#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <aten/Linear.h>
#include <aten/utils/woq.h>
#include <emmintrin.h>
#include <libxsmm.h>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include "mkl.h"

namespace torch_ipex {
namespace cpu {
namespace {

void print_matrix(float* A, int m, int n, int ld) {
  for (int i = 0; i < m; i++) {
    float* A_ = A + i * ld;
    for (int j = 0; j < n; j++) {
      std::cout << std::setprecision(23) << A_[j] << " ";
    }
    std::cout << std::endl;
  }
}

void print_matrix(const uint8_t* A, int m, int n, int ld) {
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      const uint8_t* A_ = A + i * ld;
      uint8_t tmp = (A_[j] << 4) | ((A_[j] & 0xF0) >> 4);
      printf("%02X ", tmp);
    }
    printf("\n");
  }
}

// TODO implement optimized kernels for fused op
// then the following part will be discarded
using PostopFunc = std::function<at::Tensor&(at::Tensor&)>;
using PostopFuncGetter = std::function<PostopFunc(
    const torch::List<c10::optional<at::Scalar>>&,
    const c10::optional<c10::string_view>&)>;

static PostopFuncGetter postop_func_none =
    [](const torch::List<c10::optional<at::Scalar>>&,
       const c10::optional<c10::string_view>&) {
      return [](at::Tensor& t) -> at::Tensor& { return t; };
    };

static PostopFuncGetter postop_func_relu =
    [](const torch::List<c10::optional<at::Scalar>>&,
       const c10::optional<c10::string_view>&) { return at::relu_; };

static PostopFuncGetter postop_func_gelu =
    [](const torch::List<c10::optional<at::Scalar>>&,
       const c10::optional<c10::string_view>& algorithm) {
      assert(
          algorithm.has_value() &&
          (algorithm == "none" || algorithm == "tanh"));
      return [=](at::Tensor& t) -> at::Tensor& {
        return at::gelu_(t, algorithm.value());
      };
    };

static std::map<c10::string_view, PostopFuncGetter> postop_func_map = {
    {"none", postop_func_none},
    {"relu", postop_func_relu},
    {"gelu", postop_func_gelu}};

#if defined(CPU_CAPABILITY_AVX512)
#include <immintrin.h>

inline __m256i cvt_fp32_to_bf16(const __m512 src) {
#if (defined CPU_CAPABILITY_AVX512_BF16)
  return reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(src));
#else
  __m512i value = _mm512_castps_si512(src);
  __m512i nan = _mm512_set1_epi32(0xffff);
  auto mask_value = _mm512_cmp_ps_mask(src, src, _CMP_ORD_Q);
  __m512i ones = _mm512_set1_epi32(0x1);
  __m512i vec_bias = _mm512_set1_epi32(0x7fff);
  // uint32_t lsb = (input >> 16) & 1;
  auto t_value = _mm512_and_si512(_mm512_srli_epi32(value, 16), ones);
  // uint32_t rounding_bias = 0x7fff + lsb;
  t_value = _mm512_add_epi32(t_value, vec_bias);
  // input += rounding_bias;
  t_value = _mm512_add_epi32(t_value, value);
  // input = input >> 16;
  t_value = _mm512_srli_epi32(t_value, 16);
  // Check NaN before converting back to bf16
  t_value = _mm512_mask_blend_epi32(mask_value, nan, t_value);
  return _mm512_cvtusepi32_epi16(t_value);
#endif
}

inline void cvt_fp32_to_bf16(
    BFloat16* dst,
    const float* src,
    int m,
    int n,
    int ld_dst,
    int ld_src) {
  for (int i = 0; i < m; i++) {
    const float* src0 = src + i * ld_src;
    BFloat16* dst0 = dst + i * ld_dst;
    int j;
    for (j = 0; j < n - 15; j += 16) {
      auto f32 = _mm512_loadu_ps(src0 + j);
      _mm256_storeu_si256((__m256i*)(dst0 + j), cvt_fp32_to_bf16(f32));
    }
    if (j < n) {
      auto mask = (1 << (n - j)) - 1;
      auto f32 = _mm512_maskz_loadu_ps(mask, src0 + j);
      _mm256_mask_storeu_epi16(dst0 + j, mask, cvt_fp32_to_bf16(f32));
    }
  }
}

inline __m512 cvt_bf16_to_fp32(const __m256i src) {
  auto y = _mm512_cvtepu16_epi32(src);
  return _mm512_castsi512_ps(_mm512_bslli_epi128(y, 2));
}

inline void cvt_bf16_to_fp32(
    float* dst,
    const at::BFloat16* src,
    int m,
    int n,
    int ld_dst,
    int ld_src) {
  for (int i = 0; i < m; i++) {
    float* dst0 = dst + i * ld_dst;
    const BFloat16* src0 = src + i * ld_src;
    int j = 0;
    for (; j < n - 15; j += 16) {
      auto f32 = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(src0 + j)));
      _mm512_storeu_ps(dst0 + j, f32);
    }
    if (j < n) {
      auto mask = (1 << (n - j)) - 1;
      auto f32 = cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, src0 + j));
      _mm512_mask_storeu_ps(dst0 + j, mask, f32);
    }
  }
}

void print_m512(__m512 reg) {
  float temp[16];
  _mm512_store_ps(temp, reg);
  for (int i = 0; i < 16; ++i) {
    printf("%.2f ", temp[i]);
  }
  printf("\n");
}

void print_m512i(__m512i reg) {
  int temp[16];
  _mm512_store_epi32(temp, reg);
  for (int i = 0; i < 16; ++i) {
    printf("%08X ", temp[i]);
  }
  printf("\n");
}

void print_m128i(__m128i* reg) {
  uint8_t temp[32];
  _mm_store_si128((__m128i*)temp, *reg);
  for (int i = 0; i < 32; ++i) {
    std::bitset<8> b(temp[i]);
    std::cout << b << " ";
  }
  std::cout << std::endl;
}
// This function is for the case of very small M.
// LINES should be  smaller than 4.
// N must be smaller than 64 and must be multiple of 16.
// PREFETCH_K_DIST means prefetch distance in K.
// ACC means accumulate to C or not.
// actualN , rowOff are not used in current code version.
template <
    int LINES,
    int N,
    int PREFETCH_K_DIST,
    bool ACC,
    bool bias_add = false>
void small_gemm_smallm(
    const float* A,
    const int8_t* B,
    float* C,
    int lda,
    int ldb,
    int ldc,
    int actualN,
    int K,
    float* scale,
    float* zero_point,
    float* bias = NULL,
    int rowOff = 0) {
  constexpr const int COLS = N / 16;

  __m512 va;
  __m512 vb[COLS];
  __m512 vc[LINES * COLS];
  __m512 float_scale[COLS];
  __m512 float_zero_point[COLS];

  // Load scale
  auto load_scale = [&](auto i) {
    float_scale[i] = _mm512_loadu_ps(scale + 16 * i);
  };
  compile_time_for<COLS>::op(load_scale);

  // Load zero point
  auto load_zp = [&](auto i) {
    float_zero_point[i] = _mm512_loadu_ps(zero_point + 16 * i);
  };
  compile_time_for<COLS>::op(load_zp);

  // Load from C or set to 0
  if constexpr (ACC) {
    auto loadc = [&](auto i) {
      constexpr const int row = i / COLS;
      constexpr const int col = i % COLS;
      vc[i] = _mm512_loadu_ps(ADDRESS(C, row, col * 16, ldc));
    };
    compile_time_for<LINES * COLS>::op(loadc);
  } else {
    auto set0 = [&](auto i) { vc[i] = _mm512_setzero_ps(); };
    compile_time_for<LINES * COLS>::op(set0);
  }

  auto compute = [&](auto i, int k) {
    constexpr const int row = i / COLS;
    constexpr const int col = i % COLS;

    if constexpr (col == 0) {
      va = _mm512_set1_ps(*ADDRESS(A, row, k, lda));
    }

    if constexpr (row == 0) {
      const __m128i b_ =
          _mm_loadu_si128((const __m128i*)ADDRESS(B, k, col * 16, ldb));
      _mm_prefetch(ADDRESS(B, k + PREFETCH_K_DIST, col * 16, ldb), _MM_HINT_T0);
      vb[col] = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b_));
      vb[col] = _mm512_sub_ps(vb[col], float_zero_point[col]);
      vb[col] = _mm512_mul_ps(vb[col], float_scale[col]);
    }

    constexpr const int idx = INDEX(row, col, COLS);
    vc[idx] = _mm512_fmadd_ps(va, vb[col], vc[idx]);
  };

// Accumulate along k
#pragma unroll(4)
  for (int k = 0; k < K; ++k) {
    compile_time_for<LINES * COLS>::op(compute, k);
  }

  // Store to C
  auto store = [&](auto i) {
    constexpr const int line = i / COLS;
    constexpr const int col = i % COLS;
    if constexpr (bias_add) {
      __m512 bias_ = _mm512_loadu_ps(bias + col * 16);
      vc[i] = _mm512_add_ps(vc[i], bias_);
      _mm512_storeu_ps(ADDRESS(C, line, col * 16, ldc), vc[i]);
    } else {
      _mm512_storeu_ps(ADDRESS(C, line, col * 16, ldc), vc[i]);
    }
  };

  compile_time_for<LINES * COLS>::op(store);
}

// bf16 * int8 -> fp32
template <
    int LINES,
    int N,
    int PREFETCH_K_DIST,
    bool ACC,
    bool bias_add = false>
void small_gemm_smallm(
    const BFloat16* A,
    const int8_t* B,
    float* C,
    int lda,
    int ldb,
    int ldc,
    int actualN,
    int K,
    float* scale,
    float* zero_point,
    float* bias = NULL,
    int rowOff = 0) {
  constexpr const int COLS = N / 16;

  __m512 va;
  __m512 vb[COLS];
  __m512 vc[LINES * COLS];
  __m512 float_scale[COLS];
  __m512 float_zero_point[COLS];

  // Load scale
  auto load_scale = [&](auto i) {
    float_scale[i] = _mm512_loadu_ps(scale + 16 * i);
  };
  compile_time_for<COLS>::op(load_scale);

  // Load zero point
  auto load_zp = [&](auto i) {
    float_zero_point[i] = _mm512_loadu_ps(zero_point + 16 * i);
  };
  compile_time_for<COLS>::op(load_zp);

  // Load from C or set to 0
  if constexpr (ACC) {
    auto loadc = [&](auto i) {
      constexpr const int row = i / COLS;
      constexpr const int col = i % COLS;
      vc[i] = _mm512_loadu_ps(ADDRESS(C, row, col * 16, ldc));
    };
    compile_time_for<LINES * COLS>::op(loadc);
  } else {
    auto set0 = [&](auto i) { vc[i] = _mm512_setzero_ps(); };
    compile_time_for<LINES * COLS>::op(set0);
  }

  auto compute = [&](auto i, int k) {
    constexpr const int row = i / COLS;
    constexpr const int col = i % COLS;

    if constexpr (col == 0) {
      float aa = *ADDRESS(A, row, k, lda); // convert from bf16 to fp32
      va = _mm512_set1_ps(aa);
    }

    if constexpr (row == 0) {
      const __m128i b_ =
          _mm_loadu_si128((const __m128i*)ADDRESS(B, k, col * 16, ldb));
      _mm_prefetch(ADDRESS(B, k + PREFETCH_K_DIST, col * 16, ldb), _MM_HINT_T0);
      vb[col] = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b_));
      vb[col] = _mm512_sub_ps(vb[col], float_zero_point[col]);
      vb[col] = _mm512_mul_ps(vb[col], float_scale[col]);
    }

    constexpr const int idx = INDEX(row, col, COLS);
    vc[idx] = _mm512_fmadd_ps(va, vb[col], vc[idx]);
  };

// Accumulate along k
#pragma unroll(4)
  for (int k = 0; k < K; ++k) {
    compile_time_for<LINES * COLS>::op(compute, k);
  }

  // Store to C
  auto store = [&](auto i) {
    constexpr const int line = i / COLS;
    constexpr const int col = i % COLS;
    if constexpr (bias_add) {
      __m512 bias_ = _mm512_loadu_ps(bias + col * 16);
      vc[i] = _mm512_add_ps(vc[i], bias_);
      _mm512_storeu_ps(ADDRESS(C, line, col * 16, ldc), vc[i]);
    } else {
      _mm512_storeu_ps(ADDRESS(C, line, col * 16, ldc), vc[i]);
    }
  };

  compile_time_for<LINES * COLS>::op(store);
}

static inline __m128i bytesFromNibbles(const uint8_t* rsi) {
  __m128i tmp = _mm_loadu_si64((const __m128i*)rsi);
  __m128i bytes = _mm_cvtepu8_epi16(tmp);
  const __m128i lowMask = _mm_set1_epi8(0xF);
  __m128i high = _mm_andnot_si128(lowMask, bytes);
  __m128i low = _mm_and_si128(lowMask, bytes);
  high = _mm_slli_epi16(high, 4);
  bytes = _mm_or_si128(low, high);
  return bytes;
}

// fp32 * int4 -> fp32
template <
    int LINES,
    int N,
    int PREFETCH_K_DIST,
    bool ACC,
    bool bias_add = false>
void small_gemm_smallm(
    const float* A,
    const uint8_t* B,
    float* C,
    int lda,
    int ldb,
    int ldc,
    int actualN,
    int K,
    float* scale,
    float* zero_point,
    float* bias = NULL,
    int rowOff = 0) {
  constexpr int COLS = N / 16;
  __m512 va;
  __m512 vb[COLS];
  __m512 vc[LINES * COLS];
  __m512 float_scale[COLS];
  __m512 float_zero_point[COLS];
  // lookup table converting uint8 to float, 15.0f - 0.0f
  __m512 lut = _mm512_set_ps(
      15.0f,
      14.0f,
      13.0f,
      12.0f,
      11.0f,
      10.0f,
      9.0f,
      8.0f,
      7.0f,
      6.0f,
      5.0f,
      4.0f,
      3.0f,
      2.0f,
      1.0f,
      0.0f);

  // Load scale
  auto load_scale = [&](auto i) {
    float_scale[i] = _mm512_loadu_ps(scale + 16 * i);
  };
  compile_time_for<COLS>::op(load_scale);

  // Load zero point
  auto load_zp = [&](auto i) {
    float_zero_point[i] = _mm512_loadu_ps(zero_point + 16 * i);
  };
  compile_time_for<COLS>::op(load_zp);

  // Load from C or set to 0
  if constexpr (ACC) {
    auto loadc = [&](auto i) {
      constexpr const int row = i / COLS;
      constexpr const int col = i % COLS;
      vc[i] = _mm512_loadu_ps(ADDRESS(C, row, col * 16, ldc));
    };
    compile_time_for<LINES * COLS>::op(loadc);
  } else {
    auto set0 = [&](auto i) { vc[i] = _mm512_setzero_ps(); };
    compile_time_for<LINES * COLS>::op(set0);
  }

  auto compute = [&](auto i, int k) {
    constexpr const int row = i / COLS;
    constexpr const int col = i % COLS;

    if constexpr (col == 0) {
      va = _mm512_set1_ps(*ADDRESS(A, row, k, lda));
    }

#define ALGORITHM 1

#if ALGORITHM == 0
    if constexpr (row == 0) {
      __m128i b_ =
          bytesFromNibbles(ADDRESS(B, k, col * 8, ldb / 2)); // int4 -> int8
      _mm_prefetch(
          ADDRESS(B, k + PREFETCH_K_DIST, col * 8, ldb / 2), _MM_HINT_T0);
      vb[col] = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(b_));
      vb[col] = _mm512_sub_ps(vb[col], float_zero_point[col]);
      vb[col] = _mm512_mul_ps(vb[col], float_scale[col]);
    }
#else
    // GCC < 11.3 internal compiler error with constexpr
    if (col == 0 && row == 0) {
      static_assert(COLS == 4, "expect register block size 4 for weights");
      _mm_prefetch(ADDRESS(B, k + PREFETCH_K_DIST, 0, ldb / 2), _MM_HINT_T0);
      // load 64 elements from ADDRESS(B, k, 0 ldb / 2) with 4-bit each
      // and then, unpack them and convert them into 64 fp32 numbers held in
      // four avx512 registers: vb[0] - vb[3]
      // Load the buffer into a 256-bit register
      __m256i packed = _mm256_load_si256((__m256i*)ADDRESS(B, k, 0, ldb / 2));
      __m512i int32[4];
      {
        auto low_4bit = _mm512_cvtepu8_epi32(_mm256_castsi256_si128(packed));
        auto high_4bit = _mm512_srli_epi32(low_4bit, 4);
        int32[0] = low_4bit;
        int32[2] = high_4bit;
      }
      {
        auto low_4bit =
            _mm512_cvtepu8_epi32(_mm256_extracti128_si256(packed, 1));
        auto high_4bit = _mm512_srli_epi32(low_4bit, 4);
        int32[1] = low_4bit;
        int32[3] = high_4bit;
      }

      auto dequant_int32 = [&](auto idx) {
        vb[idx] = _mm512_permutexvar_ps(int32[idx], lut);
        vb[idx] = _mm512_sub_ps(vb[idx], float_zero_point[idx]);
        vb[idx] = _mm512_mul_ps(vb[idx], float_scale[idx]);
      };
      compile_time_for<COLS>::op(dequant_int32);
    }
#endif

    constexpr const int idx = INDEX(row, col, COLS);
    vc[idx] = _mm512_fmadd_ps(va, vb[col], vc[idx]);
  };

// Accumulate along k
#pragma unroll(4)
  for (int k = 0; k < K; ++k) {
    compile_time_for<LINES * COLS>::op(compute, k);
  }

  // Store to C
  auto store = [&](auto i) {
    constexpr const int line = i / COLS;
    constexpr const int col = i % COLS;
    if constexpr (bias_add) {
      __m512 bias_ = _mm512_loadu_ps(bias + col * 16);
      vc[i] = _mm512_add_ps(vc[i], bias_);
      _mm512_storeu_ps(ADDRESS(C, line, col * 16, ldc), vc[i]);
    } else {
      _mm512_storeu_ps(ADDRESS(C, line, col * 16, ldc), vc[i]);
    }
  };

  compile_time_for<LINES * COLS>::op(store);
}

// bf16 * int4 -> fp32
template <
    int LINES,
    int N,
    int PREFETCH_K_DIST,
    bool ACC,
    bool bias_add = false>
void small_gemm_smallm(
    const BFloat16* A,
    const uint8_t* B,
    float* C,
    int lda,
    int ldb,
    int ldc,
    int actualN,
    int K,
    float* scale,
    float* zero_point,
    float* bias = NULL,
    int rowOff = 0) {
  constexpr int COLS = N / 16;
  __m512 va;
  __m512 vb[COLS];
  __m512 vc[LINES * COLS];
  __m512 float_scale[COLS];
  __m512 float_zero_point[COLS];
  // lookup table converting uint8 to float, 15.0f - 0.0f
  __m512 lut = _mm512_set_ps(
      15.0f,
      14.0f,
      13.0f,
      12.0f,
      11.0f,
      10.0f,
      9.0f,
      8.0f,
      7.0f,
      6.0f,
      5.0f,
      4.0f,
      3.0f,
      2.0f,
      1.0f,
      0.0f);

  // Load scale
  auto load_scale = [&](auto i) {
    float_scale[i] = _mm512_loadu_ps(scale + 16 * i);
  };
  compile_time_for<COLS>::op(load_scale);

  // Load zero point
  auto load_zp = [&](auto i) {
    float_zero_point[i] = _mm512_loadu_ps(zero_point + 16 * i);
  };
  compile_time_for<COLS>::op(load_zp);

  // Load from C or set to 0
  if constexpr (ACC) {
    auto loadc = [&](auto i) {
      constexpr const int row = i / COLS;
      constexpr const int col = i % COLS;
      vc[i] = _mm512_loadu_ps(ADDRESS(C, row, col * 16, ldc));
    };
    compile_time_for<LINES * COLS>::op(loadc);
  } else {
    auto set0 = [&](auto i) { vc[i] = _mm512_setzero_ps(); };
    compile_time_for<LINES * COLS>::op(set0);
  }

  auto compute = [&](auto i, int k) {
    constexpr const int row = i / COLS;
    constexpr const int col = i % COLS;

    if constexpr (col == 0) {
      float aa = *ADDRESS(A, row, k, lda); // convert from bf16 to fp32
      va = _mm512_set1_ps(aa);
    }

#define ALGORITHM 1

#if ALGORITHM == 0
    if constexpr (row == 0) {
      __m128i b_ =
          bytesFromNibbles(ADDRESS(B, k, col * 8, ldb / 2)); // int4 -> int8
      _mm_prefetch(
          ADDRESS(B, k + PREFETCH_K_DIST, col * 8, ldb / 2), _MM_HINT_T0);
      vb[col] = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(b_));
      vb[col] = _mm512_sub_ps(vb[col], float_zero_point[col]);
      vb[col] = _mm512_mul_ps(vb[col], float_scale[col]);
    }
#else
    // GCC < 11.3 internal compiler error with constexpr
    if (col == 0 && row == 0) {
      static_assert(COLS == 4, "expect register block size 4 for weights");
      _mm_prefetch(ADDRESS(B, k + PREFETCH_K_DIST, 0, ldb / 2), _MM_HINT_T0);
      // load 64 elements from ADDRESS(B, k, 0 ldb / 2) with 4-bit each
      // and then, unpack them and convert them into 64 fp32 numbers held in
      // four avx512 registers: vb[0] - vb[3]
      // Load the buffer into a 256-bit register
      __m256i packed = _mm256_load_si256((__m256i*)ADDRESS(B, k, 0, ldb / 2));
      __m512i int32[4];
      {
        auto low_4bit = _mm512_cvtepu8_epi32(_mm256_castsi256_si128(packed));
        auto high_4bit = _mm512_srli_epi32(low_4bit, 4);
        int32[0] = low_4bit;
        int32[2] = high_4bit;
      }
      {
        auto low_4bit =
            _mm512_cvtepu8_epi32(_mm256_extracti128_si256(packed, 1));
        auto high_4bit = _mm512_srli_epi32(low_4bit, 4);
        int32[1] = low_4bit;
        int32[3] = high_4bit;
      }

      auto dequant_int32 = [&](auto idx) {
        vb[idx] = _mm512_permutexvar_ps(int32[idx], lut);
        vb[idx] = _mm512_sub_ps(vb[idx], float_zero_point[idx]);
        vb[idx] = _mm512_mul_ps(vb[idx], float_scale[idx]);
      };
      compile_time_for<COLS>::op(dequant_int32);
    }
#endif

    constexpr const int idx = INDEX(row, col, COLS);
    vc[idx] = _mm512_fmadd_ps(va, vb[col], vc[idx]);
  };

// Accumulate along k
#pragma unroll(4)
  for (int k = 0; k < K; ++k) {
    compile_time_for<LINES * COLS>::op(compute, k);
  }

  // Store to C
  auto store = [&](auto i) {
    constexpr const int line = i / COLS;
    constexpr const int col = i % COLS;
    if constexpr (bias_add) {
      __m512 bias_ = _mm512_loadu_ps(bias + col * 16);
      vc[i] = _mm512_add_ps(vc[i], bias_);
      _mm512_storeu_ps(ADDRESS(C, line, col * 16, ldc), vc[i]);
    } else {
      _mm512_storeu_ps(ADDRESS(C, line, col * 16, ldc), vc[i]);
    }
  };

  compile_time_for<LINES * COLS>::op(store);
}

inline void dequant_(
    int8_t* B,
    float* b,
    __m512 float_scale,
    __m512 float_zero_point) {
  const __m128i b_ = _mm_loadu_si128((const __m128i*)B);
  __m512 vb;
  vb = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b_));
  vb = _mm512_sub_ps(vb, float_zero_point);
  vb = _mm512_mul_ps(vb, float_scale);
  _mm512_storeu_ps(b, vb);
}

inline void dequant_(
    int8_t* B,
    float* b,
    __m512 float_scale,
    __m512 float_zero_point,
    unsigned short mask) {
  const __m128i b_ = _mm_maskz_loadu_epi8(mask, (const __m128i*)B);
  __m512 vb;
  vb = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b_));
  vb = _mm512_maskz_sub_ps(mask, vb, float_zero_point);
  vb = _mm512_maskz_mul_ps(mask, vb, float_scale);
  _mm512_mask_storeu_ps(b, mask, vb);
}

inline void dequant_(
    uint8_t* B,
    float* b,
    __m512 float_scale,
    __m512 float_zero_point) {
  __m128i b_ = bytesFromNibbles(B); // 32 int4 -> 32 int8
  __m512 vb;
  vb = _mm512_mul_ps(
      _mm512_sub_ps(
          _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(b_)), float_zero_point),
      float_scale);
  _mm512_storeu_ps(b, vb);
}

inline void dequant_to_bf16_(
    uint8_t* B,
    BFloat16* b,
    __m512 float_scale,
    __m512 float_zero_point) {
  __m128i b_ = bytesFromNibbles(B); // 32 int4 -> 32 int8
  __m512 vb;
  vb = _mm512_mul_ps(
      _mm512_sub_ps(
          _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(b_)), float_zero_point),
      float_scale);
  _mm256_storeu_si256((__m256i*)b, cvt_fp32_to_bf16(vb));
}

inline void dequant_to_bf16_(
    int8_t* B,
    BFloat16* b,
    __m512 float_scale,
    __m512 float_zero_point) {
  const __m128i b_ = _mm_loadu_si128((const __m128i*)B);
  __m512 vb;
  vb = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b_));
  vb = _mm512_sub_ps(vb, float_zero_point);
  vb = _mm512_mul_ps(vb, float_scale);
  _mm256_storeu_si256((__m256i*)b, cvt_fp32_to_bf16(vb));
}

inline void dequant_to_bf16_(
    int8_t* B,
    BFloat16* b,
    __m512 float_scale,
    __m512 float_zero_point,
    unsigned short mask) {
  const __m128i b_ = _mm_maskz_loadu_epi8(mask, (const __m128i*)B);
  __m512 vb;
  vb = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b_));
  vb = _mm512_maskz_sub_ps(mask, vb, float_zero_point);
  vb = _mm512_maskz_mul_ps(mask, vb, float_scale);
  _mm256_mask_storeu_epi16(b, mask, cvt_fp32_to_bf16(vb));
}

// per channel
// B is packed and not transposed, shape:[BLOCK_K x BLOCK_N]
template <int BLOCK_K, int BLOCK_N>
void dequant(int8_t* B, float* b, float* scale, float* zero_point) {
  const int COLS = BLOCK_N / 16;
  for (int k = 0; k < BLOCK_K; k++) {
    int8_t* src = B;
    float* dst = b;
    int j, idx;
    for (idx = 0, j = 0; j < COLS * 16; j += 16) {
      __m512 float_scale = _mm512_loadu_ps(scale + j);
      __m512 float_zero_point = _mm512_loadu_ps(zero_point + j);
      dequant_(src, dst, float_scale, float_zero_point);
      src += 16;
      dst += 16;
    }
    if (j < BLOCK_N) {
      const int res = BLOCK_N - j;
      unsigned short mask = 0xffff;
      mask = (1 << res) - 1;
      __m512 float_scale = _mm512_maskz_loadu_ps(mask, scale + j);
      __m512 float_zero_point = _mm512_maskz_loadu_ps(mask, zero_point + j);
      dequant_(src, dst, float_scale, float_zero_point, mask);
    }
    B += BLOCK_N;
    b += BLOCK_N;
  }
}

// per channel dequant to fp32
// handle edge cases
void dequant(
    int8_t* B,
    float* b,
    int K,
    int N,
    float* scale,
    float* zero_point) {
  const int COLS = N / 16;
  for (int k = 0; k < K; k++) {
    int8_t* src = B;
    float* dst = b;
    int j;
    for (j = 0; j < COLS * 16; j += 16) {
      __m512 float_scale = _mm512_loadu_ps(scale + j);
      __m512 float_zero_point = _mm512_loadu_ps(zero_point + j);
      dequant_(src, dst, float_scale, float_zero_point);
      src += 16;
      dst += 16;
    }
    if (j < N) {
      const int res = N - j;
      unsigned short mask = 0xffff;
      mask = (1 << res) - 1;
      __m512 float_scale = _mm512_maskz_loadu_ps(mask, scale + j);
      __m512 float_zero_point = _mm512_maskz_loadu_ps(mask, zero_point + j);
      dequant_(src, dst, float_scale, float_zero_point, mask);
    }
    B += N;
    b += N;
  }
}

// per channel dequant to bf16
// handle edge cases
void dequant(
    int8_t* B,
    BFloat16* b,
    int K,
    int N,
    float* scale,
    float* zero_point) {
  const int COLS = N / 16;
  for (int k = 0; k < K; k++) {
    int8_t* src = B;
    BFloat16* dst = b;
    int j = 0;
    for (; j < COLS * 16; j += 16) {
      __m512 float_scale = _mm512_loadu_ps(scale + j);
      __m512 float_zero_point = _mm512_loadu_ps(zero_point + j);
      dequant_to_bf16_(src, dst, float_scale, float_zero_point);
      src += 16;
      dst += 16;
    }
    if (j < N) {
      const int res = N - j;
      unsigned short mask = 0xffff;
      mask = (1 << res) - 1;
      __m512 float_scale = _mm512_maskz_loadu_ps(mask, scale + j);
      __m512 float_zero_point = _mm512_maskz_loadu_ps(mask, zero_point + j);
      dequant_to_bf16_(src, dst, float_scale, float_zero_point, mask);
    }
    B += N;
    b += N;
  }
}

// per tensor
// B is packed and not transposed, shape:[BLOCK_K x BLOCK_N]
template <int BLOCK_K, int BLOCK_N>
void dequant(int8_t* B, float* b, float scale, float zero_point) {
  __m512 float_scale = _mm512_set1_ps(scale);
  __m512 float_zero_point = _mm512_set1_ps(zero_point);
  int COLS = BLOCK_N / 16;
  for (int k = 0; k < BLOCK_K; k++) {
    int8_t* src = B;
    float* dst = b;
    int j;
    for (j = 0; j < COLS * 16; j += 16) {
      dequant_(src, dst, float_scale, float_zero_point);
      src += 16;
      dst += 16;
    }
    if (j < BLOCK_N) { // elements < 16
      const int res = BLOCK_N - j;
      unsigned short mask = 0xffff;
      mask = (1 << res) - 1;
      dequant_(src, dst, float_scale, float_zero_point, mask);
    }
    B += BLOCK_N;
    b += BLOCK_N;
  }
}

// per tensor
// handle edge cases
void dequant(int8_t* B, float* b, int K, int N, float scale, float zero_point) {
  __m512 float_scale = _mm512_set1_ps(scale);
  __m512 float_zero_point = _mm512_set1_ps(zero_point);
  int COLS = N / 16;
  for (int k = 0; k < K; k++) {
    int8_t* src = B;
    float* dst = b;
    int j;
    for (j = 0; j < COLS * 16; j += 16) {
      dequant_(src, dst, float_scale, float_zero_point);
      src += 16;
      dst += 16;
    }
    if (j < N) { // elements < 16
      const int res = N - j;
      unsigned short mask = 0xffff;
      mask = (1 << res) - 1;
      dequant_(src, dst, float_scale, float_zero_point, mask);
    }
    B += N;
    b += N;
  }
}

// per channel dequantize for int4
// B is packed and not transposed, shape:[BLOCK_K x BLOCK_N]
template <int BLOCK_K, int BLOCK_N>
void dequant(uint8_t* B, float* b, float* scale, float* zero_point) {
  if constexpr (BLOCK_N == 64) {
    const int COLS = 4;
    // lookup table converting uint8 to float, 15.0f - 0.0f
    __m512 lut = _mm512_set_ps(
        15.0f,
        14.0f,
        13.0f,
        12.0f,
        11.0f,
        10.0f,
        9.0f,
        8.0f,
        7.0f,
        6.0f,
        5.0f,
        4.0f,
        3.0f,
        2.0f,
        1.0f,
        0.0f);
    __m512 float_scale[4] = {
        _mm512_loadu_ps(scale),
        _mm512_loadu_ps(scale + 16),
        _mm512_loadu_ps(scale + 32),
        _mm512_loadu_ps(scale + 48)};
    __m512 float_zero_point[4] = {
        _mm512_loadu_ps(zero_point),
        _mm512_loadu_ps(zero_point + 16),
        _mm512_loadu_ps(zero_point + 32),
        _mm512_loadu_ps(zero_point + 48)};
    for (int k = 0; k < BLOCK_K; k++) {
      uint8_t* src = B;
      float* dst = b;
      __m256i packed = _mm256_load_si256((__m256i*)src);
      __m512i int32[4];
      {
        auto low_4bit = _mm512_cvtepu8_epi32(_mm256_castsi256_si128(packed));
        auto high_4bit = _mm512_srli_epi32(low_4bit, 4);
        int32[0] = low_4bit;
        int32[2] = high_4bit;
      }
      {
        auto low_4bit =
            _mm512_cvtepu8_epi32(_mm256_extracti128_si256(packed, 1));
        auto high_4bit = _mm512_srli_epi32(low_4bit, 4);
        int32[1] = low_4bit;
        int32[3] = high_4bit;
      }
      for (int idx = 0; idx < 4; idx++) {
        __m512 vb = _mm512_permutexvar_ps(int32[idx], lut);
        vb = _mm512_sub_ps(vb, float_zero_point[idx]);
        vb = _mm512_mul_ps(vb, float_scale[idx]);
        _mm512_storeu_ps(b + idx * 16, vb);
      }
      B += BLOCK_N / 2;
      b += BLOCK_N;
    }
  } else {
    const int COLS = BLOCK_N / 16;
    for (int k = 0; k < BLOCK_K; k++) {
      uint8_t* src = B;
      float* dst = b;
      int j, idx;
      for (idx = 0, j = 0; j < COLS * 16; j += 16) {
        __m512 float_scale = _mm512_loadu_ps(scale + j);
        __m512 float_zero_point = _mm512_loadu_ps(zero_point + j);
        dequant_(src, dst, float_scale, float_zero_point);
        src += 8;
        dst += 16;
      }
      if (j < BLOCK_N) {
        const int res = BLOCK_N - j;
        for (int l = 0; l < res; l += 2) {
          const uint8_t vi = src[l / 2];
          const int8_t vi0 = vi & 0xf;
          const int8_t vi1 = vi >> 4;
          const float v0 = (vi0 - zero_point[j + l]) * scale[j + l];
          const float v1 = (vi1 - zero_point[j + l + 1]) * scale[j + l + 1];
          dst[l + 0] = v0;
          dst[l + 1] = v1;
        }
      }
      B += BLOCK_N / 2;
      b += BLOCK_N;
    }
  }
}

// per channel dequantize for int4
// handle edge cases
void dequant(
    uint8_t* B,
    float* b,
    int K,
    int N,
    float* scale,
    float* zero_point) {
  if (N % 2 == 0) {
    if (N == 64) {
      const int COLS = 4;
      // lookup table converting uint8 to float, 15.0f - 0.0f
      __m512 lut = _mm512_set_ps(
          15.0f,
          14.0f,
          13.0f,
          12.0f,
          11.0f,
          10.0f,
          9.0f,
          8.0f,
          7.0f,
          6.0f,
          5.0f,
          4.0f,
          3.0f,
          2.0f,
          1.0f,
          0.0f);
      __m512 float_scale[4] = {
          _mm512_loadu_ps(scale),
          _mm512_loadu_ps(scale + 16),
          _mm512_loadu_ps(scale + 32),
          _mm512_loadu_ps(scale + 48)};
      __m512 float_zero_point[4] = {
          _mm512_loadu_ps(zero_point),
          _mm512_loadu_ps(zero_point + 16),
          _mm512_loadu_ps(zero_point + 32),
          _mm512_loadu_ps(zero_point + 48)};
      for (int k = 0; k < K; k++) {
        uint8_t* src = B;
        float* dst = b;
        __m256i packed = _mm256_load_si256((__m256i*)src);
        __m512i int32[4];
        {
          auto low_4bit = _mm512_cvtepu8_epi32(_mm256_castsi256_si128(packed));
          auto high_4bit = _mm512_srli_epi32(low_4bit, 4);
          int32[0] = low_4bit;
          int32[2] = high_4bit;
        }
        {
          auto low_4bit =
              _mm512_cvtepu8_epi32(_mm256_extracti128_si256(packed, 1));
          auto high_4bit = _mm512_srli_epi32(low_4bit, 4);
          int32[1] = low_4bit;
          int32[3] = high_4bit;
        }
        for (int idx = 0; idx < 4; idx++) {
          __m512 vb = _mm512_permutexvar_ps(int32[idx], lut);
          vb = _mm512_sub_ps(vb, float_zero_point[idx]);
          vb = _mm512_mul_ps(vb, float_scale[idx]);
          _mm512_storeu_ps(b + idx * 16, vb);
        }
        B += N / 2;
        b += N;
      }
    } else {
      const int COLS = N / 16;
      for (int k = 0; k < K; k++) {
        uint8_t* src = B;
        float* dst = b;
        int j, idx;
        for (idx = 0, j = 0; j < COLS * 16; j += 16) {
          __m512 float_scale = _mm512_loadu_ps(scale + j);
          __m512 float_zero_point = _mm512_loadu_ps(zero_point + j);
          dequant_(src, dst, float_scale, float_zero_point);
          src += 8;
          dst += 16;
        }
        if (j < N) {
          const int res = N - j;
          int rr = res / 2;
          int l = 0;
          for (; l < rr * 2; l += 2) {
            const uint8_t vi = src[l / 2];
            const int8_t vi0 = vi & 0xf;
            const int8_t vi1 = vi >> 4;
            const float v0 = (vi0 - zero_point[j + l]) * scale[j + l];
            const float v1 = (vi1 - zero_point[j + l + 1]) * scale[j + l + 1];
            dst[l + 0] = v0;
            dst[l + 1] = v1;
          }
          if (l < res) {
            const uint8_t vi = src[l / 2];
            const int8_t vi0 = vi & 0xf;
            const float v0 = (vi0 - zero_point[j + l]) * scale[j + l];
            dst[l + 0] = v0;
          }
        }
        B += N / 2;
        b += N;
      }
    }
  } else {
    int i = 0;
    for (; i < K * N / 2 * 2; i += 2) {
      const uint8_t vi = B[i / 2];
      const int8_t vi0 = vi & 0xf;
      const int8_t vi1 = vi >> 4;
      const float v0 = (vi0 - zero_point[i % N]) * scale[i % N];
      const float v1 = (vi1 - zero_point[(i + 1) % N]) * scale[(i + 1) % N];
      b[i + 0] = v0;
      b[i + 1] = v1;
    }
    if (i < K * N) {
      const uint8_t vi = B[i / 2];
      const int8_t vi0 = vi & 0xf;
      const float v0 = (vi0 - zero_point[i % N]) * scale[i % N];
      b[i + 0] = v0;
    }
  }
}

// dequant uint4 weight to bf16
void dequant(
    uint8_t* B,
    BFloat16* b,
    int K,
    int N,
    float* scale,
    float* zero_point) {
  if (N % 2 == 0) {
    if (N == 64) {
      const int COLS = 4;
      // lookup table converting uint8 to float, 15.0f - 0.0f
      __m512 lut = _mm512_set_ps(
          15.0f,
          14.0f,
          13.0f,
          12.0f,
          11.0f,
          10.0f,
          9.0f,
          8.0f,
          7.0f,
          6.0f,
          5.0f,
          4.0f,
          3.0f,
          2.0f,
          1.0f,
          0.0f);
      __m512 float_scale[4] = {
          _mm512_loadu_ps(scale),
          _mm512_loadu_ps(scale + 16),
          _mm512_loadu_ps(scale + 32),
          _mm512_loadu_ps(scale + 48)};
      __m512 float_zero_point[4] = {
          _mm512_loadu_ps(zero_point),
          _mm512_loadu_ps(zero_point + 16),
          _mm512_loadu_ps(zero_point + 32),
          _mm512_loadu_ps(zero_point + 48)};
      for (int k = 0; k < K; k++) {
        uint8_t* src = B;
        BFloat16* dst = b;
        __m256i packed = _mm256_load_si256((__m256i*)src);
        __m512i int32[4];
        {
          auto low_4bit = _mm512_cvtepu8_epi32(_mm256_castsi256_si128(packed));
          auto high_4bit = _mm512_srli_epi32(low_4bit, 4);
          int32[0] = low_4bit;
          int32[2] = high_4bit;
        }
        {
          auto low_4bit =
              _mm512_cvtepu8_epi32(_mm256_extracti128_si256(packed, 1));
          auto high_4bit = _mm512_srli_epi32(low_4bit, 4);
          int32[1] = low_4bit;
          int32[3] = high_4bit;
        }
        for (int idx = 0; idx < 4; idx++) {
          __m512 vb = _mm512_permutexvar_ps(int32[idx], lut);
          vb = _mm512_sub_ps(vb, float_zero_point[idx]);
          vb = _mm512_mul_ps(vb, float_scale[idx]);
          _mm256_storeu_si256((__m256i*)(b + idx * 16), cvt_fp32_to_bf16(vb));
        }
        B += N / 2;
        b += N;
      }
    } else {
      const int COLS = N / 16;
      for (int k = 0; k < K; k++) {
        uint8_t* src = B;
        BFloat16* dst = b;
        int j, idx;
        for (idx = 0, j = 0; j < COLS * 16; j += 16) {
          __m512 float_scale = _mm512_loadu_ps(scale + j);
          __m512 float_zero_point = _mm512_loadu_ps(zero_point + j);
          dequant_to_bf16_(src, dst, float_scale, float_zero_point);
          src += 8;
          dst += 16;
        }
        if (j < N) {
          const int res = N - j;
          int rr = res / 2;
          int l = 0;
          for (; l < rr * 2; l += 2) {
            const uint8_t vi = src[l / 2];
            const int8_t vi0 = vi & 0xf;
            const int8_t vi1 = vi >> 4;
            const float v0 = (vi0 - zero_point[j + l]) * scale[j + l];
            const float v1 = (vi1 - zero_point[j + l + 1]) * scale[j + l + 1];
            dst[l + 0] = v0;
            dst[l + 1] = v1;
          }
          if (l < res) {
            const uint8_t vi = src[l / 2];
            const int8_t vi0 = vi & 0xf;
            const float v0 = (vi0 - zero_point[j + l]) * scale[j + l];
            dst[l + 0] = v0;
          }
        }
        B += N / 2;
        b += N;
      }
    }
  } else {
    int i = 0;
    for (; i < K * N / 2 * 2; i += 2) {
      const uint8_t vi = B[i / 2];
      const int8_t vi0 = vi & 0xf;
      const int8_t vi1 = vi >> 4;
      const float v0 = (vi0 - zero_point[i % N]) * scale[i % N];
      const float v1 = (vi1 - zero_point[(i + 1) % N]) * scale[(i + 1) % N];
      b[i + 0] = v0;
      b[i + 1] = v1;
    }
    if (i < K * N) {
      const uint8_t vi = B[i / 2];
      const int8_t vi0 = vi & 0xf;
      const float v0 = (vi0 - zero_point[i % N]) * scale[i % N];
      b[i + 0] = v0;
    }
  }
}

void add_bias(float* C, float* bias, int M, int N, int ldc) {
  int COLS = N / 16;
  int j;
  for (j = 0; j < COLS * 16; j += 16) {
    __m512 float_bias = _mm512_loadu_ps(bias + j);
    float* c = C + j;
    for (int m = 0; m < M; m++) {
      __m512 vc = _mm512_loadu_ps(c);
      vc = _mm512_add_ps(vc, float_bias);
      _mm512_storeu_ps(c, vc);
      c += ldc;
    }
  }
  if (j < N) {
    const int res = N - j;
    unsigned short mask = 0xffff;
    mask = (1 << res) - 1;
    __m512 float_bias = _mm512_maskz_loadu_ps(mask, bias + j);
    float* c = C + j;
    for (int m = 0; m < M; m++) {
      __m512 vc = _mm512_maskz_loadu_ps(mask, c);
      vc = _mm512_mask_add_ps(vc, mask, vc, float_bias);
      _mm512_mask_storeu_ps(c, mask, vc);
      c += ldc;
    }
  }
}

#else // not support AVX512

// per channel
template <int BLOCK_N, int BLOCK_K>
void dequant(int8_t* B, float* b, float* scale, float* zero_point) {
  AT_ASSERTM(false, "Unable to support AVX512!");
}

void dequant(
    int8_t* B,
    float* b,
    int K,
    int N,
    float* scale,
    float* zero_point) {
  AT_ASSERTM(false, "Unable to support AVX512!");
}

void dequant(
    int8_t* B,
    BFloat16* b,
    int K,
    int N,
    float* scale,
    float* zero_point) {
  AT_ASSERTM(false, "Unable to support AVX512!");
}

// per tensor
template <int BLOCK_N, int BLOCK_K>
void dequant(int8_t* B, float* b, float scale, float zero_point) {
  AT_ASSERTM(false, "Unable to support AVX512!");
}

void dequant(int8_t* B, float* b, int K, int N, float scale, float zero_point) {
  AT_ASSERTM(false, "Unable to support AVX512!");
}

// per channel dequantize for int4
// B is packed and not transposed, shape:[BLOCK_K x BLOCK_N]
template <int BLOCK_K, int BLOCK_N>
void dequant(uint8_t* B, float* b, float* scale, float* zero_point) {
  AT_ASSERTM(false, "Unable to support AVX512!");
}

// per channel dequantize for int4
// handle edge cases
void dequant(
    uint8_t* B,
    float* b,
    int K,
    int N,
    float* scale,
    float* zero_point) {
  AT_ASSERTM(false, "Unable to support AVX512!");
}

// dequant uint4 weight to bf16
void dequant(
    uint8_t* B,
    BFloat16* b,
    int K,
    int N,
    float* scale,
    float* zero_point) {
  AT_ASSERTM(false, "Unable to support AVX512!");
}

void convert_bf16_to_fp32(
    const BFloat16* src,
    float* dst,
    int M,
    int K,
    int ld_src) {
  AT_ASSERTM(false, "Unable to support AVX512!");
}

void convert_fp32_to_bf16(
    const float* src,
    BFloat16* dst,
    int M,
    int N,
    int ld) {
  AT_ASSERTM(false, "Unable to support AVX512!");
}

void add_bias(float* C, float* bias, int M, int N, int ldc) {
  AT_ASSERTM(false, "Unable to support AVX512!");
}

template <
    int LINES,
    int N,
    int PREFETCH_K_DIST,
    bool ACC,
    bool bias_add = false>
void small_gemm_smallm(
    const float* A,
    const int8_t* B,
    float* C,
    int lda,
    int ldb,
    int ldc,
    int actualN,
    int K,
    float* scale,
    float* zero_point,
    float* bias,
    int rowOff = 0) {
  AT_ASSERTM(false, "Unable to support AVX512!");
}

// bf16 * int8 -> bf16
template <
    int LINES,
    int N,
    int PREFETCH_K_DIST,
    bool ACC,
    bool bias_add = false>
void small_gemm_smallm(
    const BFloat16* A,
    const int8_t* B,
    float* C,
    int lda,
    int ldb,
    int ldc,
    int actualN,
    int K,
    float* scale,
    float* zero_point,
    float* bias = NULL,
    int rowOff = 0) {
  AT_ASSERTM(false, "Unable to support AVX512!");
}

inline void cvt_fp32_to_bf16(
    BFloat16* dst,
    const float* src,
    int m,
    int n,
    int ld_dst,
    int ld_src) {
  AT_ASSERTM(false, "Unable to support AVX512!");
}

inline void cvt_bf16_to_fp32(
    float* dst,
    const at::BFloat16* src,
    int m,
    int n,
    int ld_dst,
    int ld_src) {
  AT_ASSERTM(false, "Unable to support AVX512!");
}
#endif

static void print_mat(uint8_t* src, int size) {
  std::cout << "B mat:" << std::endl;
  for (int i = 0; i < size; i++) {
    std::cout << (int)src[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "mat:" << std::endl;
  for (int i = 0; i < size; i++) {
    std::bitset<8> b(src[i]);
    std::cout << b << " ";
  }
  std::cout << std::endl;
}

// fp32 * fp32 -> fp32
template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
void dot_tile_update(
    float* A,
    float* B,
    float* C,
    bool trans_a,
    bool trans_b,
    int lda,
    int ldb,
    int ldc) {
  auto&& kernel = create_or_get_dot_microkernel<BLOCK_M, BLOCK_N, BLOCK_K>(
      trans_a, trans_b, lda, ldb, ldc); // nonblock
  (*kernel)(A, B, C);
}

void dot_update(
    float* A,
    float* B,
    float* C,
    int M,
    int N,
    int K,
    bool trans_a,
    bool trans_b,
    int lda,
    int ldb,
    int ldc) {
  const char transa = trans_a ? 'Y' : 'N';
  const char transb = trans_b ? 'Y' : 'N';
  libxsmm_blasint BM = M, BN = N, BK = K, LDA = lda, LDB = ldb, LDC = ldc;
  const float alpha = 1.0, beta = 1.0;
  libxsmm_sgemm(
      &transa,
      &transb,
      &BM,
      &BN,
      &BK,
      &alpha,
      A,
      &LDA,
      B,
      &LDB,
      &beta,
      C,
      &LDC);
}

template <typename T>
void zero_fill(T* C, int M, int N, int stride) {
  for (int m = 0; m < M; m++) {
    memset(C + m * stride, 0, sizeof(T) * N);
  }
}

// TODO: optimize with vectorized transposition
void pack(
    const int8_t* B,
    int8_t* packed_B,
    int K,
    int N,
    int ldb,
    bool trans_B) {
  AT_ASSERTM(
      trans_B, "B must be transposed!"); // B must be transposed, shape:[N x K]
  const int blks = (N + BLOCK_N - 1) / BLOCK_N;
#pragma omp parallel for
  for (int i = 0; i < blks; ++i) {
    int rows = BLOCK_N; // each time pack BLOCK_N elements in N dimension
    if (i == blks - 1) { // last block
      rows = N - i * BLOCK_N;
    }

    const int8_t* psrc = B + i * BLOCK_N * ldb;
    int8_t* pdst = packed_B + i * K * BLOCK_N;

    for (int c = 0; c < K; ++c) {
      for (int r = 0; r < rows; ++r) {
        pdst[r] = psrc[r * ldb];
      }
      psrc += 1;
      pdst += rows;
    }
  }
}

inline uint8_t extract_element(const uint8_t* src, int c, int r, int K) {
  int offset_int4 = r * K + c;
  int offset_int8 = offset_int4 / 2;
  if (offset_int4 % 2 == 0) {
    uint8_t elem = src[offset_int8] & 0xf;
    return elem;
  } else {
    uint8_t elem = src[offset_int8] >> 4;
    return elem;
  }
}

inline void insert_element(uint8_t* dst, int c, int r, int rows, uint8_t elem) {
  int offset_int4 = c * rows + r;
  int offset_int8 = offset_int4 / 2;
  if (offset_int4 % 2 == 0) { // in last 4 bits
    dst[offset_int8] &= 0xf0;
    dst[offset_int8] |= elem;

  } else { // in first 4 bits
    elem = elem << 4;
    dst[offset_int8] &= 0xf;
    dst[offset_int8] |= elem;
  }
}

// TODO: optimize with vectorized transposition
// pack int4 weight
void pack(
    const uint8_t* B,
    uint8_t* packed_B,
    int K,
    int N,
    int ldb,
    bool trans_B) {
  AT_ASSERTM(
      trans_B, "B must be transposed!"); // B must be transposed, shape:[N x K]
  static_assert(BLOCK_N == 64, "BLOCK_N must be 64");
  const int blks = (N + BLOCK_N - 1) / BLOCK_N;
#pragma omp parallel for
  for (int i = 0; i < blks; i++) {
    int rows = BLOCK_N;
    if (i == blks - 1) {
      rows = N - i * BLOCK_N;
    }
    const uint8_t* psrc = B + i * BLOCK_N * K / 2;
    uint8_t* pdst = packed_B + i * K * BLOCK_N / 2;
    for (int c = 0; c < K; c++) {
      if (rows != BLOCK_N) {
        for (int r = 0; r < rows; r++) {
          uint8_t tmp = extract_element(psrc, c, r, K);
          insert_element(pdst, c, r, rows, tmp);
        }
      } else {
        for (int r = 0; r < BLOCK_N / 2; r++) {
          uint8_t tmp = extract_element(psrc, c, r, K);
          insert_element(pdst, c, r * 2, rows, tmp);
          tmp = extract_element(psrc, c, r + BLOCK_N / 2, K);
          insert_element(pdst, c, r * 2 + 1, rows, tmp);
        }
      }
    }
  }
}

// TODO: optimize with vectorized transposition
void unpack(
    const int8_t* packed_B,
    int8_t* unpacked_B,
    int K,
    int N,
    int ldb,
    bool trans_B) {
  AT_ASSERTM(
      trans_B, "B must be transposed!"); // B must be transposed, shape:[N x K]
  const int blks = (N + BLOCK_N - 1) / BLOCK_N;
#pragma omp parallel for
  for (int i = 0; i < blks; ++i) {
    int rows = BLOCK_N; // each time pack BLOCK_N elements in N dimension
    if (i == blks - 1) { // last block
      rows = N - i * BLOCK_N;
    }

    const int8_t* psrc = packed_B + i * K * BLOCK_N;
    int8_t* pdst = unpacked_B + i * BLOCK_N * ldb;

    for (int c = 0; c < K; ++c) {
      for (int r = 0; r < rows; ++r) {
        pdst[r * ldb] = psrc[r];
      }
      psrc += rows;
      pdst += 1;
    }
  }
}

// TODO: rewrite this
// unpack uint4 weight
void unpack(
    const uint8_t* packed_B,
    uint8_t* unpacked_B,
    int K,
    int N,
    int ldb,
    bool trans_B) {
  AT_ASSERTM(
      trans_B, "B must be transposed!"); // B must be transposed, shape:[N x K]
  static_assert(BLOCK_N == 64, "BLOCK_N must be 64");
  const int blks = (N + BLOCK_N - 1) / BLOCK_N;
#pragma omp parallel for
  for (int i = 0; i < blks; i++) {
    int rows = BLOCK_N;
    if (i == blks - 1) {
      rows = N - i * BLOCK_N;
    }
    uint8_t* pdst = unpacked_B + i * BLOCK_N * K / 2;
    const uint8_t* psrc = packed_B + i * K * BLOCK_N / 2;
    for (int c = 0; c < K; c++) {
      if (rows != BLOCK_N) {
        for (int r = 0; r < rows; r++) {
          uint8_t tmp = extract_element(psrc, r, c, rows);
          insert_element(pdst, r, c, K, tmp);
        }
      } else {
        for (int r = 0; r < BLOCK_N / 2; r++) {
          uint8_t tmp = extract_element(psrc, r * 2, c, rows);
          insert_element(pdst, r, c, K, tmp);
          tmp = extract_element(psrc, r * 2 + 1, c, rows);
          insert_element(pdst, r + BLOCK_N / 2, c, K, tmp);
        }
      }
    }
  }
}

// dequant per channel
template <bool has_bias, int BLOCK_M>
void woq_gemm_intrinsic(
    float* A,
    int8_t* B,
    float* C,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    float* scale,
    float* zero_point,
    float* bias = NULL) {
#define PTR_OFFSET(base, offset0, offset1, stride0) \
  (base) + (offset0) * (stride0) + (offset1)

  const int MB = (M + BLOCK_M - 1) / BLOCK_M, NB = (N + BLOCK_N - 1) / BLOCK_N,
            KB = (K + BLOCK_K - 1) / BLOCK_K;

#pragma omp parallel for collapse(2)
  for (int mb = 0; mb < MB; mb++) {
    for (int nb = 0; nb < NB; nb++) {
      int mb_start = mb * BLOCK_M;
      int m_bs = std::min(BLOCK_M, M - mb_start);
      int nb_start = nb * BLOCK_N;
      int n_bs = std::min(BLOCK_N, N - nb_start);
      float* C_offset = PTR_OFFSET(C, mb_start, nb_start, ldc);
      float* bi_offset =
          (float*)aligned_alloc(64, BLOCK_K * BLOCK_N * sizeof(float));
      zero_fill(C_offset, m_bs, n_bs, ldc);
      for (int kb = 0; kb < KB; kb++) {
        int kb_start = kb * BLOCK_K;
        int k_bs = std::min(BLOCK_K, K - kb_start);
        float* A_offset = PTR_OFFSET(A, mb_start, kb_start, lda);
        int8_t* B_offset = B + nb_start * K + kb_start * n_bs;
        if (m_bs == BLOCK_M && n_bs == BLOCK_N) {
          small_gemm_smallm<BLOCK_M, BLOCK_N, PREFETCH_K, true>(
              A_offset,
              B_offset,
              C_offset,
              lda,
              n_bs,
              ldc,
              BLOCK_N,
              k_bs,
              scale + nb_start,
              zero_point + nb_start);
        } else { // edge case
          dequant(
              B_offset,
              bi_offset,
              k_bs,
              n_bs,
              scale + nb_start,
              zero_point + nb_start);
          dot_update( // libxsmm is col major
              bi_offset,
              A_offset,
              C_offset,
              n_bs,
              m_bs,
              k_bs,
              false,
              false,
              n_bs,
              lda,
              ldc);
        }
      }
      if constexpr (has_bias) {
        add_bias(C_offset, bias + nb_start, m_bs, n_bs, ldc);
      }
      free(bi_offset);
    }
  }
}

// dequant per channel
template <bool has_bias, int BLOCK_M>
void woq_gemm_intrinsic(
    BFloat16* A,
    int8_t* B,
    BFloat16* C,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    float* scale,
    float* zero_point,
    float* bias = NULL) {
#define PTR_OFFSET(base, offset0, offset1, stride0) \
  (base) + (offset0) * (stride0) + (offset1)

  const int MB = (M + BLOCK_M - 1) / BLOCK_M, NB = (N + BLOCK_N - 1) / BLOCK_N,
            KB = (K + BLOCK_K - 1) / BLOCK_K;

#pragma omp parallel for collapse(2)
  for (int mb = 0; mb < MB; mb++) {
    for (int nb = 0; nb < NB; nb++) {
      int mb_start = mb * BLOCK_M;
      int m_bs = std::min(BLOCK_M, M - mb_start);
      int nb_start = nb * BLOCK_N;
      int n_bs = std::min(BLOCK_N, N - nb_start);
      BFloat16* C_offset = PTR_OFFSET(C, mb_start, nb_start, ldc);
      float* bi_offset =
          (float*)aligned_alloc(64, BLOCK_K * BLOCK_N * sizeof(float));
      float* ci_offset = (float*)aligned_alloc(64, m_bs * n_bs * sizeof(float));
      zero_fill(ci_offset, m_bs, n_bs, n_bs);
      if (m_bs == BLOCK_M && n_bs == BLOCK_N) {
        for (int kb = 0; kb < KB; kb++) {
          int kb_start = kb * BLOCK_K;
          int k_bs = std::min(BLOCK_K, K - kb_start);
          BFloat16* A_offset = PTR_OFFSET(A, mb_start, kb_start, lda);
          int8_t* B_offset = B + nb_start * K + kb_start * n_bs;
          if (kb != KB - 1) {
            small_gemm_smallm<BLOCK_M, BLOCK_N, PREFETCH_K, true>(
                A_offset,
                B_offset,
                ci_offset,
                lda,
                n_bs,
                n_bs,
                BLOCK_N,
                k_bs,
                scale + nb_start,
                zero_point + nb_start);
          } else { // last block in K, add bias
            small_gemm_smallm<BLOCK_M, BLOCK_N, PREFETCH_K, true, has_bias>(
                A_offset,
                B_offset,
                ci_offset,
                lda,
                n_bs,
                n_bs,
                BLOCK_N,
                k_bs,
                scale + nb_start,
                zero_point + nb_start,
                bias + nb_start);
          }
        }
      } else { // edge case
        float* ai_offset =
            (float*)aligned_alloc(64, m_bs * BLOCK_K * sizeof(float));
        for (int kb = 0; kb < KB; kb++) {
          int kb_start = kb * BLOCK_K;
          int k_bs = std::min(BLOCK_K, K - kb_start);
          BFloat16* A_offset = PTR_OFFSET(A, mb_start, kb_start, lda);
          int8_t* B_offset = B + nb_start * K + kb_start * n_bs;
          cvt_bf16_to_fp32(ai_offset, A_offset, m_bs, k_bs, k_bs, lda);
          dequant(
              B_offset,
              bi_offset,
              k_bs,
              n_bs,
              scale + nb_start,
              zero_point + nb_start);
          dot_update( // libxsmm is col major
              bi_offset,
              ai_offset,
              ci_offset,
              n_bs,
              m_bs,
              k_bs,
              false,
              false,
              n_bs,
              k_bs,
              n_bs);
        }
        if constexpr (has_bias) {
          add_bias(ci_offset, bias + nb_start, m_bs, n_bs, n_bs);
        }
        free(ai_offset);
      }
      cvt_fp32_to_bf16(C_offset, ci_offset, m_bs, n_bs, ldc, n_bs);
      free(ci_offset);
      free(bi_offset);
    }
  }
}

// dequant per channel
template <bool has_bias, int BLOCK_M>
void woq_gemm_intrinsic(
    float* A,
    uint8_t* B,
    float* C,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    float* scale,
    float* zero_point,
    float* bias = NULL) {
#define PTR_OFFSET(base, offset0, offset1, stride0) \
  (base) + (offset0) * (stride0) + (offset1)

  const int MB = (M + BLOCK_M - 1) / BLOCK_M, NB = (N + BLOCK_N - 1) / BLOCK_N,
            KB = (K + BLOCK_K - 1) / BLOCK_K;

#pragma omp parallel for collapse(2)
  for (int mb = 0; mb < MB; mb++) {
    for (int nb = 0; nb < NB; nb++) {
      int mb_start = mb * BLOCK_M;
      int m_bs = std::min(BLOCK_M, M - mb_start);
      int nb_start = nb * BLOCK_N;
      int n_bs = std::min(BLOCK_N, N - nb_start);
      float* C_offset = PTR_OFFSET(C, mb_start, nb_start, ldc);
      float* bi_offset =
          (float*)aligned_alloc(64, BLOCK_K * BLOCK_N * sizeof(float));
      zero_fill(C_offset, m_bs, n_bs, ldc);
      for (int kb = 0; kb < KB; kb++) {
        int kb_start = kb * BLOCK_K;
        int k_bs = std::min(BLOCK_K, K - kb_start);
        float* A_offset = PTR_OFFSET(A, mb_start, kb_start, lda);
        uint8_t* B_offset = B + nb_start / 2 * K + kb_start * n_bs / 2;
        if (m_bs == BLOCK_M && n_bs == BLOCK_N) {
          small_gemm_smallm<BLOCK_M, BLOCK_N, PREFETCH_K, true>(
              A_offset,
              B_offset,
              C_offset,
              lda,
              n_bs,
              ldc,
              BLOCK_N,
              k_bs,
              scale + nb_start,
              zero_point + nb_start);
        } else { // edge case
          dequant(
              B_offset,
              bi_offset,
              k_bs,
              n_bs,
              scale + nb_start,
              zero_point + nb_start);
          dot_update( // libxsmm is col major
              bi_offset,
              A_offset,
              C_offset,
              n_bs,
              m_bs,
              k_bs,
              false,
              false,
              n_bs,
              lda,
              ldc);
        }
      }
      if constexpr (has_bias) {
        add_bias(C_offset, bias + nb_start, m_bs, n_bs, ldc);
      }
      free(bi_offset);
    }
  }
}

// dequant per channel
template <bool has_bias, int BLOCK_M>
void woq_gemm_intrinsic(
    BFloat16* A,
    uint8_t* B,
    BFloat16* C,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    float* scale,
    float* zero_point,
    float* bias = NULL) {
#define PTR_OFFSET(base, offset0, offset1, stride0) \
  (base) + (offset0) * (stride0) + (offset1)

  const int MB = (M + BLOCK_M - 1) / BLOCK_M, NB = (N + BLOCK_N - 1) / BLOCK_N,
            KB = (K + BLOCK_K - 1) / BLOCK_K;

#pragma omp parallel for collapse(2)
  for (int mb = 0; mb < MB; mb++) {
    for (int nb = 0; nb < NB; nb++) {
      int mb_start = mb * BLOCK_M;
      int m_bs = std::min(BLOCK_M, M - mb_start);
      int nb_start = nb * BLOCK_N;
      int n_bs = std::min(BLOCK_N, N - nb_start);
      BFloat16* C_offset = PTR_OFFSET(C, mb_start, nb_start, ldc);
      float* bi_offset =
          (float*)aligned_alloc(64, BLOCK_K * BLOCK_N * sizeof(float));
      float* ci_offset = (float*)aligned_alloc(64, m_bs * n_bs * sizeof(float));
      zero_fill(ci_offset, m_bs, n_bs, n_bs);
      if (m_bs == BLOCK_M && n_bs == BLOCK_N) {
        for (int kb = 0; kb < KB; kb++) {
          int kb_start = kb * BLOCK_K;
          int k_bs = std::min(BLOCK_K, K - kb_start);
          BFloat16* A_offset = PTR_OFFSET(A, mb_start, kb_start, lda);
          uint8_t* B_offset = B + nb_start / 2 * K + kb_start * n_bs / 2;
          if (kb != KB - 1) {
            small_gemm_smallm<BLOCK_M, BLOCK_N, PREFETCH_K, true>(
                A_offset,
                B_offset,
                ci_offset,
                lda,
                n_bs,
                n_bs,
                BLOCK_N,
                k_bs,
                scale + nb_start,
                zero_point + nb_start);
          } else { // last block in K, add bias
            small_gemm_smallm<BLOCK_M, BLOCK_N, PREFETCH_K, true, has_bias>(
                A_offset,
                B_offset,
                ci_offset,
                lda,
                n_bs,
                n_bs,
                BLOCK_N,
                k_bs,
                scale + nb_start,
                zero_point + nb_start,
                bias + nb_start);
          }
        }
      } else { // edge case
        float* ai_offset =
            (float*)aligned_alloc(64, m_bs * BLOCK_K * sizeof(float));
        for (int kb = 0; kb < KB; kb++) {
          int kb_start = kb * BLOCK_K;
          int k_bs = std::min(BLOCK_K, K - kb_start);
          BFloat16* A_offset = PTR_OFFSET(A, mb_start, kb_start, lda);
          uint8_t* B_offset = B + nb_start / 2 * K + kb_start * n_bs / 2;
          cvt_bf16_to_fp32(ai_offset, A_offset, m_bs, k_bs, k_bs, lda);
          dequant(
              B_offset,
              bi_offset,
              k_bs,
              n_bs,
              scale + nb_start,
              zero_point + nb_start);
          dot_update( // libxsmm is col major
              bi_offset,
              ai_offset,
              ci_offset,
              n_bs,
              m_bs,
              k_bs,
              false,
              false,
              n_bs,
              k_bs,
              n_bs);
        }
        if constexpr (has_bias) {
          add_bias(ci_offset, bias + nb_start, m_bs, n_bs, n_bs);
        }
        free(ai_offset);
      }
      cvt_fp32_to_bf16(C_offset, ci_offset, m_bs, n_bs, ldc, n_bs);
      free(ci_offset);
      free(bi_offset);
    }
  }
}

// dequant per channel
// for large M
template <bool has_bias, int BLOCK_M>
void woq_gemm_brgemm(
    float* A,
    int8_t* B,
    float* C,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    float* scale,
    float* zero_point,
    float* bias = NULL) {
#define PTR_OFFSET(base, offset0, offset1, stride0) \
  (base) + (offset0) * (stride0) + (offset1)

  const int MB = (M + BLOCK_M - 1) / BLOCK_M, NB = (N + BLOCK_N - 1) / BLOCK_N,
            KB = (K + BLOCK_K - 1) / BLOCK_K;

#pragma omp parallel for collapse(2)
  for (int mb = 0; mb < MB; mb++) {
    for (int nb = 0; nb < NB; nb++) {
      int mb_start = mb * BLOCK_M;
      int m_bs = std::min(BLOCK_M, M - mb_start);
      int nb_start = nb * BLOCK_N;
      int n_bs = std::min(BLOCK_N, N - nb_start);
      float* C_offset = PTR_OFFSET(C, mb_start, nb_start, ldc);
      zero_fill(C_offset, m_bs, n_bs, ldc);
      float* bi_offset =
          (float*)aligned_alloc(64, BLOCK_K * BLOCK_N * sizeof(float));
      for (int kb = 0; kb < KB; kb++) {
        int kb_start = kb * BLOCK_K;
        int k_bs = std::min(BLOCK_K, K - kb_start);
        float* A_offset = PTR_OFFSET(A, mb_start, kb_start, lda);
        int8_t* B_offset = B + nb_start * K + kb_start * n_bs;
        dequant(
            B_offset,
            bi_offset,
            k_bs,
            n_bs,
            scale + nb_start,
            zero_point + nb_start);
        if (m_bs == BLOCK_M && n_bs == BLOCK_N && k_bs == BLOCK_K) {
          dot_tile_update<BLOCK_N, BLOCK_M, BLOCK_K>(
              bi_offset, A_offset, C_offset, false, false, n_bs, lda, ldc);
        } else {
          dot_update( // libxsmm is col major
              bi_offset,
              A_offset,
              C_offset,
              n_bs,
              m_bs,
              k_bs,
              false,
              false,
              n_bs,
              lda,
              ldc);
        }
      }
      if constexpr (has_bias) {
        add_bias(C_offset, bias + nb_start, m_bs, n_bs, ldc);
      }
      free(bi_offset);
    }
  }
}

// dequant per channel
// for large M
template <bool has_bias, int BLOCK_M>
void woq_gemm_brgemm(
    BFloat16* A,
    int8_t* B,
    BFloat16* C,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    float* scale,
    float* zero_point,
    float* bias = NULL) {
#define PTR_OFFSET(base, offset0, offset1, stride0) \
  (base) + (offset0) * (stride0) + (offset1)

  // Number of blocks
  const int MB = (M + BLOCK_M - 1) / BLOCK_M;
  const int NB = (N + BLOCK_N - 1) / BLOCK_N;
  const int KB = (K + BLOCK_K - 1) / BLOCK_K;

#pragma omp parallel for collapse(2)
  for (int mb = 0; mb < MB; mb++) {
    for (int nb = 0; nb < NB; nb++) {
      int mb_start = mb * BLOCK_M;
      int m_bs = std::min(BLOCK_M, M - mb_start);
      int nb_start = nb * BLOCK_N;
      int n_bs = std::min(BLOCK_N, N - nb_start);
      BFloat16* C_offset = PTR_OFFSET(C, mb_start, nb_start, ldc);
      BFloat16* bi_offset =
          (BFloat16*)aligned_alloc(64, BLOCK_K * n_bs * sizeof(BFloat16));
      float* ci_offset = (float*)aligned_alloc(64, m_bs * n_bs * sizeof(float));
      zero_fill(ci_offset, m_bs, n_bs, n_bs);
      auto compute_block = [&](int kb) {
        int kb_start = kb * BLOCK_K;
        int k_bs = std::min(BLOCK_K, K - kb_start);
        BFloat16* A_offset = PTR_OFFSET(A, mb_start, kb_start, lda);
        int8_t* B_offset = B + nb_start * K + kb_start * n_bs;
        dequant(
            B_offset,
            bi_offset,
            k_bs,
            n_bs,
            scale + nb_start,
            zero_point + nb_start);
        // MKL gemm
        // C := alpha*op(A) *op(B) + beta*C
        // op(A) is m-by-k, op(B) is k-by-n, C is m-by-n.
        cblas_gemm_bf16bf16f32(
            CblasRowMajor, // Row/col major
            CblasNoTrans, // Trans A
            CblasNoTrans, // Trans B
            m_bs, // M
            n_bs, // N
            k_bs, // K
            1.f, // alpha = 1.0
            (const MKL_BF16*)A_offset, // A
            lda, // lda
            (const MKL_BF16*)bi_offset, // B
            n_bs, // ldb
            1.f, // beta = 1.0 as we need to accumulate
            ci_offset, // C
            n_bs); // ldc
      };
      int kb = 0;
      for (; kb < KB; kb++) {
        compute_block(kb);
      }
      if constexpr (has_bias) {
        add_bias(ci_offset, bias + nb_start, m_bs, n_bs, n_bs);
      }
      cvt_fp32_to_bf16(C_offset, ci_offset, m_bs, n_bs, ldc, n_bs);
      free(ci_offset);
      free(bi_offset);
    }
  }
}

// dequant per channel
// for large M
template <bool has_bias, int BLOCK_M>
void woq_gemm_brgemm(
    float* A,
    uint8_t* B,
    float* C,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    float* scale,
    float* zero_point,
    float* bias = NULL) {
#define PTR_OFFSET(base, offset0, offset1, stride0) \
  (base) + (offset0) * (stride0) + (offset1)

  const int MB = (M + BLOCK_M - 1) / BLOCK_M, NB = (N + BLOCK_N - 1) / BLOCK_N,
            KB = (K + BLOCK_K - 1) / BLOCK_K;

#pragma omp parallel for collapse(2)
  for (int mb = 0; mb < MB; mb++) {
    for (int nb = 0; nb < NB; nb++) {
      int mb_start = mb * BLOCK_M;
      int m_bs = std::min(BLOCK_M, M - mb_start);
      int nb_start = nb * BLOCK_N;
      int n_bs = std::min(BLOCK_N, N - nb_start);
      float* C_offset = PTR_OFFSET(C, mb_start, nb_start, ldc);
      zero_fill(C_offset, m_bs, n_bs, ldc);
      float* bi_offset =
          (float*)aligned_alloc(64, BLOCK_K * BLOCK_N * sizeof(float));
      for (int kb = 0; kb < KB; kb++) {
        int kb_start = kb * BLOCK_K;
        int k_bs = std::min(BLOCK_K, K - kb_start);
        float* A_offset = PTR_OFFSET(A, mb_start, kb_start, lda);
        uint8_t* B_offset = B + nb_start / 2 * K + kb_start * n_bs / 2;
        dequant(
            B_offset,
            bi_offset,
            k_bs,
            n_bs,
            scale + nb_start,
            zero_point + nb_start);
        if (m_bs == BLOCK_M && n_bs == BLOCK_N && k_bs == BLOCK_K) {
          dot_tile_update<BLOCK_N, BLOCK_M, BLOCK_K>(
              bi_offset, A_offset, C_offset, false, false, n_bs, lda, ldc);
        } else {
          dot_update( // libxsmm is col major
              bi_offset,
              A_offset,
              C_offset,
              n_bs,
              m_bs,
              k_bs,
              false,
              false,
              n_bs,
              lda,
              ldc);
        }
      }
      if constexpr (has_bias) {
        add_bias(C_offset, bias + nb_start, m_bs, n_bs, ldc);
      }
      free(bi_offset);
    }
  }
}

// dequant per channel
// for large M
template <bool has_bias, int BLOCK_M>
void woq_gemm_brgemm(
    BFloat16* A,
    uint8_t* B,
    BFloat16* C,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    float* scale,
    float* zero_point,
    float* bias = NULL) {
#define PTR_OFFSET(base, offset0, offset1, stride0) \
  (base) + (offset0) * (stride0) + (offset1)

  // Number of blocks
  const int MB = (M + BLOCK_M - 1) / BLOCK_M;
  const int NB = (N + BLOCK_N - 1) / BLOCK_N;
  const int KB = (K + BLOCK_K - 1) / BLOCK_K;

#pragma omp parallel for collapse(2)
  for (int mb = 0; mb < MB; mb++) {
    for (int nb = 0; nb < NB; nb++) {
      int mb_start = mb * BLOCK_M;
      int m_bs = std::min(BLOCK_M, M - mb_start);
      int nb_start = nb * BLOCK_N;
      int n_bs = std::min(BLOCK_N, N - nb_start);
      BFloat16* C_offset = PTR_OFFSET(C, mb_start, nb_start, ldc);
      BFloat16* bi_offset =
          (BFloat16*)aligned_alloc(64, BLOCK_K * n_bs * sizeof(BFloat16));
      float* ci_offset = (float*)aligned_alloc(64, m_bs * n_bs * sizeof(float));
      zero_fill(ci_offset, m_bs, n_bs, n_bs);
      auto compute_block = [&](int kb) {
        int kb_start = kb * BLOCK_K;
        int k_bs = std::min(BLOCK_K, K - kb_start);
        BFloat16* A_offset = PTR_OFFSET(A, mb_start, kb_start, lda);
        uint8_t* B_offset = B + nb_start / 2 * K + kb_start * n_bs / 2;
        dequant(
            B_offset,
            bi_offset,
            k_bs,
            n_bs,
            scale + nb_start,
            zero_point + nb_start);
        // MKL gemm
        // C := alpha*op(A) *op(B) + beta*C
        // op(A) is m-by-k, op(B) is k-by-n, C is m-by-n.
        cblas_gemm_bf16bf16f32(
            CblasRowMajor, // Row/col major
            CblasNoTrans, // Trans A
            CblasNoTrans, // Trans B
            m_bs, // M
            n_bs, // N
            k_bs, // K
            1.f, // alpha = 1.0
            (const MKL_BF16*)A_offset, // A
            lda, // lda
            (const MKL_BF16*)bi_offset, // B
            n_bs, // ldb
            1.f, // beta = 1.0 as we need to accumulate
            ci_offset, // C
            n_bs); // ldc
      };
      int kb = 0;
      for (; kb < KB; kb++) {
        compute_block(kb);
      }
      if constexpr (has_bias) {
        add_bias(ci_offset, bias + nb_start, m_bs, n_bs, n_bs);
      }
      cvt_fp32_to_bf16(C_offset, ci_offset, m_bs, n_bs, ldc, n_bs);
      free(ci_offset);
      free(bi_offset);
    }
  }
}

// per tensor
template <bool has_bias, int BLOCK_M>
void woq_gemm_brgemm_per_tensor(
    float* A,
    int8_t* B,
    float* C,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    float scale,
    float zero_point,
    float* bias = NULL) {
#define PTR_OFFSET(base, offset0, offset1, stride0) \
  (base) + (offset0) * (stride0) + (offset1)

  const int MB = (M + BLOCK_M - 1) / BLOCK_M, NB = (N + BLOCK_N - 1) / BLOCK_N,
            KB = (K + BLOCK_K - 1) / BLOCK_K; // num of blks

#pragma omp parallel for collapse(2)
  for (int mb = 0; mb < MB; mb++) {
    for (int nb = 0; nb < NB; nb++) {
      int mb_start = mb * BLOCK_M;
      int m_bs = std::min(BLOCK_M, M - mb_start);
      int nb_start = nb * BLOCK_N;
      int n_bs = std::min(BLOCK_N, N - nb_start);
      float* C_offset = PTR_OFFSET(C, mb_start, nb_start, ldc);
      zero_fill(C_offset, m_bs, n_bs, ldc);
      float* bi_offset =
          (float*)aligned_alloc(64, BLOCK_K * BLOCK_N * sizeof(float));
      for (int kb = 0; kb < KB; kb++) {
        int kb_start = kb * BLOCK_K;
        int k_bs = std::min(BLOCK_K, K - kb_start);
        float* A_offset = PTR_OFFSET(A, mb_start, kb_start, lda);
        int8_t* B_offset = B + nb_start * K + kb_start * n_bs;
        dequant(B_offset, bi_offset, k_bs, n_bs, scale, zero_point);
        if (m_bs == BLOCK_M && n_bs == BLOCK_N && k_bs == BLOCK_K) {
          dot_tile_update<BLOCK_N, BLOCK_M, BLOCK_K>(
              bi_offset, A_offset, C_offset, false, false, n_bs, lda, ldc);
        } else {
          dot_update( // libxsmm is col major
              bi_offset,
              A_offset,
              C_offset,
              n_bs,
              m_bs,
              k_bs,
              false,
              false,
              n_bs,
              lda,
              ldc);
        }
      }
      if constexpr (has_bias) {
        add_bias(C_offset, bias + nb_start, m_bs, n_bs, ldc);
      }
      free(bi_offset);
    }
  }
}

template <typename T1, typename T2, bool has_bias, int BM>
struct Function_matching {
  static void func_match(
      T1* A,
      T2* B,
      T1* C,
      int M,
      int N,
      int K,
      int lda,
      int ldb,
      int ldc,
      float* scale,
      float* zero_point,
      float* bias = NULL) {
    switch (M) {
      case BM:
        woq_gemm_intrinsic<has_bias, BM>(
            A, B, C, M, N, K, K, K, N, scale, zero_point, bias);
        break;
      default:
        Function_matching<T1, T2, has_bias, BM - 1>::func_match(
            A, B, C, M, N, K, lda, ldb, ldc, scale, zero_point, bias);
        break;
    }
  }
};

template <typename T1, typename T2, bool has_bias>
struct Function_matching<T1, T2, has_bias, 0> {
  static void func_match(
      T1* A,
      T2* B,
      T1* C,
      int M,
      int N,
      int K,
      int lda,
      int ldb,
      int ldc,
      float* scale,
      float* zero_point,
      float* bias = NULL) {
    return; // do nothing
  }
};

void woq_gemm_kernel_impl(
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& scales_float,
    const at::Tensor& zero_points_float,
    const at::Tensor& bias,
    int64_t lowp_mode,
    at::Tensor& output) {
  auto self_ = self.is_contiguous() ? self : self.contiguous();
  const int64_t dim = self.dim();
  auto self_reshaped =
      dim == 2 ? self_ : self_.reshape({-1, self.size(self.dim() - 1)});
  auto M = self_reshaped.size(0);
  auto K = self_reshaped.size(1);
  auto N = weight.size(0);
#if defined(CPU_CAPABILITY_AVX512)
  const auto qtype = weight.qscheme();

  // TODO: per-tensor block size tuning
  // dequant per tensor
  // only fp32 activation and int8 weight is supported
  if (qtype == c10::kPerTensorAffine) {
    auto in_ptr = self_.data_ptr<float>();
    auto weight_ptr = weight.data_ptr<int8_t>();
    auto out_ptr = output.data_ptr<float>();
    auto scales_float_ptr = scales_float.data_ptr<float>();
    auto zero_points_float_ptr = zero_points_float.data_ptr<float>();
    if (bias.defined()) {
      auto bias_ = bias.is_contiguous() ? bias : bias.contiguous();
      auto bias_ptr = bias_.data_ptr<float>();
      return woq_gemm_brgemm_per_tensor<true, 24>(
          in_ptr,
          weight_ptr,
          out_ptr,
          M,
          N,
          K,
          K,
          K,
          N,
          scales_float_ptr[0],
          zero_points_float_ptr[0],
          bias_ptr);
    } else {
      return woq_gemm_brgemm_per_tensor<false, 24>(
          in_ptr,
          weight_ptr,
          out_ptr,
          M,
          N,
          K,
          K,
          K,
          N,
          scales_float_ptr[0],
          zero_points_float_ptr[0]);
    }
  }
  // TODO: per-channel block size tuning
  // dequant per channel
  // activation of both fp32 and bf16, weight of int8 is supported
  else if (qtype == c10::kPerChannelAffine) {
    auto weight_ptr = weight.data_ptr<int8_t>();
    auto scales_float_ptr = scales_float.data_ptr<float>();
    auto zero_points_float_ptr = zero_points_float.data_ptr<float>();
    if (self_.scalar_type() == at::kFloat) {
      auto in_ptr = self_.data_ptr<float>();
      auto out_ptr = output.data_ptr<float>();
      if (bias.defined()) { // case with bias
        auto bias_ = bias.is_contiguous() ? bias : bias.contiguous();
        auto bias_ptr = bias_.data_ptr<float>();
        if (M <= 4) { // small M
          Function_matching<float, int8_t, true, 4>::func_match(
              in_ptr,
              weight_ptr,
              out_ptr,
              M,
              N,
              K,
              K,
              K,
              N,
              scales_float_ptr,
              zero_points_float_ptr,
              bias_ptr);
          return;
        } else { // large M
          return woq_gemm_brgemm<true, 196>(
              in_ptr,
              weight_ptr,
              out_ptr,
              M,
              N,
              K,
              K,
              K,
              N,
              scales_float_ptr,
              zero_points_float_ptr,
              bias_ptr);
        }
      } else { // case without bias
        if (M <= 4) { // small M
          Function_matching<float, int8_t, false, 4>::func_match(
              in_ptr,
              weight_ptr,
              out_ptr,
              M,
              N,
              K,
              K,
              K,
              N,
              scales_float_ptr,
              zero_points_float_ptr);
          return;
        } else { // large M
          return woq_gemm_brgemm<false, 196>(
              in_ptr,
              weight_ptr,
              out_ptr,
              M,
              N,
              K,
              K,
              K,
              N,
              scales_float_ptr,
              zero_points_float_ptr);
        }
      }
    } else if (self_.scalar_type() == at::kBFloat16) {
      auto in_ptr = self_.data_ptr<BFloat16>();
      auto out_ptr = output.data_ptr<BFloat16>();
      if (bias.defined()) { // case with bias
        auto bias_ = bias.is_contiguous() ? bias : bias.contiguous();
        auto bias_ptr = bias_.data_ptr<float>();
        if (M <= 4) { // small M
          return Function_matching<BFloat16, int8_t, true, 4>::func_match(
              in_ptr,
              weight_ptr,
              out_ptr,
              M,
              N,
              K,
              K,
              K,
              N,
              scales_float_ptr,
              zero_points_float_ptr,
              bias_ptr);
        } else { // large M
          return woq_gemm_brgemm<true, 196>(
              in_ptr,
              weight_ptr,
              out_ptr,
              M,
              N,
              K,
              K,
              K,
              N,
              scales_float_ptr,
              zero_points_float_ptr,
              bias_ptr);
        }
      } else { // case without bias
        if (M <= 4) { // small M
          return Function_matching<BFloat16, int8_t, false, 4>::func_match(
              in_ptr,
              weight_ptr,
              out_ptr,
              M,
              N,
              K,
              K,
              K,
              N,
              scales_float_ptr,
              zero_points_float_ptr);
        } else { // large M
          return woq_gemm_brgemm<false, 196>(
              in_ptr,
              weight_ptr,
              out_ptr,
              M,
              N,
              K,
              K,
              K,
              N,
              scales_float_ptr,
              zero_points_float_ptr);
        }
      }
    } else {
      auto qw = woq_linear_unpackB_impl(weight);
      auto w = qw.dequantize().to(self_.scalar_type()).to(c10::kFloat);
      auto x = self.to(c10::ScalarType::Float);
      auto out = at::linear(x, w);
      if (bias.defined()) {
        auto b = bias.to(self_.scalar_type()).to(c10::kFloat);
        out = at::add(out, b);
      }
      output = out.to(self.scalar_type());
    }
  } else { // kPerChannelAffineFloatQParams

    auto weight_ptr = weight.data_ptr<uint8_t>();
    auto scales_float_ptr = scales_float.data_ptr<float>();
    auto zero_points_float_ptr = zero_points_float.data_ptr<float>();

    if (self_.scalar_type() == at::kFloat) {
      auto in_ptr = self_.data_ptr<float>();
      auto out_ptr = output.data_ptr<float>();
      if (bias.defined()) { // case with bias
        auto bias_ = bias.is_contiguous() ? bias : bias.contiguous();
        auto bias_ptr = bias_.data_ptr<float>();
        if (M <= 4) { // small M
          Function_matching<float, uint8_t, true, 4>::func_match(
              in_ptr,
              weight_ptr,
              out_ptr,
              M,
              N,
              K,
              K,
              K,
              N,
              scales_float_ptr,
              zero_points_float_ptr,
              bias_ptr);
          return;
        } else { // large M
          return woq_gemm_brgemm<true, 196>(
              in_ptr,
              weight_ptr,
              out_ptr,
              M,
              N,
              K,
              K,
              K,
              N,
              scales_float_ptr,
              zero_points_float_ptr,
              bias_ptr);
        }
      } else { // case without bias
        if (M <= 4) { // small M
          Function_matching<float, uint8_t, false, 4>::func_match(
              in_ptr,
              weight_ptr,
              out_ptr,
              M,
              N,
              K,
              K,
              K,
              N,
              scales_float_ptr,
              zero_points_float_ptr);
          return;
        } else { // large M
          return woq_gemm_brgemm<false, 196>(
              in_ptr,
              weight_ptr,
              out_ptr,
              M,
              N,
              K,
              K,
              K,
              N,
              scales_float_ptr,
              zero_points_float_ptr);
        }
      }
    } else if (self_.scalar_type() == at::kBFloat16) {
      auto in_ptr = self_.data_ptr<BFloat16>();
      auto out_ptr = output.data_ptr<BFloat16>();
      if (bias.defined()) { // case with bias
        auto bias_ = bias.is_contiguous() ? bias : bias.contiguous();
        auto bias_ptr = bias_.data_ptr<float>();
        if (M <= 4) { // small M
          return Function_matching<BFloat16, uint8_t, true, 4>::func_match(
              in_ptr,
              weight_ptr,
              out_ptr,
              M,
              N,
              K,
              K,
              K,
              N,
              scales_float_ptr,
              zero_points_float_ptr,
              bias_ptr);
        } else { // large M
          return woq_gemm_brgemm<true, 196>(
              in_ptr,
              weight_ptr,
              out_ptr,
              M,
              N,
              K,
              K,
              K,
              N,
              scales_float_ptr,
              zero_points_float_ptr,
              bias_ptr);
        }
      } else { // case without bias
        if (M <= 4) { // small M
          return Function_matching<BFloat16, uint8_t, false, 4>::func_match(
              in_ptr,
              weight_ptr,
              out_ptr,
              M,
              N,
              K,
              K,
              K,
              N,
              scales_float_ptr,
              zero_points_float_ptr);
        } else { // large M
          return woq_gemm_brgemm<false, 196>(
              in_ptr,
              weight_ptr,
              out_ptr,
              M,
              N,
              K,
              K,
              K,
              N,
              scales_float_ptr,
              zero_points_float_ptr);
        }
      }
    }
  }
#else
  if (self.scalar_type() == c10::ScalarType::Float) {
    auto w = weight.dequantize();
    if (bias.defined()) {
      at::linear_out(output, self, w, bias.detach());
    } else {
      at::linear_out(output, self, w);
    }
  } else if (self_.scalar_type() == at::kBFloat16) {
    auto w = weight.dequantize();
    auto x = self.to(c10::ScalarType::Float);
    // This is to align with the AVX512 kernel
    // so that UT test_weight_only_quantization_autocast can pass
    if (M > 4) {
      w = w.to(c10::kBFloat16).to(c10::kFloat);
    }
    auto out = at::linear(x, w);
    if (bias.defined()) {
      out = at::add(out, bias);
    }
    output = out.to(self.scalar_type());
  } else {
    auto w = weight.dequantize().to(self_.scalar_type()).to(c10::kFloat);
    auto x = self.to(c10::ScalarType::Float);
    auto out = at::linear(x, w);
    if (bias.defined()) {
      auto b = bias.to(self_.scalar_type()).to(c10::kFloat);
      out = at::add(out, b);
    }
    output = out.to(self.scalar_type());
  }

#endif
}

void woq_gemm_eltwise_kernel_impl(
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& scales_float,
    const at::Tensor& zero_points_float,
    const at::Tensor& bias,
    const c10::string_view& post_op,
    const torch::List<c10::optional<at::Scalar>>& scalars,
    const c10::optional<c10::string_view>& algorithm,
    int64_t lowp_mode,
    at::Tensor& output) {
  // TODO Postop not yet implemented in kernel
  // Here we apply post op after GEMM
  woq_gemm_kernel_impl(
      self, weight, scales_float, zero_points_float, bias, lowp_mode, output);
  auto postop_func = postop_func_map[post_op](scalars, algorithm);
  postop_func(output);
}

at::Tensor woq_linear_packB_impl(
    const at::Tensor& weight,
    const at::Tensor& scales,
    const at::Tensor& zero_points) {
#if defined(CPU_CAPABILITY_AVX512)
  auto N = weight.size(0);
  auto K = weight.size(1);
  auto weight_size = weight.sizes().vec();
  auto weight_contig = weight.contiguous();
  const auto qtype = weight.qscheme();
  if (weight.scalar_type() == c10::ScalarType::QUInt4x2) { // int4 weight
    auto weight_packed = at::_empty_per_channel_affine_quantized(
        weight_size,
        scales,
        zero_points,
        1,
        device(c10::kCPU).dtype(c10::kQUInt4x2));
    auto weight_ptr = weight_contig.data_ptr<uint8_t>();
    auto weightpacked_ptr =
        reinterpret_cast<uint8_t*>(weight_packed.data_ptr());
    pack(weight_ptr, weightpacked_ptr, K, N, (K + 1) / 2, true);
    return weight_packed;
  } else { // int8 weight
    auto weight_packed = at::_empty_per_channel_affine_quantized(
        weight_size,
        scales,
        zero_points,
        1,
        device(c10::kCPU).dtype(c10::kQInt8));

    auto weight_ptr = weight_contig.data_ptr<int8_t>();
    auto weightpacked_ptr = reinterpret_cast<int8_t*>(weight_packed.data_ptr());
    pack(weight_ptr, weightpacked_ptr, K, N, K, true);
    return weight_packed;
  }
#else
  return weight;
#endif
}

at::Tensor woq_linear_unpackB_impl(const at::Tensor& weight) {
#if defined(CPU_CAPABILITY_AVX512)
  auto N = weight.size(0);
  auto K = weight.size(1);
  const auto qtype = weight.qscheme();

  if (weight.scalar_type() == c10::ScalarType::QUInt4x2) { // int4 weight
    std::vector<float> weight_scales_float(1, 0.0);
    if (qtype == c10::kPerChannelAffineFloatQParams) {
      weight_scales_float.resize(N, 0.0);
      for (const auto i : c10::irange(N)) {
        weight_scales_float[i] = weight.q_per_channel_scales()[i].item<float>();
      }
    }
    at::Tensor scales = at::empty(
        {static_cast<long>(weight_scales_float.size())},
        at::device(c10::kCPU).dtype(c10::kFloat));
    std::copy(
        weight_scales_float.begin(),
        weight_scales_float.end(),
        scales.data_ptr<float>());

    std::vector<float> weight_zero_points_float(1, 0); // zero_points is float
    if (qtype == c10::kPerChannelAffineFloatQParams) {
      weight_zero_points_float.resize(N, 0);
      for (const auto i : c10::irange(N)) {
        weight_zero_points_float[i] =
            weight.q_per_channel_zero_points()[i].item<float>();
      }
    }
    at::Tensor zero_points = at::empty(
        {static_cast<long>(weight_zero_points_float.size())},
        at::device(c10::kCPU).dtype(c10::kFloat));
    std::copy(
        weight_zero_points_float.begin(),
        weight_zero_points_float.end(),
        zero_points.data_ptr<float>());

    auto weight_size = weight.sizes().vec();
    auto weight_unpacked = at::_empty_per_channel_affine_quantized(
        weight_size,
        scales,
        zero_points,
        0,
        device(c10::kCPU).dtype(c10::kQUInt4x2));

    auto weight_contig = weight.contiguous();

    auto weight_ptr = weight_contig.data_ptr<uint8_t>();
    auto weight_unpacked_ptr =
        reinterpret_cast<uint8_t*>(weight_unpacked.data_ptr());
    unpack(weight_ptr, weight_unpacked_ptr, K, N, K, true);
    return weight_unpacked;
  } else { // int8 weight
    std::vector<float> weight_scales_float(1, 0.0);
    if (qtype == c10::kPerTensorAffine) {
      weight_scales_float[0] = weight.q_scale();
    } else if (qtype == c10::kPerChannelAffine) {
      weight_scales_float.resize(N, 0.0);
      for (const auto i : c10::irange(N)) {
        weight_scales_float[i] = weight.q_per_channel_scales()[i].item<float>();
      }
    }
    at::Tensor scales = at::empty(
        {static_cast<long>(weight_scales_float.size())},
        at::device(c10::kCPU).dtype(c10::kFloat));
    std::copy(
        weight_scales_float.begin(),
        weight_scales_float.end(),
        scales.data_ptr<float>());

    std::vector<int32_t> weight_zero_points_int32(1, 0); // zero points is int32
    if (qtype == c10::kPerTensorAffine) {
      weight_zero_points_int32[0] = weight.q_zero_point();
    } else if (qtype == c10::kPerChannelAffine) {
      weight_zero_points_int32.resize(N, 0);
      for (const auto i : c10::irange(N)) {
        weight_zero_points_int32[i] =
            weight.q_per_channel_zero_points()[i].item<int32_t>();
      }
    }
    at::Tensor zero_points = at::empty(
        {static_cast<long>(weight_zero_points_int32.size())},
        at::device(c10::kCPU).dtype(c10::kInt));
    std::copy(
        weight_zero_points_int32.begin(),
        weight_zero_points_int32.end(),
        zero_points.data_ptr<int32_t>());

    auto weight_size = weight.sizes().vec();
    auto weight_unpacked = at::_empty_per_channel_affine_quantized(
        weight_size,
        scales,
        zero_points,
        0,
        device(c10::kCPU).dtype(c10::kQInt8));

    auto weight_contig = weight.contiguous();

    auto weight_ptr = weight_contig.data_ptr<int8_t>();
    auto weight_unpacked_ptr =
        reinterpret_cast<int8_t*>(weight_unpacked.data_ptr());
    unpack(weight_ptr, weight_unpacked_ptr, K, N, K, true);
    return weight_unpacked;
  }

#else
  return weight;
#endif
}
} // namespace

IPEX_REGISTER_DISPATCH(woq_gemm_kernel_stub, &woq_gemm_kernel_impl);
IPEX_REGISTER_DISPATCH(
    woq_gemm_eltwise_kernel_stub,
    &woq_gemm_eltwise_kernel_impl);
IPEX_REGISTER_DISPATCH(woq_linear_unpackB_stub, &woq_linear_unpackB_impl);
IPEX_REGISTER_DISPATCH(woq_linear_packB_stub, &woq_linear_packB_impl);
} // namespace cpu
} // namespace torch_ipex
#endif