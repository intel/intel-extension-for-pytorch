// weight-only quantization gemm kernel (int8, int4 etc.)
// #include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/cpu/vec/vec.h>
#include <aten/Linear.h>
#include "csrc/cpu/tpp/woq/tla.h"

#ifdef __GNUC__
#include <features.h>
#if __GNUC_PREREQ(12, 3)
#define COMPILER_PREREQ_MET
#endif
#endif

namespace torch_ipex {
namespace cpu {
namespace {

using namespace tpp;
using TensorList = std::vector<at::Tensor>;

// We only build optimized kernels if AVX512_FP16 is supported and gcc>=12.3
// Otherwise we just return empty results
// TODO(Weiwen) Merge WoqTppKrnl.cpp and WoqLinearKrnl.cpp and put the latter in
// the #else part
#if defined(CPU_CAPABILITY_AVX512_FP16) && defined(COMPILER_PREREQ_MET)

#define SMALL_BATCH_THRESHOLD 32
#define PARALLEL_M_THRESHOLD 128
constexpr long PREFETCH_K_DIST = 64; // TODO(jgong5): do not hard-code
constexpr long LOOP_K_UNROLL = 4; // TODO(jgong5): do not hard-code

template <long N_GROUP_SIZE, typename VAT, typename LUT>
inline VAT load_dequant_zp_only_int4(uint8_t* p, VAT vzps, LUT lut) {
  TLA_ASSERT(false, "not implemented");
}

template <long N_GROUP_SIZE, typename VAT>
inline VAT load_dequant_zp_only_int8(uint8_t* p, VAT vzps) {
  TLA_ASSERT(false, "not implemented");
}

// TODO(jgong5): further simplify the dequant intrinsics below with VecOps
#ifdef __AVX512F__
template <>
inline std::array<__m512, 4> load_dequant_zp_only_int4<64>(
    uint8_t* p,
    std::array<__m512, 4> vzps,
    __m512 lut) {
  using T = float;
  using VA = VecArray<64, T>;
  using VAT = typename VA::type;
  constexpr long COLS = VA::num_vec;
  auto packed = _mm256_loadu_si256((__m256i*)p);
  __m512i int32[COLS];
  {
    auto low_4bit = _mm512_cvtepu8_epi32(_mm256_castsi256_si128(packed));
    auto high_4bit = _mm512_srli_epi32(low_4bit, 4);
    int32[0] = low_4bit;
    int32[2] = high_4bit;
  }
  {
    auto low_4bit = _mm512_cvtepu8_epi32(_mm256_extracti128_si256(packed, 1));
    auto high_4bit = _mm512_srli_epi32(low_4bit, 4);
    int32[1] = low_4bit;
    int32[3] = high_4bit;
  }
  VAT vbs;
  compile_time_for<COLS>::op([&](auto idx) {
    vbs[idx] = _mm512_permutexvar_ps(int32[idx], lut);
    vbs[idx] = _mm512_sub_ps(vbs[idx], vzps[idx]);
  });
  return vbs;
}

template <>
inline std::array<__m512, 2> load_dequant_zp_only_int4<32>(
    uint8_t* p,
    std::array<__m512, 2> vzps,
    __m512 lut) {
  using T = float;
  using VA = VecArray<32, T>;
  using VAT = typename VA::type;
  constexpr long COLS = VA::num_vec;
  auto packed = _mm_loadu_si128((__m128i*)p);
  __m512i int32[COLS];
  {
    auto low_4bit = _mm512_cvtepu8_epi32(packed);
    auto high_4bit = _mm512_srli_epi32(low_4bit, 4);
    int32[0] = low_4bit;
    int32[1] = high_4bit;
  }
  VAT vbs;
  compile_time_for<COLS>::op([&](auto idx) {
    vbs[idx] = _mm512_permutexvar_ps(int32[idx], lut);
    vbs[idx] = _mm512_sub_ps(vbs[idx], vzps[idx]);
  });
  return vbs;
}

template <>
inline std::array<__m512, 1> load_dequant_zp_only_int4<16>(
    uint8_t* p,
    std::array<__m512, 1> vzps,
    __m512 lut) {
  using T = float;
  using VA = VecArray<16, T>;
  using VAT = typename VA::type;
  constexpr long COLS = VA::num_vec;
  static_assert(COLS == 1, "COLS must be 1");
  uint64_t packed = reinterpret_cast<uint64_t*>(p)[0];
  uint64_t high = packed >> 4;
  __m128i packed_128 = _mm_set_epi64x(high, packed);
  __m512i int32 = _mm512_cvtepu8_epi32(packed_128);
  VAT vbs;
  vbs[0] = _mm512_permutexvar_ps(int32, lut);
  vbs[0] = _mm512_sub_ps(vbs[0], vzps[0]);
  return vbs;
}

template <>
inline std::array<__m512, 4> load_dequant_zp_only_int8<64>(
    uint8_t* p,
    std::array<__m512, 4> vzps) {
  using T = float;
  using VA = VecArray<64, T>;
  using VAT = typename VA::type;
  constexpr long COLS = VA::num_vec;
  auto packed = _mm512_loadu_si512((__m512i*)p);
  VAT vbs;
  compile_time_for<COLS>::op([&](auto i) {
    constexpr long imm = i;
    auto int8 = _mm512_extracti32x4_epi32(packed, imm);
    vbs[i] = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(int8));
    vbs[i] = _mm512_sub_ps(vbs[i], vzps[i]);
  });
  return vbs;
}

template <>
inline std::array<__m512, 2> load_dequant_zp_only_int8<32>(
    uint8_t* p,
    std::array<__m512, 2> vzps) {
  using T = float;
  using VA = VecArray<32, T>;
  using VAT = typename VA::type;
  constexpr long COLS = VA::num_vec;
  auto packed = _mm256_loadu_si256((__m256i*)p);
  VAT vbs;
  compile_time_for<COLS>::op([&](auto i) {
    constexpr long imm = i;
    auto int8 = _mm256_extracti128_si256(packed, imm);
    vbs[i] = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(int8));
    vbs[i] = _mm512_sub_ps(vbs[i], vzps[i]);
  });
  return vbs;
}

template <>
inline std::array<__m512, 1> load_dequant_zp_only_int8<16>(
    uint8_t* p,
    std::array<__m512, 1> vzps) {
  using T = float;
  using VA = VecArray<16, T>;
  using VAT = typename VA::type;
  constexpr long COLS = VA::num_vec;
  static_assert(COLS == 1);
  auto packed = _mm_loadu_si128((__m128i*)p);
  VAT vbs;
  vbs[0] = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(packed));
  vbs[0] = _mm512_sub_ps(vbs[0], vzps[0]);
  return vbs;
}

inline __m512i combine_m256i(__m256i a, __m256i b) {
  __m512i c = _mm512_castsi256_si512(a);
  return _mm512_inserti64x4(c, b, 1);
}

inline __m512i combine_m256i(std::array<__m256i, 2> two_256) {
  return combine_m256i(two_256[0], two_256[1]);
}

inline std::array<__m256i, 2> load_zps_4vnni(int8_t* zps) {
  // broadcast 01234567 to
  // 01234567012345670123456701234567
  __m256i vzps_low = _mm256_set1_epi64x(*reinterpret_cast<long*>(zps));
  __m256i vzps_high = _mm256_set1_epi64x(*reinterpret_cast<long*>(zps + 8));
  // shuffle from
  // 01234567012345670123456701234567
  // to
  // 00001111222233334444555566667777
  __m256i shuffle_mask = _mm256_set_epi8(
      7,
      7,
      7,
      7,
      6,
      6,
      6,
      6,
      5,
      5,
      5,
      5,
      4,
      4,
      4,
      4,
      3,
      3,
      3,
      3,
      2,
      2,
      2,
      2,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0);
  vzps_low = _mm256_shuffle_epi8(vzps_low, shuffle_mask);
  vzps_high = _mm256_shuffle_epi8(vzps_high, shuffle_mask);
  return {vzps_low, vzps_high};
}

inline std::array<__m256i, 2> load_int4_as_int8(uint8_t* qB) {
  __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qB));
  const __m256i low_mask = _mm256_set1_epi8(0x0f);
  __m256i high = _mm256_srli_epi16(packed, 4);
  high = _mm256_and_si256(high, low_mask);
  __m256i low = _mm256_and_si256(packed, low_mask);
  return {low, high};
}

#else
inline std::array<__m256i, 2> load_zps_4vnni(int8_t* zps) {
  TLA_ASSERT(false, "not implemented");
  return std::array<__m256i, 2>();
}

inline std::array<__m256i, 2> load_int4_as_int8(uint8_t* qB) {
  TLA_ASSERT(false, "not implemented");
  return std::array<__m256i, 2>();
}

#endif

#ifdef __AVX512FP16__
template <>
inline std::array<__m512h, 2> load_dequant_zp_only_int4<64>(
    uint8_t* p,
    std::array<__m512h, 2> vzps,
    __m512h lut) {
  using T = tpp::half;
  using VA = VecArray<64, T>;
  using VAT = typename VA::type;
  constexpr long COLS = VA::num_vec;
  auto packed = _mm256_loadu_si256((__m256i*)p);
  __m512i int32[COLS];
  {
    auto low_4bit = _mm512_cvtepu8_epi16(packed);
    auto high_4bit = _mm512_srli_epi16(low_4bit, 4);
    int32[0] = low_4bit;
    int32[1] = high_4bit;
  }
  VAT vbs;
  compile_time_for<COLS>::op([&](auto idx) {
    vbs[idx] = _mm512_permutexvar_ph(int32[idx], lut);
    vbs[idx] = _mm512_sub_ph(vbs[idx], vzps[idx]);
  });
  return vbs;
}

template <>
inline std::array<__m512h, 1> load_dequant_zp_only_int4<32>(
    uint8_t* p,
    std::array<__m512h, 1> vzps,
    __m512h lut) {
  using T = tpp::half;
  using VA = VecArray<32, T>;
  using VAT = typename VA::type;
  constexpr long COLS = VA::num_vec;
  auto packed = _mm_loadu_si128((__m128i*)p);
  __m512i int32[COLS];
  {
    auto low_4bit = _mm256_cvtepu8_epi16(packed);
    auto high_4bit = _mm256_srli_epi16(low_4bit, 4);
    // combine low_4bit and high_4bit into __m512i
    int32[0] =
        _mm512_inserti64x4(_mm512_castsi256_si512(low_4bit), high_4bit, 1);
  }
  VAT vbs;
  compile_time_for<COLS>::op([&](auto idx) {
    vbs[idx] = _mm512_permutexvar_ph(int32[idx], lut);
    vbs[idx] = _mm512_sub_ph(vbs[idx], vzps[idx]);
  });
  return vbs;
}

template <>
inline std::array<__m512h, 2> load_dequant_zp_only_int8<64>(
    uint8_t* p,
    std::array<__m512h, 2> vzps) {
  using T = tpp::half;
  using VA = VecArray<64, T>;
  using VAT = typename VA::type;
  constexpr long COLS = VA::num_vec;
  auto packed = _mm512_loadu_si512((__m512i*)p);
  VAT vbs;
  compile_time_for<COLS>::op([&](auto i) {
    constexpr long imm = i;
    auto int8 = _mm512_extracti64x4_epi64(packed, imm);
    vbs[i] = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(int8));
    vbs[i] = _mm512_sub_ph(vbs[i], vzps[i]);
  });
  return vbs;
}

template <>
inline std::array<__m512h, 1> load_dequant_zp_only_int8<32>(
    uint8_t* p,
    std::array<__m512h, 1> vzps) {
  using T = tpp::half;
  using VA = VecArray<32, T>;
  using VAT = typename VA::type;
  constexpr long COLS = VA::num_vec;
  auto packed = _mm256_loadu_si256((__m256i*)p);
  VAT vbs;
  compile_time_for<COLS>::op([&](auto i) {
    constexpr long imm = i;
    vbs[i] = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(packed));
    vbs[i] = _mm512_sub_ph(vbs[i], vzps[i]);
  });
  return vbs;
}
#endif

template <long N, typename T>
struct load_dequant_int4 {
  using VT = typename VecType<T>::type;
  using V = VecOps<VT>;
  using VA = VecArray<N, T>;
  using VAT = typename VA::type;
  constexpr static long COLS = VA::num_vec;

  static inline VAT call(uint8_t* p, VAT vscales, VAT vzps, VT lut) {
    auto vbs = load_dequant_zp_only_int4<N>(p, vzps, lut);
    compile_time_for<COLS>::op(
        [&](auto idx) { vbs[idx] = V::mul(vbs[idx], vscales[idx]); });
    return vbs;
  }
};

template <long N, typename T>
struct load_dequant_int8 {
  using VT = typename VecType<T>::type;
  using V = VecOps<VT>;
  using VA = VecArray<N, T>;
  using VAT = typename VA::type;
  constexpr static long COLS = VA::num_vec;

  static inline VAT call(uint8_t* p, VAT vscales, VAT vzps) {
    auto vbs = load_dequant_zp_only_int8<N>(p, vzps);
    compile_time_for<COLS>::op(
        [&](auto idx) { vbs[idx] = V::mul(vbs[idx], vscales[idx]); });
    return vbs;
  }
};

constexpr int get_n_group_size(int N) {
  return N == 16 ? 16 : (N == 32 ? 32 : 64);
}

// TODO(jgong5): move to tpp.h
// TODO(jgong5): add pre/post op fusion
template <
    typename T,
    typename Tout,
    typename TScale,
    typename TZero,
    long M,
    long N,
    long ldb,
    bool transA = false,
    bool ACC = false,
    long PREFETCH_K_DIST = 0,
    typename Enabled = void>
struct GemmMicroKernel {
  template <bool is_int4>
  static inline void call(
      long K,
      T* A,
      long lda,
      uint8_t* B,
      Tout* C,
      long ldc,
      TScale* scales,
      TZero* zps) {
    TLA_ASSERT(false, "Not implemented");
  }
};

template <
    typename T,
    long M,
    long N,
    long ldb,
    bool transA,
    bool ACC,
    long PREFETCH_K_DIST>
struct GemmMicroKernel<
    T,
    T,
    T,
    T,
    M,
    N,
    ldb,
    transA,
    ACC,
    PREFETCH_K_DIST,
    typename std::enable_if_t<
        std::is_same<T, float>::value || std::is_same<T, half>::value>> {
  // TODO(jgong5): generalize this with pre/post op handlers
  template <bool is_int4>
  static inline void call(
      long K,
      T* A,
      long lda,
      uint8_t* B,
      T* C,
      long ldc,
      T* scales,
      T* zps) {
#define INDEX(x, y, ld) ((x) * (ld) + (y))
#define ADDRESS(p, x, y, ld) ((p) + (x) * (ld) + (y))

    static_assert(N % 16 == 0, "N must be a multiple of 16");
    constexpr const int N_GROUP_SIZE = get_n_group_size(N);

    using VT = typename VecType<T>::type;
    using V = VecOps<VT>;
    using ST = typename V::ST;
    using VArray = VecArray<N_GROUP_SIZE, T>;
    using VArrayT = typename VArray::type;

    constexpr const int COLS = N / V::VLEN;
    constexpr const int CBLOCK = N_GROUP_SIZE / V::VLEN;
    constexpr const int CNBLOCKS = N / N_GROUP_SIZE;
    VT va[M];
    VArrayT vb[CNBLOCKS];
    VT vc[M * COLS];
    VArrayT vscales[CNBLOCKS];
    VArrayT vzps[CNBLOCKS];

    VT lut = V::set_0_to_15();

    // Load scales and zps
    compile_time_for<CNBLOCKS>::op([&](auto i) {
      constexpr const int col = i * CBLOCK;
      vscales[i] = VArray::load1d(scales + col * V::VLEN);
      vzps[i] = VArray::load1d(zps + col * V::VLEN);
    });

    // NB: For fp16 in int8 woq, we do not delay the scale to the post-op but
    // leave it to the dequant otherwise the weight value might be too large to
    // overflow fp16 range.
    constexpr bool scale_as_post_op = !std::is_same<T, half>() || is_int4;

    compile_time_for<M * COLS>::op([&](auto i) { vc[i] = V::setzero(); });

    auto compute = [&](auto i, int k) {
      constexpr const int row = i / CNBLOCKS;
      constexpr const int cbidx = i % CNBLOCKS;

      if constexpr (cbidx == 0) {
        if constexpr (transA) {
          va[row] = V::set1(*(ST*)ADDRESS(A, k, row, lda));
        } else {
          va[row] = V::set1(*(ST*)ADDRESS(A, row, k, lda));
        }
      }

      if constexpr (row == 0) {
        constexpr const int col = cbidx * CBLOCK;
        if constexpr (scale_as_post_op) {
          if constexpr (is_int4) {
            vb[cbidx] = load_dequant_zp_only_int4<N_GROUP_SIZE>(
                ADDRESS(B, k, col * V::VLEN / 2, ldb / 2), vzps[cbidx], lut);
          } else {
            vb[cbidx] = load_dequant_zp_only_int8<N_GROUP_SIZE>(
                ADDRESS(B, k, col * V::VLEN, ldb), vzps[cbidx]);
          }
        } else {
          if constexpr (is_int4) {
            vb[cbidx] = load_dequant_int4<N_GROUP_SIZE, T>::call(
                ADDRESS(B, k, col * V::VLEN / 2, ldb / 2),
                vscales[cbidx],
                vzps[cbidx],
                lut);
          } else {
            vb[cbidx] = load_dequant_int8<N_GROUP_SIZE, T>::call(
                ADDRESS(B, k, col * V::VLEN, ldb), vscales[cbidx], vzps[cbidx]);
          }
        }
        if constexpr (PREFETCH_K_DIST > 0) {
          if constexpr (is_int4) {
            _mm_prefetch(
                ADDRESS(B, k + PREFETCH_K_DIST, col * V::VLEN / 2, ldb / 2),
                _MM_HINT_T0);
          } else {
            _mm_prefetch(
                ADDRESS(B, k + PREFETCH_K_DIST, col * V::VLEN, ldb),
                _MM_HINT_T0);
          }
        }
      }

      compile_time_for<CBLOCK>::op([&](auto col) {
        constexpr const int idx = INDEX(row, INDEX(cbidx, col, CBLOCK), COLS);
        vc[idx] = V::fmadd(va[row], vb[cbidx][col], vc[idx]);
      });
    };

    // Accumulate along k
    constexpr const int unroll = LOOP_K_UNROLL;
    int k = 0;
    for (; k < K / unroll; k++) {
      compile_time_for<unroll>::op([&](auto i) {
        compile_time_for<M * CNBLOCKS>::op(compute, k * unroll + i);
      });
    }
    k *= unroll;
    for (; k < K; k++) {
      compile_time_for<M * CNBLOCKS>::op(compute, k);
    }

    // Store to C
    auto store = [&](auto i) {
      constexpr const int row = i / COLS;
      constexpr const int col = i % COLS;
      if constexpr (ACC) {
        auto vc_old = V::loadu(ADDRESS(C, row, col * V::VLEN, ldc));
        if constexpr (scale_as_post_op) {
          vc[i] = V::fmadd(vscales[col / CBLOCK][col % CBLOCK], vc[i], vc_old);
        } else {
          vc[i] = V::fmadd(V::set1(1.0f), vc[i], vc_old);
        }
      } else if constexpr (scale_as_post_op) {
        vc[i] = V::mul(vscales[col / CBLOCK][col % CBLOCK], vc[i]);
      }
      V::storeu(ADDRESS(C, row, col * V::VLEN, ldc), vc[i]);
    };

    compile_time_for<M * COLS>::op(store);
  }
};

#ifdef __AVX512VNNI__
template <long M, long N, long ldb, bool transA, bool ACC, long PREFETCH_K_DIST>
struct GemmMicroKernel<
    /*Tin*/ uint8_t,
    /*Tout*/ float,
    /*TScale*/ float,
    /*TZero*/ int8_t,
    M,
    N,
    ldb,
    transA,
    ACC,
    PREFETCH_K_DIST> {
  template <bool is_int4>
  static inline void call(
      long K,
      uint8_t* A,
      long lda,
      uint8_t* B,
      float* C,
      long ldc,
      float* scales,
      int8_t* zps,
      float scale_a,
      int32_t zp_a) {
    auto pqB = GetVLAPtr<uint8_t>(B, {ldb, 2}); // [K/4,N,4] packed in 4-bit

    static_assert(N % 16 == 0, "N must be a multiple of 16");
    constexpr const int COLS = N / 16;

    __m512i ones = _mm512_set1_epi8(1); // used for computing compensation
    __m512i va;
    __m512i vb[COLS];
    __m512i vc[M * COLS];
    __m512 vscales[COLS];
    __m512i vzps[COLS];
    __m512i vcompensate[COLS];

    // Load scales and zps
    compile_time_for<COLS>::op([&](auto i) {
      vscales[i] = _mm512_loadu_ps(scales + i * 16);
      // TODO(jgong5): should we use 512 or two 256 here?
      vzps[i] = combine_m256i(load_zps_4vnni(zps + i * 16));
      vcompensate[i] = _mm512_setzero_epi32();
    });

    compile_time_for<M * COLS>::op(
        [&](auto i) { vc[i] = _mm512_setzero_epi32(); });

    auto compute = [&](auto i, int k) {
      constexpr const int row = i / COLS;
      constexpr const int col = i % COLS;

      if constexpr (col == 0) {
        if constexpr (transA) {
          va = _mm512_set1_epi32(*(int32_t*)ADDRESS(A, k, row, lda));
        } else {
          va = _mm512_set1_epi32(*(int32_t*)ADDRESS(A, row, k, lda));
        }
      }

      if constexpr (row == 0) {
        vb[col] = combine_m256i(load_int4_as_int8(pqB[k / 4][col * 16]));
        vb[col] = _mm512_sub_epi8(vb[col], vzps[col]);
        vcompensate[col] = _mm512_dpbusd_epi32(vcompensate[col], ones, vb[col]);
        if constexpr (PREFETCH_K_DIST > 0) {
          _mm_prefetch(pqB[(k + PREFETCH_K_DIST) / 4][col * 16], _MM_HINT_T0);
        }
      }

      vc[i] = _mm512_dpbusd_epi32(vc[i], va, vb[col]);
    };

    // Accumulate along k
    constexpr const int unroll = LOOP_K_UNROLL;
    int k = 0;
    for (; k < K / 4 / unroll; k++) {
      compile_time_for<unroll>::op([&](auto i) {
        compile_time_for<M * COLS>::op(compute, 4 * (k * unroll + i));
      });
    }
    k *= 4 * unroll;
    for (; k < K; k += 4) {
      compile_time_for<M * COLS>::op(compute, k);
    }

    // Store to C
    auto store = [&](auto i) {
      constexpr const int row = i / COLS;
      constexpr const int col = i % COLS;
      // compute (qC - compensate * zp_a) * scale_a * scale_b
      // where compensate = sum(qB)
      vc[i] = _mm512_sub_epi32(
          vc[i], _mm512_mullo_epi32(vcompensate[col], _mm512_set1_epi32(zp_a)));
      __m512 vc_float = _mm512_cvtepi32_ps(vc[i]);
      vc_float = _mm512_mul_ps(vc_float, _mm512_set1_ps(scale_a));
      vc_float = _mm512_mul_ps(vc_float, vscales[col]);
      if constexpr (ACC) {
        auto vc_old = _mm512_loadu_ps(C + row * ldc + col * 16);
        vc_float = _mm512_add_ps(vc_float, vc_old);
      }
      _mm512_storeu_ps(C + row * ldc + col * 16, vc_float);
    };
    compile_time_for<M * COLS>::op(store);
  }
};
#endif

// a dequant function the requires N to be a multiple of N_GROUP_SIZE
template <typename Tin, long ldb, long N_GROUP_SIZE, bool is_int4>
struct dequant_n_grouped {
  template <typename Lambda1, typename Lambda2, typename Lambda3>
  static inline void call(
      uint8_t* qB,
      long K,
      long N,
      Tin* scales,
      Tin* zps,
      Tin* B,
      const Lambda1& load_qparam,
      const Lambda2& load_qint_as_fp,
      const Lambda3& store) {
    for (int n = 0; n < N; n += N_GROUP_SIZE) {
      // load scales and zps
      auto vscales = load_qparam(scales + n);
      auto vzps = load_qparam(zps + n);
      for (int k = 0; k < K; k++) {
        // load and dequant qB to vb
        auto vbs = load_qint_as_fp(
            is_int4 ? &qB[k * ldb / 2 + n / 2] : &qB[k * ldb + n],
            vscales,
            vzps);
        // store vb to B
        store(B + k * N + n, vbs);
      }
    }
  }
};

#ifdef __AVX512F__
template <long ldb, long N_GROUP_SIZE, bool is_int4>
struct dequant_n_grouped<bfloat16, ldb, N_GROUP_SIZE, is_int4> {
  template <typename Lambda1, typename Lambda2, typename Lambda3>
  static inline void call(
      uint8_t* qB,
      long K,
      long N,
      bfloat16* scales,
      bfloat16* zps,
      bfloat16* B,
      const Lambda1& load_qparam,
      const Lambda2& load_qint_as_fp,
      const Lambda3& store) {
#define ADDRESS(p, x, y, ld) ((p) + (x) * (ld) + (y))

    using VA = VecArray<N_GROUP_SIZE, float>;
    constexpr long COLS = VA::num_vec;

    for (int n = 0; n < N; n += N_GROUP_SIZE) {
      // load scales and zps
      auto vscales = load_qparam(scales + n);
      auto vzps = load_qparam(zps + n);
      // convert to vnni: [K/2, N, 2]
      for (int k = 0; k < K; k += 2) {
        auto interleave = [](__m512 v0, __m512 v1) {
          __m512i idx_low = _mm512_set_epi32(
              0x17,
              0x07,
              0x16,
              0x06,
              0x15,
              0x05,
              0x14,
              0x04,
              0x13,
              0x03,
              0x12,
              0x02,
              0x11,
              0x01,
              0x10,
              0x00);
          __m512i idx_high = _mm512_set_epi32(
              0x1f,
              0x0f,
              0x1e,
              0x0e,
              0x1d,
              0x0d,
              0x1c,
              0x0c,
              0x1b,
              0x0b,
              0x1a,
              0x0a,
              0x19,
              0x09,
              0x18,
              0x08);
          return std::array<__m512, 2>(
              {_mm512_permutex2var_ps(v0, idx_low, v1),
               _mm512_permutex2var_ps(v0, idx_high, v1)});
        };
        // load and dequant qB to vb
        auto vbs_k0 = load_qint_as_fp(
            is_int4 ? &qB[k * ldb / 2 + n / 2] : &qB[k * ldb + n],
            vscales,
            vzps);
        auto vbs_k1 = load_qint_as_fp(
            is_int4 ? &qB[(k + 1) * ldb / 2 + n / 2] : &qB[(k + 1) * ldb + n],
            vscales,
            vzps);
        typename VA::type vbs[2];
        compile_time_for<COLS>::op([&](auto i) {
          auto [low, high] = interleave(vbs_k0[i], vbs_k1[i]);
          vbs[i * 2 / COLS][i * 2 % COLS] = low;
          vbs[(i * 2 + 1) / COLS][(i * 2 + 1) % COLS] = high;
        });
        // store vb to B: low: [k + n*2 / N, n*2 % N], high: [k +
        // (n*2+N_GROUP_SIZE) / N, (n*2+N_GROUP_SIZE) % N]
        store(ADDRESS(B, k + (n * 2) / N, (n * 2) % N, N), vbs[0]);
        store(
            ADDRESS(
                B,
                k + (n * 2 + N_GROUP_SIZE) / N,
                (n * 2 + N_GROUP_SIZE) % N,
                N),
            vbs[1]);
      }
    }
  }
};
#endif

template <typename Tin, long ldb, long N_GROUP_SIZE, bool is_int4>
struct Dequantize {
  static void call(uint8_t* qB, long K, long N, Tin* scales, Tin* zps, Tin* B);
};

template <long ldb, long N_GROUP_SIZE, bool is_int4>
struct Dequantize<float, ldb, N_GROUP_SIZE, is_int4> {
  static inline void call(
      uint8_t* qB,
      long K,
      long N,
      float* scales,
      float* zps,
      float* B) {
#if defined(__AVX512F__)
    using T = float;
    using VA = VecArray<N_GROUP_SIZE, T>;
    constexpr int VLEN = VA::vec_ops::VLEN;
    constexpr long COLS = VA::num_vec;

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
    dequant_n_grouped<float, ldb, N_GROUP_SIZE, is_int4>::call(
        qB,
        K,
        N,
        scales,
        zps,
        B,
        [&](float* p) { return VA::load1d(p); },
        [&](uint8_t* p, auto vscales, auto vzps) {
          if constexpr (is_int4) {
            return load_dequant_int4<N_GROUP_SIZE, T>::call(
                p, vscales, vzps, lut);
          } else {
            return load_dequant_int8<N_GROUP_SIZE, T>::call(p, vscales, vzps);
          }
        },
        [&](auto p, auto vbs) {
          compile_time_for<COLS>::op(
              [&](auto idx) { _mm512_storeu_ps(p + idx * VLEN, vbs[idx]); });
        });
#else
    TLA_ASSERT(false, "not implemented");
#endif
  }
};

template <long ldb, long N_GROUP_SIZE, bool is_int4>
struct Dequantize<bfloat16, ldb, N_GROUP_SIZE, is_int4> {
  static inline void call(
      uint8_t* qB,
      long K,
      long N,
      bfloat16* scales,
      bfloat16* zps,
      bfloat16* B) {
#ifdef __AVX512F__
    using T = bfloat16;
    using VA = VecArray<N_GROUP_SIZE, T>;
    constexpr long COLS = VA::num_vec;

    // lookup table converting uint8 to float, 15.0f - 0.0f
    // _mm512_permutexvar_ph needs 5 bits while we only need 4 bits, init the
    // table to honor the lower 4 bits regardless of the the highest bit, thus
    // saving an "and" op
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
    dequant_n_grouped<bfloat16, ldb, N_GROUP_SIZE, is_int4>::call(
        qB,
        K,
        N,
        scales,
        zps,
        B,
        [&](bfloat16* p) { return VA::load1d(p); },
        [&](uint8_t* p, auto vscales, auto vzps) {
          if constexpr (is_int4) {
            return load_dequant_int4<N_GROUP_SIZE, float>::call(
                p, vscales, vzps, lut);
          } else {
            return load_dequant_int8<N_GROUP_SIZE, float>::call(
                p, vscales, vzps);
          }
        },
        [&](auto p, auto vbs) {
          compile_time_for<COLS / 2>::op([&](auto idx) {
            _vec_store_two_floats_as_bfloat16(
                p + idx * 32, vbs[idx * 2], vbs[idx * 2 + 1]);
          });
        });
#else
    TLA_ASSERT(false, "not implemented");
#endif
  }
};

template <long ldb, long N_GROUP_SIZE, bool is_int4>
struct Dequantize<half, ldb, N_GROUP_SIZE, is_int4> {
  static inline void call(
      uint8_t* qB,
      long K,
      long N,
      half* scales,
      half* zps,
      half* B) {
#ifdef __AVX512FP16__
    using T = half;
    using VA = VecArray<N_GROUP_SIZE, T>;
    constexpr int VLEN = VA::vec_ops::VLEN;
    constexpr long COLS = VA::num_vec;

    // lookup table converting uint8 to float, 15.0f - 0.0f
    // _mm512_permutexvar_ph needs 5 bits while we only need 4 bits, init the
    // table to honor the lower 4 bits regardless of the the highest bit, thus
    // saving an "and" op
    __m512h lut = _mm512_set_ph(
        15.0,
        14.0,
        13.0,
        12.0,
        11.0,
        10.0,
        9.0,
        8.0,
        7.0,
        6.0,
        5.0,
        4.0,
        3.0,
        2.0,
        1.0,
        0.0,
        15.0,
        14.0,
        13.0,
        12.0,
        11.0,
        10.0,
        9.0,
        8.0,
        7.0,
        6.0,
        5.0,
        4.0,
        3.0,
        2.0,
        1.0,
        0.0);
    dequant_n_grouped<half, ldb, N_GROUP_SIZE, is_int4>::call(
        qB,
        K,
        N,
        scales,
        zps,
        B,
        [&](half* p) { return VA::load1d(p); },
        [&](uint8_t* p, auto vscales, auto vzps) {
          if constexpr (is_int4) {
            return load_dequant_int4<N_GROUP_SIZE, T>::call(
                p, vscales, vzps, lut);
          } else {
            return load_dequant_int8<N_GROUP_SIZE, T>::call(p, vscales, vzps);
          }
        },
        [&](auto p, auto vbs) {
          compile_time_for<COLS>::op(
              [&](auto idx) { _mm512_storeu_ph(p + idx * VLEN, vbs[idx]); });
        });
#else
    TLA_ASSERT(false, "not implemented");
#endif
  }
};

template <long ldb>
struct Dequantize<int8_t, ldb, /*N_GROUP_SIZE*/ 16, /*is_int4*/ true> {
  static inline void call(
      uint8_t* qB,
      long K,
      long N,
      int8_t* zps,
      int8_t* B,
      int32_t* compensation) {
#ifdef __AVX512VNNI__
    auto pqB = GetVLAPtr<uint8_t>(qB, {ldb, 2}); // [K/4,N,4] packed in 4-bit
    auto pB = GetVLAPtr<int8_t>(B, {ldb, 4}); // [K/4,N,4]
    __m256i ones = _mm256_set1_epi8(1);
    for (int n = 0; n < N; n += 16) {
      auto [vzps_low, vzps_high] = load_zps_4vnni(&zps[n]);
      __m256i vcompensate[2];
      vcompensate[0] = _mm256_setzero_si256();
      vcompensate[1] = _mm256_setzero_si256();
      // TODO(jgong5): unroll k?
      for (int k = 0; k < K / 4; k++) {
        // TODO(jgong5): consider optimize the instruction sequence below, e.g,
        // use avx512? load 64 (N:16, K:4) int4 values from qB
        auto [low, high] = load_int4_as_int8(pqB[k][n]);
        high = _mm256_sub_epi8(high, vzps_high);
        low = _mm256_sub_epi8(low, vzps_low);
        vcompensate[0] = _mm256_dpbusd_epi32(vcompensate[0], ones, low);
        vcompensate[1] = _mm256_dpbusd_epi32(vcompensate[1], ones, high);
        // store vb to B
        _mm256_storeu_si256(reinterpret_cast<__m256i_u*>(pB[k][n]), low);
        _mm256_storeu_si256(reinterpret_cast<__m256i_u*>(pB[k][n + 8]), high);
      }
      _mm256_storeu_si256(
          reinterpret_cast<__m256i_u*>(&compensation[n]), vcompensate[0]);
      _mm256_storeu_si256(
          reinterpret_cast<__m256i_u*>(&compensation[n + 8]), vcompensate[1]);
    }
#else
    TLA_ASSERT(false, "not implemented");
#endif
  }
};

// TODO(jgong5): move to tpp.h
template <
    typename Tin,
    typename Tout,
    typename TScale,
    typename TZero,
    long BLOCK_M,
    long N,
    long ldb,
    bool transA,
    bool ACC,
    bool is_int4,
    long PREFETCH_K_DIST = 0>
class DequantGemmTPP {
 public:
  DequantGemmTPP(long M, long K, long lda, long ldc) {
    TLA_ASSERT(false, "not implemented");
  }

  void operator()(
      Tin* A,
      uint8_t* qB,
      TScale* scales,
      TZero* zps,
      Tout* C,
      bool no_tile_cfg = true,
      float scale_a = 1.0,
      int32_t zp_a = 0) {
    TLA_ASSERT(false, "not implemented");
  }

  void config() {
    TLA_ASSERT(false, "not implemented");
  }

  void release() {
    TLA_ASSERT(false, "not implemented");
  }
};

template <
    typename Tin,
    typename Tout,
    long BLOCK_M,
    long N,
    long ldb,
    bool transA,
    bool ACC,
    bool is_int4,
    long PREFETCH_K_DIST>
class DequantGemmTPP<
    Tin,
    Tout,
    Tin,
    Tin,
    BLOCK_M,
    N,
    ldb,
    transA,
    ACC,
    is_int4,
    PREFETCH_K_DIST> {
 public:
  DequantGemmTPP(long M, long K, long lda, long ldc)
      : M(M), K(K), lda(lda), ldc(ldc) {
    static_assert(N % 16 == 0, "N must be a multiple of 16");
    if (std::is_same<Tin, bfloat16>())
      TLA_ASSERT(K % 2 == 0, "Kb must be a multiple of 2 for bfloat16");
    pgemm = std::make_shared<BrgemmTPP<Tin, Tout>>(
        M,
        N,
        K,
        1,
        1,
        lda,
        ldb,
        ldc,
        ACC ? 1 : 0,
        transA,
        1,
        /*b_vnni*/ std::is_same<Tin, bfloat16>());
  }

  inline void operator()(
      Tin* A,
      uint8_t* qB,
      Tin* scales,
      Tin* zps,
      Tout* C,
      bool no_tile_cfg = true,
      float scale_a = 1.0,
      int32_t zp_a = 0) {
    if (M < SMALL_BATCH_THRESHOLD &&
        ((std::is_same<Tin, half>() && std::is_same<Tout, half>()) ||
         (std::is_same<Tin, float>() && std::is_same<Tout, float>()))) {
      for (long m = 0; m < M; m += BLOCK_M) {
        long block_m = std::min(M - m, BLOCK_M);
        enumerate_dispatcher<long, 4, BLOCK_M>::call(
            block_m,
            [&](auto i) {
              GemmMicroKernel<
                  Tin,
                  Tin,
                  Tin,
                  Tin,
                  i,
                  N,
                  ldb,
                  transA,
                  ACC,
                  PREFETCH_K_DIST>::
                  template call<is_int4>(
                      K,
                      transA ? (Tin*)A + m : (Tin*)A + m * lda,
                      lda,
                      qB,
                      (Tin*)C + m * ldc,
                      ldc,
                      scales,
                      zps);
            },
            [&](auto i) {
              range_dispatcher<long, 1, BLOCK_M - 1>::call(
                  i,
                  [&](auto j) {
                    GemmMicroKernel<
                        Tin,
                        Tin,
                        Tin,
                        Tin,
                        j,
                        N,
                        ldb,
                        transA,
                        ACC,
                        PREFETCH_K_DIST>::
                        template call<is_int4>(
                            K,
                            transA ? (Tin*)A + m : (Tin*)A + m * lda,
                            lda,
                            qB,
                            (Tin*)C + m * ldc,
                            ldc,
                            scales,
                            zps);
                  },
                  [&](auto j) { failing_fallback(); });
            });
      }
    } else {
      constexpr const int N_GROUP_SIZE = get_n_group_size(N);
      Tin B[K][N];
      // TODO(jgong5): add prefetch
      Dequantize<Tin, ldb, N_GROUP_SIZE, is_int4>::call(
          qB, K, N, scales, zps, B[0]);
      (*pgemm)(A, B[0], C, 1, no_tile_cfg);
    }
  }

  void config() {
    if (pgemm) {
      pgemm->config();
    }
  }

  void release() {
    if (pgemm) {
      pgemm->release();
    }
  }

 private:
  std::shared_ptr<BrgemmTPP<Tin, Tout>> pgemm;
  long M;
  long K;
  long lda;
  long ldc;
};

template <
    long BLOCK_M,
    long N,
    long ldb,
    bool transA,
    bool ACC,
    long PREFETCH_K_DIST>
class DequantGemmTPP<
    /*Tin*/ uint8_t,
    /*Tout*/ float,
    /*TScale*/ float,
    /*TZero*/ int8_t,
    BLOCK_M,
    N,
    ldb,
    transA,
    ACC,
    /*is_int4*/ true,
    PREFETCH_K_DIST> {
  using TBrgemmTPP = BrgemmTPP<int8_t, int32_t>;

 public:
  DequantGemmTPP(long M, long K, long lda, long ldc)
      : M(M), K(K), lda(lda), ldc(ldc) {
    static_assert(N % 16 == 0, "N must be a multiple of 16");
    TLA_ASSERT(K % 4 == 0, "Kb must be a multiple of 4 for int8 VNNI");
    // TODO(jgong5): output fp32 directly
    pgemm = std::make_shared<TBrgemmTPP>(
        M,
        N,
        K,
        1,
        1,
        lda,
        N,
        N,
        /*ACC*/ 0,
        /*transA*/ false,
        1,
        /*b_vnni*/ true);
  }

  inline void operator()(
      uint8_t* A,
      uint8_t* qB,
      float* scales,
      int8_t* zps,
      float* C,
      bool no_tile_cfg = true,
      float scale_a = 1.0,
      int32_t zp_a = 0) {
    auto qA = GetVLAPtr<uint8_t>(A, {lda});
#ifdef __AVX512VNNI__
    if (M < SMALL_BATCH_THRESHOLD) {
      constexpr long PREFERRED_BLOCK_M =
          BLOCK_M * N / 16 >= 16 ? BLOCK_M / 2 : BLOCK_M;
      for (long m = 0; m < M; m += PREFERRED_BLOCK_M) {
        long block_m = std::min(M - m, PREFERRED_BLOCK_M);
        enumerate_dispatcher<long, 4, PREFERRED_BLOCK_M>::call(
            block_m,
            [&](auto i) {
              GemmMicroKernel<
                  /*Tin*/ uint8_t,
                  /*Tout*/ float,
                  /*TScale*/ float,
                  /*TZero*/ int8_t,
                  /*M*/ i,
                  N,
                  ldb,
                  /*transA*/ false,
                  ACC,
                  PREFETCH_K_DIST>::
                  template call<true>(
                      K,
                      qA[m],
                      lda,
                      qB,
                      C + m * ldc,
                      ldc,
                      scales,
                      zps,
                      scale_a,
                      zp_a);
            },
            [&](auto i) {
              range_dispatcher<long, 1, PREFERRED_BLOCK_M - 1>::call(
                  i,
                  [&](auto j) {
                    GemmMicroKernel<
                        /*Tin*/ uint8_t,
                        /*Tout*/ float,
                        /*TScale*/ float,
                        /*TZero*/ int8_t,
                        /*M*/ j,
                        N,
                        ldb,
                        /*transA*/ false,
                        ACC,
                        PREFETCH_K_DIST>::
                        template call<true>(
                            K,
                            qA[m],
                            lda,
                            qB,
                            C + m * ldc,
                            ldc,
                            scales,
                            zps,
                            scale_a,
                            zp_a);
                  },
                  [&](auto j) { failing_fallback(); });
            });
      }
    } else
#endif
    {
      constexpr const int N_GROUP_SIZE = 16;
      int8_t B[K / 4][N][4];
      int32_t qC[M][N];
      int32_t compensation[N];
      // TODO(jgong5): add prefetch
      Dequantize<int8_t, ldb, N_GROUP_SIZE, /*is_int4*/ true>::call(
          qB, K, N, zps, B[0][0], compensation);
      (*pgemm)((int8_t*)qA[0], B[0][0], qC[0], 1, no_tile_cfg);
      // post-op and convert back to C
      for (long m = 0; m < M; ++m) {
#pragma omp simd
        for (long n = 0; n < N; ++n) {
          float c = (qC[m][n] - compensation[n] * zp_a) * scale_a * scales[n];
          if constexpr (ACC) {
            C[m * ldc + n] += c;
          } else {
            C[m * ldc + n] = c;
          }
        }
      }
    }
  }

  void config() {
    if (pgemm) {
      pgemm->config();
    }
  }

  void release() {
    if (pgemm) {
      pgemm->release();
    }
  }

 private:
  std::shared_ptr<TBrgemmTPP> pgemm;
  long M;
  long K;
  long lda;
  long ldc;
};

#define FUSE_GELU 1
#define FUSE_ADD 2
#define FUSE_ADD_ADD 3

// If T != TComp
//   T -> TComp -> GEMM -> TComp -> bias/PostOp -> Tout
// If T == TComp (we can save intermediate output buffer and schedule M/N/K
// loops together)
//   T -> GEMM -> T -> bias/PostOp -> Tout
template <
    typename T,
    typename TComp,
    typename TGemmOut,
    typename Tout,
    typename TScale,
    typename TZero>
void qlinear_woq_affine_impl(
    const at::Tensor& x,
    const at::Tensor& qw_packed,
    const at::Tensor& scales, // dtype is TComp
    const at::Tensor& zps, // dtype is TComp
    const at::Tensor& b, // dtype is TComp
    at::Tensor y,
    bool is_int4,
    int k_splits,
    int num_concats,
    int fusion_type,
    const TensorList& others_list,
    float scale_a = 1.0f,
    int32_t zp_a = 0) {
  auto x_sizes = x.sizes();
  auto w_sizes = qw_packed.sizes();
  auto M = x_sizes[0];
  auto Nc = w_sizes[0];
  auto Nb = is_int4 ? w_sizes[3] * 2 : w_sizes[3];
  auto Kc = w_sizes[1];
  auto Kb = w_sizes[2];
  auto N = Nc * Nb;
  auto K = Kc * Kb;

  TLA_ASSERT(Nb % 16 == 0, "Nb must be a multiple of 16");
  TLA_ASSERT(
      num_concats <= 1 || Nc % num_concats == 0,
      "Nc must be a multiple of num_concats");

  // select BLOCK_M according to M
  // TODO(jgong5): improve the heuristic
  auto BLOCK_M = [&]() -> long {
    if (M < 32) {
      return M;
    } else if (M < 64) {
      return 32;
    } else {
      return 64;
    }
  }();

  auto BLOCK_M_rem = M % BLOCK_M;

  // TODO(jgong5): use heuristics to decide k_splits
  if (k_splits <= 0 || num_concats > 1 || M >= 32 || BLOCK_M_rem) {
    k_splits = 1;
  }
  TLA_ASSERT(Kc % k_splits == 0, "Kc must be a multiple of k_splits");
  TLA_ASSERT(
      !(std::is_same<T, uint8_t>()) || (std::is_same<T, TComp>()),
      "T must be TComp if T is uint8_t");

  bool no_x_buf = std::is_same<T, TComp>();
  bool no_y_buf = std::is_same<T, TComp>() && std::is_same<Tout, TGemmOut>() &&
      k_splits == 1;

  auto lda = no_x_buf ? K : Kb;
  auto ldy = num_concats <= 1 ? N : Nc / num_concats * Nb;
  auto ldc = (no_y_buf || k_splits > 1) ? ldy : Nb;

  auto px = GetVLAPtr<T>(x, {Kc, Kb});
  auto pw = GetVLAPtr<uint8_t>(
      (uint8_t*)qw_packed.data_ptr(), {Kc, Kb * (is_int4 ? Nb / 2 : Nb)});
  auto py = GetVLAPtr<Tout>(y, {Nc, Nb}); /*[M, Nc, Nb]*/
  auto py_concat = GetVLAPtr<Tout>(
      y, {M, Nc / num_concats, Nb}); /*[num_concats, M, Nc/num_concats, Nb]*/
  auto pscales = GetVLAPtr<TScale>(scales, {Nb});
  auto pzps = GetVLAPtr<TZero>(zps, {Nb});
  auto pb = GetVLAPtr<TGemmOut>(b, {Nb});
  auto tin0 = others_list.size() > 0 ? others_list[0] : at::Tensor{};
  auto pin0 = GetVLAPtr<Tout>(tin0, {Nc, Nb}); /*[M, Nc, Nb]*/
  auto pin0_concat = GetVLAPtr<Tout>(
      tin0, {M, Nc / num_concats, Nb}); /*[num_concats, M, Nc/num_concats, Nb]*/
  auto tin1 = others_list.size() > 1 ? others_list[1] : at::Tensor{};
  auto pin1 = GetVLAPtr<Tout>(tin1, {Nc, Nb}); /*[M, Nc, Nb]*/
  auto pin1_concat = GetVLAPtr<Tout>(
      tin1, {M, Nc / num_concats, Nb}); /*[num_concats, M, Nc/num_concats, Nb]*/

  auto copy_bias_out_tpp = CpyBiasTPP<TGemmOut>(BLOCK_M, Nb, ldy);
  auto copy_bias_buf_tpp = CpyBiasTPP<TGemmOut>(BLOCK_M, Nb, Nb);
  auto copy_bias_out_rem_tpp = CpyBiasTPP<TGemmOut>(BLOCK_M_rem, Nb, ldy);
  auto copy_bias_buf_rem_tpp = CpyBiasTPP<TGemmOut>(BLOCK_M_rem, Nb, Nb);
  auto zero_out_tpp = SetZeroTPP<TGemmOut>(BLOCK_M, Nb, ldy);
  auto zero_buf_tpp = SetZeroTPP<TGemmOut>(BLOCK_M, Nb, Nb);
  auto zero_out_rem_tpp = SetZeroTPP<TGemmOut>(BLOCK_M_rem, Nb, ldy);
  auto zero_buf_rem_tpp = SetZeroTPP<TGemmOut>(BLOCK_M_rem, Nb, Nb);
  auto gelu_fwd_tpp = GeluFwdTPP<Tout>(BLOCK_M, Nb, ldy, ldy);
  auto gelu_fwd_rem_tpp = GeluFwdTPP<Tout>(BLOCK_M_rem, Nb, ldy, ldy);
  auto add_tpp = AddTPP<Tout>(BLOCK_M, Nb, ldy, ldy);
  auto add_rem_tpp = AddTPP<Tout>(BLOCK_M_rem, Nb, ldy, ldy);
  auto post_ops_fn = [&](int m, int nc) {
    Tout* y_ptr = num_concats <= 1
        ? (Tout*)py[m][nc]
        : (Tout*)py_concat[nc / (Nc / num_concats)][m][nc % (Nc / num_concats)];
    Tout* tin0_ptr = fusion_type > 1 ? num_concats <= 1
            ? (Tout*)pin0[m][nc]
            : (Tout*)pin0_concat[nc / (Nc / num_concats)][m]
                                [nc % (Nc / num_concats)]
                                     : nullptr;
    Tout* tin1_ptr = fusion_type > 2 ? num_concats <= 1
            ? (Tout*)pin1[m][nc]
            : (Tout*)pin1_concat[nc / (Nc / num_concats)][m]
                                [nc % (Nc / num_concats)]
                                     : nullptr;
    if (fusion_type == FUSE_GELU) {
      gelu_fwd_tpp(y_ptr, y_ptr);
    } else if (fusion_type == FUSE_ADD) {
      add_tpp(y_ptr, tin0_ptr, y_ptr);
    } else if (fusion_type == FUSE_ADD_ADD) {
      add_tpp(y_ptr, tin0_ptr, y_ptr);
      add_tpp(y_ptr, tin1_ptr, y_ptr);
    }
  };
  auto post_ops_rem_fn = [&](int m, int nc) {
    Tout* y_ptr = num_concats <= 1
        ? (Tout*)py[m][nc]
        : (Tout*)py_concat[nc / (Nc / num_concats)][m][nc % (Nc / num_concats)];
    Tout* tin0_ptr = fusion_type > 1 ? num_concats <= 1
            ? (Tout*)pin0[m][nc]
            : (Tout*)pin0_concat[nc / (Nc / num_concats)][m]
                                [nc % (Nc / num_concats)]
                                     : nullptr;
    Tout* tin1_ptr = fusion_type > 2 ? num_concats <= 1
            ? (Tout*)pin1[m][nc]
            : (Tout*)pin1_concat[nc / (Nc / num_concats)][m]
                                [nc % (Nc / num_concats)]
                                     : nullptr;
    if (fusion_type == FUSE_GELU) {
      gelu_fwd_rem_tpp(y_ptr, y_ptr);
    } else if (fusion_type == FUSE_ADD) {
      add_rem_tpp(y_ptr, tin0_ptr, y_ptr);
    } else if (fusion_type == FUSE_ADD_ADD) {
      add_rem_tpp(y_ptr, tin0_ptr, y_ptr);
      add_rem_tpp(y_ptr, tin1_ptr, y_ptr);
    }
  };

  constexpr long MICRO_BLOCK_M = 8;
  product_dispatcher<
      std::tuple</*BLOCK_N*/ long, /*is_int4*/ bool>,
      std::tuple<
          enumerate_dispatcher<long, 16, 32, 64, 128>,
          boolean_dispatcher>>::
      call(
          std::make_tuple(Nb, is_int4),
          [&](auto tuple) {
            auto BLOCK_N = std::get<0>(tuple);
            auto is_int4 = std::get<1>(tuple);
            // TODO(jgong5): design API to avoid duplicate code of defining
            // similar kernel object
            auto dequant_gemm_tpp = DequantGemmTPP<
                TComp,
                TGemmOut,
                TScale,
                TZero,
                MICRO_BLOCK_M,
                BLOCK_N,
                /*ldb*/ BLOCK_N,
                /*transA*/ false,
                /*ACC*/ true,
                is_int4,
                PREFETCH_K_DIST>(
                /*M*/ BLOCK_M,
                /*K*/ Kb,
                /*lda*/ lda,
                /*ldc*/ ldc);
            auto dequant_gemm_no_prefetch_tpp = DequantGemmTPP<
                TComp,
                TGemmOut,
                TScale,
                TZero,
                MICRO_BLOCK_M,
                BLOCK_N,
                /*ldb*/ BLOCK_N,
                /*transA*/ false,
                /*ACC*/ true,
                is_int4,
                0>(
                /*M*/ BLOCK_M,
                /*K*/ Kb,
                /*lda*/ lda,
                /*ldc*/ ldc);
            auto dequant_gemm_rem_tpp = DequantGemmTPP<
                TComp,
                TGemmOut,
                TScale,
                TZero,
                MICRO_BLOCK_M,
                BLOCK_N,
                /*ldb*/ BLOCK_N,
                /*transA*/ false,
                /*ACC*/ true,
                is_int4,
                PREFETCH_K_DIST>(
                /*M*/ BLOCK_M_rem,
                /*K*/ Kb,
                /*lda*/ lda,
                /*ldc*/ ldc);
            auto dequant_gemm_no_prefetch_rem_tpp = DequantGemmTPP<
                TComp,
                TGemmOut,
                TScale,
                TZero,
                MICRO_BLOCK_M,
                BLOCK_N,
                /*ldb*/ BLOCK_N,
                /*transA*/ false,
                /*ACC*/ true,
                is_int4,
                0>(
                /*M*/ BLOCK_M_rem,
                /*K*/ Kb,
                /*lda*/ lda,
                /*ldc*/ ldc);

            auto pcvt_x_tpp = std::is_same<T, uint8_t>()
                ? nullptr
                : std::make_shared<ConvertTPP<T, TComp>>(BLOCK_M, Kb, K, Kb);
            auto pcvt_x_rem_tpp = std::is_same<T, uint8_t>()
                ? nullptr
                : std::make_shared<ConvertTPP<T, TComp>>(
                      BLOCK_M_rem, Kb, K, Kb);
            auto cvt_y_tpp = ConvertTPP<TGemmOut, Tout>(BLOCK_M, Nb, Nb, ldy);
            auto cvt_y_rem_tpp =
                ConvertTPP<TGemmOut, Tout>(BLOCK_M_rem, Nb, Nb, ldy);
            auto cvt_y_private_tpp =
                ConvertTPP<TGemmOut, Tout>(BLOCK_M, Nb, N, N);
            auto add_y_tpp = BinaryTPP(
                BLOCK_M, /*row*/
                Nb, /*col*/
                N, /*ldi0*/
                N, /*ldi1*/
                N, /*ldo*/
                XsmmDtype<TGemmOut>(), /*dt_in0*/
                XsmmDtype<Tout>(), /*dt_in1*/
                XsmmDtype<Tout>(), /*dt_out*/
                XsmmDtype<float>(), /*dt_compute*/
                LIBXSMM_MELTW_FLAG_BINARY_NONE,
                LIBXSMM_MELTW_TYPE_BINARY_ADD);

            // TODO(jgong5): parallelize over M on large BS
            if (no_y_buf) {
              auto loop_scheme = M >= PARALLEL_M_THRESHOLD ? "ACb" : "aCb";
              auto gemm_loop = ThreadedLoop<3>(
                  {{0, M, BLOCK_M, false}, {Kc}, {Nc}}, loop_scheme);
              gemm_loop(
                  [&](int* idx) {
                    int m = idx[0];
                    int kc = idx[1];
                    int nc = idx[2];
                    bool is_rem = (m + BLOCK_M > M);
                    TGemmOut* y_ptr = num_concats <= 1
                        ? (TGemmOut*)py[m][nc]
                        : (TGemmOut*)py_concat[nc / (Nc / num_concats)][m]
                                              [nc % (Nc / num_concats)];
                    if (!is_rem) {
                      if (kc == 0) {
                        if (b.defined()) {
                          copy_bias_out_tpp(pb[nc], y_ptr);
                        } else {
                          zero_out_tpp(y_ptr);
                        }
                      }
                      TComp* x_ptr = (TComp*)px[m][kc];
                      if (kc < Kc - 1) {
                        dequant_gemm_tpp(
                            x_ptr,
                            pw[nc][kc],
                            pscales[nc],
                            pzps[nc],
                            y_ptr,
                            true,
                            scale_a,
                            zp_a);
                      } else {
                        dequant_gemm_no_prefetch_tpp(
                            x_ptr,
                            pw[nc][kc],
                            pscales[nc],
                            pzps[nc],
                            y_ptr,
                            true,
                            scale_a,
                            zp_a);
                        if (fusion_type > 0) {
                          post_ops_fn(m, nc);
                        }
                      }
                    } else {
                      if (kc == 0) {
                        if (b.defined()) {
                          copy_bias_out_rem_tpp(pb[nc], y_ptr);
                        } else {
                          zero_out_rem_tpp(y_ptr);
                        }
                      }
                      TComp* x_ptr = (TComp*)px[m][kc];
                      if (kc < Kc - 1) {
                        dequant_gemm_rem_tpp(
                            x_ptr,
                            pw[nc][kc],
                            pscales[nc],
                            pzps[nc],
                            y_ptr,
                            false,
                            scale_a,
                            zp_a);
                        dequant_gemm_tpp.config();
                      } else {
                        dequant_gemm_no_prefetch_rem_tpp(
                            x_ptr,
                            pw[nc][kc],
                            pscales[nc],
                            pzps[nc],
                            y_ptr,
                            false,
                            scale_a,
                            zp_a);
                        dequant_gemm_no_prefetch_tpp.config();
                        if (fusion_type > 0) {
                          post_ops_rem_fn(m, nc);
                        }
                      }
                    }
                    // TODO(jgong5): post-op fusion
                  },
                  [&]() { dequant_gemm_tpp.config(); },
                  [&]() { dequant_gemm_tpp.release(); });
            } else {
              auto num_threads = omp_get_max_threads();
              TGemmOut* y_private = nullptr;
              bool* y_private_valid = nullptr;
              if (k_splits > 1) {
                // TODO(jgong5): if we know the thread decomposition, we can
                // allocate a smaller buffer
                y_private = (TGemmOut*)std::aligned_alloc(
                    64, num_threads * M * N * sizeof(TGemmOut));
                y_private_valid = (bool*)std::aligned_alloc(
                    64, num_threads * (M / BLOCK_M) * Nc * sizeof(bool));
                memset(
                    y_private_valid,
                    0,
                    sizeof(bool) * num_threads * (M / BLOCK_M) * Nc);
              }
              auto y_private_ptr = GetVLAPtr<TGemmOut>(y_private, {M, Nc, Nb});
              auto y_private_valid_ptr =
                  GetVLAPtr<bool>(y_private_valid, {M / BLOCK_M, Nc});
              auto loop_scheme = M >= PARALLEL_M_THRESHOLD ? "CAB" : "ABc";
              auto gemm_loop = ThreadedLoop<3>(
                  {{Nc}, {0, Kc, Kc / k_splits, true}, {0, M, BLOCK_M, false}},
                  loop_scheme);
              gemm_loop(
                  [&](int* idx) {
                    int my_id = omp_get_thread_num();
                    int nc = idx[0];
                    int kc_start = idx[1];
                    int kc_end = kc_start + Kc / k_splits;
                    int m = idx[2];
                    bool is_rem = (m + BLOCK_M > M);
                    auto y_out_ptr = num_concats <= 1
                        ? py[m][nc]
                        : py_concat[nc / (Nc / num_concats)][m]
                                   [nc % (Nc / num_concats)];
                    alignas(64) TGemmOut y_buf[BLOCK_M][Nb];
                    TGemmOut* y_ptr = y_private_ptr[my_id][m][nc];
                    if (k_splits > 1) {
                      if (!y_private_valid_ptr[my_id][m / BLOCK_M][nc]) {
                        if (kc_start == 0 && b.defined()) {
                          copy_bias_out_tpp(pb[nc], y_ptr);
                        } else {
                          zero_out_tpp(y_ptr);
                        }
                        y_private_valid_ptr[my_id][m / BLOCK_M][nc] = true;
                      }
                    } else {
                      y_ptr = y_buf[0];
                      if (b.defined()) {
                        if (!is_rem) {
                          copy_bias_buf_tpp(pb[nc], y_buf[0]);
                        } else {
                          copy_bias_buf_rem_tpp(pb[nc], y_buf[0]);
                        }
                      } else {
                        if (!is_rem) {
                          zero_buf_tpp(y_buf[0]);
                        } else {
                          zero_buf_rem_tpp(y_buf[0]);
                        }
                      }
                    }
                    for (int kc = kc_start; kc < kc_end; kc++) {
                      TComp* x_ptr = (TComp*)px[m][kc];
                      if (!is_rem) {
                        alignas(64) TComp x_buf[BLOCK_M][Kb];
                        if (!no_x_buf) {
                          (*pcvt_x_tpp)(px[m][kc], x_buf[0]);
                          x_ptr = x_buf[0];
                        }
                        if (kc < Kc - 1) {
                          dequant_gemm_tpp(
                              x_ptr,
                              pw[nc][kc],
                              pscales[nc],
                              pzps[nc],
                              y_ptr,
                              true,
                              scale_a,
                              zp_a);
                        } else {
                          dequant_gemm_no_prefetch_tpp(
                              x_ptr,
                              pw[nc][kc],
                              pscales[nc],
                              pzps[nc],
                              y_ptr,
                              true,
                              scale_a,
                              zp_a);
                        }
                      } else {
                        alignas(64) TComp x_buf[BLOCK_M][Kb];
                        if (!no_x_buf) {
                          (*pcvt_x_rem_tpp)(px[m][kc], x_buf[0]);
                          x_ptr = x_buf[0];
                        }
                        if (kc < Kc - 1) {
                          dequant_gemm_rem_tpp(
                              x_ptr,
                              pw[nc][kc],
                              pscales[nc],
                              pzps[nc],
                              y_ptr,
                              false,
                              scale_a,
                              zp_a);
                          dequant_gemm_tpp.config();
                        } else {
                          dequant_gemm_no_prefetch_rem_tpp(
                              x_ptr,
                              pw[nc][kc],
                              pscales[nc],
                              pzps[nc],
                              y_ptr,
                              false,
                              scale_a,
                              zp_a);
                          dequant_gemm_no_prefetch_tpp.config();
                        }
                      }
                    }
                    // TODO(jgong5): post-op fusion
                    if (k_splits <= 1) {
                      if (!is_rem) {
                        cvt_y_tpp(y_buf[0], y_out_ptr);
                        if (fusion_type > 0) {
                          post_ops_fn(m, nc);
                        }
                      } else {
                        cvt_y_rem_tpp(y_buf[0], y_out_ptr);
                        if (fusion_type > 0) {
                          post_ops_rem_fn(m, nc);
                        }
                      }
                    }
                  },
                  [&]() { dequant_gemm_tpp.config(); },
                  [&]() { dequant_gemm_tpp.release(); });
              if (k_splits > 1) {
                TLA_ASSERT(
                    M % BLOCK_M == 0,
                    "M must be divisible by BLOCK_M for k_splits > 1");
                auto reduce_loop =
                    ThreadedLoop<2>({{0, M, BLOCK_M, true}, {Nc}}, "AB");
                reduce_loop([&](int* idx) {
                  int m = idx[0];
                  int nc = idx[1];
                  bool init = false;
                  for (int id = 0; id < num_threads; id++) {
                    if (y_private_valid_ptr[id][m / BLOCK_M][nc]) {
                      if (!init) {
                        cvt_y_private_tpp(y_private_ptr[id][m][nc], py[m][nc]);
                        init = true;
                      } else {
                        add_y_tpp(
                            y_private_ptr[id][m][nc], py[m][nc], py[m][nc]);
                      }
                    }
                  }
                  if (fusion_type > 0) {
                    post_ops_fn(m, nc);
                  }
                });
                std::free(y_private);
                std::free(y_private_valid);
              }
            }
          },
          [](auto tuple) { failing_fallback(); });
}

#define LOWP_MODE_NONE 0
#define LOWP_MODE_FP16 1
#define LOWP_MODE_BF16 2
#define LOWP_MODE_INT8 3

/**
 * @brief pack the weight in quantized format.
 * @param qw quantized weight with shape [N, K]
 * @param block_n block size along N, N % block_n == 0, block_n % 16 == 0
 * @param block_k block size along K, K % block_k == 0. block_k % 2 == 0 for
 * bf16 compute_dtype. false if activation is expected to be float32.
 */
at::Tensor qlinear_woq_pack(
    const at::Tensor& qw,
    bool is_int4,
    size_t block_n,
    size_t block_k,
    int64_t lowp_mode) {
  TLA_ASSERT(qw.is_contiguous(), "qw must be contiguous");
  auto sizes = qw.sizes();
  auto N = sizes[0];
  auto K = is_int4 ? sizes[1] * 2 : sizes[1];
  TLA_ASSERT(N % block_n == 0, "N must be multiple of block_n");
  TLA_ASSERT(K % block_k == 0, "K must be multiple of block_k");
  TLA_ASSERT(block_n % 16 == 0, "block_n must be multiple of 16 for int4");
  if (lowp_mode == LOWP_MODE_INT8) {
    TLA_ASSERT(
        block_k % 4 == 0,
        "block_k must be multiple of 4 for int8 for LOWP_MODE_INT8");
  }
  const int N_GROUP_SIZE =
      lowp_mode != LOWP_MODE_INT8 ? get_n_group_size(block_n) : 16;
  const int Nc = N / block_n;
  const int Kc = K / block_k;
  if (is_int4) {
    // TODO(jgong5): support lowp_mode == LOWP_MODE_INT8
    auto result = at::empty({Nc, Kc, block_k, block_n / 2}, qw.options());
    // Pack weight in [N,K] to [N/block_n, K/block_k, block_k, block_n]
    // And then, pre-shuffle per 32 or 64 4-bit values to save shuffle at
    // runtime Take 32 4-bit values as an example below: x0 x1 x2 x3 x4 x5 x6 x7
    // x8 x9 x10 x11 x12 x13 x14 x15 y0 y1 y2 y3 y4 y5 y6 y7 y8 y9 y10 y11 y12
    // y13 y14 y15 becomes x0 y0 x1 y1 x2 y2 x3 y3 x4 y4 x5 y5 x6 y6 x7 y7 x8 y8
    // x9 y9 x10 y10 x11 y11 x12 y12 x13 y13 x14 y14 x15 y15 Here, xi and yj are
    // 4-bit values.
    uint8_t* src_data = (uint8_t*)qw.data_ptr();
    uint8_t* dst_data = (uint8_t*)result.data_ptr();
    auto psrc = GetVLAPtr<uint8_t>(src_data, {block_n, Kc, block_k / 2});
    auto pdst = GetVLAPtr<uint8_t>(dst_data, {Kc, block_k, block_n / 2});
    auto pdst_4vnni =
        GetVLAPtr<uint8_t>(dst_data, {Kc, block_k / 4, block_n / 2, 4});
    auto pack_loop =
        ThreadedLoop<3>({{Nc}, {Kc}, {0, block_n, N_GROUP_SIZE, false}}, "ABc");
    pack_loop([&](int* idx) {
      int nc = idx[0];
      int kc = idx[1];
      int nb = idx[2];
      for (int i = 0; i < N_GROUP_SIZE / 2; i++) {
        for (int kb = 0; kb < block_k; kb += 2) {
          auto src0 = psrc[nc][nb + i][kc][kb / 2];
          auto src1 = psrc[nc][nb + i + N_GROUP_SIZE / 2][kc][kb / 2];
          auto dst0 = (src0 & 0xf) | ((src1 & 0xf) << 4);
          auto dst1 = (src0 >> 4) | ((src1 >> 4) << 4);
          if (lowp_mode != LOWP_MODE_INT8) {
            pdst[nc][kc][kb][nb / 2 + i] = dst0;
            pdst[nc][kc][kb + 1][nb / 2 + i] = dst1;
          } else {
            pdst_4vnni[nc][kc][kb / 4][nb / 2 + i][kb % 4] = dst0;
            pdst_4vnni[nc][kc][(kb + 1) / 4][nb / 2 + i][(kb + 1) % 4] = dst1;
          }
        }
      }
    });
    return result;
  } else {
    TLA_ASSERT(
        lowp_mode != LOWP_MODE_INT8,
        "lowp mode int8 is not supported yet with int8 weight");
    auto result = at::empty({Nc, Kc, block_k, block_n}, qw.options());
    // Pack weight in [N,K] to [N/block_n, K/block_k, block_k, block_n]
    int8_t* src_data = (int8_t*)qw.data_ptr();
    int8_t* dst_data = (int8_t*)result.data_ptr();
    auto psrc = GetVLAPtr<int8_t>(src_data, {block_n, Kc, block_k});
    auto pdst = GetVLAPtr<int8_t>(dst_data, {Kc, block_k, block_n});
    auto pack_loop =
        ThreadedLoop<3>({{Nc}, {Kc}, {0, block_n, N_GROUP_SIZE, false}}, "ABc");
    pack_loop([&](int* idx) {
      int nc = idx[0];
      int kc = idx[1];
      int nb = idx[2];
      for (int i = 0; i < N_GROUP_SIZE; i++) {
        for (int kb = 0; kb < block_k; kb++) {
          pdst[nc][kc][kb][nb + i] = psrc[nc][nb + i][kc][kb];
        }
      }
    });
    return result;
  }
}

at::Tensor qlinear_woq_unpack(
    const at::Tensor& qw_packed,
    bool is_int4,
    int64_t lowp_mode) {
  if (qw_packed.dim() == 4) {
    auto w_sizes = qw_packed.sizes();
    auto Nc = w_sizes[0];
    auto Nb = is_int4 ? w_sizes[3] * 2 : w_sizes[3];
    auto Kc = w_sizes[1];
    auto Kb = w_sizes[2];
    auto N = Nc * Nb;
    auto K = Kc * Kb;
    const int N_GROUP_SIZE =
        lowp_mode != LOWP_MODE_INT8 ? get_n_group_size(Nb) : 16;
    if (is_int4) {
      // TODO: support lowp_mode == 3
      auto result = at::empty({N, K / 2}, qw_packed.options());
      uint8_t* src_data = (uint8_t*)qw_packed.data_ptr();
      uint8_t* dst_data = (uint8_t*)result.data_ptr();
      auto psrc = GetVLAPtr<uint8_t>(src_data, {Kc, Kb, Nb / 2});
      auto psrc_4vnni = GetVLAPtr<uint8_t>(src_data, {Kc, Kb / 4, Nb / 2, 4});
      auto pdst = GetVLAPtr<uint8_t>(dst_data, {Nb, Kc, Kb / 2});
      auto unpack_loop =
          ThreadedLoop<3>({{Nc}, {Kc}, {0, Nb, N_GROUP_SIZE, false}}, "ABc");
      unpack_loop([&](int* idx) {
        int nc = idx[0];
        int kc = idx[1];
        int nb = idx[2];
        for (int kb = 0; kb < Kb; kb += 2) {
          for (int i = 0; i < N_GROUP_SIZE / 2; i++) {
            uint8_t src0, src1;
            if (lowp_mode != LOWP_MODE_INT8) {
              src0 = psrc[nc][kc][kb][nb / 2 + i];
              src1 = psrc[nc][kc][kb + 1][nb / 2 + i];
            } else {
              src0 = psrc_4vnni[nc][kc][kb / 4][nb / 2 + i][kb % 4];
              src1 = psrc_4vnni[nc][kc][(kb + 1) / 4][nb / 2 + i][(kb + 1) % 4];
            }
            pdst[nc][nb + i][kc][kb / 2] = (src0 & 0xf) | ((src1 & 0xf) << 4);
            pdst[nc][nb + i + N_GROUP_SIZE / 2][kc][kb / 2] =
                (src0 >> 4) | ((src1 >> 4) << 4);
          }
        }
      });
      return result;
    } else {
      TLA_ASSERT(
          lowp_mode != LOWP_MODE_INT8,
          "lowp mode int8 is not supported yet with int8 weight");
      auto result = at::empty({N, K}, qw_packed.options());
      int8_t* src_data = (int8_t*)qw_packed.data_ptr();
      int8_t* dst_data = (int8_t*)result.data_ptr();
      auto psrc = GetVLAPtr<int8_t>(src_data, {Kc, Kb, Nb});
      auto pdst = GetVLAPtr<int8_t>(dst_data, {Nb, Kc, Kb});
      auto unpack_loop =
          ThreadedLoop<3>({{Nc}, {Kc}, {0, Nb, N_GROUP_SIZE, false}}, "ABc");
      unpack_loop([&](int* idx) {
        int nc = idx[0];
        int kc = idx[1];
        int nb = idx[2];
        for (int kb = 0; kb < Kb; kb++) {
          for (int i = 0; i < N_GROUP_SIZE; i++) {
            pdst[nc][nb + i][kc][kb] = psrc[nc][kc][kb][nb + i];
          }
        }
      });
      return result;
    }
  } else {
    TLA_ASSERT(qw_packed.dim() == 2, "qw_packed must be 2D or 4D");
    return qw_packed;
  }
}

void compute_int8_qparams_per_tensor(
    const at::Tensor& t,
    float* scale,
    int32_t* zp) {
  auto [t_min, t_max] = at::aminmax(t);
  auto min = t_min.item<float>();
  auto max = t_max.item<float>();
  min = std::min(min, 0.0f);
  max = std::max(max, 0.0f);
  *scale = (max - min) / 255.0f;
  *zp = (int32_t)(-std::nearbyint(min / *scale));
}

template <typename T>
at::Tensor quantize_per_tensor(const at::Tensor& t, float scale, int32_t zp) {
  // TODO(jgong5): optimize me
  auto t_q = t / scale + zp;
  t_q = at::clamp(at::round(t_q), 0, 255);
  return t_q.to(at::kByte);
}

template <>
at::Tensor quantize_per_tensor<bfloat16>(
    const at::Tensor& t,
    float scale,
    int32_t zp) {
#ifdef __AVX512F__
  // modified based on inductor codegen...
  auto convert_float_to_uint8 =
      [](at::vec::Vectorized<float> src) -> at::vec::Vectorized<uint8_t> {
    // Convert from float32 to int32
    __m512i x_values_int32 = _mm512_cvtps_epi32(src);

    // Convert from int32 to int16 using signed saturation
    __m512i xy_packed_v = _mm512_packs_epi32(x_values_int32, x_values_int32);

    constexpr auto min_val = std::numeric_limits<uint8_t>::min();
    constexpr auto max_val = std::numeric_limits<uint8_t>::max();

    // Convert from int16 to uint8 using unsigned saturation
    __m512i packed_and_sat = _mm512_packus_epi16(xy_packed_v, xy_packed_v);
    __m512i xyzw_clamped_v = _mm512_max_epu8(
        _mm512_set1_epi8(min_val),
        _mm512_min_epu8(packed_and_sat, _mm512_set1_epi8(max_val)));
    __m512i permute_mask_v = _mm512_set_epi32(
        0x0f,
        0x0b,
        0x07,
        0x03,
        0x0e,
        0x0a,
        0x06,
        0x02,
        0x0d,
        0x09,
        0x05,
        0x01,
        0x0c,
        0x08,
        0x04,
        0x00);
    return _mm512_permutexvar_epi32(permute_mask_v, xyzw_clamped_v);
  };
  at::Tensor out = at::empty_like(t, at::kByte);
  auto in_ptr0 = t.data_ptr<at::BFloat16>();
  auto out_ptr0 = out.data_ptr<uint8_t>();
  auto n = t.numel();
  auto vecsize = at::vec::Vectorized<float>::size();
  auto vec_end = 0;
#pragma omp parallel for
  for (long i0 = 0; i0 < static_cast<long>(n) / vecsize * vecsize;
       i0 += static_cast<long>(vecsize)) {
    auto tmp0 = at::vec::Vectorized<at::BFloat16>::loadu(
        in_ptr0 + static_cast<long>(i0), vecsize);
    at::vec::Vectorized<float> res_vec1(0);
    at::vec::Vectorized<float> res_vec2(0);
    std::tie(res_vec1, res_vec2) = at::vec::convert_bfloat16_float(tmp0);
    auto tmp1 = res_vec1;
    // auto tmp1 = cvt_bf16_to_fp32(tmp0);
    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(scale));
    auto tmp3 = tmp1 / tmp2;
    auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(zp));
    auto tmp5 = tmp3 + tmp4;
    auto tmp6 = tmp5.round();
    auto tmp7 = (tmp6);
    auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.0));
    auto tmp9 = at::vec::maximum(tmp7, tmp8);
    auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(255.0));
    auto tmp11 = at::vec::minimum(tmp9, tmp10);
    auto tmp12 = (tmp11);
    auto tmp13 = convert_float_to_uint8(tmp12);
    tmp13.store(out_ptr0 + static_cast<long>(i0), vecsize);
  }
  for (long i0 = static_cast<long>(n) / vecsize * vecsize;
       i0 < static_cast<long>(n);
       i0 += static_cast<long>(1)) {
    auto tmp0 = in_ptr0[static_cast<long>(i0)];
    auto tmp1 = static_cast<float>(tmp0);
    auto tmp2 = static_cast<float>(0.05);
    auto tmp3 = tmp1 / tmp2;
    auto tmp4 = static_cast<float>(1.0);
    auto tmp5 = tmp3 + tmp4;
    auto tmp6 = std::nearbyint(tmp5);
    auto tmp7 = static_cast<float>(tmp6);
    auto tmp8 = static_cast<float>(0.0);
    // auto tmp9 = max_propagate_nan(tmp7, tmp8);
    auto tmp9 = 0;
    if (at::_isnan(tmp7)) {
      tmp9 = tmp7;
    }
    tmp9 = tmp7 > tmp8 ? tmp7 : tmp8;
    auto tmp10 = static_cast<float>(255.0);
    auto tmp11 = 0;
    if (at::_isnan(tmp9)) {
      tmp11 = tmp9;
    }
    tmp11 = tmp9 < tmp10 ? tmp9 : tmp10;
    // auto tmp11 = min_propagate_nan(tmp9, tmp10);
    auto tmp12 = static_cast<float>(tmp11);
    auto tmp13 = static_cast<unsigned char>(tmp12);
    out_ptr0[static_cast<long>(i0)] = tmp13;
  }
  return out;
#else
  return at::quantize_per_tensor(t.to(c10::kFloat), scale, zp, c10::kQUInt8);
#endif
}

/**
 * @brief quantized linear with weight in affine quantized format (scale +
 * zero-point) but activation in floating point format.
 * TODO(jgong5): support epilogue fusion
 *
 * @param x input activation in floating point format, 2D plain format [M,K]
 * @param qw weight in affine quantized format, could be 4-bit or 8-bit
 * quantized in 4D blocked format [Nc,Kc,Kb,Nb] or 2D plain format [N,K].
 * @param scales_list a list of fp32/fp16/bf16 scales tensors
 * @param zp_list a list of fp32/fp16/bf16/int8 zero points tensors
 * @param bias_list a list of fp32/fp16/bf16 bias tensors
 * @param lowp_mode decide the compute dtype to use.
 *        LOWP_MODE_NONE: keep activation dtype
 *        LOWP_MODE_FP16: use FP16 or FP32 as compute dtype
 *        LOWP_MODE_BF16: use BF16, FP16 or FP32 as compute dtype
 * @return at::Tensor output activation in same dtype as `x`, 2D plain format
 * [M,N]
 */
at::Tensor qlinear_woq_affine(
    const at::Tensor& x,
    const at::Tensor& qw,
    const TensorList& scales_list,
    const TensorList& zp_list,
    const TensorList& bias_list,
    bool is_int4,
    int64_t lowp_mode,
    int64_t num_concats,
    int64_t fusion_type,
    const TensorList& others_list) {
  const int64_t k_splits = 0;
  // int8_idx is only valid with zp_list when lowp_mode == LOWP_MODE_INT8
  constexpr size_t fp32_idx = 0, fp16_idx = 1, bf16_idx = 2, int8_idx = 3;
  auto biases = bias_list.empty()
      ? TensorList({at::Tensor(), at::Tensor(), at::Tensor()})
      : bias_list;
  if (qw.dim() == 4) {
    auto w_sizes = qw.sizes();
    auto K = x.size(-1);
    auto M = x.numel() / K;
    auto N = w_sizes[0] * w_sizes[3];
    if (is_int4) {
      N *= 2;
    }
    auto out_sizes = x.sizes().vec();
    out_sizes.back() = N;
    auto y = at::empty(out_sizes, x.options());
    auto x_reshape = x.reshape({M, K});
    enumerate_dispatcher<at::ScalarType, at::kFloat, at::kBFloat16, at::kHalf>::
        call(
            x.scalar_type(),
            [&](auto act_dtype) {
              using act_type =
                  typename c10::impl::ScalarTypeToCPPType<act_dtype>::type;
              auto try_compute_in_half = [&]() {
#ifdef __AVX512FP16__
                qlinear_woq_affine_impl<
                    act_type,
                    half,
                    /*TGemmOut*/ half,
                    act_type,
                    half,
                    half>(
                    x_reshape,
                    qw,
                    scales_list[fp16_idx],
                    zp_list[fp16_idx],
                    biases[fp16_idx],
                    y,
                    is_int4,
                    k_splits,
                    num_concats,
                    fusion_type,
                    others_list);
#else
                qlinear_woq_affine_impl<
                    act_type,
                    float,
                    /*TGemmOut*/ float,
                    act_type,
                    float,
                    float>(
                    x_reshape,
                    qw,
                    scales_list[fp32_idx],
                    zp_list[fp32_idx],
                    biases[fp32_idx],
                    y,
                    is_int4,
                    k_splits,
                    num_concats,
                    fusion_type,
                    others_list);
#endif
              };
              if (lowp_mode == LOWP_MODE_NONE) {
                if (std::is_same<act_type, half>()) {
                  try_compute_in_half();
                } else if (std::is_same<act_type, bfloat16>()) {
                  qlinear_woq_affine_impl<
                      bfloat16,
                      bfloat16,
                      /*TGemmOut*/ float,
                      bfloat16,
                      bfloat16,
                      bfloat16>(
                      x_reshape,
                      qw,
                      scales_list[bf16_idx],
                      zp_list[bf16_idx],
                      biases[fp32_idx],
                      y,
                      is_int4,
                      k_splits,
                      num_concats,
                      fusion_type,
                      others_list);
                } else {
                  qlinear_woq_affine_impl<
                      float,
                      float,
                      /*TGemmOut*/ float,
                      float,
                      float,
                      float>(
                      x_reshape,
                      qw,
                      scales_list[fp32_idx],
                      zp_list[fp32_idx],
                      biases[fp32_idx],
                      y,
                      is_int4,
                      k_splits,
                      num_concats,
                      fusion_type,
                      others_list);
                }
              } else if (lowp_mode == LOWP_MODE_FP16) {
                try_compute_in_half();
              } else if (lowp_mode == LOWP_MODE_BF16) {
                if (M >= SMALL_BATCH_THRESHOLD) {
                  // compute in bfloat16 for large bs
                  qlinear_woq_affine_impl<
                      act_type,
                      bfloat16,
                      /*TGemmOut*/ float,
                      act_type,
                      bfloat16,
                      bfloat16>(
                      x_reshape,
                      qw,
                      scales_list[bf16_idx],
                      zp_list[bf16_idx],
                      biases[fp32_idx],
                      y,
                      is_int4,
                      k_splits,
                      num_concats,
                      fusion_type,
                      others_list);
                } else {
                  try_compute_in_half();
                }
              } else {
                TLA_ASSERT(lowp_mode == LOWP_MODE_INT8, "invalid lowp_mode");
                TLA_ASSERT(is_int4, "LOWP_MODE_INT8 only support is_int4=true");
                float scale_a;
                int32_t zp_a;
                auto x_reshape_contig = x_reshape.contiguous();
                compute_int8_qparams_per_tensor(
                    x_reshape_contig, &scale_a, &zp_a);
                auto x_quantized = quantize_per_tensor<act_type>(
                    x_reshape_contig, scale_a, zp_a);
                qlinear_woq_affine_impl<
                    uint8_t,
                    uint8_t,
                    /*TGemmOut*/ float,
                    act_type,
                    float,
                    int8_t>(
                    x_quantized,
                    qw,
                    scales_list[fp32_idx],
                    zp_list[int8_idx],
                    biases[fp32_idx],
                    y,
                    is_int4,
                    k_splits,
                    num_concats,
                    fusion_type,
                    others_list,
                    scale_a,
                    zp_a);
              }
            },
            failing_fallback<at::ScalarType>);
    return y;
  } else {
    TLA_ASSERT(
        qw.dim() == 2,
        "weight must be in 4D blocked format or 2D plain format");
    auto compute_dtype = x.scalar_type();
    if (lowp_mode == LOWP_MODE_FP16) {
      compute_dtype = at::kHalf;
    } else if (lowp_mode == LOWP_MODE_BF16) {
      compute_dtype = at::kBFloat16;
    }
    auto w =
        [&]() {
          if (is_int4) {
            using namespace at::indexing;
            auto w_int8 = at::empty(
                {qw.size(0), qw.size(1) * 2}, qw.options().dtype(at::kByte));
            w_int8.index({Slice(), Slice(None, None, 2)})
                .copy_(qw.bitwise_and(0xf));
            w_int8.index({Slice(), Slice(1, None, 2)})
                .copy_(qw.bitwise_right_shift(4));
            return (w_int8.to(at::kFloat) - zp_list[fp32_idx]) *
                scales_list[fp32_idx];
          } else {
            return (qw.to(at::kFloat) - zp_list[fp32_idx]) *
                scales_list[fp32_idx];
          }
        }()
            .to(compute_dtype);
    auto x_fp = x.to(compute_dtype);
    auto y = at::linear(x_fp, w);
    if (biases[0].defined()) {
      auto b_index = compute_dtype == at::kFloat ? fp32_idx
          : compute_dtype == at::kHalf           ? fp16_idx
                                                 : bf16_idx;
      y = at::add(y, biases[b_index]);
    }
    if (fusion_type == FUSE_GELU) {
      y = at::gelu(y);
    } else if (fusion_type == FUSE_ADD || fusion_type == FUSE_ADD_ADD) {
      for (auto& tin : others_list)
        y = at::add(y, tin);
    }
    if (num_concats > 1) {
      y = y.view({-1, num_concats, y.size(-1) / num_concats})
              .transpose(0, 1)
              .contiguous()
              .view({-1, y.size(-1)});
    }
    return y.to(x.scalar_type());
  }
}

#else // defined(CPU_CAPABILITY_AVX512_FP16) && defined(COMPILER_PREREQ_MET)

static at::Tensor empty_tensor;

at::Tensor qlinear_woq_affine(
    const at::Tensor& x,
    const at::Tensor& qw,
    const TensorList& scales_list,
    const TensorList& zp_list,
    const TensorList& bias_list,
    bool is_int4,
    int64_t lowp_mode,
    int64_t num_concats,
    int64_t fusion_type,
    const TensorList& others_list) {
  return empty_tensor;
}

at::Tensor qlinear_woq_pack(
    const at::Tensor& qw,
    bool is_int4,
    size_t block_n,
    size_t block_k,
    int64_t lowp_mode) {
  return empty_tensor;
}

at::Tensor qlinear_woq_unpack(
    const at::Tensor& qw_packed,
    bool is_int4,
    int64_t lowp_mode) {
  return empty_tensor;
}
#endif // defined(CPU_CAPABILITY_AVX512_FP16) && defined(COMPILER_PREREQ_MET)

} // namespace

REGISTER_DISPATCH(woq_tpp_gemm_kernel_stub, &qlinear_woq_affine);
REGISTER_DISPATCH(woq_tpp_gemm_packB_stub, &qlinear_woq_pack);
REGISTER_DISPATCH(woq_tpp_gemm_unpackB_stub, &qlinear_woq_unpack);

} // namespace cpu
} // namespace torch_ipex