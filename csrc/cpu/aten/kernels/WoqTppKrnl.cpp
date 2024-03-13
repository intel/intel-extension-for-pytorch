// weight-only quantization gemm kernel (int8, int4 etc.)
#ifdef USE_LIBXSMM
#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/cpu/vec/functional.h>
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

#define FUSE_GELU_ERF 1
#define FUSE_ADD 2
#define FUSE_ADD_ADD 3
#define FUSE_GELU_TANH 4

#define LOWP_MODE_NONE 0
#define LOWP_MODE_FP16 1
#define LOWP_MODE_BF16 2
#define LOWP_MODE_INT8 3

#define QINT8 1
#define QINT4 2
#define NF4 3

constexpr bool is_4bit(const int qw_type) {
  return qw_type == QINT4 || qw_type == NF4;
}

constexpr bool is_sym_quant(const int qw_type) {
  return qw_type == NF4;
}

static constexpr std::array<float, 16> NF4_QUANT_TABLE = {
    -1.0 - 1e-2, // 0b0000
    -0.8480964004993439, // 0b0001
    -0.6106329262256622, // 0b0010
    -0.4599952697753906, // 0b0011
    -0.33967943489551544, // 0b0100
    -0.23460740596055984, // 0b0101
    -0.13791173323988914, // 0b0110
    -0.045525018125772476, // 0b0111
    0.03979014977812767, // 0b1000
    0.1202552504837513, // 0b1001
    0.2035212516784668, // 0b1010
    0.2920137718319893, // 0b1011
    0.3893125355243683, // 0b1100
    0.5016634166240692, // 0b1101
    0.6427869200706482, // 0b1110
    0.8614784181118011, // 0b1111
};

static constexpr std::array<float, 16> NF4_DEQUANT_TABLE = {
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
};

at::Tensor map_float_tensor_to_nf4(const at::Tensor& t) {
  // Map [-1, 1] to nf4. Assume t in [-1, 1]
  // Logic:
  // for i in range(len(NF4_QUANT_TABLE)):
  //     out_uint8[t > NF4_QUANT_TABLE[i]] = i
  using namespace at::indexing;
  auto out_uint8 = at::empty(t.sizes(), t.options().dtype(at::kByte));
  for (size_t i = 0; i < NF4_QUANT_TABLE.size(); ++i) {
    out_uint8.index_put_({t.greater(NF4_QUANT_TABLE[i])}, i);
  }
  return out_uint8;
}

at::Tensor map_nf4_tensor_to_float(const at::Tensor& t) {
  // Map nf4 to [-1, 1], t is already unpacked as uint8
  // Logic:
  // for i in range(len(NF4_DEQUANT_TABLE)):
  //     out_dq[t == i] = NF4_DEQUANT_TABLE[i]
  using namespace at::indexing;
  auto out_dq = at::empty(t.sizes(), t.options().dtype(at::kFloat));
  for (size_t i = 0; i < NF4_DEQUANT_TABLE.size(); ++i) {
    out_dq.index_put_({t.eq(i)}, NF4_DEQUANT_TABLE[i]);
  }
  return out_dq;
}

// We only build optimized kernels if AVX512_FP16 is supported and gcc>=12.3
// Otherwise we just return empty results
// TODO(Weiwen) Merge WoqTppKrnl.cpp and WoqLinearKrnl.cpp and put the latter in
// the #else part
#if defined(CPU_CAPABILITY_AVX512_FP16) && defined(COMPILER_PREREQ_MET)

#define QUANT_A_THRESHOLD 30720
#define SMALL_BATCH_THRESHOLD 32
#define PARALLEL_M_THRESHOLD 128
constexpr long PREFETCH_K_DIST = 64; // TODO(jgong5): do not hard-code
constexpr long LOOP_K_UNROLL = 4; // TODO(jgong5): do not hard-code

#define UNQUANT_A -1
#define QUANT_A_PER_TENSOR 0
#define QUANT_A_PER_K_BLOCK 1
#define QUANT_A_PER_M 2
#define QUANT_A_PER_M_K_BLOCK 3

#define QUANT_W_PER_CHANNEL 0
#define QUANT_W_PER_K_BLOCK 1

template <long N_GROUP_SIZE, bool sym_quant>
struct load_dequant_zp_only_4bit {
  template <typename LUT, typename VAT>
  static inline VAT call(uint8_t* p, LUT lut, VAT vzps) {
    TLA_ASSERT(false, "not implemented");
  }
};

template <long N_GROUP_SIZE, bool sym_quant>
struct load_dequant_zp_only_int8 {
  template <typename VAT>
  static inline VAT call(uint8_t* p, VAT vzps) {
    TLA_ASSERT(false, "not implemented");
  }
};

template <bool sym_quant>
struct load_dequant_zp_only_4bit<64, sym_quant> {
// TODO(jgong5): further simplify the dequant intrinsics below with VecOps
#ifdef __AVX512F__
  static inline std::array<__m512, 4> call(
      uint8_t* p,
      __m512 lut,
      std::array<__m512, 4> vzps) {
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
      if constexpr (!sym_quant) {
        vbs[idx] = _mm512_sub_ps(vbs[idx], vzps[idx]);
      }
    });
    return vbs;
  }
#endif

#ifdef __AVX512FP16__
  static inline std::array<__m512h, 2> call(
      uint8_t* p,
      __m512h lut,
      std::array<__m512h, 2> vzps) {
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
      if constexpr (!sym_quant) {
        vbs[idx] = _mm512_sub_ph(vbs[idx], vzps[idx]);
      }
    });
    return vbs;
  }
#endif
};

template <bool sym_quant>
struct load_dequant_zp_only_int8<64, sym_quant> {
// TODO(jgong5): further simplify the dequant intrinsics below with VecOps
#ifdef __AVX512F__
  static inline std::array<__m512, 4> call(
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
      if constexpr (!sym_quant) {
        vbs[i] = _mm512_sub_ps(vbs[i], vzps[i]);
      }
    });
    return vbs;
  }
#endif

#ifdef __AVX512FP16__
  static inline std::array<__m512h, 2> call(
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
      if constexpr (!sym_quant) {
        vbs[i] = _mm512_sub_ph(vbs[i], vzps[i]);
      }
    });
    return vbs;
  }
#endif
};

template <bool sym_quant>
struct load_dequant_zp_only_4bit<32, sym_quant> {
#ifdef __AVX512F__
  static inline std::array<__m512, 2> call(
      uint8_t* p,
      __m512 lut,
      std::array<__m512, 2> vzps) {
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
      if constexpr (!sym_quant) {
        vbs[idx] = _mm512_sub_ps(vbs[idx], vzps[idx]);
      }
    });
    return vbs;
  }
#endif

#ifdef __AVX512FP16__
  static inline std::array<__m512h, 1> call(
      uint8_t* p,
      __m512h lut,
      std::array<__m512h, 1> vzps) {
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
      if constexpr (!sym_quant) {
        vbs[idx] = _mm512_sub_ph(vbs[idx], vzps[idx]);
      }
    });
    return vbs;
  }
#endif
};

template <bool sym_quant>
struct load_dequant_zp_only_int8<32, sym_quant> {
#ifdef __AVX512F__
  static inline std::array<__m512, 2> call(
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
      if constexpr (!sym_quant) {
        vbs[i] = _mm512_sub_ps(vbs[i], vzps[i]);
      }
    });
    return vbs;
  }
#endif

#ifdef __AVX512FP16__
  static inline std::array<__m512h, 1> call(
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
      if constexpr (!sym_quant) {
        vbs[i] = _mm512_sub_ph(vbs[i], vzps[i]);
      }
    });
    return vbs;
  }
#endif
};

template <bool sym_quant>
struct load_dequant_zp_only_4bit<16, sym_quant> {
#ifdef __AVX512F__
  static inline std::array<__m512, 1> call(
      uint8_t* p,
      __m512 lut,
      std::array<__m512, 1> vzps) {
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
    if constexpr (!sym_quant) {
      vbs[0] = _mm512_sub_ps(vbs[0], vzps[0]);
    }
    return vbs;
  }
#endif

#ifdef __AVX512FP16__
  static inline std::array<__m512h, 0> call(
      uint8_t* p,
      __m512h lut,
      std::array<__m512h, 0> vzps) {
    TLA_ASSERT(false, "not implemented");
  }
#endif
};

template <bool sym_quant>
struct load_dequant_zp_only_int8<16, sym_quant> {
#ifdef __AVX512F__
  static inline std::array<__m512, 1> call(
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
    if constexpr (!sym_quant) {
      vbs[0] = _mm512_sub_ps(vbs[0], vzps[0]);
    }
    return vbs;
  }
#endif

#ifdef __AVX512FP16__
  static inline std::array<__m512h, 0> call(
      uint8_t* p,
      std::array<__m512h, 0> vzps) {
    TLA_ASSERT(false, "not implemented");
  }
#endif
};

#ifdef __AVX512F__
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

template <long N, bool sym_quant, typename T>
struct load_dequant_4bit {
  using VT = typename VecType<T>::type;
  using V = VecOps<VT>;
  using VA = VecArray<N, T>;
  using VAT = typename VA::type;
  constexpr static long COLS = VA::num_vec;

  static inline VAT call(uint8_t* p, VAT vscales, VT lut, VAT vzps) {
    auto vbs = load_dequant_zp_only_4bit<N, sym_quant>::call(p, lut, vzps);
    compile_time_for<COLS>::op(
        [&](auto idx) { vbs[idx] = V::mul(vbs[idx], vscales[idx]); });
    return vbs;
  }
};

template <long N, bool sym_quant, typename T>
struct load_dequant_int8 {
  using VT = typename VecType<T>::type;
  using V = VecOps<VT>;
  using VA = VecArray<N, T>;
  using VAT = typename VA::type;
  constexpr static long COLS = VA::num_vec;

  static inline VAT call(uint8_t* p, VAT vscales, VAT vzps) {
    auto vbs = load_dequant_zp_only_int8<N, sym_quant>::call(p, vzps);
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
    int quant_a_mode = -1,
    long PREFETCH_K_DIST = 0,
    typename Enabled = void>
struct GemmMicroKernel {
  template <int qw_type>
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
    int quant_a_mode,
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
    quant_a_mode,
    PREFETCH_K_DIST,
    typename std::enable_if_t<
        std::is_same<T, float>::value || std::is_same<T, half>::value>> {
  // TODO(jgong5): generalize this with pre/post op handlers
  template <int qw_type>
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

    VT lut;
    constexpr bool is_4bit_flag = is_4bit(qw_type);
    constexpr bool sym_quant = is_sym_quant(qw_type);
    if constexpr (is_4bit_flag) {
      lut = qw_type == NF4 ? V::set_nf4_lut() : V::set_0_to_15();
    }

    // Load scales and zps
    compile_time_for<CNBLOCKS>::op([&](auto i) {
      constexpr const int col = i * CBLOCK;
      vscales[i] = VArray::load1d(scales + col * V::VLEN);
      if constexpr (!sym_quant) {
        vzps[i] = VArray::load1d(zps + col * V::VLEN);
      }
    });

    // NB: For fp16 in int8 woq, we do not delay the scale to the post-op but
    // leave it to the dequant otherwise the weight value might be too large to
    // overflow fp16 range.
    constexpr bool scale_as_post_op = !std::is_same<T, half>() || is_4bit_flag;

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
          if constexpr (is_4bit_flag) {
            vb[cbidx] =
                load_dequant_zp_only_4bit<N_GROUP_SIZE, sym_quant>::call(
                    ADDRESS(B, k, col * V::VLEN / 2, ldb / 2),
                    lut,
                    vzps[cbidx]);
          } else {
            vb[cbidx] =
                load_dequant_zp_only_int8<N_GROUP_SIZE, sym_quant>::call(
                    ADDRESS(B, k, col * V::VLEN, ldb), vzps[cbidx]);
          }
        } else {
          if constexpr (is_4bit_flag) {
            vb[cbidx] = load_dequant_4bit<N_GROUP_SIZE, sym_quant, T>::call(
                ADDRESS(B, k, col * V::VLEN / 2, ldb / 2),
                vscales[cbidx],
                lut,
                vzps[cbidx]);
          } else {
            vb[cbidx] = load_dequant_int8<N_GROUP_SIZE, sym_quant, T>::call(
                ADDRESS(B, k, col * V::VLEN, ldb), vscales[cbidx], vzps[cbidx]);
          }
        }
        if constexpr (PREFETCH_K_DIST > 0) {
          if constexpr (is_4bit_flag) {
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
template <
    long M,
    long N,
    long ldb,
    bool transA,
    bool ACC,
    int quant_a_mode,
    long PREFETCH_K_DIST>
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
    quant_a_mode,
    PREFETCH_K_DIST> {
  template <int qw_type>
  static inline void call(
      long K,
      uint8_t* A,
      long lda,
      uint8_t* B,
      float* C,
      long ldc,
      float* scales,
      int8_t* zps,
      float* scale_a,
      int32_t* zp_a,
      int32_t k_groups) {
    TLA_ASSERT(zps, "Calculation of uint8 does not support symmetric quant.");
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
      __m512 vc_float;
      if constexpr (
          quant_a_mode == QUANT_A_PER_TENSOR ||
          quant_a_mode == QUANT_A_PER_K_BLOCK) {
        vc[i] = _mm512_sub_epi32(
            vc[i],
            _mm512_mullo_epi32(vcompensate[col], _mm512_set1_epi32(*zp_a)));
        vc_float = _mm512_cvtepi32_ps(vc[i]);
        vc_float = _mm512_mul_ps(vc_float, _mm512_set1_ps(*scale_a));
      } else if constexpr (quant_a_mode == QUANT_A_PER_M) {
        vc[i] = _mm512_sub_epi32(
            vc[i],
            _mm512_mullo_epi32(
                vcompensate[col], _mm512_set1_epi32(*(zp_a + row))));
        vc_float = _mm512_cvtepi32_ps(vc[i]);
        vc_float = _mm512_mul_ps(vc_float, _mm512_set1_ps(*(scale_a + row)));
      } else {
        vc[i] = _mm512_sub_epi32(
            vc[i],
            _mm512_mullo_epi32(
                vcompensate[col], _mm512_set1_epi32(*(zp_a + row * k_groups))));
        vc_float = _mm512_cvtepi32_ps(vc[i]);
        vc_float = _mm512_mul_ps(
            vc_float, _mm512_set1_ps(*(scale_a + row * k_groups)));
      }

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
template <typename Tin, long ldb, long N_GROUP_SIZE, int qw_type>
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
    using VA = VecArray<N_GROUP_SIZE, Tin>;
    using VAT = typename VA::type;
    constexpr bool is_4bit_flag = is_4bit(qw_type);
    constexpr bool sym_quant = is_sym_quant(qw_type);
    for (int n = 0; n < N; n += N_GROUP_SIZE) {
      // load scales and zps
      auto vscales = load_qparam(scales + n);
      VAT vzps;
      if constexpr (!sym_quant) {
        vzps = load_qparam(zps + n);
      }
      for (int k = 0; k < K; k++) {
        // load and dequant qB to vb
        auto vbs = load_qint_as_fp(
            is_4bit_flag ? &qB[k * ldb / 2 + n / 2] : &qB[k * ldb + n],
            vscales,
            vzps);
        // prefetch qB data
        if constexpr (PREFETCH_K_DIST > 0) {
          auto prefetch_addr = is_4bit_flag
              ? &qB[(k + PREFETCH_K_DIST) * ldb / 2 + n / 2]
              : &qB[(k + PREFETCH_K_DIST) * ldb + n];
          _mm_prefetch(prefetch_addr, _MM_HINT_T0);
        }
        // store vb to B
        store(B + k * N + n, vbs);
      }
    }
  }
};

#ifdef __AVX512F__
template <long ldb, long N_GROUP_SIZE, int qw_type>
struct dequant_n_grouped<bfloat16, ldb, N_GROUP_SIZE, qw_type> {
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
    using VAT = typename VA::type;
    constexpr long COLS = VA::num_vec;
    constexpr bool is_4bit_flag = is_4bit(qw_type);
    constexpr bool sym_quant = is_sym_quant(qw_type);

    for (int n = 0; n < N; n += N_GROUP_SIZE) {
      // load scales and zps
      auto vscales = load_qparam(scales + n);
      VAT vzps;
      if constexpr (!sym_quant) {
        vzps = load_qparam(zps + n);
      }
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
            is_4bit_flag ? &qB[k * ldb / 2 + n / 2] : &qB[k * ldb + n],
            vscales,
            vzps);
        auto vbs_k1 = load_qint_as_fp(
            is_4bit_flag ? &qB[(k + 1) * ldb / 2 + n / 2]
                         : &qB[(k + 1) * ldb + n],
            vscales,
            vzps);
        // prefetch qB data
        if constexpr (PREFETCH_K_DIST > 0) {
          auto prefetch_addr = is_4bit_flag
              ? &qB[(k + PREFETCH_K_DIST) * ldb / 2 + n / 2]
              : &qB[(k + PREFETCH_K_DIST) * ldb + n];
          _mm_prefetch(prefetch_addr, _MM_HINT_T0);
        }
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

template <typename Tin, long ldb, long N_GROUP_SIZE, int qw_type>
struct Dequantize {
  static void call(uint8_t* qB, long K, long N, Tin* scales, Tin* zps, Tin* B);
};

template <long ldb, long N_GROUP_SIZE, int qw_type>
struct Dequantize<float, ldb, N_GROUP_SIZE, qw_type> {
  static inline void call(
      uint8_t* qB,
      long K,
      long N,
      float* scales,
      float* zps,
      float* B) {
#if defined(__AVX512F__)
    using T = float;
    using VT = typename VecType<T>::type;
    using V = VecOps<VT>;
    using VA = VecArray<N_GROUP_SIZE, T>;
    constexpr int VLEN = VA::vec_ops::VLEN;
    constexpr long COLS = VA::num_vec;

    VT lut;
    constexpr bool is_4bit_flag = is_4bit(qw_type);
    constexpr bool sym_quant = is_sym_quant(qw_type);
    if constexpr (is_4bit_flag) {
      lut = qw_type == NF4 ? V::set_nf4_lut() : V::set_0_to_15();
    }

    dequant_n_grouped<float, ldb, N_GROUP_SIZE, qw_type>::call(
        qB,
        K,
        N,
        scales,
        zps,
        B,
        [&](float* p) { return VA::load1d(p); },
        [&](uint8_t* p, auto vscales, auto vzps) {
          if constexpr (is_4bit_flag) {
            return load_dequant_4bit<N_GROUP_SIZE, sym_quant, T>::call(
                p, vscales, lut, vzps);
          } else {
            return load_dequant_int8<N_GROUP_SIZE, sym_quant, T>::call(
                p, vscales, vzps);
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

template <long ldb, long N_GROUP_SIZE, int qw_type>
struct Dequantize<bfloat16, ldb, N_GROUP_SIZE, qw_type> {
  static inline void call(
      uint8_t* qB,
      long K,
      long N,
      bfloat16* scales,
      bfloat16* zps,
      bfloat16* B) {
#ifdef __AVX512F__
    using T = bfloat16;
    using VT = typename VecType<T>::type;
    using V = VecOps<VT>;
    using VA = VecArray<N_GROUP_SIZE, T>;
    constexpr long COLS = VA::num_vec;

    // lookup table converting uint8 to float, 15.0f - 0.0f
    // _mm512_permutexvar_ph needs 5 bits while we only need 4 bits, init the
    // table to honor the lower 4 bits regardless of the the highest bit, thus
    // saving an "and" op
    VT lut;
    constexpr bool is_4bit_flag = is_4bit(qw_type);
    constexpr bool sym_quant = is_sym_quant(qw_type);
    if constexpr (is_4bit_flag) {
      lut = qw_type == NF4 ? V::set_nf4_lut() : V::set_0_to_15();
    }

    dequant_n_grouped<bfloat16, ldb, N_GROUP_SIZE, qw_type>::call(
        qB,
        K,
        N,
        scales,
        zps,
        B,
        [&](bfloat16* p) { return VA::load1d(p); },
        [&](uint8_t* p, auto vscales, auto vzps) {
          if constexpr (is_4bit_flag) {
            return load_dequant_4bit<N_GROUP_SIZE, sym_quant, float>::call(
                p, vscales, lut, vzps);
          } else {
            return load_dequant_int8<N_GROUP_SIZE, sym_quant, float>::call(
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

template <long ldb, long N_GROUP_SIZE, int qw_type>
struct Dequantize<half, ldb, N_GROUP_SIZE, qw_type> {
  static inline void call(
      uint8_t* qB,
      long K,
      long N,
      half* scales,
      half* zps,
      half* B) {
#ifdef __AVX512FP16__
    using T = half;
    using VT = typename VecType<T>::type;
    using V = VecOps<VT>;
    using VA = VecArray<N_GROUP_SIZE, T>;
    constexpr int VLEN = VA::vec_ops::VLEN;
    constexpr long COLS = VA::num_vec;

    // lookup table converting uint8 to float, 15.0f - 0.0f
    // _mm512_permutexvar_ph needs 5 bits while we only need 4 bits, init the
    // table to honor the lower 4 bits regardless of the the highest bit, thus
    // saving an "and" op
    VT lut;
    constexpr bool is_4bit_flag = is_4bit(qw_type);
    constexpr bool sym_quant = is_sym_quant(qw_type);
    if constexpr (is_4bit_flag) {
      lut = qw_type == NF4 ? V::set_nf4_lut() : V::set_0_to_15();
    }

    dequant_n_grouped<half, ldb, N_GROUP_SIZE, qw_type>::call(
        qB,
        K,
        N,
        scales,
        zps,
        B,
        [&](half* p) { return VA::load1d(p); },
        [&](uint8_t* p, auto vscales, auto vzps) {
          if constexpr (is_4bit_flag) {
            return load_dequant_4bit<N_GROUP_SIZE, sym_quant, T>::call(
                p, vscales, lut, vzps);
          } else {
            return load_dequant_int8<N_GROUP_SIZE, sym_quant, T>::call(
                p, vscales, vzps);
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
struct Dequantize<int8_t, ldb, /*N_GROUP_SIZE*/ 16, /*qw_type*/ QINT4> {
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
    int qw_type,
    int quant_a_mode,
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
      float* scale_a = nullptr,
      int32_t* zp_a = nullptr,
      int32_t k_groups = -1) {
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
    int qw_type,
    int quant_a_mode,
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
    qw_type,
    quant_a_mode,
    PREFETCH_K_DIST> {
 public:
  DequantGemmTPP(long M, long K, long lda, long ldc)
      : M(M), K(K), lda(lda), ldc(ldc) {
    static_assert(N % 16 == 0, "N must be a multiple of 16");
    if (std::is_same<Tin, bfloat16>())
      TLA_ASSERT(K % 2 == 0, "Kb must be a multiple of 2 for bfloat16");
    pgemm = new BrgemmTPP<Tin, Tout>(
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

  ~DequantGemmTPP() {
    delete pgemm;
  }

  inline void operator()(
      Tin* A,
      uint8_t* qB,
      Tin* scales,
      Tin* zps,
      Tout* C,
      bool no_tile_cfg = true,
      float* scale_a = nullptr,
      int32_t* zp_a = nullptr,
      int32_t k_groups = -1) {
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
                  quant_a_mode,
                  PREFETCH_K_DIST>::
                  template call<qw_type>(
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
                        quant_a_mode,
                        PREFETCH_K_DIST>::
                        template call<qw_type>(
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
      Dequantize<Tin, ldb, N_GROUP_SIZE, qw_type>::call(
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
  BrgemmTPP<Tin, Tout>* pgemm;
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
    int quant_a_mode,
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
    /*qw_type*/ QINT4,
    quant_a_mode,
    PREFETCH_K_DIST> {
  using TBrgemmTPP = BrgemmTPP<int8_t, int32_t>;

 public:
  DequantGemmTPP(long M, long K, long lda, long ldc)
      : M(M), K(K), lda(lda), ldc(ldc) {
    static_assert(N % 16 == 0, "N must be a multiple of 16");
    TLA_ASSERT(K % 4 == 0, "Kb must be a multiple of 4 for int8 VNNI");
    // TODO(jgong5): output fp32 directly
    pgemm = new TBrgemmTPP(
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

  ~DequantGemmTPP() {
    delete pgemm;
  }

  inline void operator()(
      uint8_t* A,
      uint8_t* qB,
      float* scales,
      int8_t* zps,
      float* C,
      bool no_tile_cfg = true,
      float* scale_a = nullptr,
      int32_t* zp_a = nullptr,
      int32_t k_groups = -1) {
    auto qA = GetVLAPtr<uint8_t>(A, {lda});
#ifdef __AVX512VNNI__
    if (M < SMALL_BATCH_THRESHOLD) {
      constexpr long PREFERRED_BLOCK_M =
          BLOCK_M * N / 16 >= 16 ? BLOCK_M / 2 : BLOCK_M;
      for (long m = 0; m < M; m += PREFERRED_BLOCK_M) {
        long block_m = std::min(M - m, PREFERRED_BLOCK_M);
        float* scale_a_m;
        int32_t* zp_a_m;
        if constexpr (
            quant_a_mode == QUANT_A_PER_M ||
            quant_a_mode == QUANT_A_PER_M_K_BLOCK) {
          scale_a_m = scale_a + m * k_groups;
          zp_a_m = zp_a + m * k_groups;
        } else {
          scale_a_m = scale_a;
          zp_a_m = zp_a;
        }
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
                  quant_a_mode,
                  PREFETCH_K_DIST>::
                  template call<QINT4>(
                      K,
                      qA[m],
                      lda,
                      qB,
                      C + m * ldc,
                      ldc,
                      scales,
                      zps,
                      scale_a_m,
                      zp_a_m,
                      k_groups);
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
                        quant_a_mode,
                        PREFETCH_K_DIST>::
                        template call<QINT4>(
                            K,
                            qA[m],
                            lda,
                            qB,
                            C + m * ldc,
                            ldc,
                            scales,
                            zps,
                            scale_a_m,
                            zp_a_m,
                            k_groups);
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
      Dequantize<int8_t, ldb, N_GROUP_SIZE, /*qw_type*/ QINT4>::call(
          qB, K, N, zps, B[0][0], compensation);
      (*pgemm)((int8_t*)qA[0], B[0][0], qC[0], 1, no_tile_cfg);
      // post-op and convert back to C
      for (long m = 0; m < M; ++m) {
#pragma omp simd
        for (long n = 0; n < N; ++n) {
          float* scale_a_m;
          int32_t* zp_a_m;
          if constexpr (
              quant_a_mode == QUANT_A_PER_M ||
              quant_a_mode == QUANT_A_PER_M_K_BLOCK) {
            scale_a_m = scale_a + m * k_groups;
            zp_a_m = zp_a + m * k_groups;
          } else {
            scale_a_m = scale_a;
            zp_a_m = zp_a;
          }
          float c = (qC[m][n] - compensation[n] * (*zp_a_m)) * (*scale_a_m) *
              scales[n];
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
  TBrgemmTPP* pgemm;
  long M;
  long K;
  long lda;
  long ldc;
};

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
    typename TZero,
    int quant_a_mode = -1,
    int quant_w_mode = 0>
void qlinear_woq_affine_impl(
    const at::Tensor& x,
    const at::Tensor& qw_packed,
    const at::Tensor& scales, // dtype is TComp
    const at::Tensor& b, // dtype is TComp
    at::Tensor y,
    const int qw_type,
    int k_splits,
    int fusion_type,
    const TensorList& others_list,
    int64_t quant_block_k,
    const std::optional<at::Tensor>& zps = std::nullopt, // dtype is TComp
    float* scales_a_ptr = nullptr,
    int32_t* zps_a_ptr = nullptr) {
  const bool is_4bit_flag = is_4bit(qw_type);
  const bool sym_quant = is_sym_quant(qw_type);
  auto w_sizes = qw_packed.sizes();
  auto Nc = w_sizes[0];
  auto Nb = is_4bit_flag ? w_sizes[3] * 2 : w_sizes[3];
  auto Kc = w_sizes[1];
  auto Kb = w_sizes[2];
  auto N = Nc * Nb;
  auto K = Kc * Kb;
  auto M = x.numel() / K;
  assert(quant_block_k % Kb == 0);
  auto quant_block_multiple = quant_block_k == 0 ? 1 : quant_block_k / Kb;
  auto quant_k_blocks =
      quant_block_k == 0 ? 1 : (K + quant_block_k - 1) / quant_block_k;

  TLA_ASSERT(Nb % 16 == 0, "Nb must be a multiple of 16");

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
  if (k_splits <= 0 || M >= 32 || BLOCK_M_rem) {
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
  auto ldy = N;
  auto ldc = (no_y_buf || k_splits > 1) ? ldy : Nb;

  auto px = GetVLAPtr<T>(x, {Kc, Kb});
  auto pw = GetVLAPtr<uint8_t>(
      (uint8_t*)qw_packed.data_ptr(), {Kc, Kb * (is_4bit_flag ? Nb / 2 : Nb)});
  auto py = GetVLAPtr<Tout>(y, {Nc, Nb}); /*[M, Nc, Nb]*/
  int scales_kc = quant_w_mode == QUANT_W_PER_CHANNEL ? QUANT_W_PER_K_BLOCK
                                                      : quant_k_blocks;
  auto pscales = GetVLAPtr<TScale>(scales, {scales_kc, Nb});
  auto pzps = sym_quant ? GetVLAPtr<TZero>(nullptr, {1, 1})
                        : GetVLAPtr<TZero>(zps.value(), {scales_kc, Nb});
  auto pb = GetVLAPtr<TGemmOut>(b, {Nb});
  auto tin0 = others_list.size() > 0 ? others_list[0] : at::Tensor{};
  auto pin0 = GetVLAPtr<Tout>(tin0, {Nc, Nb}); /*[M, Nc, Nb]*/
  auto tin1 = others_list.size() > 1 ? others_list[1] : at::Tensor{};
  auto pin1 = GetVLAPtr<Tout>(tin1, {Nc, Nb}); /*[M, Nc, Nb]*/

  auto copy_bias_out_tpp = CpyBiasTPP<TGemmOut>(BLOCK_M, Nb, ldy);
  auto copy_bias_buf_tpp = CpyBiasTPP<TGemmOut>(BLOCK_M, Nb, Nb);
  auto copy_bias_out_rem_tpp = CpyBiasTPP<TGemmOut>(BLOCK_M_rem, Nb, ldy);
  auto copy_bias_buf_rem_tpp = CpyBiasTPP<TGemmOut>(BLOCK_M_rem, Nb, Nb);
  auto zero_out_tpp = SetZeroTPP<TGemmOut>(BLOCK_M, Nb, ldy);
  auto zero_buf_tpp = SetZeroTPP<TGemmOut>(BLOCK_M, Nb, Nb);
  auto zero_out_rem_tpp = SetZeroTPP<TGemmOut>(BLOCK_M_rem, Nb, ldy);
  auto zero_buf_rem_tpp = SetZeroTPP<TGemmOut>(BLOCK_M_rem, Nb, Nb);
  auto gelu_erf_fwd_tpp = GeluFwdTPP<Tout>(BLOCK_M, Nb, ldy, ldy);
  auto gelu_erf_fwd_rem_tpp = GeluFwdTPP<Tout>(BLOCK_M_rem, Nb, ldy, ldy);
  auto gelu_tanh_fwd_tpp = GeluTanhFwdTPP<Tout>(BLOCK_M, Nb, ldy, ldy);
  auto gelu_tanh_fwd_rem_tpp = GeluTanhFwdTPP<Tout>(BLOCK_M_rem, Nb, ldy, ldy);
  auto add_tpp = AddTPP<Tout>(BLOCK_M, Nb, ldy, ldy);
  auto add_rem_tpp = AddTPP<Tout>(BLOCK_M_rem, Nb, ldy, ldy);
  bool is_fusion_type_addrelated =
      fusion_type == FUSE_ADD || fusion_type == FUSE_ADD_ADD;
  auto post_ops_fn = [&](int m, int nc) {
    Tout* y_ptr = (Tout*)py[m][nc];
    Tout* tin0_ptr = is_fusion_type_addrelated ? (Tout*)pin0[m][nc] : nullptr;
    Tout* tin1_ptr = fusion_type == FUSE_ADD_ADD ? (Tout*)pin1[m][nc] : nullptr;
    if (fusion_type == FUSE_GELU_ERF) {
      gelu_erf_fwd_tpp(y_ptr, y_ptr);
    } else if (fusion_type == FUSE_ADD) {
      add_tpp(y_ptr, tin0_ptr, y_ptr);
    } else if (fusion_type == FUSE_ADD_ADD) {
      add_tpp(y_ptr, tin0_ptr, y_ptr);
      add_tpp(y_ptr, tin1_ptr, y_ptr);
    } else if (fusion_type == FUSE_GELU_TANH) {
      gelu_tanh_fwd_tpp(y_ptr, y_ptr);
    }
  };
  auto post_ops_rem_fn = [&](int m, int nc) {
    Tout* y_ptr = (Tout*)py[m][nc];
    Tout* tin0_ptr = (fusion_type == FUSE_ADD || fusion_type == FUSE_ADD_ADD)
        ? (Tout*)pin0[m][nc]
        : nullptr;
    Tout* tin1_ptr = fusion_type == FUSE_ADD_ADD ? (Tout*)pin1[m][nc] : nullptr;
    if (fusion_type == FUSE_GELU_ERF) {
      gelu_erf_fwd_rem_tpp(y_ptr, y_ptr);
    } else if (fusion_type == FUSE_ADD) {
      add_rem_tpp(y_ptr, tin0_ptr, y_ptr);
    } else if (fusion_type == FUSE_ADD_ADD) {
      add_rem_tpp(y_ptr, tin0_ptr, y_ptr);
      add_rem_tpp(y_ptr, tin1_ptr, y_ptr);
    } else if (fusion_type == FUSE_GELU_TANH) {
      gelu_tanh_fwd_rem_tpp(y_ptr, y_ptr);
    }
  };

  constexpr long MICRO_BLOCK_M = 8;
  product_dispatcher<
      std::tuple</*BLOCK_N*/ long, /*qw_type*/ int>,
      std::tuple<
          enumerate_dispatcher<long, 16, 32, 64, 128>,
          enumerate_dispatcher<int, QINT8, QINT4, NF4>>>::
      call(
          std::make_tuple(Nb, qw_type),
          [&](auto tuple) {
            auto BLOCK_N = std::get<0>(tuple);
            auto qw_type = std::get<1>(tuple);
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
                qw_type,
                quant_a_mode,
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
                qw_type,
                quant_a_mode,
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
                qw_type,
                quant_a_mode,
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
                qw_type,
                quant_a_mode,
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
                    float* scale_a = nullptr;
                    int32_t* zp_a = nullptr;
                    int32_t k_groups = -1;
                    int32_t quant_offset = kc / quant_block_multiple;
                    if constexpr (std::is_same<TComp, uint8_t>()) {
                      TLA_ASSERT(
                          !sym_quant,
                          "Calculation of uint8 does not support symmetric quant.");
                      if constexpr (quant_a_mode == QUANT_A_PER_TENSOR) {
                        scale_a = scales_a_ptr;
                        zp_a = zps_a_ptr;
                      } else if constexpr (
                          quant_a_mode == QUANT_A_PER_K_BLOCK) {
                        scale_a = scales_a_ptr + quant_offset;
                        zp_a = zps_a_ptr + quant_offset;
                      } else if constexpr (quant_a_mode == QUANT_A_PER_M) {
                        scale_a = scales_a_ptr + m;
                        zp_a = zps_a_ptr + m;
                        k_groups = 1;
                      } else {
                        scale_a =
                            scales_a_ptr + m * quant_k_blocks + quant_offset;
                        zp_a = zps_a_ptr + m * quant_k_blocks + quant_offset;
                        k_groups = quant_k_blocks;
                      }
                    }
                    TScale* scale_w = nullptr;
                    TZero* zp_w = nullptr;
                    if constexpr (quant_w_mode == QUANT_W_PER_CHANNEL) {
                      scale_w = pscales[nc][0];
                      if (!sym_quant) {
                        zp_w = pzps[nc][0];
                      }
                    } else {
                      scale_w = pscales[nc][quant_offset];
                      if (!sym_quant) {
                        zp_w = pzps[nc][quant_offset];
                      }
                    }
                    bool is_rem = (m + BLOCK_M > M);
                    TGemmOut* y_ptr = (TGemmOut*)py[m][nc];
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
                            scale_w,
                            zp_w,
                            y_ptr,
                            true,
                            scale_a,
                            zp_a,
                            k_groups);
                      } else {
                        dequant_gemm_no_prefetch_tpp(
                            x_ptr,
                            pw[nc][kc],
                            scale_w,
                            zp_w,
                            y_ptr,
                            true,
                            scale_a,
                            zp_a,
                            k_groups);
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
                            scale_w,
                            zp_w,
                            y_ptr,
                            false,
                            scale_a,
                            zp_a,
                            k_groups);
                        dequant_gemm_tpp.config();
                      } else {
                        dequant_gemm_no_prefetch_rem_tpp(
                            x_ptr,
                            pw[nc][kc],
                            scale_w,
                            zp_w,
                            y_ptr,
                            false,
                            scale_a,
                            zp_a,
                            k_groups);
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
                    auto y_out_ptr = py[m][nc];
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
                      float* scale_a = nullptr;
                      int32_t* zp_a = nullptr;
                      int32_t k_groups = -1;
                      int32_t quant_offset = kc / quant_block_multiple;
                      if constexpr (std::is_same<TComp, uint8_t>()) {
                        TLA_ASSERT(
                            !sym_quant,
                            "Calculation of uint8 does not support symmetric quant.");
                        if constexpr (quant_a_mode == QUANT_A_PER_TENSOR) {
                          scale_a = scales_a_ptr;
                          zp_a = zps_a_ptr;
                        } else if constexpr (
                            quant_a_mode == QUANT_A_PER_K_BLOCK) {
                          scale_a = scales_a_ptr + quant_offset;
                          zp_a = zps_a_ptr + quant_offset;
                        } else if constexpr (quant_a_mode == QUANT_A_PER_M) {
                          scale_a = scales_a_ptr + m;
                          zp_a = zps_a_ptr + m;
                          k_groups = 1;
                        } else {
                          scale_a =
                              scales_a_ptr + m * quant_k_blocks + quant_offset;
                          zp_a = zps_a_ptr + m * quant_k_blocks + quant_offset;
                          k_groups = quant_k_blocks;
                        }
                      }
                      TScale* scale_w = nullptr;
                      TZero* zp_w = nullptr;
                      if constexpr (quant_w_mode == QUANT_W_PER_CHANNEL) {
                        scale_w = pscales[nc][0];
                        if (!sym_quant) {
                          zp_w = pzps[nc][0];
                        }
                      } else {
                        scale_w = pscales[nc][quant_offset];
                        if (!sym_quant) {
                          zp_w = pzps[nc][quant_offset];
                        }
                      }
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
                              scale_w,
                              zp_w,
                              y_ptr,
                              true,
                              scale_a,
                              zp_a,
                              k_groups);
                        } else {
                          dequant_gemm_no_prefetch_tpp(
                              x_ptr,
                              pw[nc][kc],
                              scale_w,
                              zp_w,
                              y_ptr,
                              true,
                              scale_a,
                              zp_a,
                              k_groups);
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
                              scale_w,
                              zp_w,
                              y_ptr,
                              false,
                              scale_a,
                              zp_a,
                              k_groups);
                          dequant_gemm_tpp.config();
                        } else {
                          dequant_gemm_no_prefetch_rem_tpp(
                              x_ptr,
                              pw[nc][kc],
                              scale_w,
                              zp_w,
                              y_ptr,
                              false,
                              scale_a,
                              zp_a,
                              k_groups);
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

/**
 * @brief pack the weight in quantized format.
 * @param qw quantized weight with shape [N, K]
 * @param block_n block size along N, N % block_n == 0, block_n % 16 == 0
 * @param block_k block size along K, K % block_k == 0. block_k % 2 == 0 for
 * bf16 compute_dtype. false if activation is expected to be float32.
 */
at::Tensor qlinear_woq_pack(
    const at::Tensor& qw,
    int qw_type,
    size_t block_n,
    size_t block_k,
    int64_t lowp_mode) {
  TLA_ASSERT(qw.is_contiguous(), "qw must be contiguous");
  bool is_4bit_flag = is_4bit(qw_type);
  auto sizes = qw.sizes();
  auto N = sizes[0];
  auto K = is_4bit_flag ? sizes[1] * 2 : sizes[1];
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
  if (is_4bit_flag) {
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
    int qw_type,
    int64_t lowp_mode) {
  bool is_4bit_flag = is_4bit(qw_type);
  if (qw_packed.dim() == 4) {
    auto w_sizes = qw_packed.sizes();
    auto Nc = w_sizes[0];
    auto Nb = is_4bit_flag ? w_sizes[3] * 2 : w_sizes[3];
    auto Kc = w_sizes[1];
    auto Kb = w_sizes[2];
    auto N = Nc * Nb;
    auto K = Kc * Kb;
    const int N_GROUP_SIZE =
        lowp_mode != LOWP_MODE_INT8 ? get_n_group_size(Nb) : 16;
    if (is_4bit_flag) {
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

template <typename scalar_t>
inline scalar_t max_propagate_nan(scalar_t a, scalar_t b) {
  if (at::_isnan(a)) {
    return a;
  }
  return a > b ? a : b;
}

template <typename scalar_t>
inline scalar_t min_propagate_nan(scalar_t a, scalar_t b) {
  if (at::_isnan(a)) {
    return a;
  }
  return a < b ? a : b;
}

template <typename T>
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

template <>
void compute_int8_qparams_per_tensor<float>(
    const at::Tensor& t,
    float* scale,
    int32_t* zp) {
  auto in_ptr0 = t.data_ptr<float>();
  auto n = t.numel();
  auto K = t.size(-1);
  auto M = t.numel() / K;
  auto vecsize = at::vec::Vectorized<float>::size();
  auto compute_block = [&](float* in_ptr, int start, int end) {
    float min_val = std::numeric_limits<float>::infinity();
    float max_val = -std::numeric_limits<float>::infinity();
    auto min_vec = at::vec::Vectorized(min_val);
    auto max_vec = at::vec::Vectorized(max_val);
    int i1;
    for (i1 = start; i1 < end / n * n; i1 += vecsize) {
      auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr + i1, vecsize);
      min_vec = at::vec::minimum(min_vec, tmp0);
      max_vec = at::vec::maximum(tmp0, max_vec);
    }
    for (; i1 < end; i1++) {
      auto tmp0 = in_ptr[i1];
      min_val = std::min(min_val, tmp0);
      max_val = std::max(tmp0, max_val);
    }
    min_val = min_propagate_nan(
        min_val,
        at::vec::vec_reduce_all<float>(
            [](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) {
              return at::vec::minimum(x, y);
            },
            min_vec));
    max_val = max_propagate_nan(
        max_val,
        at::vec::vec_reduce_all<float>(
            [](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) {
              return at::vec::maximum(x, y);
            },
            max_vec));
    return std::make_pair(min_val, max_val);
  };
  if (n > QUANT_A_THRESHOLD) {
    int num_threads = omp_get_max_threads();
    int vec_per_thread = std::ceil((float)n / vecsize / num_threads);
    int thread_used = std::ceil((float)n / vecsize / vec_per_thread);
    float min_vals[thread_used];
    float max_vals[thread_used];
#pragma omp parallel for
    for (int i0 = 0; i0 < n; i0 += vec_per_thread * vecsize) {
      auto vec_start = i0;
      auto vec_end = std::min(i0 + vec_per_thread * vecsize, (int)n);
      auto [min_val, max_val] = compute_block(in_ptr0, vec_start, vec_end);
      min_vals[i0 / vec_per_thread / vecsize] = min_val;
      max_vals[i0 / vec_per_thread / vecsize] = max_val;
    }
    auto min_elem_ptr = std::min_element(min_vals, min_vals + thread_used);
    auto max_elem_ptr = std::max_element(max_vals, max_vals + thread_used);
    *scale = (*max_elem_ptr - *min_elem_ptr) / 255.0f;
    *zp = (int32_t)(-std::nearbyint(*min_elem_ptr / *scale));
  } else {
    auto [min_val, max_val] = compute_block(in_ptr0, 0, n);
    *scale = (max_val - min_val) / 255.0f;
    *zp = (int32_t)(-std::nearbyint(min_val / *scale));
  }
}

template <>
void compute_int8_qparams_per_tensor<bfloat16>(
    const at::Tensor& t,
    float* scale,
    int32_t* zp) {
  auto in_ptr0 = t.data_ptr<at::BFloat16>();
  auto n = t.numel();
  auto K = t.size(-1);
  auto M = t.numel() / K;
  auto vecsize = at::vec::Vectorized<float>::size();
  auto compute_block = [&](at::BFloat16* in_ptr, int start, int end) {
    float min_val = std::numeric_limits<float>::infinity();
    float max_val = -std::numeric_limits<float>::infinity();
    auto min_vec = at::vec::Vectorized(min_val);
    auto max_vec = at::vec::Vectorized(max_val);
    int i1;
    for (i1 = start; i1 < end / n * n; i1 += vecsize) {
      auto tmp0 =
          at::vec::Vectorized<at::BFloat16>::loadu(in_ptr + i1, vecsize);
      at::vec::Vectorized<float> res_vec1(0);
      at::vec::Vectorized<float> res_vec2(0);
      std::tie(res_vec1, res_vec2) = at::vec::convert_bfloat16_float(tmp0);
      min_vec = at::vec::minimum(min_vec, res_vec1);
      max_vec = at::vec::maximum(res_vec1, max_vec);
    }
    for (; i1 < end; i1++) {
      auto tmp0 = in_ptr[i1];
      min_val = std::min(min_val, (float)tmp0);
      max_val = std::max((float)tmp0, max_val);
    }
    min_val = min_propagate_nan(
        min_val,
        at::vec::vec_reduce_all<float>(
            [](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) {
              return at::vec::minimum(x, y);
            },
            min_vec));
    max_val = max_propagate_nan(
        max_val,
        at::vec::vec_reduce_all<float>(
            [](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) {
              return at::vec::maximum(x, y);
            },
            max_vec));
    return std::make_pair(min_val, max_val);
  };
  if (n > QUANT_A_THRESHOLD) {
    int num_threads = omp_get_max_threads();
    int vec_per_thread = std::ceil((float)n / vecsize / num_threads);
    int thread_used = std::ceil((float)n / vecsize / vec_per_thread);
    float min_vals[thread_used];
    float max_vals[thread_used];
#pragma omp parallel for
    for (int i0 = 0; i0 < n; i0 += vec_per_thread * vecsize) {
      auto vec_start = i0;
      auto vec_end = std::min(i0 + vec_per_thread * vecsize, (int)n);
      auto [min_val, max_val] = compute_block(in_ptr0, vec_start, vec_end);
      min_vals[i0 / vec_per_thread / vecsize] = min_val;
      max_vals[i0 / vec_per_thread / vecsize] = max_val;
    }
    auto min_elem_ptr = std::min_element(min_vals, min_vals + thread_used);
    auto max_elem_ptr = std::max_element(max_vals, max_vals + thread_used);
    *scale = (*max_elem_ptr - *min_elem_ptr) / 255.0f;
    *zp = (int32_t)(-std::nearbyint(*min_elem_ptr / *scale));
  } else {
    auto [min_val, max_val] = compute_block(in_ptr0, 0, n);
    *scale = (max_val - min_val) / 255.0f;
    *zp = (int32_t)(-std::nearbyint(min_val / *scale));
  }
}

template <typename T>
std::pair<at::Tensor, at::Tensor> compute_int8_qparams_per_block(
    const at::Tensor& t,
    int quant_block_k,
    int quant_a_mode) {
  auto K = t.size(-1);
  auto n = t.numel();
  auto M = n / K;
  auto t_reshape = t.reshape({M, K});
  if (quant_a_mode == QUANT_A_PER_M) {
    auto grouped_min = std::get<0>(t_reshape.min(-1));
    auto grouped_max = std::get<0>(t_reshape.max(-1));
    auto zeros = at::zeros_like(grouped_min);
    auto min = at::minimum(grouped_min, zeros);
    auto max = at::maximum(grouped_max, zeros);
    auto scales = (max - min) / 255;
    auto zps = -at::round(min / scales);
    return std::make_pair<at::Tensor&&, at::Tensor&&>(
        std::move(scales.to(c10::kFloat)), std::move(zps.to(c10::kInt)));
  }
  int k_rem = K % quant_block_k;
  int block_k = quant_block_k;
  auto grouped =
      t_reshape
          .index({at::indexing::Slice(), at::indexing::Slice(0, K - k_rem)})
          .view({M, K / quant_block_k, quant_block_k});
  at::Tensor grouped_min, grouped_max;
  if (quant_a_mode == QUANT_A_PER_K_BLOCK) {
    grouped_min = std::get<0>(std::get<0>(grouped.min(-1)).min(0));
    grouped_max = std::get<0>(std::get<0>(grouped.max(-1)).max(0));
  } else {
    grouped_min = std::get<0>(grouped.min(-1));
    grouped_max = std::get<0>(grouped.max(-1));
  }
  auto zeros = at::zeros_like(grouped_min);
  auto min = at::minimum(grouped_min, zeros);
  auto max = at::maximum(grouped_max, zeros);
  auto scales = (max - min) / 255.0f;
  auto zps = -at::round(min / scales);
  if (k_rem) {
    auto grouped_rem =
        t_reshape
            .index({at::indexing::Slice(), at::indexing::Slice(K - k_rem, K)})
            .view({M, 1, k_rem});
    at::Tensor grouped_rem_min, grouped_rem_max;
    if (quant_a_mode == QUANT_A_PER_K_BLOCK) {
      grouped_rem_min = std::get<0>(std::get<0>(grouped_rem.min(-1)).min(0));
      grouped_rem_max = std::get<0>(std::get<0>(grouped_rem.max(-1)).max(0));
    } else {
      grouped_rem_min = std::get<0>(grouped_rem.min(-1));
      grouped_rem_max = std::get<0>(grouped_rem.max(-1));
    }
    auto min_rem = at::minimum(grouped_rem_min, at::tensor({0}));
    auto max_rem = at::maximum(grouped_rem_max, at::tensor({0}));
    auto scales_rem = (max_rem - min_rem) / 255;
    auto zps_rem = -at::round(min_rem / scales_rem);
    scales = at::cat({scales, scales_rem}, -1).contiguous();
    zps = at::cat({zps, zps_rem}, -1).contiguous();
  }
  return std::make_pair<at::Tensor&&, at::Tensor&&>(
      std::move(scales.to(c10::kFloat)), std::move(zps.to(c10::kInt)));
}

template <>
std::pair<at::Tensor, at::Tensor> compute_int8_qparams_per_block<bfloat16>(
    const at::Tensor& t,
    int quant_block_k,
    int quant_a_mode) {
  auto in_ptr = t.data_ptr<at::BFloat16>();
  int K = t.size(-1);
  int n = t.numel();
  int M = n / K;
  int Kc = (K + quant_block_k - 1) / quant_block_k;
  auto vecsize = at::vec::Vectorized<float>::size();
  at::Tensor scales, zps;
  if (quant_a_mode == QUANT_A_PER_K_BLOCK) {
    scales = at::empty({Kc}, t.options().dtype(at::kFloat));
    zps = at::empty({Kc}, t.options().dtype(at::kInt));
  } else if (quant_a_mode == QUANT_A_PER_M) {
    scales = at::empty({M}, t.options().dtype(at::kFloat));
    zps = at::empty({M}, t.options().dtype(at::kInt));
  } else {
    scales = at::empty({M, Kc}, t.options().dtype(at::kFloat));
    zps = at::empty({M, Kc}, t.options().dtype(at::kInt));
  }
  auto scales_ptr = scales.data_ptr<float>();
  auto zps_ptr = zps.data_ptr<int32_t>();
  auto compute_minmax = [vecsize, scales_ptr, zps_ptr](
                            at::BFloat16* ptr,
                            int M,
                            int K,
                            int scale_offset,
                            int zp_offset,
                            int ld) {
    float min_val = std::numeric_limits<float>::infinity();
    float max_val = -std::numeric_limits<float>::infinity();
    auto in_ptr_ = ptr;
    auto min_vec = at::vec::Vectorized(min_val);
    auto max_vec = at::vec::Vectorized(max_val);
    for (int m = 0; m < M; m++) {
      auto in_ptr0 = in_ptr_;
      int k;
      for (k = 0; k < K / vecsize * vecsize; k += vecsize) {
        auto tmp0 = at::vec::Vectorized<at::BFloat16>::loadu(in_ptr0, vecsize);
        at::vec::Vectorized<float> res_vec1(0);
        at::vec::Vectorized<float> res_vec2(0);
        std::tie(res_vec1, res_vec2) = at::vec::convert_bfloat16_float(tmp0);
        auto tmp1 = res_vec1;
        min_vec = at::vec::minimum(min_vec, tmp1);
        max_vec = at::vec::maximum(tmp1, max_vec);
        in_ptr0 += vecsize;
      }
      for (; k < K; k++) {
        auto tmp0 = in_ptr0[k];
        min_val = std::min(min_val, (float)tmp0);
        max_val = std::max(max_val, (float)tmp0);
      }
      in_ptr_ += ld;
    }
    min_val = min_propagate_nan(
        min_val,
        at::vec::vec_reduce_all<float>(
            [](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) {
              return at::vec::minimum(x, y);
            },
            min_vec));
    max_val = max_propagate_nan(
        max_val,
        at::vec::vec_reduce_all<float>(
            [](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) {
              return at::vec::maximum(x, y);
            },
            max_vec));
    scales_ptr[scale_offset] = (max_val - min_val) / 255.0f;
    zps_ptr[zp_offset] =
        (int32_t)(-std::nearbyint(min_val / scales_ptr[scale_offset]));
  };
  if (quant_a_mode == QUANT_A_PER_K_BLOCK) {
#pragma omp parallel for
    for (int kc = 0; kc < Kc; kc++) {
      int offset = kc * quant_block_k;
      int block_k = std::min(quant_block_k, K - offset);
      compute_minmax(in_ptr + offset, M, block_k, kc, kc, K);
    }
  } else if (quant_a_mode == QUANT_A_PER_M) {
#pragma omp parallel for
    for (int m = 0; m < M; m++) {
      int offset = m * K;
      compute_minmax(in_ptr + offset, 1, K, m, m, K);
    }
  } else {
#pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++) {
      for (int kc = 0; kc < Kc; kc++) {
        auto in_ptr0 = in_ptr + m * K + kc * quant_block_k;
        auto scale_offset = m * Kc + kc;
        auto zp_offset = m * Kc + kc;
        int block_k = std::min(quant_block_k, K - kc * quant_block_k);
        compute_minmax(in_ptr0, 1, block_k, scale_offset, zp_offset, K);
      }
    }
  }
  return std::make_pair<at::Tensor&&, at::Tensor&&>(
      std::move(scales), std::move(zps));
}

template <>
std::pair<at::Tensor, at::Tensor> compute_int8_qparams_per_block<float>(
    const at::Tensor& t,
    int quant_block_k,
    int quant_a_mode) {
  auto in_ptr = t.data_ptr<float>();
  int K = t.size(-1);
  int n = t.numel();
  int M = n / K;
  int Kc = (K + quant_block_k - 1) / quant_block_k;
  auto vecsize = at::vec::Vectorized<float>::size();
  at::Tensor scales, zps;
  if (quant_a_mode == QUANT_A_PER_K_BLOCK) {
    scales = at::empty({Kc}, t.options().dtype(at::kFloat));
    zps = at::empty({Kc}, t.options().dtype(at::kInt));
  } else if (quant_a_mode == QUANT_A_PER_M) {
    scales = at::empty({M}, t.options().dtype(at::kFloat));
    zps = at::empty({M}, t.options().dtype(at::kInt));
  } else {
    scales = at::empty({M, Kc}, t.options().dtype(at::kFloat));
    zps = at::empty({M, Kc}, t.options().dtype(at::kInt));
  }
  auto scales_ptr = scales.data_ptr<float>();
  auto zps_ptr = zps.data_ptr<int32_t>();
  auto compute_minmax = [vecsize, scales_ptr, zps_ptr](
                            float* ptr,
                            int M,
                            int K,
                            int scale_offset,
                            int zp_offset,
                            int ld) {
    float min_val = std::numeric_limits<float>::infinity();
    float max_val = -std::numeric_limits<float>::infinity();
    auto in_ptr_ = ptr;
    auto min_vec = at::vec::Vectorized(min_val);
    auto max_vec = at::vec::Vectorized(max_val);
    for (int m = 0; m < M; m++) {
      auto in_ptr0 = in_ptr_;
      int k;
      for (k = 0; k < K / vecsize * vecsize; k += vecsize) {
        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0, vecsize);
        min_vec = at::vec::minimum(min_vec, tmp0);
        max_vec = at::vec::maximum(tmp0, max_vec);
        in_ptr0 += vecsize;
      }
      for (; k < K; k++) {
        auto tmp0 = in_ptr0[k];
        min_val = std::min(min_val, tmp0);
        max_val = std::max(max_val, tmp0);
      }
      in_ptr_ += ld;
    }
    min_val = min_propagate_nan(
        min_val,
        at::vec::vec_reduce_all<float>(
            [](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) {
              return at::vec::minimum(x, y);
            },
            min_vec));
    max_val = max_propagate_nan(
        max_val,
        at::vec::vec_reduce_all<float>(
            [](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) {
              return at::vec::maximum(x, y);
            },
            max_vec));
    scales_ptr[scale_offset] = (max_val - min_val) / 255.0f;
    zps_ptr[zp_offset] =
        (int32_t)(-std::nearbyint(min_val / scales_ptr[scale_offset]));
  };
  if (quant_a_mode == QUANT_A_PER_K_BLOCK) {
#pragma omp parallel for
    for (int kc = 0; kc < Kc; kc++) {
      int offset = kc * quant_block_k;
      int block_k = std::min(quant_block_k, K - offset);
      compute_minmax(in_ptr + offset, M, block_k, kc, kc, K);
    }
  } else if (quant_a_mode == QUANT_A_PER_M) {
#pragma omp parallel for
    for (int m = 0; m < M; m++) {
      int offset = m * K;
      compute_minmax(in_ptr + offset, 1, K, m, m, K);
    }
  } else {
#pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++) {
      for (int kc = 0; kc < Kc; kc++) {
        auto in_ptr0 = in_ptr + m * K + kc * quant_block_k;
        auto scale_offset = m * Kc + kc;
        auto zp_offset = m * Kc + kc;
        int block_k = std::min(quant_block_k, K - kc * quant_block_k);
        compute_minmax(in_ptr0, 1, block_k, scale_offset, zp_offset, K);
      }
    }
  }
  return std::make_pair<at::Tensor&&, at::Tensor&&>(
      std::move(scales), std::move(zps));
}

template <typename T>
at::Tensor quantize_per_tensor(const at::Tensor& t, float scale, int32_t zp) {
  // TODO(jgong5): optimize me
  auto t_q = t / scale + zp;
  t_q = at::clamp(at::round(t_q), 0, 255);
  return t_q.to(at::kByte);
}

template <>
at::Tensor quantize_per_tensor<float>(
    const at::Tensor& t,
    float scale,
    int32_t zp) {
#ifdef __AVX512F__
  at::Tensor out = at::empty_like(t, at::kByte);
  auto in_ptr0 = t.data_ptr<float>();
  auto out_ptr0 = out.data_ptr<uint8_t>();
  auto n = t.numel();
  auto K = t.size(-1);
  auto M = t.numel() / K;
  auto vecsize = at::vec::Vectorized<float>::size();
  auto quantize_block =
      [vecsize, scale, zp](
          float* in_ptr, int start, int end, uint8_t* out_ptr) {
        int i1;
        for (i1 = start; i1 < end / vecsize * vecsize; i1 += vecsize) {
          auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr + i1, vecsize);
          auto tmp1 =
              tmp0 / at::vec::Vectorized<float>(static_cast<float>(scale));
          auto tmp2 = tmp1 + at::vec::Vectorized<float>(static_cast<float>(zp));
          auto tmp3 = tmp2.round();
          auto tmp4 = (tmp3);
          auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.0));
          auto tmp6 = at::vec::maximum(tmp4, tmp5);
          auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(255.0));
          auto tmp8 = at::vec::minimum(tmp6, tmp7);
          auto tmp9 = (tmp8);
          auto tmp10 = at::vec::convert_float_to_int8<uint8_t>(tmp9);
          tmp10.store(out_ptr + i1, vecsize);
        }
        for (; i1 < end; i1++) {
          auto tmp0 = in_ptr[i1];
          auto tmp1 = tmp0 / static_cast<float>(scale);
          auto tmp2 = tmp1 + static_cast<float>(zp);
          auto tmp3 = std::nearbyint(tmp2);
          auto tmp4 = static_cast<float>(tmp3);
          auto tmp5 = static_cast<float>(0.0);
          auto tmp6 = 0;
          if (at::_isnan(tmp4)) {
            tmp6 = tmp4;
          }
          tmp6 = tmp4 > tmp5 ? tmp4 : tmp5;
          auto tmp7 = static_cast<float>(255.0);
          auto tmp8 = 0;
          if (at::_isnan(tmp6)) {
            tmp8 = tmp6;
          }
          tmp8 = tmp6 < tmp7 ? tmp6 : tmp7;
          auto tmp9 = static_cast<float>(tmp8);
          auto tmp10 = static_cast<unsigned char>(tmp9);
          out_ptr[i1] = tmp10;
        }
      };
  if (n > QUANT_A_THRESHOLD) {
    int num_threads = omp_get_max_threads();
    int vec_per_thread = std::ceil((float)n / vecsize / num_threads);
#pragma omp parallel for
    for (int i0 = 0; i0 < n; i0 += vec_per_thread * vecsize) {
      auto vec_start = i0;
      auto vec_end = std::min(i0 + vec_per_thread * vecsize, (int)n);
      quantize_block(in_ptr0, vec_start, vec_end, out_ptr0);
    }
  } else {
    quantize_block(in_ptr0, 0, n, out_ptr0);
  }
  return out;
#else
  return at::quantize_per_tensor(t, scale, zp, c10::kQUInt8);
#endif
}

template <>
at::Tensor quantize_per_tensor<bfloat16>(
    const at::Tensor& t,
    float scale,
    int32_t zp) {
#ifdef __AVX512F__
  at::Tensor out = at::empty_like(t, at::kByte);
  auto in_ptr0 = t.data_ptr<at::BFloat16>();
  auto out_ptr0 = out.data_ptr<uint8_t>();
  auto n = t.numel();
  auto K = t.size(-1);
  auto M = t.numel() / K;
  auto vecsize = at::vec::Vectorized<float>::size();
  auto quantize_block =
      [vecsize, scale, zp](
          at::BFloat16* in_ptr, int start, int end, uint8_t* out_ptr) {
        int i1;
        for (i1 = start; i1 < end / vecsize * vecsize; i1 += vecsize) {
          auto tmp0 =
              at::vec::Vectorized<at::BFloat16>::loadu(in_ptr + i1, vecsize);
          at::vec::Vectorized<float> res_vec1(0);
          at::vec::Vectorized<float> res_vec2(0);
          std::tie(res_vec1, res_vec2) = at::vec::convert_bfloat16_float(tmp0);
          auto tmp1 = res_vec1;
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
          auto tmp13 = at::vec::convert_float_to_int8<uint8_t>(tmp12);
          tmp13.store(out_ptr + i1, vecsize);
        }
        for (; i1 < end; i1++) {
          auto tmp0 = in_ptr[i1];
          auto tmp1 = static_cast<float>(tmp0);
          auto tmp2 = static_cast<float>(scale);
          auto tmp3 = tmp1 / tmp2;
          auto tmp4 = static_cast<float>(zp);
          auto tmp5 = tmp3 + tmp4;
          auto tmp6 = std::nearbyint(tmp5);
          auto tmp7 = static_cast<float>(tmp6);
          auto tmp8 = static_cast<float>(0.0);
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
          auto tmp12 = static_cast<float>(tmp11);
          auto tmp13 = static_cast<unsigned char>(tmp12);
          out_ptr[i1] = tmp13;
        }
      };
  if (n > QUANT_A_THRESHOLD) {
    auto num_threads = omp_get_max_threads();
    int vec_per_thread = std::ceil((float)n / vecsize / num_threads);
#pragma omp parallel for
    for (int i0 = 0; i0 < n; i0 += vec_per_thread * vecsize) {
      auto vec_start = i0;
      auto vec_end = std::min(i0 + vec_per_thread * vecsize, (int)n);
      quantize_block(in_ptr0, vec_start, vec_end, out_ptr0);
    }
  } else {
    quantize_block(in_ptr0, 0, n, out_ptr0);
  }
  return out;
#else
  return at::quantize_per_tensor(t.to(c10::kFloat), scale, zp, c10::kQUInt8);
#endif
}

template <typename T>
at::Tensor quantize_per_block(
    const at::Tensor& t,
    const at::Tensor& scale,
    const at::Tensor& zp,
    int quant_block_k,
    int quant_a_mode) {
  auto K = t.size(-1);
  auto n = t.numel();
  auto M = n / K;
  auto k_rem = K % quant_block_k;
  bool has_rem = k_rem != 0;
  auto K_padded = has_rem ? K + quant_block_k - k_rem : K;
  auto t_padded = has_rem
      ? at::cat(
            {t.reshape({M, K}),
             at::zeros({M, quant_block_k - k_rem}, t.options())},
            -1)
      : t;
  auto grouped = t_padded.view({-1, K_padded / quant_block_k, quant_block_k});
  at::Tensor out;
  if (quant_a_mode == QUANT_A_PER_K_BLOCK) {
    out = at::clamp(
        at::round(grouped / scale.unsqueeze(1)) + zp.unsqueeze(1), 0, 255);
  } else if (quant_a_mode == QUANT_A_PER_M) {
    out = at::clamp(
        at::round(grouped / scale.unsqueeze(1).unsqueeze(2)) +
            zp.unsqueeze(1).unsqueeze(2),
        0,
        255);
  } else {
    out = at::clamp(
        at::round(grouped / scale.unsqueeze(-1)) + zp.unsqueeze(-1), 0, 255);
  }
  out = out.view({-1, K_padded})
            .index({at::indexing::Slice(), at::indexing::Slice(0, K)});
  return out.to(at::kByte).contiguous();
}

template <>
at::Tensor quantize_per_block<bfloat16>(
    const at::Tensor& t,
    const at::Tensor& scale,
    const at::Tensor& zp,
    int quant_block_k,
    int quant_a_mode) {
  int K = t.size(-1);
  int n = t.numel();
  int M = n / K;
  at::Tensor out = at::empty_like(t, at::kByte);
  int Kc = (K + quant_block_k - 1) / quant_block_k;
  auto scale_ptr = scale.data_ptr<float>();
  auto zp_ptr = zp.data_ptr<int32_t>();
  auto in_ptr = t.data_ptr<at::BFloat16>();
  auto out_ptr = out.data_ptr<uint8_t>();
  auto vecsize = at::vec::Vectorized<float>::size();
  auto quantize_block = [vecsize](
                            at::BFloat16* in_ptr,
                            uint8_t* out_ptr,
                            int block_k,
                            float scale_,
                            int zp_) {
    int k;
    for (k = 0; k < block_k / vecsize * vecsize; k += vecsize) {
      auto in_ptr0 = in_ptr + k;
      auto out_ptr0 = out_ptr + k;
      auto tmp0 = at::vec::Vectorized<at::BFloat16>::loadu(in_ptr0, vecsize);
      at::vec::Vectorized<float> res_vec1(0);
      at::vec::Vectorized<float> res_vec2(0);
      std::tie(res_vec1, res_vec2) = at::vec::convert_bfloat16_float(tmp0);
      auto tmp1 = res_vec1;
      auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(scale_));
      auto tmp3 = tmp1 / tmp2;
      auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(zp_));
      auto tmp5 = tmp3 + tmp4;
      auto tmp6 = tmp5.round();
      auto tmp7 = (tmp6);
      auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.0));
      auto tmp9 = at::vec::maximum(tmp7, tmp8);
      auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(255.0));
      auto tmp11 = at::vec::minimum(tmp9, tmp10);
      auto tmp12 = (tmp11);
      auto tmp13 = at::vec::convert_float_to_int8<uint8_t>(tmp12);
      tmp13.store(out_ptr0, vecsize);
    }
    for (; k < block_k; k++) {
      auto tmp0 = in_ptr[k];
      auto tmp1 = static_cast<float>(tmp0);
      auto tmp2 = static_cast<float>(scale_);
      auto tmp3 = tmp1 / tmp2;
      auto tmp4 = static_cast<float>(zp_);
      auto tmp5 = tmp3 + tmp4;
      auto tmp6 = std::nearbyint(tmp5);
      auto tmp7 = static_cast<float>(tmp6);
      auto tmp8 = static_cast<float>(0.0);
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
      auto tmp12 = static_cast<float>(tmp11);
      auto tmp13 = static_cast<unsigned char>(tmp12);
      out_ptr[k] = tmp13;
    }
  };
  if (quant_a_mode == QUANT_A_PER_K_BLOCK) {
#pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++) {
      for (int kc = 0; kc < Kc; kc++) {
        auto in_ptr0 = in_ptr + m * K + kc * quant_block_k;
        auto out_ptr0 = out_ptr + m * K + kc * quant_block_k;
        auto scale_ = scale_ptr[kc];
        auto zp_ = zp_ptr[kc];
        int block_k = std::min(quant_block_k, (int)K - kc * quant_block_k);
        quantize_block(in_ptr0, out_ptr0, block_k, scale_, zp_);
      }
    }
  } else if (quant_a_mode == QUANT_A_PER_M) {
#pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++) {
      for (int kc = 0; kc < Kc; kc++) {
        auto in_ptr0 = in_ptr + m * K + kc * quant_block_k;
        auto out_ptr0 = out_ptr + m * K + kc * quant_block_k;
        auto scale_ = scale_ptr[m];
        auto zp_ = zp_ptr[m];
        int block_k = std::min(quant_block_k, (int)K - kc * quant_block_k);
        quantize_block(in_ptr0, out_ptr0, block_k, scale_, zp_);
      }
    }
  } else {
#pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++) {
      for (int kc = 0; kc < Kc; kc++) {
        auto in_ptr0 = in_ptr + m * K + kc * quant_block_k;
        auto out_ptr0 = out_ptr + m * K + kc * quant_block_k;
        auto scale_ = scale_ptr[m * Kc + kc];
        auto zp_ = zp_ptr[m * Kc + kc];
        int block_k = std::min(quant_block_k, (int)K - kc * quant_block_k);
        quantize_block(in_ptr0, out_ptr0, block_k, scale_, zp_);
      }
    }
  }
  return out;
}

template <>
at::Tensor quantize_per_block<float>(
    const at::Tensor& t,
    const at::Tensor& scale,
    const at::Tensor& zp,
    int quant_block_k,
    int quant_a_mode) {
  int K = t.size(-1);
  int n = t.numel();
  int M = n / K;
  at::Tensor out = at::empty_like(t, at::kByte);
  int Kc = (K + quant_block_k - 1) / quant_block_k;
  auto scale_ptr = scale.data_ptr<float>();
  auto zp_ptr = zp.data_ptr<int32_t>();
  auto in_ptr = t.data_ptr<float>();
  auto out_ptr = out.data_ptr<uint8_t>();
  auto vecsize = at::vec::Vectorized<float>::size();
  auto quantize_block =
      [vecsize](
          float* in_ptr, uint8_t* out_ptr, int block_k, float scale_, int zp_) {
        int k;
        for (k = 0; k < block_k / vecsize * vecsize; k += vecsize) {
          auto in_ptr0 = in_ptr + k;
          auto out_ptr0 = out_ptr + k;
          auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0, vecsize);
          auto tmp1 =
              tmp0 / at::vec::Vectorized<float>(static_cast<float>(scale_));
          auto tmp2 =
              tmp1 + at::vec::Vectorized<float>(static_cast<float>(zp_));
          auto tmp3 = tmp2.round();
          auto tmp4 = (tmp3);
          auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.0));
          auto tmp6 = at::vec::maximum(tmp4, tmp5);
          auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(255.0));
          auto tmp8 = at::vec::minimum(tmp6, tmp7);
          auto tmp9 = (tmp8);
          auto tmp10 = at::vec::convert_float_to_int8<uint8_t>(tmp9);
          tmp10.store(out_ptr0, vecsize);
        }
        for (; k < block_k; k++) {
          auto tmp0 = in_ptr[k];
          auto tmp1 = tmp0 / static_cast<float>(scale_);
          auto tmp2 = tmp1 + static_cast<float>(zp_);
          auto tmp3 = std::nearbyint(tmp2);
          auto tmp4 = static_cast<float>(tmp3);
          auto tmp5 = static_cast<float>(0.0);
          auto tmp6 = 0;
          if (at::_isnan(tmp4)) {
            tmp6 = tmp4;
          }
          tmp6 = tmp4 > tmp5 ? tmp4 : tmp5;
          auto tmp7 = static_cast<float>(255.0);
          auto tmp8 = 0;
          if (at::_isnan(tmp6)) {
            tmp8 = tmp6;
          }
          tmp8 = tmp6 < tmp7 ? tmp6 : tmp7;
          auto tmp9 = static_cast<float>(tmp8);
          auto tmp10 = static_cast<unsigned char>(tmp9);
          out_ptr[k] = tmp10;
        }
      };
  if (quant_a_mode == QUANT_A_PER_K_BLOCK) {
#pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++) {
      for (int kc = 0; kc < Kc; kc++) {
        auto in_ptr0 = in_ptr + m * K + kc * quant_block_k;
        auto out_ptr0 = out_ptr + m * K + kc * quant_block_k;
        auto scale_ = scale_ptr[kc];
        auto zp_ = zp_ptr[kc];
        int block_k = std::min(quant_block_k, (int)K - kc * quant_block_k);
        quantize_block(in_ptr0, out_ptr0, block_k, scale_, zp_);
      }
    }
  } else if (quant_a_mode == QUANT_A_PER_M) {
#pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++) {
      for (int kc = 0; kc < Kc; kc++) {
        auto in_ptr0 = in_ptr + m * K + kc * quant_block_k;
        auto out_ptr0 = out_ptr + m * K + kc * quant_block_k;
        auto scale_ = scale_ptr[m];
        auto zp_ = zp_ptr[m];
        int block_k = std::min(quant_block_k, (int)K - kc * quant_block_k);
        quantize_block(in_ptr0, out_ptr0, block_k, scale_, zp_);
      }
    }
  } else {
#pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++) {
      for (int kc = 0; kc < Kc; kc++) {
        auto in_ptr0 = in_ptr + m * K + kc * quant_block_k;
        auto out_ptr0 = out_ptr + m * K + kc * quant_block_k;
        auto scale_ = scale_ptr[m * Kc + kc];
        auto zp_ = zp_ptr[m * Kc + kc];
        int block_k = std::min(quant_block_k, (int)K - kc * quant_block_k);
        quantize_block(in_ptr0, out_ptr0, block_k, scale_, zp_);
      }
    }
  }
  return out;
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
    const int qw_type,
    int64_t lowp_mode,
    int64_t fusion_type,
    const TensorList& others_list,
    int64_t quant_a_mode = -1,
    int64_t quant_w_mode = 0,
    int64_t quant_block_k = 0) {
  const int64_t k_splits = 0;
  // int8_idx is only valid with zp_list when lowp_mode == LOWP_MODE_INT8
  constexpr size_t fp32_idx = 0, fp16_idx = 1, bf16_idx = 2, int8_idx = 3;
  auto biases = bias_list.empty()
      ? TensorList({at::Tensor(), at::Tensor(), at::Tensor()})
      : bias_list;
  const bool is_4bit_flag = is_4bit(qw_type);
  const bool sym_quant = is_sym_quant(qw_type);
  if (qw.dim() == 4) {
    auto w_sizes = qw.sizes();
    auto K = x.size(-1);
    auto M = x.numel() / K;
    auto N = w_sizes[0] * w_sizes[3];
    if (is_4bit_flag) {
      N *= 2;
    }
    auto out_sizes = x.sizes().vec();
    out_sizes.back() = N;
    auto y = at::empty(out_sizes, x.options());
    product_dispatcher<
        std::tuple<at::ScalarType, long>,
        std::tuple<
            enumerate_dispatcher<
                at::ScalarType,
                at::kFloat,
                at::kBFloat16,
                at::kHalf>,
            range_dispatcher<long, 0, 1>>>::
        call(
            std::make_tuple(x.scalar_type(), quant_w_mode),
            [&](auto tuple) {
              auto act_dtype = std::get<0>(tuple);
              auto quant_w_mode_ = std::get<1>(tuple);
              using act_type =
                  typename c10::impl::ScalarTypeToCPPType<act_dtype>::type;
              auto try_compute_in_half = [&]() {
#ifdef __AVX512FP16__
                if (sym_quant) {
                  qlinear_woq_affine_impl<
                      act_type,
                      half,
                      /*TGemmOut*/ half,
                      act_type,
                      half,
                      half,
                      UNQUANT_A,
                      quant_w_mode_>(
                      x,
                      qw,
                      scales_list[fp16_idx],
                      biases[fp16_idx],
                      y,
                      qw_type,
                      k_splits,
                      fusion_type,
                      others_list,
                      quant_block_k);
                } else {
                  qlinear_woq_affine_impl<
                      act_type,
                      half,
                      /*TGemmOut*/ half,
                      act_type,
                      half,
                      half,
                      UNQUANT_A,
                      quant_w_mode_>(
                      x,
                      qw,
                      scales_list[fp16_idx],
                      biases[fp16_idx],
                      y,
                      qw_type,
                      k_splits,
                      fusion_type,
                      others_list,
                      quant_block_k,
                      zp_list[fp16_idx]);
                }
#else
                if (sym_quant) {
                  qlinear_woq_affine_impl<
                      act_type,
                      float,
                      /*TGemmOut*/ float,
                      act_type,
                      float,
                      float,
                      UNQUANT_A,
                      quant_w_mode_>(
                      x,
                      qw,
                      scales_list[fp32_idx],
                      biases[fp32_idx],
                      y,
                      qw_type,
                      k_splits,
                      fusion_type,
                      others_list,
                      quant_block_k);
                } else {
                  qlinear_woq_affine_impl<
                      act_type,
                      float,
                      /*TGemmOut*/ float,
                      act_type,
                      float,
                      float,
                      UNQUANT_A,
                      quant_w_mode_>(
                      x,
                      qw,
                      scales_list[fp32_idx],
                      biases[fp32_idx],
                      y,
                      qw_type,
                      k_splits,
                      fusion_type,
                      others_list,
                      quant_block_k,
                      zp_list[fp32_idx]);
                }
#endif
              };
              if (lowp_mode == LOWP_MODE_NONE) {
                if (std::is_same<act_type, half>()) {
                  try_compute_in_half();
                } else if (std::is_same<act_type, bfloat16>()) {
                  if (sym_quant) {
                    qlinear_woq_affine_impl<
                        bfloat16,
                        bfloat16,
                        /*TGemmOut*/ float,
                        bfloat16,
                        bfloat16,
                        bfloat16,
                        UNQUANT_A,
                        quant_w_mode_>(
                        x,
                        qw,
                        scales_list[bf16_idx],
                        biases[fp32_idx],
                        y,
                        qw_type,
                        k_splits,
                        fusion_type,
                        others_list,
                        quant_block_k);
                  } else {
                    qlinear_woq_affine_impl<
                        bfloat16,
                        bfloat16,
                        /*TGemmOut*/ float,
                        bfloat16,
                        bfloat16,
                        bfloat16,
                        UNQUANT_A,
                        quant_w_mode_>(
                        x,
                        qw,
                        scales_list[bf16_idx],
                        biases[fp32_idx],
                        y,
                        qw_type,
                        k_splits,
                        fusion_type,
                        others_list,
                        quant_block_k,
                        zp_list[bf16_idx]);
                  }
                } else {
                  if (sym_quant) {
                    qlinear_woq_affine_impl<
                        float,
                        float,
                        /*TGemmOut*/ float,
                        float,
                        float,
                        float,
                        UNQUANT_A,
                        quant_w_mode_>(
                        x,
                        qw,
                        scales_list[fp32_idx],
                        biases[fp32_idx],
                        y,
                        qw_type,
                        k_splits,
                        fusion_type,
                        others_list,
                        quant_block_k);
                  } else {
                    qlinear_woq_affine_impl<
                        float,
                        float,
                        /*TGemmOut*/ float,
                        float,
                        float,
                        float,
                        UNQUANT_A,
                        quant_w_mode_>(
                        x,
                        qw,
                        scales_list[fp32_idx],
                        biases[fp32_idx],
                        y,
                        qw_type,
                        k_splits,
                        fusion_type,
                        others_list,
                        quant_block_k,
                        zp_list[fp32_idx]);
                  }
                }
              } else if (lowp_mode == LOWP_MODE_FP16) {
                try_compute_in_half();
              } else if (lowp_mode == LOWP_MODE_BF16) {
                if (M >= SMALL_BATCH_THRESHOLD) {
                  // compute in bfloat16 for large bs
                  if (sym_quant) {
                    qlinear_woq_affine_impl<
                        act_type,
                        bfloat16,
                        /*TGemmOut*/ float,
                        act_type,
                        bfloat16,
                        bfloat16,
                        UNQUANT_A,
                        quant_w_mode_>(
                        x,
                        qw,
                        scales_list[bf16_idx],
                        biases[fp32_idx],
                        y,
                        qw_type,
                        k_splits,
                        fusion_type,
                        others_list,
                        quant_block_k);
                  } else {
                    qlinear_woq_affine_impl<
                        act_type,
                        bfloat16,
                        /*TGemmOut*/ float,
                        act_type,
                        bfloat16,
                        bfloat16,
                        UNQUANT_A,
                        quant_w_mode_>(
                        x,
                        qw,
                        scales_list[bf16_idx],
                        biases[fp32_idx],
                        y,
                        qw_type,
                        k_splits,
                        fusion_type,
                        others_list,
                        quant_block_k,
                        zp_list[bf16_idx]);
                  }
                } else {
                  try_compute_in_half();
                }
              } else {
                TLA_ASSERT(lowp_mode == LOWP_MODE_INT8, "invalid lowp_mode");
                TLA_ASSERT(
                    qw_type == QINT4,
                    "LOWP_MODE_INT8 only support qw_type = QINT4");
                TLA_ASSERT(!sym_quant, "qw_type = QINT4 is asymmetric quant");
                if (quant_a_mode == QUANT_A_PER_TENSOR) {
                  float scale_a;
                  int32_t zp_a;
                  compute_int8_qparams_per_tensor<act_type>(x, &scale_a, &zp_a);
                  auto x_quantized =
                      quantize_per_tensor<act_type>(x, scale_a, zp_a);
                  qlinear_woq_affine_impl<
                      uint8_t,
                      uint8_t,
                      /*TGemmOut*/ float,
                      act_type,
                      float,
                      int8_t,
                      QUANT_A_PER_TENSOR,
                      quant_w_mode_>(
                      x_quantized,
                      qw,
                      scales_list[fp32_idx],
                      biases[fp32_idx],
                      y,
                      qw_type,
                      k_splits,
                      fusion_type,
                      others_list,
                      quant_block_k,
                      zp_list[int8_idx],
                      &scale_a,
                      &zp_a);
                } else {
                  auto block_k = w_sizes[2];
                  if (quant_block_k <= 0)
                    quant_block_k = block_k;
                  auto [scale_a, zp_a] =
                      compute_int8_qparams_per_block<act_type>(
                          x, quant_block_k, quant_a_mode);
                  auto x_quantized = quantize_per_block<act_type>(
                      x, scale_a, zp_a, quant_block_k, quant_a_mode);
                  float* scale_a_ptr = (float*)scale_a.data_ptr();
                  int32_t* zp_a_ptr = (int32_t*)zp_a.data_ptr();
                  range_dispatcher<
                      long,
                      QUANT_A_PER_K_BLOCK,
                      QUANT_A_PER_M_K_BLOCK>::
                      call(
                          quant_a_mode,
                          [&](auto quant_a_mode_) {
                            qlinear_woq_affine_impl<
                                uint8_t,
                                uint8_t,
                                /*TGemmOut*/ float,
                                act_type,
                                float,
                                int8_t,
                                quant_a_mode_,
                                quant_w_mode_>(
                                x_quantized,
                                qw,
                                scales_list[fp32_idx],
                                biases[fp32_idx],
                                y,
                                qw_type,
                                k_splits,
                                fusion_type,
                                others_list,
                                quant_block_k,
                                zp_list[int8_idx],
                                scale_a_ptr,
                                zp_a_ptr);
                          },
                          [&](auto quant_a_mode_) { failing_fallback(); });
                }
              }
            },
            [](auto tuple) { failing_fallback(); });
    return y;
  } else {
    TLA_ASSERT(
        qw.dim() == 2,
        "weight must be in 4D blocked format or 2D plain format");
    auto K = x.size(-1);
    auto M = x.numel() / K;
    auto N = qw.size(0);
    auto compute_dtype = x.scalar_type();
    if (lowp_mode == LOWP_MODE_FP16) {
      compute_dtype = at::kHalf;
    } else if (lowp_mode == LOWP_MODE_BF16) {
      compute_dtype = K >= SMALL_BATCH_THRESHOLD ? at::kBFloat16 : at::kHalf;
    }
    at::Tensor scale, zp;
    scale = scales_list[fp32_idx].unsqueeze(-1);
    if (qw_type != NF4) {
      zp = zp_list[fp32_idx].unsqueeze(-1);
    }
    auto w =
        [&]() {
          at::Tensor dqw;
          if (qw_type == NF4) {
            TLA_ASSERT(
                sym_quant, "Weight must be symmetrically quantized for NF4");
            using namespace at::indexing;
            auto w_int8 =
                at::empty({N, qw.size(1) * 2}, qw.options().dtype(at::kByte));
            w_int8.index({Slice(), Slice(None, None, 2)})
                .copy_(qw.bitwise_and(0xf));
            w_int8.index({Slice(), Slice(1, None, 2)})
                .copy_(qw.bitwise_right_shift(4));
            auto w_ret = map_nf4_tensor_to_float(w_int8);
            if (quant_w_mode == 0) {
              dqw = w_ret * scale;
            } else {
              int64_t num_blocks = scale.size(-2);
              auto w_int8_view = w_ret.view({N, num_blocks, -1});
              dqw = w_int8_view * scale;
              dqw = dqw.view({N, -1});
            }
          } else if (qw_type == QINT4) {
            TLA_ASSERT(
                !sym_quant, "Weight must be asymmetrically quantized for INT4");
            using namespace at::indexing;
            auto w_int8 =
                at::empty({N, qw.size(1) * 2}, qw.options().dtype(at::kByte));
            w_int8.index({Slice(), Slice(None, None, 2)})
                .copy_(qw.bitwise_and(0xf));
            w_int8.index({Slice(), Slice(1, None, 2)})
                .copy_(qw.bitwise_right_shift(4));
            if (quant_w_mode == 0) {
              dqw = (w_int8.to(at::kFloat) - zp) * scale;
            } else {
              int64_t num_blocks = scale.size(-2);
              auto w_int8_view = w_int8.view({N, num_blocks, -1});
              dqw = (w_int8_view.to(at::kFloat) - zp) * scale;
              dqw = dqw.view({N, -1});
            }
          } else {
            TLA_ASSERT(
                !sym_quant, "Weight must be asymmetrically quantized for INT8");
            if (quant_w_mode == 0) {
              dqw = sym_quant ? qw.to(at::kFloat) * scale
                              : (qw.to(at::kFloat) - zp) * scale;
            } else {
              int64_t num_blocks = scale.size(-2);
              auto w_int8_view = qw.view({N, num_blocks, -1});
              dqw = sym_quant ? w_int8_view.to(at::kFloat) * scale
                              : (w_int8_view.to(at::kFloat) - zp) * scale;
              dqw = dqw.view({N, -1});
            }
          }
          if (K != qw.size(1) * 2) {
            TORCH_CHECK(
                K < qw.size(1) * 2,
                'WOQ Linear kernel: Unexpected weight shape');
            dqw = dqw.narrow(1, 0, K);
          }
          return dqw;
        }()
            .to(compute_dtype);
    auto x_reshape = x.reshape({M, K});
    auto x_fp = x_reshape.to(compute_dtype);
    // PyTorch does not support computing in half yet
    auto y = compute_dtype == at::kHalf
        ? at::linear(x_fp.to(c10::kFloat), w.to(c10::kFloat))
        : at::linear(x_fp, w);
    if (biases[0].defined()) {
      auto b_index = compute_dtype == at::kFloat ? fp32_idx
          : compute_dtype == at::kHalf           ? fp16_idx
                                                 : bf16_idx;
      y = at::add(y, biases[b_index]);
    }
    if (fusion_type == FUSE_GELU_ERF) {
      y = at::gelu(y);
    } else if (fusion_type == FUSE_ADD || fusion_type == FUSE_ADD_ADD) {
      for (auto& tin : others_list) {
        y = at::add(y, tin.view(y.sizes()));
      }
    } else if (fusion_type == FUSE_GELU_TANH) {
      y = at::gelu(y, "tanh");
    }
    auto out_sizes = x.sizes().vec();
    out_sizes.back() = N;
    y = y.view(out_sizes);
    return y.to(x.scalar_type());
  }
}

#else // defined(CPU_CAPABILITY_AVX512_FP16) && defined(COMPILER_PREREQ_MET)

#define SMALL_BATCH_THRESHOLD 32

at::Tensor qlinear_woq_affine(
    const at::Tensor& x,
    const at::Tensor& qw,
    const TensorList& scales_list,
    const TensorList& zp_list,
    const TensorList& bias_list,
    const int qw_type,
    int64_t lowp_mode,
    int64_t fusion_type,
    const TensorList& others_list,
    int64_t quant_a_mode = -1,
    int64_t quant_w_mode = 0,
    int64_t quant_block_k = 0) {
  constexpr size_t fp32_idx = 0, fp16_idx = 1, bf16_idx = 2, int8_idx = 3;
  auto biases = bias_list.empty()
      ? TensorList({at::Tensor(), at::Tensor(), at::Tensor()})
      : bias_list;
  const bool sym_quant = is_sym_quant(qw_type);
  TLA_ASSERT(
      qw.dim() == 2, "weight must be in 4D blocked format or 2D plain format");
  auto K = x.size(-1);
  auto M = x.numel() / K;
  auto N = qw.size(0);
  auto compute_dtype = x.scalar_type();
  if (lowp_mode == LOWP_MODE_FP16) {
    compute_dtype = at::kHalf;
  } else if (lowp_mode == LOWP_MODE_BF16) {
    compute_dtype = K >= SMALL_BATCH_THRESHOLD ? at::kBFloat16 : at::kHalf;
  }
  at::Tensor scale, zp;
  scale = scales_list[fp32_idx].unsqueeze(-1);
  if (qw_type != NF4) {
    zp = zp_list[fp32_idx].unsqueeze(-1);
  }
  auto w =
      [&]() {
        at::Tensor dqw;
        if (qw_type == NF4) {
          TLA_ASSERT(
              sym_quant, "Weight must be symmetrically quantized for NF4");
          using namespace at::indexing;
          auto w_int8 =
              at::empty({N, qw.size(1) * 2}, qw.options().dtype(at::kByte));
          w_int8.index({Slice(), Slice(None, None, 2)})
              .copy_(qw.bitwise_and(0xf));
          w_int8.index({Slice(), Slice(1, None, 2)})
              .copy_(qw.bitwise_right_shift(4));
          auto w_ret = map_nf4_tensor_to_float(w_int8);
          if (quant_w_mode == 0) {
            dqw = w_ret * scale;
          } else {
            int64_t num_blocks = scale.size(-2);
            auto w_int8_view = w_ret.view({N, num_blocks, -1});
            dqw = w_int8_view * scale;
            dqw = dqw.view({N, -1});
          }
        } else if (qw_type == QINT4) {
          TLA_ASSERT(
              !sym_quant, "Weight must be asymmetrically quantized for INT4");
          using namespace at::indexing;
          auto w_int8 =
              at::empty({N, qw.size(1) * 2}, qw.options().dtype(at::kByte));
          w_int8.index({Slice(), Slice(None, None, 2)})
              .copy_(qw.bitwise_and(0xf));
          w_int8.index({Slice(), Slice(1, None, 2)})
              .copy_(qw.bitwise_right_shift(4));
          if (quant_w_mode == 0) {
            dqw = (w_int8.to(at::kFloat) - zp) * scale;
          } else {
            int64_t num_blocks = scale.size(-2);
            auto w_int8_view = w_int8.view({N, num_blocks, -1});
            dqw = (w_int8_view.to(at::kFloat) - zp) * scale;
            dqw = dqw.view({N, -1});
          }
        } else {
          TLA_ASSERT(
              !sym_quant, "Weight must be asymmetrically quantized for INT8");
          if (quant_w_mode == 0) {
            dqw = sym_quant ? qw.to(at::kFloat) * scale
                            : (qw.to(at::kFloat) - zp) * scale;
          } else {
            int64_t num_blocks = scale.size(-2);
            auto w_int8_view = qw.view({N, num_blocks, -1});
            dqw = sym_quant ? w_int8_view.to(at::kFloat) * scale
                            : (w_int8_view.to(at::kFloat) - zp) * scale;
            dqw = dqw.view({N, -1});
          }
        }
        if (K != qw.size(1) * 2) {
          TORCH_CHECK(
              K < qw.size(1) * 2, 'WOQ Linear kernel: Unexpected weight shape');
          dqw = dqw.narrow(1, 0, K);
        }
        return dqw;
      }()
          .to(compute_dtype);
  auto x_reshape = x.reshape({M, K});
  auto x_fp = x_reshape.to(compute_dtype);
  // PyTorch does not support computing in half yet
  auto y = compute_dtype == at::kHalf
      ? at::linear(x_fp.to(c10::kFloat), w.to(c10::kFloat))
      : at::linear(x_fp, w);
  if (biases[0].defined()) {
    auto b_index = compute_dtype == at::kFloat ? fp32_idx
        : compute_dtype == at::kHalf           ? fp16_idx
                                               : bf16_idx;
    y = at::add(y, biases[b_index]);
  }
  if (fusion_type == FUSE_GELU_ERF) {
    y = at::gelu(y);
  } else if (fusion_type == FUSE_ADD || fusion_type == FUSE_ADD_ADD) {
    for (auto& tin : others_list) {
      y = at::add(y, tin.view(y.sizes()));
    }
  } else if (fusion_type == FUSE_GELU_TANH) {
    y = at::gelu(y, "tanh");
  }
  auto out_sizes = x.sizes().vec();
  out_sizes.back() = N;
  y = y.view(out_sizes);
  return y.to(x.scalar_type());
}

at::Tensor qlinear_woq_pack(
    const at::Tensor& qw,
    int qw_type,
    size_t block_n,
    size_t block_k,
    int64_t lowp_mode) {
  return qw;
}

at::Tensor qlinear_woq_unpack(
    const at::Tensor& qw_packed,
    int qw_type,
    int64_t lowp_mode) {
  return qw_packed;
}
#endif // defined(CPU_CAPABILITY_AVX512_FP16) && defined(COMPILER_PREREQ_MET)

} // namespace

IPEX_REGISTER_DISPATCH(woq_tpp_gemm_kernel_stub, &qlinear_woq_affine);
IPEX_REGISTER_DISPATCH(woq_tpp_gemm_packB_stub, &qlinear_woq_pack);
IPEX_REGISTER_DISPATCH(woq_tpp_gemm_unpackB_stub, &qlinear_woq_unpack);

} // namespace cpu
} // namespace torch_ipex

#endif