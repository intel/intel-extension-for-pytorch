#include <aten/Linear.h>
#include "aten/utils/fp8_bf16_gemm.h"

// #include "common.h"
// #include "vec.h"
// #include "gemm.h"

namespace torch_ipex {
namespace cpu {
namespace {

// global float8 LUT
alignas(64) static uint16_t e4m3_to_16bit[256];

template <typename T>
static void initialize_e4m3_to_16bit_tables() {
  // run only once
  static bool initialized_16bit = false;
  if (!initialized_16bit) {
    std::cout << "\n@@@@ doing lut init ..." << std::endl;
    for (uint8_t u8 = 0; u8 < 256; ++u8) {
      auto value = static_cast<T>(c10::bit_cast<c10::Float8_e4m3fn>(u8));
      uint16_t value_bits = c10::bit_cast<uint16_t>(value);
      e4m3_to_16bit[u8] = value_bits;
      if (u8 == 255) {
        break;
      }
    }
    initialized_16bit = true;
  }
}

#if defined(CPU_CAPABILITY_AVX512_BF16)
template <typename scalar_t>
inline void copy_stub(
    scalar_t* __restrict__ out,
    const float* __restrict__ input,
    int size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();

  int d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec data0 = fVec::loadu(input + d);
    fVec data1 = fVec::loadu(input + d + fVec::size());
    bVec out_vec = convert_from_float_ext<scalar_t>(data0, data1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d]);
  }
}

template <typename scalar_t>
inline void copy_mul_stub(
    scalar_t* __restrict__ out,
    const float* __restrict__ input,
    int size,
    float scale) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec vscale = fVec(scale);

  int d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec data0 = fVec::loadu(input + d) * vscale;
    fVec data1 = fVec::loadu(input + d + fVec::size()) * vscale;
    bVec out_vec = convert_from_float_ext<scalar_t>(data0, data1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d] * scale);
  }
}

// convert to vnni format
// from [N, K] to [K/2, N, 2] for bfloat16 and float16
template <typename scalar_t>
inline void pack_vnni(
    scalar_t* __restrict__ packed,
    const scalar_t* __restrict__ weight,
    int N,
    int K) {
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K / VNNI_BLK; ++k) {
      for (int d = 0; d < VNNI_BLK; ++d) {
        packed[k * N * VNNI_BLK + n * VNNI_BLK + d] =
            weight[n * K + k * VNNI_BLK + d];
      }
    }
  }
}

// for fp8, shuffle per 64
//   [0, 1, ... 31][32, 33, ... 63]
//   [0, 2, ... 62][ 1,  3, ... 63]
template <>
inline void pack_vnni<at::Float8_e4m3fn>(
    at::Float8_e4m3fn* __restrict__ packed,
    const at::Float8_e4m3fn* __restrict__ weight,
    int N,
    int K) {
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K / VNNI_BLK; ++k) {
      for (int d = 0; d < VNNI_BLK; ++d) {
        packed[k * N * VNNI_BLK + n * VNNI_BLK + d] =
            weight[n * K + k * VNNI_BLK + d];
      }
    }
  }

  at::Float8_e4m3fn arr[64];
  for (int i = 0; i < N * K / 64; ++i) {
    auto row_ptr = packed + i * 64;
    memcpy(arr, row_ptr, 64 * sizeof(at::Float8_e4m3fn));
    // from [32, 2] to [2, 32]
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 32; ++k) {
        row_ptr[j * 32 + k] = arr[k * 2 + j];
      }
    }
  }
}

template <typename scalar_t, typename packed_t, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const packed_t* __restrict__ B,
      scalar_t* __restrict__ C,
      float scale,
      int K,
      int lda,
      int ldb,
      int ldc) {
    TORCH_CHECK(false, "tinygemm_kernel_nn: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn<at::BFloat16, at::BFloat16, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const at::BFloat16* __restrict__ B,
      at::BFloat16* __restrict__ C,
      float scale,
      int K,
      int lda,
      int ldb,
      int ldc) {
    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;

    // prefetch distance
    constexpr int PREFETCH_SIZE_K = 0;

    __m512bh va;
    __m512bh vb[COLS];
    __m512 vc[ROWS * COLS];

    auto loadc = [&](auto i) { vc[i] = _mm512_set1_ps(0.f); };
    Unroll<ROWS * COLS>{}(loadc);

    const int K2 = K >> 1;
    const int lda2 = lda >> 1;
    const int ldb2 = ldb; // ldb * 2 >> 1;
    const float* a_ptr = reinterpret_cast<const float*>(A);
    const float* b_ptr = reinterpret_cast<const float*>(B);

    auto compute = [&](auto i, int k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = (__m512bh)(_mm512_set1_ps(a_ptr[row * lda2 + k]));
      }
      if constexpr (row == 0) {
        vb[col] = (__m512bh)(_mm512_loadu_si512(b_ptr + k * ldb2 + col * 16));
        if constexpr (PREFETCH_SIZE_K > 0) {
          _mm_prefetch(
              b_ptr + (k + PREFETCH_SIZE_K) * ldb2 + col * 16, _MM_HINT_T0);
        }
      }
      vc[i] = _mm512_dpbf16_ps(vc[i], va, vb[col]);
    };
    for (int k = 0; k < K2; ++k) {
      Unroll<ROWS * COLS>{}(compute, k);
    }

    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      // for COLS = 2, 4 use 512bit store
      // for COLS = 1, 3 use 256bit store
      if constexpr (COLS % 2 == 0) {
        if constexpr (col % 2 == 0) {
          _mm512_storeu_si512(
              reinterpret_cast<__m512i*>((C + row * ldc + col * 16)),
              (__m512i)(_mm512_cvtne2ps_pbh(
                  vc[row * COLS + col + 1], vc[row * COLS + col])));
        }
      } else {
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(C + row * ldc + col * 16),
            (__m256i)(_mm512_cvtneps_pbh(vc[i])));
      }
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};

void print_16x16(const __m256i x) {
  at::BFloat16 a[16];
  _mm256_storeu_si256((__m256i*)a, x);

  for (int i = 0; i < 16; i++) {
    std::cout << a[i] << " ";
  }
  std::cout << std::endl;
}

void print_32x16(const __m512i x) {
  at::BFloat16 a[32];
  _mm512_storeu_si512((__m512i*)a, x);

  for (int i = 0; i < 32; i++) {
    std::cout << a[i] << " ";
  }
  std::cout << std::endl;
}

template <int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn<at::BFloat16, at::Float8_e4m3fn, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const at::Float8_e4m3fn* __restrict__ B,
      at::BFloat16* __restrict__ C,
      float scale,
      int K,
      int lda,
      int ldb,
      int ldc) {
    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;

    // prefetch distance
    constexpr int PREFETCH_SIZE_K = 0;

    __m512bh va;
    __m512bh vb[COLS];
    __m512 vc[ROWS * COLS];

    const __m512 vscale = _mm512_set1_ps(scale);
    const __m512i mask = _mm512_set1_epi32(0xFFFF);

    auto loadc = [&](auto i) { vc[i] = _mm512_set1_ps(0.f); };
    Unroll<ROWS * COLS>{}(loadc);

    const int K2 = K >> 1;
    const int lda2 = lda >> 1;
    const int ldb2 = ldb; // ldb * 2 >> 1;
    const float* a_ptr = reinterpret_cast<const float*>(A);
    const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(B);

    auto compute = [&](auto i, int k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = (__m512bh)(_mm512_set1_ps(a_ptr[row * lda2 + k]));
      }
      if constexpr (row == 0) {
        if constexpr (col % 2 == 0) {
          __m512i b8 = _mm512_loadu_si512(b_ptr + k * ldb2 + col * 16);
          if constexpr (PREFETCH_SIZE_K > 0) {
            _mm_prefetch(
                b_ptr + (k + PREFETCH_SIZE_K) * ldb2 + col * 16, _MM_HINT_T0);
          }
          __m512i idx0 = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(b8));
          __m512i idx1 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(b8, 1));
          __m512i idx2 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(b8, 2));
          __m512i idx3 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(b8, 3));

          __m512i b16_0 = _mm512_i32gather_epi32(idx0, e4m3_to_16bit, 2);
          __m512i b16_1 = _mm512_i32gather_epi32(idx1, e4m3_to_16bit, 2);
          __m512i b16_2 = _mm512_i32gather_epi32(idx2, e4m3_to_16bit, 2);
          __m512i b16_3 = _mm512_i32gather_epi32(idx3, e4m3_to_16bit, 2);

          vb[col + 0] = (__m512bh)(_mm512_or_epi32(
              _mm512_slli_epi32(b16_2, 16), _mm512_and_epi32(b16_0, mask)));
          vb[col + 1] = (__m512bh)(_mm512_or_epi32(
              _mm512_slli_epi32(b16_3, 16), _mm512_and_epi32(b16_1, mask)));
        }
      }
      vc[i] = _mm512_dpbf16_ps(vc[i], va, vb[col]);
    };
    for (int k = 0; k < K2; ++k) {
      Unroll<ROWS * COLS>{}(compute, k);
    }

    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      // for COLS = 2, 4 use 512bit store
      if constexpr (col % 2 == 0) {
        __m512 vc0 = _mm512_mul_ps(vc[row * COLS + col + 0], vscale);
        __m512 vc1 = _mm512_mul_ps(vc[row * COLS + col + 1], vscale);
        _mm512_storeu_si512(
            reinterpret_cast<__m512i*>((C + row * ldc + col * 16)),
            (__m512i)(_mm512_cvtne2ps_pbh(vc1, vc0)));
      }
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};
#endif

#define LAUNCH_TINYGEMM_KERNEL_NN(MB_SIZE, NB_SIZE)                \
  tinygemm_kernel_nn<scalar_t, packed_t, MB_SIZE, NB_SIZE>::apply( \
      A + mb_start * lda,                                          \
      B + nb_start * 2,                                            \
      C + mb_start * ldc + nb_start,                               \
      scale,                                                       \
      K,                                                           \
      lda,                                                         \
      ldb,                                                         \
      ldc);

template <typename scalar_t, typename packed_t>
struct brgemm {};

template <typename scalar_t>
struct brgemm<scalar_t, scalar_t> {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const scalar_t* __restrict__ B,
      scalar_t* __restrict__ C,
      scalar_t* __restrict__ Btmp,
      float* __restrict__ Ctmp,
      float scale,
      int M,
      int N,
      int K,
      int lda,
      int ldb,
      int ldc) {
    TORCH_UNUSED(scale);

    constexpr int BLOCK_N = block_size_n();
    at::native::cpublas::brgemm(
        M, N, K, lda, ldb, BLOCK_N, /* add_C */ false, A, B, Ctmp);

    // copy from Ctmp to C
    for (int m = 0; m < M; ++m) {
      copy_stub(C + m * ldc, Ctmp + m * BLOCK_N, N);
    }
  }
};

inline void unpack_B(
    at::BFloat16* __restrict__ Btmp,
    const at::Float8_e4m3fn* __restrict__ packed_B,
    int N,
    int K,
    int ldb,
    int ldb_tmp) {
  // [K/2, N, 2]
  const int K2 = K >> 1;
  const int ldb2 = ldb; // ldb * 2 >> 1;
  const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(packed_B);

  const __m512i mask = _mm512_set1_epi32(0xFFFF);

#pragma GCC unroll 4
  for (int k = 0; k < K2; ++k) {
    for (int n = 0; n < N; n += 32) {
      __m512i b8 = _mm512_loadu_si512(b_ptr + k * ldb2 + n);
      __m512i idx0 = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(b8));
      __m512i idx1 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(b8, 1));
      __m512i idx2 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(b8, 2));
      __m512i idx3 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(b8, 3));

      __m512i b16_0 = _mm512_i32gather_epi32(idx0, e4m3_to_16bit, 2);
      __m512i b16_1 = _mm512_i32gather_epi32(idx1, e4m3_to_16bit, 2);
      __m512i b16_2 = _mm512_i32gather_epi32(idx2, e4m3_to_16bit, 2);
      __m512i b16_3 = _mm512_i32gather_epi32(idx3, e4m3_to_16bit, 2);

      __m512i b16_02 = _mm512_or_epi32(
          _mm512_slli_epi32(b16_2, 16), _mm512_and_epi32(b16_0, mask));
      __m512i b16_13 = _mm512_or_epi32(
          _mm512_slli_epi32(b16_3, 16), _mm512_and_epi32(b16_1, mask));

      _mm512_storeu_si512(Btmp + k * ldb_tmp * 2 + n * 2 + 0, b16_02);
      _mm512_storeu_si512(Btmp + k * ldb_tmp * 2 + n * 2 + 32, b16_13);
    }
  }
}

template <>
struct brgemm<at::BFloat16, at::Float8_e4m3fn> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const at::Float8_e4m3fn* __restrict__ B,
      at::BFloat16* __restrict__ C,
      at::BFloat16* __restrict__ Btmp,
      float* __restrict__ Ctmp,
      float scale,
      int M,
      int N,
      int K,
      int lda,
      int ldb,
      int ldc) {
    constexpr int BLOCK_N = block_size_n();

    // [BLOCK_K, BLOCK_N] -> [BLOCK_K / 2, BLOCK_N * 2]
    const int ldb_tmp = block_size_n();

    // accumulate across K per BLOCK_K
    for (int k = 0; k < K; k += BLOCK_K) {
      int kb_size = std::min(BLOCK_K, K - k);
      unpack_B(Btmp, B + k * ldb, N, kb_size, ldb, ldb_tmp);

      const bool add_C = (k != 0);
      at::native::cpublas::brgemm(
          M, N, kb_size, lda, ldb_tmp, BLOCK_N, add_C, A + k, Btmp, Ctmp);
    }

    // copy from Ctmp to C and mul scale
    for (int m = 0; m < M; ++m) {
      copy_mul_stub(C + m * ldc, Ctmp + m * BLOCK_N, N, scale);
    }
  }
};

template <typename scalar_t, typename packed_t>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const packed_t* __restrict__ B,
    scalar_t* __restrict__ C,
    scalar_t* __restrict__ Btmp,
    float* __restrict__ Ctmp,
    float scale,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    bool brg) {
  if (brg) {
    brgemm<scalar_t, packed_t>::apply(
        A, B, C, Btmp, Ctmp, scale, M, N, K, lda, ldb, ldc);
    return;
  }

  // pattern: 1-8-8
  if (M == 1) {
    constexpr int BLOCK_N = 128;
    const int NB = div_up(N, BLOCK_N);
    int mb_start = 0;

    for (int nb = 0; nb < NB; ++nb) {
      int nb_start = nb * BLOCK_N;
      int nb_size = std::min(BLOCK_N, N - nb_start);

      switch (nb_size >> 4) {
        case 2:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 32);
          break;
        case 4:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 64);
          break;
        case 6:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 96);
          break;
        case 8:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 128);
          break;
        default:
          TORCH_CHECK(false, "Unexpected block size, 1x", "nb_size");
      }
    }
    return;
  }

  // pattern: 1-4-16
  constexpr int BLOCK_M = 4;
  constexpr int BLOCK_N = 64;
  const int MB = div_up(M, BLOCK_M);
  const int NB = div_up(N, BLOCK_N);
  for (int mb = 0; mb < MB; ++mb) {
    int mb_start = mb * BLOCK_M;
    int mb_size = std::min(BLOCK_M, M - mb_start);
    for (int nb = 0; nb < NB; ++nb) {
      int nb_start = nb * BLOCK_N;
      int nb_size = std::min(BLOCK_N, N - nb_start);

      switch (mb_size << 4 | nb_size >> 4) {
        // mb_size = 1
        case 0x12:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 32);
          break;
        case 0x14:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 64);
          break;
        // mb_size = 2
        case 0x22:
          LAUNCH_TINYGEMM_KERNEL_NN(2, 32);
          break;
        case 0x24:
          LAUNCH_TINYGEMM_KERNEL_NN(2, 64);
          break;
        // mb_size = 3
        case 0x32:
          LAUNCH_TINYGEMM_KERNEL_NN(3, 32);
          break;
        case 0x34:
          LAUNCH_TINYGEMM_KERNEL_NN(3, 64);
          break;
        // mb_size = 4
        case 0x42:
          LAUNCH_TINYGEMM_KERNEL_NN(4, 32);
          break;
        case 0x44:
          LAUNCH_TINYGEMM_KERNEL_NN(4, 64);
          break;
        default:
          TORCH_CHECK(
              false, "Unexpected block size, ", mb_size, "x", "nb_size");
      }
    }
  }
}

// bmm
template <typename scalar_t, typename packed_t>
void bmm_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ mat1,
    const packed_t* __restrict__ mat2,
    int B,
    int M,
    int N,
    int K,
    int mat1_strideB,
    int mat1_strideM,
    int out_strideB,
    int out_strideM,
    const float* __restrict__ scale = nullptr) {
  constexpr int BLOCK_M = block_size_m();
  constexpr int BLOCK_N = block_size_n();
  const int MB = div_up(M, BLOCK_M);
  const int NB = div_up(N, BLOCK_N);

  // mat2 contiguous in [B, N, K]
  int mat2_strideB = N * K;
  int mat2_strideN = K;

  // use avx512-bf16 when a) M is small; b) dtype is bfloat16, otherwise use amx
  const bool use_brgemm = (M > 4) || (!std::is_same_v<scalar_t, at::BFloat16>);

  // parallel on [B, MB, NB]
  at::parallel_for(0, B * MB * NB, 0, [&](int begin, int end) {
    int bs{0}, mb{0}, nb{0};
    data_index_init(begin, bs, B, mb, MB, nb, NB);

    // for brgemm, use float32 for accumulate
    alignas(64) float Ctmp[BLOCK_M * BLOCK_N];
    // for brgemm when mat2 is float8_e4m3
    alignas(64) scalar_t Btmp[BLOCK_N * BLOCK_K];
    const float* scale_ptr = scale ? scale + nb * BLOCK_N : nullptr;

    for (int i = begin; i < end; ++i) {
      TORCH_UNUSED(i);
      int mb_start = mb * BLOCK_M;
      int mb_size = std::min(M - mb_start, BLOCK_M);
      int nb_start = nb * BLOCK_N;
      int nb_size = std::min(N - nb_start, BLOCK_N);

      tinygemm_kernel(
          /*   A */ mat1 + bs * mat1_strideB + mb_start * mat1_strideM,
          /*   B */ mat2 + bs * mat2_strideB +
              nb_start * mat2_strideN /* nb * BLOCK_N * K */,
          /*   C */ out + bs * out_strideB + mb_start * out_strideM + nb_start,
          /* Btmp*/ Btmp,
          /* Ctmp*/ Ctmp,
          /*scale*/ scale_ptr[0],
          /*   M */ mb_size,
          /*   N */ nb_size,
          /*   K */ K,
          /* lda */ mat1_strideM,
          /* ldb */ nb_size,
          /* ldc */ out_strideM,
          /* brg */ use_brgemm);

      // move to the next index
      data_index_step(bs, B, mb, MB, nb, NB);
    }

    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });
}

at::Tensor convert_weight_packed(at::Tensor& weight) {
  // weight : [E, OC, IC]
  //     w1 : [E, 2N,  K]
  //     w2 : [E,  K,  N]
  CHECK_DIM(3, weight);
  CHECK_INPUT(weight);
  const auto st = weight.scalar_type();
  const int E = weight.size(0);
  const int OC = weight.size(1);
  const int IC = weight.size(2);

  // we handle 2 TILE_N at a time.
  TORCH_CHECK(OC % TILE_N == 0, "invalid weight out features ", OC);
  TORCH_CHECK(IC % TILE_K == 0, "invalid weight input features ", IC);

  constexpr int BLOCK_N = block_size_n();

  // use phony sizes here [E, OC, IC], for each [E], [OC, IC] -> [IC / 2, OC, 2]
  auto packed_weight = at::empty({E, OC, IC}, weight.options());
  const int stride = OC * IC;

  TORCH_CHECK(
      st == at::kBFloat16 || st == at::kHalf || st == at::kFloat8_e4m3fn,
      "expect weight to be bfloat16, float16 or float8_e4m3fn.");

  CPU_DISPATCH_FLOAT_TYPES(st, [&] {
    const scalar_t* w_data = weight.data_ptr<scalar_t>();
    scalar_t* packed_data = packed_weight.data_ptr<scalar_t>();

    // parallel on {E, OC}
    int n_blocks = div_up(OC, BLOCK_N);
    int n_tasks = E * n_blocks;
    at::parallel_for(0, n_tasks, 0, [&](int begin, int end) {
      for (int nt = begin; nt < end; ++nt) {
        int e = nt / n_blocks;
        int n = nt % n_blocks;
        int n_size = std::min(BLOCK_N, OC - n * BLOCK_N);
        pack_vnni<scalar_t>(
            packed_data + e * stride + n * IC,
            w_data + e * stride + n * IC,
            n_size,
            IC);
      }
      // for (int e = begin; e < end; ++e) {
      //   for (int n = 0; n < OC; n += BLOCK_N) {
      //     int n_size = std::min(BLOCK_N, OC - n);
      //     pack_vnni<scalar_t>(
      //         packed_data + e * stride + n * IC,
      //         w_data + e * stride + n * IC,
      //         n_size,
      //         IC);
      //   }
      // }
    });
  });
  return packed_weight;
}

// mat1 : [B, M, K]
// mat2 : [B, N, K] or [B, OC, IC]
// out  : [B, M, N]
// scale: [] 0-dim tensor for per tensor quant
//
void bmm(
    at::Tensor& out,
    at::Tensor& mat1,
    at::Tensor& mat2,
    bool is_vnni,
    std::optional<at::Tensor>& scale) {
  auto packed_w = is_vnni ? mat2 : convert_weight_packed(mat2);

  // input and out could be non-contiguous
  // weight needs to be contiguous in [OC, IC] order
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(mat1);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(out);
  CHECK_INPUT(mat2);
  CHECK_DIM(3, out);
  CHECK_DIM(3, mat1);
  CHECK_DIM(3, mat2);

  int B = mat1.size(0);
  int M = mat1.size(1);
  int N = mat2.size(1);
  int K = mat1.size(2);

  const bool use_fp8_w8a16 = scale.has_value();
  TORCH_CHECK(N % 32 == 0, "tinygemm requires N to be 32x.");

  int mat1_strideB = mat1.stride(0);
  int mat1_strideM = mat1.stride(1);
  int out_strideB = out.stride(0);
  int out_strideM = out.stride(1);

  // check shapes
  TORCH_CHECK(
      mat2.size(0) == B && mat2.size(2) == K, "bmm: mat2 shape mismatch!");
  TORCH_CHECK(out.size(0) == B && out.size(1) == M, "bmm: out shape mismatch!");

  CPU_DISPATCH_PACKED_FLOAT_TYPES(mat1.scalar_type(), mat2.scalar_type(), [&] {
    // float scale_val = 0.f;
    float* scale_data = nullptr;
    if (use_fp8_w8a16) {
      initialize_e4m3_to_16bit_tables<scalar_t>();

      auto scale_tensor = scale.value();
      // TORCH_CHECK(scale_tensor.ndimension() == 0, "bmm: expect scale to be
      // 0-dim tensor."); scale_val = scale_tensor.item<float>();
      scale_data = scale_tensor.data_ptr<float>();
    }

    bmm_kernel_impl<scalar_t, packed_t>(
        out.data_ptr<scalar_t>(),
        mat1.data_ptr<scalar_t>(),
        packed_w.data_ptr<packed_t>(),
        B,
        M,
        N,
        K,
        mat1_strideB,
        mat1_strideM,
        out_strideB,
        out_strideM,
        scale_data);
  });
}

#else
at::Tensor convert_weight_packed(at::Tensor& weight) {
  return weight;
}
void bmm(
    at::Tensor& out,
    at::Tensor& mat1,
    at::Tensor& mat2,
    bool is_vnni,
    std::optional<at::Tensor>& scale) {
  at::bmm_out(out, mat1, mat2.to(mat1.scalar_type()));
  if (scale.has_value()) {
    at::mul_out(out, out, scale.value());
  }
}

#endif

} // anonymous namespace

IPEX_REGISTER_DISPATCH(fp8_bmm_stub, &bmm);

} // namespace cpu
} // namespace torch_ipex