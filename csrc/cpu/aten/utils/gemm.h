#pragma once
#include <aten/utils/common.h>
#include <aten/utils/vec.h>
#include <aten/utils/amx.h>
#include <immintrin.h>
#include <aten/utils/woq.h>
// namespace {
namespace torch_ipex {
namespace cpu {
// global float8 LUT
alignas(64) static uint16_t e4m3_to_16bit[256];
// block size for AMX gemm in moe.cpp
constexpr int block_size_m() {
  return 1 * TILE_M;
}
constexpr int block_size_n() {
  return 8 * TILE_N;
}
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
      if (u8 == 255) { break; }
    }
    initialized_16bit = true;
  }
}

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ input, int size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();

  int d;
  #pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec data0 = fVec::loadu(input + d);
    fVec data1 = fVec::loadu(input + d + fVec::size());
    bVec out_vec = torch_ipex::cpu::convert_from_float_ext<scalar_t>(data0, data1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d]);
  }
}



template <typename scalar_t, typename packed_t, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn {
  static inline void apply(
      const scalar_t* __restrict__ A, const packed_t* __restrict__ B, float* __restrict__ C,
      float scale, int K, int lda, int ldb, int ldc) {
    TORCH_CHECK(false, "tinygemm_kernel_avx: scalar path not implemented!");
  }
};
#if defined(CPU_CAPABILITY_AVX512_BF16)
static inline void tinygemm_kernel_nn_woq(
  const at::BFloat16*  A, const at::BFloat16*  B, float*  C,
   int K, int lda, int ldb, int ldc, int BLOCK_M, int BLOCK_N) {

//std::cout << "### tinygemm_kernel_nn: M = " << BLOCK_M << "; N = " << BLOCK_N << ";  K = " << K << std::endl;

constexpr int ROWS = 1;
constexpr int COLS = 32 / 16;

// prefetch distance
constexpr int PREFETCH_SIZE_K = 0;

__m512bh va;
__m512bh vb[COLS];
__m512 vc[ROWS * COLS];

auto loadc = [&](auto i) {
  vc[i] = _mm512_set1_ps(0.f);
};
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
      _mm_prefetch(b_ptr + (k + PREFETCH_SIZE_K) * ldb2 + col * 16, _MM_HINT_T0);
    }
  }
  vc[i] = _mm512_dpbf16_ps(vc[i], va, vb[col]);
  // _mm512_storeu_ps(reinterpret_cast<__m512*>(C + row * ldc + col * 16), vc[i]);
};
for (int k = 0; k < K2; ++k) {
  Unroll<ROWS * COLS>{}(compute, k);
}
auto storec = [&](auto i) {
  constexpr int row = i / COLS;
  constexpr int col = i % COLS;
  // for COLS = 2, 4 use 512bit store
  // for COLS = 1, 3 use 256bit store
  _mm512_storeu_ps(reinterpret_cast<__m512*>(C + row * ldc + col * 16), vc[i]);
};
Unroll<ROWS * COLS>{}(storec);
// auto storec = [&](auto i) {
//   constexpr int row = i / COLS;
//   constexpr int col = i % COLS;
//   // for COLS = 2, 4 use 512bit store
//   // for COLS = 1, 3 use 256bit store
//   if constexpr (COLS % 2 == 0) {
//     if constexpr (col % 2 == 0) {
//       _mm512_storeu_si512(
//           reinterpret_cast<__m512i*>((C + row * ldc + col * 16)),
//           (__m512i)(_mm512_cvtne2ps_pbh(vc[row * COLS + col + 1], vc[row * COLS + col])));
//     }
//   } else {
//     _mm256_storeu_si256(
//         reinterpret_cast<__m256i*>(C + row * ldc + col * 16),
//         (__m256i)(_mm512_cvtneps_pbh(vc[i])));
//   }
// };

}
#else
template <typename scalar_t, typename packed_t>
static inline void tinygemm_kernel_nn_woq(
  const scalar_t*  A, const packed_t*  B, float*  C,
   int K, int lda, int ldb, int ldc, int BLOCK_M, int BLOCK_N) {
    TORCH_CHECK(false, "tinygemm_kernel_avx: scalar path not implemented!");
   }
   #endif
#if defined(CPU_CAPABILITY_AVX512_BF16)
template <int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn<at::BFloat16, at::BFloat16, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const at::BFloat16* __restrict__ A, const at::BFloat16* __restrict__ B, float* __restrict__ C,
      float scale, int K, int lda, int ldb, int ldc) {
        // std::cout<<"===K:"<<K<<std::endl;
        // std::cout<<"===lda:"<<lda<<std::endl;
        // std::cout<<"===ldb:"<<ldb<<std::endl;
        // std::cout<<"===ldc:"<<ldc<<std::endl;
    //std::cout << "### tinygemm_kernel_nn: M = " << BLOCK_M << "; N = " << BLOCK_N << ";  K = " << K << std::endl;

    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;

    // prefetch distance
    constexpr int PREFETCH_SIZE_K = 0;

    __m512bh va;
    __m512bh vb[COLS];
    __m512 vc[ROWS * COLS];

    auto loadc = [&](auto i) {
      vc[i] = _mm512_set1_ps(0.f);
    };
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
          _mm_prefetch(b_ptr + (k + PREFETCH_SIZE_K) * ldb2 + col * 16, _MM_HINT_T0);
        }
      }
      vc[i] = _mm512_dpbf16_ps(vc[i], va, vb[col]);
      // _mm512_storeu_ps(reinterpret_cast<__m512*>(C + row * ldc + col * 16), vc[i]);
    };
    for (int k = 0; k < K2; ++k) {
      Unroll<ROWS * COLS>{}(compute, k);
    }
    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      // for COLS = 2, 4 use 512bit store
      // for COLS = 1, 3 use 256bit store
      _mm512_storeu_ps(reinterpret_cast<__m512*>(C + row * ldc + col * 16), vc[i]);
    };
    Unroll<ROWS * COLS>{}(storec);
    // auto storec = [&](auto i) {
    //   constexpr int row = i / COLS;
    //   constexpr int col = i % COLS;
    //   // for COLS = 2, 4 use 512bit store
    //   // for COLS = 1, 3 use 256bit store
    //   if constexpr (COLS % 2 == 0) {
    //     if constexpr (col % 2 == 0) {
    //       _mm512_storeu_si512(
    //           reinterpret_cast<__m512i*>((C + row * ldc + col * 16)),
    //           (__m512i)(_mm512_cvtne2ps_pbh(vc[row * COLS + col + 1], vc[row * COLS + col])));
    //     }
    //   } else {
    //     _mm256_storeu_si256(
    //         reinterpret_cast<__m256i*>(C + row * ldc + col * 16),
    //         (__m256i)(_mm512_cvtneps_pbh(vc[i])));
    //   }
    // };

  }
};
        // tinygemm_kernel_nn_woq(act, vbs_t, out+k*N_GROUP_SIZE, 2, 2, N_GROUP_SIZE, N_GROUP_SIZE, 1 , N_GROUP_SIZE);

// void print_16x16(const __m256i x) {
//   at::BFloat16 a[16];
//   _mm256_storeu_si256((__m256i *)a, x);

//   for (int i = 0; i < 16; i++){
//     std::cout << a[i] << " ";
//   }
//   std::cout << std::endl;
// }

// void print_32x16(const __m512i x) {
//   at::BFloat16 a[32];
//   _mm512_storeu_si512((__m512i *)a, x);

//   for (int i = 0; i < 32; i++){
//     std::cout << a[i] << " ";
//   }
//   std::cout << std::endl;
// }


#define MM512_SET_M256I(a, b) _mm512_inserti64x4(_mm512_castsi256_si512(a), b, 1)

template <int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn<at::BFloat16, at::Float8_e4m3fn, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const at::BFloat16* __restrict__ A, const at::Float8_e4m3fn* __restrict__ B, at::BFloat16* __restrict__ C,
      float scale, int K, int lda, int ldb, int ldc) {

    TORCH_CHECK(BLOCK_N % 32 == 0, "tinygemm float8 requires BLOCK_N to be 32x.");

    //std::cout << "\n### tinygemm_kernel_nn fp8, scale = " << scale << std::endl;
    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;

    // prefetch distance
    constexpr int PREFETCH_SIZE_K = 0;

    __m512bh va;
    __m512bh vb[COLS];
    __m512 vc[ROWS * COLS];

    __m512 vscale = _mm512_set1_ps(scale);

    auto loadc = [&](auto i) {
      vc[i] = _mm512_set1_ps(0.f);
    };
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
            _mm_prefetch(b_ptr + (k + PREFETCH_SIZE_K) * ldb2 + col * 16, _MM_HINT_T0);
          }
          __m512i idx0 = _mm512_cvtepu8_epi32(   _mm512_castsi512_si128(b8));
          __m512i idx1 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(b8, 1));
          __m512i idx2 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(b8, 2));
          __m512i idx3 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(b8, 3));

          __m512i bf16_i32_vec0 = _mm512_i32gather_epi32(idx0, e4m3_to_16bit, 2);
          __m512i bf16_i32_vec1 = _mm512_i32gather_epi32(idx1, e4m3_to_16bit, 2);
          __m512i bf16_i32_vec2 = _mm512_i32gather_epi32(idx2, e4m3_to_16bit, 2);
          __m512i bf16_i32_vec3 = _mm512_i32gather_epi32(idx3, e4m3_to_16bit, 2);

          __m256i bf16_i16_vec0 = _mm512_cvtepi32_epi16(bf16_i32_vec0);
          __m256i bf16_i16_vec1 = _mm512_cvtepi32_epi16(bf16_i32_vec1);
          __m256i bf16_i16_vec2 = _mm512_cvtepi32_epi16(bf16_i32_vec2);
          __m256i bf16_i16_vec3 = _mm512_cvtepi32_epi16(bf16_i32_vec3);

          vb[col + 0] = (__m512bh)(MM512_SET_M256I(bf16_i16_vec0, bf16_i16_vec1));
          vb[col + 1] = (__m512bh)(MM512_SET_M256I(bf16_i16_vec2, bf16_i16_vec3));

          //if (k == 0) {
          //  print_16x16(bf16_i16_vec0);
          //  print_16x16(bf16_i16_vec1);
          //  print_32x16((__m512i)vb[col + 0]);
          //}

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

#define LAUNCH_TINYGEMM_KERNEL_NN(MB_SIZE, NB_SIZE)                          \
    tinygemm_kernel_nn<scalar_t, packed_t, MB_SIZE, NB_SIZE>::apply(         \
        A + mb_start * lda, B + nb_start * 2, C + mb_start * ldc + nb_start, \
        scale, K, lda, ldb, ldc);

template <typename scalar_t, typename packed_t>
struct brgemm {};

template <typename scalar_t>
struct brgemm<scalar_t, scalar_t> {
  static inline void apply(
      const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* __restrict__ C, float* __restrict__ Ctmp,
      int M, int N, int K, int lda, int ldb, int ldc) {

    constexpr int BLOCK_N = block_size_n();
    at::native::cpublas::brgemm(
        M, N, K, lda, ldb, BLOCK_N, /* add_C */false,
        A, B, Ctmp);

    // copy from Ctmp to C
    for (int m = 0; m < M; ++m) {
      copy_stub(C + m * ldc, Ctmp + m * BLOCK_N, N);
    }
  }
};

template <>
struct brgemm<at::BFloat16, at::Float8_e4m3fn> {
  static inline void apply(
      const at::BFloat16* __restrict__ A, const at::Float8_e4m3fn* __restrict__ B, at::BFloat16* __restrict__ C, float* __restrict__ Ctmp,
      int M, int N, int K, int lda, int ldb, int ldc) {

    std::cout << "### brgemm fp8" << std::endl;

  }
};

template <typename scalar_t, typename packed_t>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const packed_t* __restrict__ B,
    float* __restrict__ C,
    // float* __restrict__ Ctmp,
    float scale,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc
    // bool brg
    ) {

  // if (brg) {
  //   brgemm<scalar_t, packed_t>::apply(A, B, C, Ctmp, M, N, K, lda, ldb, ldc);
  //   return;
  // }

  // pattern: 1-8-8
  if (M == 1) {
    constexpr int BLOCK_N = 32;
    const int NB = div_up(N, BLOCK_N);
    int mb_start = 0;

    for (int nb = 0; nb < NB; ++nb) {
      int nb_start = nb * BLOCK_N;
      int nb_size = std::min(BLOCK_N, N - nb_start);

      switch(nb_size >> 4) {
        case 1: LAUNCH_TINYGEMM_KERNEL_NN(1, 16); break;
        case 2: LAUNCH_TINYGEMM_KERNEL_NN(1, 32); break;
        case 3: LAUNCH_TINYGEMM_KERNEL_NN(1, 48); break;
        case 4: LAUNCH_TINYGEMM_KERNEL_NN(1, 64); break;
        case 5: LAUNCH_TINYGEMM_KERNEL_NN(1, 80); break;
        case 6: LAUNCH_TINYGEMM_KERNEL_NN(1, 96); break;
        case 7: LAUNCH_TINYGEMM_KERNEL_NN(1, 112); break;
        case 8: LAUNCH_TINYGEMM_KERNEL_NN(1, 128); break;
        default: TORCH_CHECK(false, "Unexpected block size, 1x", "nb_size");
      }
    }
    return;
  }

  // pattern: 1-4-16
  constexpr int BLOCK_M = 4;
  constexpr int BLOCK_N = 32;
  const int MB = div_up(M, BLOCK_M);
  const int NB = div_up(N, BLOCK_N);
  for (int mb = 0; mb < MB; ++mb) {
    int mb_start = mb * BLOCK_M;
    int mb_size = std::min(BLOCK_M, M - mb_start);
    for (int nb = 0; nb < NB; ++nb) {
      int nb_start = nb * BLOCK_N;
      int nb_size = std::min(BLOCK_N, N - nb_start);

      switch(mb_size << 4 | nb_size >> 4) {
        // mb_size = 1
        case 0x11: LAUNCH_TINYGEMM_KERNEL_NN(1, 16); break;
        case 0x12: LAUNCH_TINYGEMM_KERNEL_NN(1, 32); break;
        case 0x13: LAUNCH_TINYGEMM_KERNEL_NN(1, 48); break;
        case 0x14: LAUNCH_TINYGEMM_KERNEL_NN(1, 64); break;
        // mb_size = 2
        case 0x21: LAUNCH_TINYGEMM_KERNEL_NN(2, 16); break;
        case 0x22: LAUNCH_TINYGEMM_KERNEL_NN(2, 32); break;
        case 0x23: LAUNCH_TINYGEMM_KERNEL_NN(2, 48); break;
        case 0x24: LAUNCH_TINYGEMM_KERNEL_NN(2, 64); break;
        // mb_size = 3
        case 0x31: LAUNCH_TINYGEMM_KERNEL_NN(3, 16); break;
        case 0x32: LAUNCH_TINYGEMM_KERNEL_NN(3, 32); break;
        case 0x33: LAUNCH_TINYGEMM_KERNEL_NN(3, 48); break;
        case 0x34: LAUNCH_TINYGEMM_KERNEL_NN(3, 64); break;
        // mb_size = 4
        case 0x41: LAUNCH_TINYGEMM_KERNEL_NN(4, 16); break;
        case 0x42: LAUNCH_TINYGEMM_KERNEL_NN(4, 32); break;
        case 0x43: LAUNCH_TINYGEMM_KERNEL_NN(4, 48); break;
        case 0x44: LAUNCH_TINYGEMM_KERNEL_NN(4, 64); break;
        default: TORCH_CHECK(false, "Unexpected block size, ", mb_size, "x", "nb_size");
      }
    }
  }
}

static inline std::array<__m512, 2> interleave_woq(__m512 v0, __m512 v1) {
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

static inline void  dequant_n_grouped_and_compute(
  uint8_t* qB,
  long M,
  long K,
  long N,
  at::BFloat16* scales,
  at::BFloat16* zps,
  const at::BFloat16* act,
  float* out,
  long ldb,
  long N_GROUP_SIZE) {
#if defined(CPU_CAPABILITY_AVX512_BF16)
using T = at::BFloat16;
using VT = typename VecType<T>::type;
using V = VecOps<VT>;
using VA = VecArray<32, float>;
using VA_l = VecArray<32, at::BFloat16>;
using VAT = typename VA::type;
constexpr long COLS = VA::num_vec;
auto load_qparam = [&](at::BFloat16* p) { return VA_l::load1d(p); };
auto load_qint_as_fp = [&](uint8_t* p, auto vscales, auto vzps) {

  return load_dequant_int8<32, false, float>::call(
      p, vscales, vzps);
  
};


for (int m = 0; m < M; m += 1) {

for (int n = 0; n < N; n += N_GROUP_SIZE) {

  // load scales and zps
  int n_ = N-n < N_GROUP_SIZE ? N-n: N_GROUP_SIZE;
constexpr int ROWS_ = 1;  // each m
constexpr int COLS_ = get_n_group_size(WOQ_N_BLOCK_SIZE) / 16 ; //N_GROUP_SIZE / 16;

 __m512bh va;
 __m512bh vb[COLS_];
 __m512 vc[ROWS_ * COLS_];

 auto loadc = [&](auto i) {
   vc[i] = _mm512_set1_ps(0.f);
 };
 Unroll<ROWS_ * COLS_>{}(loadc);
 

  auto vscales = load_qparam(scales + n);
  VAT vzps;
  vzps = load_qparam(zps + n);
  // convert to vnni: [K/2, N, 2]
  // torch::Tensor vbs_ = torch::empty(
  //   {1, N, 2}, c10::CppTypeToScalarType<at::BFloat16>::value);
  // at::BFloat16*  vbs_t = vbs_.data_ptr<at::BFloat16>();
  for (int k = 0; k < K; k += 2) {

    // load and dequant qB to vb
    auto vbs_k0 = load_qint_as_fp(
         &qB[k * ldb + n],
        vscales,
        vzps);
    auto vbs_k1 = load_qint_as_fp(
        &qB[(k + 1) * ldb + n],
        vscales,
        vzps);
    // prefetch qB data
    if constexpr (PREFETCH_K_DIST > 0) {
      auto prefetch_addr = &qB[(k + PREFETCH_K_DIST) * ldb + n];
      _mm_prefetch(prefetch_addr, _MM_HINT_T0);
    }
    // typename VA::type vbs[2];
    __m512i vas_[2];
    compile_time_for<COLS>::op([&](auto i) {
      auto [low, high] = interleave_woq(vbs_k0[i], vbs_k1[i]);
      // already VNNI
      // n=16 x k=2
      if(i < COLS/2){
      vas_[0] = _vec_store_two_floats_as_bfloat16_no_addr(low, high);
      }else{
      vas_[1] = _vec_store_two_floats_as_bfloat16_no_addr(low, high);  
      }
      // vbs[i * 2 / COLS][i * 2 % COLS] = low;
      // vbs[(i * 2 + 1) / COLS][(i * 2 + 1) % COLS] = high;
    });
    // store vb to B: low: [k + n*2 / N, n*2 % N], high: [k +
    // (n*2+N_GROUP_SIZE) / N, (n*2+N_GROUP_SIZE) % N]
    // _vec_store_two_floats_as_bfloat16(vbs_t,vbs[0], vbs[1]);
    // auto store = [&](auto p, auto vbs) {
    //   compile_time_for<COLS / 2>::op([&](auto idx) {
    //     _vec_store_two_floats_as_bfloat16(
    //         p + idx * 32, vbs[idx * 2], vbs[idx * 2 + 1]);
    //   });
    // };
    
    // auto store = [&](auto vbs, int id) {
    //   compile_time_for<COLS / 2>::op([&](auto idx) {
    //     vas_[id] = _vec_store_two_floats_as_bfloat16_no_addr(vbs[idx * 2], vbs[idx * 2 + 1]);
    //   });
    // };
    // store(vbs[0], 0);
    // // // B + (k + (n * 2) / N)*N + (n * 2) % N

    // store(vbs[1], 1);
    // store(ADDRESS(vbs_t, (n * 2) / N, (n * 2) % N, N), vbs[0]);
    // // // B + (k + (n * 2) / N)*N + (n * 2) % N

    // store(
    //     ADDRESS(
    //       vbs_t,
    //         (n * 2 + N_GROUP_SIZE) / N,
    //         (n * 2 + N_GROUP_SIZE) % N,
    //         N),
    //     vbs[1]);

    const int K2 = 2 >> 1;
    const int lda2 = 2 >> 1;
    const int ldb2 = N; // ldb * 2 >> 1;
    // float* b_ptr = reinterpret_cast<float*>(vbs_t);
    const float* a_ptr = reinterpret_cast<const float*>(act + m*K + k);
    auto compute = [&](auto i, int k_) {
      constexpr int row = i / COLS_;
      constexpr int col = i % COLS_;

      if constexpr (col == 0) {
        va = (__m512bh)(_mm512_set1_ps(a_ptr[row * lda2 + k_]));
      }
      if constexpr (row == 0) {
        vb[col] = (__m512bh)(vas_[col]);
      }
      vc[i] = _mm512_dpbf16_ps(vc[i], va, vb[col]);
    };
    for (int k_ = 0; k_ < K2; ++k_) {
      Unroll<ROWS_ * COLS_>{}(compute, k_);
    }

  }//end of 1xKxblock_N
  auto storec = [&](auto i) {
    constexpr int row = i / COLS_;
    constexpr int col = i % COLS_;
    // for COLS = 2, 4 use 512bit store
    // for COLS = 1, 3 use 256bit store
    _mm512_storeu_ps(reinterpret_cast<__m512*>(out+ m*N + col * 16), vc[i]);
  };
  Unroll<ROWS_ * COLS_>{}(storec);
} //end of 1xKxN
} //end of MxKxN
#endif
}


// static inline void Dequantize_and_compute(
// uint8_t* qB,
// long K,
// long N,
// at::BFloat16* scales,
// at::BFloat16* zps,
// const at::BFloat16* act,
// float* out,
// long ldb,
// long N_GROUP_SIZE,
// at::BFloat16* B) {
// #if defined(CPU_CAPABILITY_AVX512_BF16)
// using T = at::BFloat16;
// using VT = typename VecType<T>::type;
// using V = VecOps<VT>;
// // lookup table converting uint8 to float, 15.0f - 0.0f
// // _mm512_permutexvar_ph needs 5 bits while we only need 4 bits, init the
// // table to honor the lower 4 bits regardless of the the highest bit, thus
// // saving an "and" op
// VT lut;
// lut = V::set_0_to_15();
// dequant_n_grouped_and_compute(qB, K, N, scales, zps, act, out, 32, 32, B);
// torch_ipex::cpu::tinygemm_kernel<at::BFloat16, at::BFloat16>(
//   /*   A */ act,
//   /*   B */ B/* nb * BLOCK_N * K */,
//   /*   C */ out,
//   /*scale*/ 0.f,
//   /*   M */ 1,
//   /*   N */ N,
//   /*   K */ K,
//   /* lda */ K,
//   /* ldb */ N,
//   /* ldc */ 32);
// #else

// #endif
// }

// }
} // namespace cpu
} // namespace torch_ipex
