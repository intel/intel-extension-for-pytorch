#include <ATen/Tensor.h>
#include <aten/Decode.h>
#include <aten/FlashAttention.h>
#include <aten/MaskedMultiHeadAttention.h>
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>
#include <limits>
#include "../../utils/isa_utils.h"
#include "vec/vec.h"

namespace torch_ipex {
namespace cpu {
namespace {
using namespace at::vec;
template <
    typename scalar_t,
    typename std::enable_if_t<c10::is_reduced_floating_point_v<scalar_t>, int> =
        0>
inline Vectorized<scalar_t> convert_from_float_ext(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return at::vec::convert_from_float<scalar_t>(a, b);
}
#if defined(CPU_CAPABILITY_AVX512_BF16)
// `at::vec::convert_from_float<>` from PyTorch doesn't have avx512-bf16
// intrinsics use native instruction for bfloat16->float32 conversion
template <>
inline Vectorized<at::BFloat16> convert_from_float_ext<at::BFloat16>(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return (__m512i)(_mm512_cvtne2ps_pbh(__m512(b), __m512(a)));
}
#define CVT_BF16_TO_FP32(a) \
  _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(a), 16))
#define CVT_FP16_TO_FP32(a) \
  _mm512_cvtps_ph(a, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC))
#endif
// debug
template <typename scalar_t>
void print_array(scalar_t* ptr, int size) {
  for (int d = 0; d < size; ++d) {
    if (d % 16 == 0) {
      std::cout << std::endl;
    }
    std::cout << ptr[d] << " ";
  }
  std::cout << std::endl;
}
} // anonymous namespace

namespace {

inline void check_shape(
    const at::Tensor& a,
    const at::Tensor& b,
    const char* a_name,
    const char* b_name) {
  TORCH_CHECK(
      a.dim() == b.dim(),
      a_name,
      ".dim() != ",
      b_name,
      ".dim(). ",
      a.dim(),
      " vs ",
      b.dim());
  for (int i = 0; i < a.dim(); ++i) {
    TORCH_CHECK(
        a.size(i) == b.size(i),
        a_name,
        ".size(",
        i,
        ") != ",
        b_name,
        ".size(",
        i,
        ")");
  }
}

inline constexpr uint32_t pack_u16(uint16_t a, uint16_t b) {
  return (uint32_t(a) << 16) | uint32_t(b);
}

#define TORCH_UNUSED(x) (void)(x)

#define CHECK_CPU(x) \
  TORCH_CHECK(x.device().type() == at::kCPU, #x " must be a CPU tensor")

#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_LAST_DIM_CONTIGUOUS(x)            \
  TORCH_CHECK(                                  \
      x.strides()[x.strides().size() - 1] == 1, \
      #x "must be contiguous at last dimention")

#define CHECK_INPUT(x) \
  CHECK_CPU(x);        \
  CHECK_CONTIGUOUS(x)
#define CHECK_LAST_DIM_CONTIGUOUS_INPUT(x) \
  CHECK_CPU(x);                            \
  CHECK_LAST_DIM_CONTIGUOUS(x)

#define CHECK_DIM(d, x) \
  TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

#define CHECK_SHAPE(a, b) check_shape(a, b, #a, #b)

#define CHECK_EQ(a, b) \
  TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

#define CHECK_GE(a, b) \
  TORCH_CHECK((a) >= (b), "CHECK_GE(" #a ", " #b ") failed. ", a, " vs ", b)
// parallel routines
template <
    typename T,
    typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
inline T div_up(T x, T y) {
  return (x + y - 1) / y;
}
template <typename T>
inline void balance211(T n, T nth, T ith, T& n_start, T& n_end) {
#if 0
    // onednn partition pattern
    T& n_my = n_end;
    if (nth <= 1 || n == 0) {
        n_start = 0;
        n_my = n;
    } else {
        T n1 = div_up(n, nth);
        T n2 = n1 - 1;
        T T1 = n - n2 * nth;
        n_my = ith < T1 ? n1 : n2;
        n_start = ith <= T1 ? ith*n1 : T1 * n1 + (ith - T1) * n2;
    }
    n_end += n_start;
#else
  // pytorch aten partition pattern
  T n_my = div_up(n, nth);
  n_start = ith * n_my;
  n_end = std::min(n_start + n_my, n);
#endif
}
template <typename func_t>
inline void parallel_for(int n, const func_t& f) {
#if defined(_OPENMP)
#pragma omp parallel
  {
    int nth = omp_get_num_threads();
    int ith = omp_get_thread_num();
    int tbegin, tend;
    balance211(n, nth, ith, tbegin, tend);
    f(tbegin, tend);
  }
#else
  f(0, n);
#endif
}
// data indexing for dimension collapse
template <typename T>
inline T data_index_init(T offset) {
  return offset;
}
template <typename T, typename... Args>
inline T data_index_init(T offset, T& x, const T& X, Args&&... args) {
  offset = data_index_init(offset, std::forward<Args>(args)...);
  x = offset % X;
  return offset / X;
}
inline bool data_index_step() {
  return true;
}

template <typename T, typename... Args>
inline bool data_index_step(T& x, const T& X, Args&&... args) {
  if (data_index_step(std::forward<Args>(args)...)) {
    x = ((x + 1) == X) ? 0 : (x + 1);
    return x == 0;
  }
  return false;
}
// forced unroll for perf critical path
#if __has_attribute(always_inline)
#define ALWAYS_INLINE __attribute__((__always_inline__)) inline
#else
#define ALWAYS_INLINE inline
#endif
// TODO: remove me
#define STRINGIFY(x) #x // Turns the argument into a string
#define TOSTRING(x) STRINGIFY(x) // Handles nested macros
template <int n>
struct Unroll {
  template <typename Func, typename... Args>
  ALWAYS_INLINE void operator()(const Func& f, Args... args) const {
    Unroll<n - 1>{}(f, args...);
    f(std::integral_constant<int, n - 1>{}, args...);
  }
};
template <>
struct Unroll<1> {
  template <typename Func, typename... Args>
  ALWAYS_INLINE void operator()(const Func& f, Args... args) const {
    f(std::integral_constant<int, 0>{}, args...);
  }
};
} // anonymous namespace
namespace {
// [NOTE] TODO list for this kernel:
//   1. tune the value for BLOCK_N
//   2. planning for {batches, num_heads, num_kv_splits}
//      and use actual num_kv_splits for small seq length
//   3. try fast impl of `.tanh()`
//   4. provide amx kernel for index_gemm_kernel_nn when M = 16
//
inline void fill_stub(float* __restrict__ out, float val, int size) {
  using Vec = at::vec::Vectorized<float>;
  const Vec data_vec(val);
  at::vec::map<float>(
      [data_vec](Vec out) { return out = data_vec; }, out, out, size);
}
template <typename scalar_t>
inline void copy_stub(
    scalar_t* __restrict__ out,
    const float* __restrict__ acc,
    float s,
    int size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  const fVec s_fvec = fVec(s);
  int d = 0;
  for (; d <= size - bVec::size(); d += bVec::size()) {
    fVec a_fvec0 = fVec::loadu(acc + d) * s_fvec;
    fVec a_fvec1 = fVec::loadu(acc + d + fVec::size()) * s_fvec;
    bVec out_bvec = convert_from_float_ext<scalar_t>(a_fvec0, a_fvec1);
    out_bvec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(acc[d] * s);
  }
}
// GEMM handles query @ key (indexed) x scale
//   A : [M, K]
//   B : [N, K] indexed
//   C : [M, N]
//
template <typename scalar_t, typename index_t, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nt {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const scalar_t* __restrict__ B,
      float* __restrict__ C,
      const index_t* __restrict__ indices,
      float scale,
      int lda,
      int ldb,
      int ldc,
      int K,
      int max_tokens) {
    for (int m = 0; m < BLOCK_M; ++m) {
      for (int n = 0; n < BLOCK_N; ++n) {
        float sum = 0.f;
        int b_idx = indices[n];
        TORCH_CHECK(b_idx < max_tokens, "token index out of scope!");
        for (int k = 0; k < K; ++k) {
          sum += scale * static_cast<float>(A[m * lda + k]) *
              static_cast<float>(B[b_idx * ldb + k]);
        }
        C[m * ldc + n] = sum;
      }
    }
  }
};
#if defined(CPU_CAPABILITY_AVX512_BF16)
template <typename index_t, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nt<at::BFloat16, index_t, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const at::BFloat16* __restrict__ B,
      float* __restrict__ C,
      const index_t* __restrict__ indices,
      float scale,
      int lda,
      int ldb,
      int ldc,
      int K,
      int max_tokens) {
    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N;
    __m512bh va;
    __m512bh vb[COLS];
    __m512 vc[ROWS * COLS];
    __m512 vscale = _mm512_set1_ps(scale);
    auto loadc = [&](auto i) { vc[i] = _mm512_setzero_ps(); };
    Unroll<ROWS * COLS>{}(loadc);
    // for main loop
    auto compute = [&](auto i, int k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      if constexpr (col == 0) {
        va = (__m512bh)(_mm512_loadu_si512(A + row * lda + k));
      }
      if constexpr (row == 0) {
        if constexpr (col + 1 < COLS) {
          int b_idx_prefetch = indices[col + 1];
          _mm_prefetch(B + b_idx_prefetch * ldb + k, _MM_HINT_T0);
        }
        int b_idx = indices[col];
        TORCH_CHECK(b_idx < max_tokens, "token index out of scope!");
        vb[col] = (__m512bh)(_mm512_loadu_si512(B + b_idx * ldb + k));
      }
      vc[i] = _mm512_dpbf16_ps(vc[i], va, vb[col]);
    };
    // for remainder
    auto compute2 = [&](auto i, int k, __mmask32 mask) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      if constexpr (col == 0) {
        va = (__m512bh)(_mm512_maskz_loadu_epi16(mask, A + row * lda + k));
      }
      if constexpr (row == 0) {
        int b_idx = indices[col];
        TORCH_CHECK(b_idx < max_tokens, "token index out of scope!");
        vb[col] =
            (__m512bh)(_mm512_maskz_loadu_epi16(mask, B + b_idx * ldb + k));
      }
      vc[i] = _mm512_dpbf16_ps(vc[i], va, vb[col]);
    };
    int k = 0;
    for (; k <= K - 32; k += 32) {
      Unroll<ROWS * COLS>{}(compute, k);
    }
    int count = K - k;
    if (count > 0) {
      __mmask32 mask = (1ULL << count) - 1;
      Unroll<ROWS * COLS>{}(compute2, k, mask);
    }
    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      C[row * ldc + col] = _mm512_reduce_add_ps(_mm512_mul_ps(vc[i], vscale));
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};
#endif
#define LAUNCH_TINYGEMM_KERNEL_NT(MB_SIZE, NB_SIZE)               \
  tinygemm_kernel_nt<scalar_t, index_t, MB_SIZE, NB_SIZE>::apply( \
      A + mb_start * lda,                                         \
      B,                                                          \
      C + mb_start * ldc + nb_start,                              \
      indices + nb_start,                                         \
      scale,                                                      \
      lda,                                                        \
      ldb,                                                        \
      ldc,                                                        \
      K,                                                          \
      max_tokens);
// this is used when N isn't multiple of 16,
// N corresponds to `head_size_v` which should be 16x
template <typename scalar_t, typename index_t>
inline void tinygemm_kernel_nn_scalar(
    const float* __restrict__ A,
    const scalar_t* __restrict__ B,
    float* __restrict__ C,
    const index_t* __restrict__ indices,
    const float* __restrict__ scale,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    int max_tokens) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      C[m * ldc + n] *= scale[m];
      for (int k = 0; k < K; ++k) {
        int b_idx = indices[k];
        TORCH_CHECK(b_idx < max_tokens, "token index out of scope!");
        C[m * ldc + n] +=
            A[m * lda + k] * static_cast<float>(B[b_idx * ldb + n]);
      }
    }
  }
}
// GEMM handles v' * scale + attn @ value (indexed)
//   A : [M, K]
//   B : [K, N] indexed
//   C ：[M, N]
//
template <typename scalar_t, typename index_t, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn {
  static inline void apply(
      const float* __restrict__ A,
      const scalar_t* __restrict__ B,
      float* __restrict__ C,
      const index_t* __restrict__ indices,
      const float* __restrict__ scale,
      int lda,
      int ldb,
      int ldc,
      int K,
      int max_tokens) {
    tinygemm_kernel_nn_scalar(
        A,
        B,
        C,
        indices,
        scale,
        BLOCK_M,
        BLOCK_N,
        K,
        lda,
        ldb,
        ldc,
        max_tokens);
  }
};
#if defined(CPU_CAPABILITY_AVX512_BF16)
template <typename index_t, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn<at::BFloat16, index_t, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const float* __restrict__ A,
      const at::BFloat16* __restrict__ B,
      float* __restrict__ C,
      const index_t* __restrict__ indices,
      const float* __restrict__ scale,
      int lda,
      int ldb,
      int ldc,
      int K,
      int max_tokens) {
    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;
    __m512 va;
    __m512 vb[COLS];
    __m512 vc[ROWS * COLS];
    __m512 vscale;
    auto loadc = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
      if constexpr (col == 0) {
        vscale = _mm512_set1_ps(scale[row]);
      }
#pragma GCC diagnostic pop
      vc[i] = _mm512_loadu_ps(C + row * ldc + col * 16);
      vc[i] = _mm512_mul_ps(vc[i], vscale);
    };
    Unroll<ROWS * COLS>{}(loadc);
    auto compute = [&](auto i, int k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      if constexpr (col == 0) {
        va = _mm512_set1_ps(A[row * lda + k]);
      }
      if constexpr (row == 0) {
        if (k + 1 < K) {
          int b_idx_prefetch = indices[k + 1];
          _mm_prefetch(B + b_idx_prefetch * ldb + col * 16, _MM_HINT_T0);
        }
        int b_idx = indices[k];
        TORCH_CHECK(b_idx < max_tokens, "token index out of scope!");
        // for COLS = 2, 4, 6, 8 use 512 bit load
        // for COLS = 1, 3, 5, 7 use 256 bit load
        if constexpr (COLS % 2 == 0) {
          if constexpr (col % 2 == 0) {
            __m512i b16 = _mm512_loadu_si512(
                reinterpret_cast<const __m512i*>(B + b_idx * ldb + col * 16));
            vb[col + 0] = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(b16, 0));
            vb[col + 1] = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(b16, 1));
          }
        } else {
          __m256i b16 = _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(B + b_idx * ldb + col * 16));
          vb[col] = CVT_BF16_TO_FP32(b16);
        }
      }
      vc[i] = _mm512_fmadd_ps(va, vb[col], vc[i]);
    };
    for (int k = 0; k < K; ++k) {
      Unroll<ROWS * COLS>{}(compute, k);
    }
    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      _mm512_storeu_ps(C + row * ldc + col * 16, vc[i]);
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};
#endif
#define LAUNCH_TINYGEMM_KERNEL_NN(MB_SIZE, NB_SIZE)               \
  tinygemm_kernel_nn<scalar_t, index_t, MB_SIZE, NB_SIZE>::apply( \
      A + mb_start * lda,                                         \
      B + nb_start,                                               \
      C + mb_start * ldc + nb_start,                              \
      indices,                                                    \
      scale + mb_start,                                           \
      lda,                                                        \
      ldb,                                                        \
      ldc,                                                        \
      K,                                                          \
      max_tokens);
template <typename scalar_t, typename index_t>
void index_gemm_kernel_nt(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    float* __restrict__ C,
    const index_t* __restrict__ indices,
    float scale,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    int max_tokens) {
  // pattern: 1-8-8
  if (M == 1) {
    constexpr int BLOCK_N = 8;
    const int NB = div_up(N, BLOCK_N);
    int mb_start = 0, lda = 1, ldc = 1;
    for (int nb = 0; nb < NB; ++nb) {
      int nb_start = nb * BLOCK_N;
      int nb_size = std::min(BLOCK_N, N - nb_start);
      switch (nb_size) {
        case 1:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 1);
          break;
        case 2:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 2);
          break;
        case 3:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 3);
          break;
        case 4:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 4);
          break;
        case 5:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 5);
          break;
        case 6:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 6);
          break;
        case 7:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 7);
          break;
        case 8:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 8);
          break;
        default:
          TORCH_CHECK(false, "Unexpected block size, 1x", "nb_size");
      }
    }
    return;
  }
  // pattern: 1-6-24
  constexpr int BLOCK_M = 4;
  constexpr int BLOCK_N = 6;
  const int MB = div_up(M, BLOCK_M);
  const int NB = div_up(N, BLOCK_N);
  for (int mb = 0; mb < MB; ++mb) {
    int mb_start = mb * BLOCK_M;
    int mb_size = std::min(BLOCK_M, M - mb_start);
    for (int nb = 0; nb < NB; ++nb) {
      int nb_start = nb * BLOCK_N;
      int nb_size = std::min(BLOCK_N, N - nb_start);
      switch (mb_size << 4 | nb_size) {
        // mb_size = 1
        case 0x11:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 1);
          break;
        case 0x12:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 2);
          break;
        case 0x13:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 3);
          break;
        case 0x14:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 4);
          break;
        case 0x15:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 5);
          break;
        case 0x16:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 6);
          break;
        // mb_size = 2
        case 0x21:
          LAUNCH_TINYGEMM_KERNEL_NT(2, 1);
          break;
        case 0x22:
          LAUNCH_TINYGEMM_KERNEL_NT(2, 2);
          break;
        case 0x23:
          LAUNCH_TINYGEMM_KERNEL_NT(2, 3);
          break;
        case 0x24:
          LAUNCH_TINYGEMM_KERNEL_NT(2, 4);
          break;
        case 0x25:
          LAUNCH_TINYGEMM_KERNEL_NT(2, 5);
          break;
        case 0x26:
          LAUNCH_TINYGEMM_KERNEL_NT(2, 6);
          break;
        // mb_size = 3
        case 0x31:
          LAUNCH_TINYGEMM_KERNEL_NT(3, 1);
          break;
        case 0x32:
          LAUNCH_TINYGEMM_KERNEL_NT(3, 2);
          break;
        case 0x33:
          LAUNCH_TINYGEMM_KERNEL_NT(3, 3);
          break;
        case 0x34:
          LAUNCH_TINYGEMM_KERNEL_NT(3, 4);
          break;
        case 0x35:
          LAUNCH_TINYGEMM_KERNEL_NT(3, 5);
          break;
        case 0x36:
          LAUNCH_TINYGEMM_KERNEL_NT(3, 6);
          break;
        // mb_size = 4
        case 0x41:
          LAUNCH_TINYGEMM_KERNEL_NT(4, 1);
          break;
        case 0x42:
          LAUNCH_TINYGEMM_KERNEL_NT(4, 2);
          break;
        case 0x43:
          LAUNCH_TINYGEMM_KERNEL_NT(4, 3);
          break;
        case 0x44:
          LAUNCH_TINYGEMM_KERNEL_NT(4, 4);
          break;
        case 0x45:
          LAUNCH_TINYGEMM_KERNEL_NT(4, 5);
          break;
        case 0x46:
          LAUNCH_TINYGEMM_KERNEL_NT(4, 6);
          break;
        default:
          TORCH_CHECK(
              false, "Unexpected block size, ", mb_size, "x", "nb_size");
      }
    }
  }
}
template <typename scalar_t, typename index_t>
void index_gemm_kernel_nn(
    const float* __restrict__ A,
    const scalar_t* __restrict__ B,
    float* __restrict__ C,
    const index_t* __restrict__ indices,
    float* __restrict__ scale,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    int max_tokens) {
  constexpr int kVecSize = 16;
  if ((N & (kVecSize - 1)) != 0) {
    tinygemm_kernel_nn_scalar(
        A, B, C, indices, scale, M, N, K, lda, ldb, ldc, max_tokens);
    return;
  }
  // pattern: 1-8-8
  if (M == 1) {
    constexpr int BLOCK_N = 8 * kVecSize;
    const int NB = div_up(N, BLOCK_N);
    int mb_start = 0, lda = 1, ldc = 1;
    for (int nb = 0; nb < NB; ++nb) {
      int nb_start = nb * BLOCK_N;
      int nb_size = std::min(BLOCK_N, N - nb_start);
      switch (nb_size >> 4) {
        case 1:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 16);
          break;
        case 2:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 32);
          break;
        case 3:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 48);
          break;
        case 4:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 64);
          break;
        case 5:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 80);
          break;
        case 6:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 96);
          break;
        case 7:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 112);
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
  constexpr int BLOCK_M = 4;
  constexpr int BLOCK_N = 6 * kVecSize;
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
        case 0x11:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 16);
          break;
        case 0x12:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 32);
          break;
        case 0x13:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 48);
          break;
        case 0x14:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 64);
          break;
        case 0x15:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 80);
          break;
        case 0x16:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 96);
          break;
        // mb_size = 2
        case 0x21:
          LAUNCH_TINYGEMM_KERNEL_NN(2, 16);
          break;
        case 0x22:
          LAUNCH_TINYGEMM_KERNEL_NN(2, 32);
          break;
        case 0x23:
          LAUNCH_TINYGEMM_KERNEL_NN(2, 48);
          break;
        case 0x24:
          LAUNCH_TINYGEMM_KERNEL_NN(2, 64);
          break;
        case 0x25:
          LAUNCH_TINYGEMM_KERNEL_NN(2, 80);
          break;
        case 0x26:
          LAUNCH_TINYGEMM_KERNEL_NN(2, 96);
          break;
        // mb_size = 3
        case 0x31:
          LAUNCH_TINYGEMM_KERNEL_NN(3, 16);
          break;
        case 0x32:
          LAUNCH_TINYGEMM_KERNEL_NN(3, 32);
          break;
        case 0x33:
          LAUNCH_TINYGEMM_KERNEL_NN(3, 48);
          break;
        case 0x34:
          LAUNCH_TINYGEMM_KERNEL_NN(3, 64);
          break;
        case 0x35:
          LAUNCH_TINYGEMM_KERNEL_NN(3, 80);
          break;
        case 0x36:
          LAUNCH_TINYGEMM_KERNEL_NN(3, 96);
          break;
        // mb_size = 4
        case 0x41:
          LAUNCH_TINYGEMM_KERNEL_NN(4, 16);
          break;
        case 0x42:
          LAUNCH_TINYGEMM_KERNEL_NN(4, 32);
          break;
        case 0x43:
          LAUNCH_TINYGEMM_KERNEL_NN(4, 48);
          break;
        case 0x44:
          LAUNCH_TINYGEMM_KERNEL_NN(4, 64);
          break;
        case 0x45:
          LAUNCH_TINYGEMM_KERNEL_NN(4, 80);
          break;
        case 0x46:
          LAUNCH_TINYGEMM_KERNEL_NN(4, 96);
          break;
        default:
          TORCH_CHECK(
              false, "Unexpected block size, ", mb_size, "x", "nb_size");
      }
    }
  }
}
template <typename scalar_t, typename index_t>
void decode_attention_kernel_impl(
    scalar_t* __restrict__ output,
    float* __restrict__ attn_logits,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ kv_cache,
    const index_t* __restrict__ beam_idx,
    int seq_len_kv,
    int batches,
    int num_heads,
    int head_size,
    int head_size_v,
    int num_kv_splits,
    float scaling,
    float logit_cap,
    int max_total_num_tokens,
    int beam_stride0) {
  using Vec = at::vec::Vectorized<float>;
  // block length for k_cache and v_cache
  constexpr int BLOCK_N = 64;
  // strides
  const int stride_q0 = num_heads * head_size;
  const int stride_q1 = head_size;
  const int stride_kv0 = num_heads * head_size;
  const int stride_kv1 = head_size;
  const int stride_l1 = num_kv_splits * (head_size_v + 1);
  const int stride_l2 = head_size_v + 1;
  const bool has_logit_cap = logit_cap > 0;
  float rlogit_cap = has_logit_cap ? 1 / logit_cap : 0.f;
  // parallel on [batches, num_heads, num_kv_splits]
  parallel_for(batches * num_heads * num_kv_splits, [&](int begin, int end) {
    int bs{0}, head_id{0}, kv_id{0};
    data_index_init(
        begin, bs, batches, head_id, num_heads, kv_id, num_kv_splits);
    // s_prime and s_delta
    static thread_local float s_i[BLOCK_N];
    float* __restrict__ s_delta = s_i;
    for (int i = begin; i < end; ++i) {
      // get query
      const scalar_t* __restrict__ q_ptr =
          query + bs * stride_q0 + head_id * stride_q1;
      // get key/value
      const int SPLIT_SIZE = div_up(seq_len_kv, num_kv_splits);
      const int kv_start = kv_id * SPLIT_SIZE;
      const int kv_end = std::min(kv_start + SPLIT_SIZE, seq_len_kv);
      float m_prime = -std::numeric_limits<float>::infinity();
      float s_prime = 0.f;
      // get v_prime, and init to zero
      float* __restrict__ v_prime = attn_logits + i * (head_size_v + 1);
      fill_stub(v_prime, 0.f, head_size_v);
      // loop over K and V sequence with BLOCK_N
      for (int n = kv_start; n < kv_end; n += BLOCK_N) {
        int n_size = std::min(BLOCK_N, kv_end - n);
        // calculate s_i <- scale * Q @ K
        index_gemm_kernel_nt<scalar_t, index_t>(
            /* A   */ q_ptr,
            /* B   */ kv_cache + head_id * stride_kv1,
            /* C   */ s_i,
            /* ind */ beam_idx + bs * beam_stride0 + n,
            /* scl */ scaling,
            /* M   */ 1,
            /* N   */ n_size,
            /* K   */ head_size,
            /* lda */ 1,
            /* ldb */ stride_kv0,
            /* ldc */ 1,
            /* mtt */ max_total_num_tokens);
        // TODO: `tanh` from torch uses sleef u10, going to be slow
        if (has_logit_cap) {
          at::vec::map<float>(
              [logit_cap, rlogit_cap](Vec x) {
                return Vec(logit_cap) * (x * Vec(rlogit_cap)).tanh();
              },
              s_i,
              s_i,
              n_size);
        }
        // m_i: max value per row
        float m_i = at::vec::reduce_all<float>(
            [](Vec& x, Vec& y) { return at::vec::maximum(x, y); }, s_i, n_size);
        m_i = std::max(m_i, m_prime);
        // m_delta <- exp(m' - m_i)
        float m_delta = std::exp(m_prime - m_i);
        // s_delta <- exp(s_i - m_i)
        at::vec::map<float>(
            [m_i](Vec x) { return (x - Vec(m_i)).exp_u20(); },
            s_delta,
            s_i,
            n_size);
        // s' <- s' * m_delta + sum(s_delta)
        s_prime *= m_delta;
        s_prime += at::vec::reduce_all<float>(
            [](Vec& x, Vec& y) { return x + y; }, s_delta, n_size);
        m_prime = m_i;
        // caculate V' <- s_delta @ V + V' * m_delta
        index_gemm_kernel_nn(
            /* A   */ s_delta,
            /* B   */ kv_cache + head_id * stride_kv1,
            /* C   */ v_prime,
            /* ind */ beam_idx + bs * beam_stride0 + n,
            /* scl */ &m_delta,
            /* M   */ 1,
            /* N   */ head_size_v,
            /* K   */ n_size,
            /* lda */ 1,
            /* ldb */ stride_kv0,
            /* ldc */ 1,
            /* mtt */ max_total_num_tokens);
      } // loop with KV blocks
      float s = 1 / s_prime;
      at::vec::map<float>(
          [s](Vec out) { return out * Vec(s); }, v_prime, v_prime, head_size_v);
      v_prime[head_size_v] = m_prime + std::log(s_prime);
      // move to the next index
      data_index_step(bs, batches, head_id, num_heads, kv_id, num_kv_splits);
    }
  });
  // parallel on [batches, num_heads]
  parallel_for(batches * num_heads, [&](int begin, int end) {
    // NB: here we use logits[b][h][0] as acc, since
    // for the first kv split (kv_id == 0):
    //   m_delta = std::exp(-inf) = 0
    //   e_logic = std::exp(0) = 1
    //   acc = acc * m_delta + tv * e_logic = tv
    for (int i = begin; i < end; ++i) {
      float* __restrict__ acc = attn_logits + i * stride_l1;
      float s_prime = 0.f;
      float m_prime = -std::numeric_limits<scalar_t>::infinity();
      // update acc with from each kv_split
      for (int kv_id = 0; kv_id < num_kv_splits; ++kv_id) {
        float* __restrict__ tv = acc + kv_id * stride_l2;
        const float tlogic = (acc + kv_id * stride_l2)[head_size_v];
        float m_i = std::max(tlogic, m_prime);
        float m_delta = std::exp(m_prime - m_i);
        float e_logic = std::exp(tlogic - m_i);
        if (kv_id != 0) {
          at::vec::map2<float>(
              [m_delta, e_logic](Vec x, Vec y) {
                return x * Vec(m_delta) + y * Vec(e_logic);
              },
              acc,
              acc,
              tv,
              head_size_v);
        }
        s_prime = s_prime * m_delta + e_logic;
        m_prime = m_i;
      }
      copy_stub<scalar_t>(
          output + i * head_size_v, acc, 1 / s_prime, head_size_v);
    }
  });
}
template <typename scalar_t, typename index_t>
void decode_attention_grouped_kernel_impl(
    scalar_t* __restrict__ output,
    float* __restrict__ attn_logits,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ kv_cache,
    const index_t* __restrict__ beam_idx,
    int seq_len_kv,
    int batches,
    int num_heads,
    int num_heads_kv,
    int head_size,
    int head_size_v,
    int num_kv_splits,
    float scaling,
    float logit_cap,
    int max_total_num_tokens,
    int beam_stride0) {
  using Vec = at::vec::Vectorized<float>;
  // block length for k_cache and v_cache
  // TODO: tune BLOCK_N 256/512
  constexpr int BLOCK_N = 512;
  // block length for heads
  constexpr int BLOCK_H = 16;
  // strides
  const int stride_q0 = num_heads * head_size;
  const int stride_q1 = head_size;
  const int stride_kv0 = num_heads_kv * head_size;
  const int stride_kv1 = head_size;
  const int stride_l0 = num_heads * num_kv_splits * (head_size_v + 1);
  const int stride_l1 = num_kv_splits * (head_size_v + 1);
  const int stride_l2 = head_size_v + 1;
  const bool has_logit_cap = logit_cap > 0;
  float rlogit_cap = has_logit_cap ? 1 / logit_cap : 0.f;
  // partition the heads into blocks for parallel
  const int num_groups = num_heads / num_heads_kv;
  const int num_blocks = div_up(num_heads, std::min(BLOCK_H, num_groups));
  const int num_groups_per_block = div_up(num_groups, BLOCK_H);
  const int num_heads_per_block = std::min(num_groups, BLOCK_H);
  // parallel on [batches, num_blocks, num_kv_splits]
  parallel_for(batches * num_blocks * num_kv_splits, [&](int begin, int end) {
    int bs{0}, head_id{0}, kv_id{0};
    data_index_init(
        begin, bs, batches, head_id, num_blocks, kv_id, num_kv_splits);
    static thread_local float s_i[BLOCK_H * BLOCK_N];
    float* __restrict__ s_delta = s_i;
    static thread_local float s_prime[BLOCK_H];
    static thread_local float m_prime[BLOCK_H];
    static thread_local float m_delta[BLOCK_H];
    for (int i = begin; i < end; ++i) {
      const int h_start = head_id * num_heads_per_block;
      const int h_end = std::min(h_start + num_heads_per_block, num_heads);
      const int h_size = h_end - h_start;
      // get query
      const scalar_t* __restrict__ q_ptr =
          query + bs * stride_q0 + h_start * stride_q1;
      // kv head id and valid block head size
      int head_kv_id = head_id / num_groups_per_block;
      const int SPLIT_SIZE = div_up(seq_len_kv, num_kv_splits);
      const int kv_start = kv_id * SPLIT_SIZE;
      const int kv_end = std::min(kv_start + SPLIT_SIZE, seq_len_kv);
      fill_stub(s_prime, 0.f, BLOCK_H);
      fill_stub(m_prime, -std::numeric_limits<float>::infinity(), BLOCK_H);
      // get v_prime, and init to zero
      float* __restrict__ v_prime = attn_logits + bs * stride_l0 +
          h_start * stride_l1 + kv_id * stride_l2;
      for (int h = 0; h < h_size; ++h) {
        fill_stub(v_prime + h * stride_l1, 0.f, head_size_v);
      }
      // loop over K and V sequence with BLOCK_N
      for (int n = kv_start; n < kv_end; n += BLOCK_N) {
        int n_size = std::min(BLOCK_N, kv_end - n);
        // calculate Q @ K
        index_gemm_kernel_nt<scalar_t, index_t>(
            /* A   */ q_ptr,
            /* B   */ kv_cache + head_kv_id * stride_kv1,
            /* C   */ s_i,
            /* ind */ beam_idx + bs * beam_stride0 + n,
            /* scl */ scaling,
            /* M   */ h_size,
            /* N   */ n_size,
            /* K   */ head_size,
            /* lda */ stride_q1,
            /* ldb */ stride_kv0,
            /* ldc */ BLOCK_N,
            /* mtt */ max_total_num_tokens);
        if (has_logit_cap) {
          at::vec::map<float>(
              [logit_cap, rlogit_cap](Vec x) {
                return Vec(logit_cap) * (x * Vec(rlogit_cap)).tanh();
              },
              s_i,
              s_i,
              n_size);
        }
        // update the scaling coefficients
        for (int h = 0; h < h_size; ++h) {
          // m_i: max value per row
          float m_i = at::vec::reduce_all<float>(
              [](Vec& x, Vec& y) { return at::vec::maximum(x, y); },
              s_i + h * BLOCK_N,
              n_size);
          m_i = std::max(m_i, m_prime[h]);
          // m_delta <- exp(m' - m_i)
          m_delta[h] = std::exp(m_prime[h] - m_i);
          // s_delta <- exp(s_i - m_i)
          at::vec::map<float>(
              [m_i](Vec x) { return (x - Vec(m_i)).exp_u20(); },
              s_delta + h * BLOCK_N,
              s_i + h * BLOCK_N,
              n_size);
          // s' <- s' * m_delta + sum(s_delta)
          s_prime[h] *= m_delta[h];
          s_prime[h] += at::vec::reduce_all<float>(
              [](Vec& x, Vec& y) { return x + y; },
              s_delta + h * BLOCK_N,
              n_size);
          m_prime[h] = m_i;
        }
        // caculate V' <- s_delta @ V + V' * m_delta
        index_gemm_kernel_nn(
            /* A   */ s_delta,
            /* B   */ kv_cache + head_kv_id * stride_kv1,
            /* C   */ v_prime,
            /* ind */ beam_idx + bs * beam_stride0 + n,
            /* scl */ m_delta,
            /* M   */ h_size,
            /* N   */ head_size_v,
            /* K   */ n_size,
            /* lda */ BLOCK_N,
            /* ldb */ stride_kv0,
            /* ldc */ stride_l1,
            /* mtt */ max_total_num_tokens);
      } // loop with KV blocks
      for (int h = 0; h < h_size; ++h) {
        float s = std::fabs(s_prime[h]) < 1e-9 ? 0 : 1 / s_prime[h];
        at::vec::map<float>(
            [s](Vec out) { return out * Vec(s); },
            v_prime + h * stride_l1,
            v_prime + h * stride_l1,
            head_size_v);
        (v_prime + h * stride_l1)[head_size_v] =
            m_prime[h] + std::log(s_prime[h]);
      }
      // move to the next index
      data_index_step(bs, batches, head_id, num_blocks, kv_id, num_kv_splits);
    }
  });
  // parallel on [batches, num_heads]
  parallel_for(batches * num_heads, [&](int begin, int end) {
    // NB: same as above
    for (int i = begin; i < end; ++i) {
      float* __restrict__ acc = attn_logits + i * stride_l1;
      float s_prime = 0.f;
      float m_prime = -std::numeric_limits<scalar_t>::infinity();
      // update acc with from each kv_split
      for (int kv_id = 0; kv_id < num_kv_splits; ++kv_id) {
        float* __restrict__ tv = acc + kv_id * stride_l2;
        const float tlogic = (acc + kv_id * stride_l2)[head_size_v];
        float m_i = std::max(tlogic, m_prime);
        float m_delta = std::exp(m_prime - m_i);
        float e_logic = std::exp(tlogic - m_i);
        if (kv_id != 0) {
          at::vec::map2<float>(
              [m_delta, e_logic](Vec x, Vec y) {
                return x * Vec(m_delta) + y * Vec(e_logic);
              },
              acc,
              acc,
              tv,
              head_size_v);
        }
        s_prime = s_prime * m_delta + e_logic;
        m_prime = m_i;
      }
      copy_stub<scalar_t>(
          output + i * head_size_v, acc, 1 / s_prime, head_size_v);
    }
  });
}
// query: [bs, cur_len, num_heads, head_size]
// output: [bs, num_heads, cur_len, head_size]
// kv_cache: [max_positions, beam_batch, kv_num_heads, head_size]
// beam_idx: [bs, offset+cur_len+1]
// attn_logits: [bs, num_heads, num_kv_splits, head_size_v + 1]
at::Tensor decode_attention(
    at::Tensor& query,
    at::Tensor& output,
    at::Tensor& kv_cache,
    at::Tensor& beam_idx,
    at::Tensor& attn_logits,
    double scaling,
    double logit_cap,
    int64_t offset) {
  RECORD_FUNCTION("ipex::decode_attention", c10::ArrayRef<c10::IValue>({}));
  CHECK_INPUT(query);
  CHECK_INPUT(kv_cache);
  CHECK_DIM(4, query);
  CHECK_DIM(4, kv_cache);
  int max_total_num_tokens = kv_cache.size(0) * kv_cache.size(1);
  int bs = query.size(0);
  int num_heads = query.size(-2);
  int num_heads_kv = kv_cache.size(2);
  int head_size = query.size(-1);
  int head_size_v = attn_logits.size(-1) - 1;
  int num_kv_splits = attn_logits.size(2);
  int beam_stride0 = beam_idx.stride(0);
  CHECK_EQ(attn_logits.size(1), num_heads);
  CHECK_EQ(attn_logits.size(3), head_size_v + 1);
  CHECK_EQ(attn_logits.scalar_type(), at::kFloat);
  // make sure all the indices have the same data type
  const auto index_dtype = beam_idx.scalar_type();
  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      query.scalar_type(), "decode_attention_kernel", [&] {
        AT_DISPATCH_INDEX_TYPES(index_dtype, "decode_attention_indices", [&] {
          if (num_heads == num_heads_kv) {
            // MHA
            decode_attention_kernel_impl<scalar_t, index_t>(
                output.data_ptr<scalar_t>(),
                attn_logits.data_ptr<float>(),
                query.data_ptr<scalar_t>(),
                kv_cache.data_ptr<scalar_t>(),
                beam_idx.data_ptr<index_t>(),
                offset,
                bs,
                num_heads,
                head_size,
                head_size_v,
                num_kv_splits,
                scaling,
                logit_cap,
                max_total_num_tokens,
                beam_stride0);
          } else {
            // GQA/MQA/MLA
            decode_attention_grouped_kernel_impl<scalar_t, index_t>(
                output.data_ptr<scalar_t>(),
                attn_logits.data_ptr<float>(),
                query.data_ptr<scalar_t>(),
                kv_cache.data_ptr<scalar_t>(),
                beam_idx.data_ptr<index_t>(),
                offset,
                bs,
                num_heads,
                num_heads_kv,
                head_size,
                head_size_v,
                num_kv_splits,
                scaling,
                logit_cap,
                max_total_num_tokens,
                beam_stride0);
          }
        });
      });
  return output;
}
} // anonymous namespace

IPEX_REGISTER_DISPATCH(decode_attention_kernel_stub, &decode_attention);

} // namespace cpu
} // namespace torch_ipex