#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <aten/Linear.h>
#include <emmintrin.h>
#include <libxsmm.h>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>
#include "assert.h"
#include "vec/vec.h"

namespace torch_ipex {
namespace cpu {
namespace {

#if defined(CPU_CAPABILITY_AVX512)
#include <immintrin.h>
#define INDEX(x, y, ld) ((x) * (ld) + (y))
#define ADDRESS(p, x, y, ld) ((p) + (x) * (ld) + (y))

// Get mask for last column
template <int EXPANDED_N, int col>
constexpr inline unsigned short get_mask(unsigned short mask) {
  // Not last column, return 0xffffff indicating load/store all 16 floats
  if constexpr (col < EXPANDED_N / 16 - 1)
    return (unsigned short)0xffff;
  else
    return mask;
}

template <int EXPANDED_N>
constexpr inline unsigned short get_mask(int col, unsigned short mask) {
  // Not last column, return 0xffffff indicating load/store all 16 floats
  if (col < EXPANDED_N / 16 - 1)
    return (unsigned short)0xffff;
  else
    return mask;
}
class IdentityOP {
 public:
  __m512 operator()(__m512& v, __mmask16 mask, int row, int col) const {
    return v;
  }
};

// This function is for the case of very small M.
// LINES should be  smaller than 4.
// N must be smaller than 64 and must be multiple of 16.
// PREFETCH_K_DIST means prefetch distance in K.
// ACC means accumulate to C or not.
// actualN , rowOff and postop are not used in current code version.
template <
    int LINES,
    int N,
    int PREFETCH_K_DIST,
    bool ACC,
    typename Lambda = IdentityOP>
void small_gemm_smallm(
    const float* A,
    const int8_t* B,
    float* C,
    int lda,
    int ldb,
    int ldc,
    int actualN,
    int K,
    float* zero_point,
    float* scale,
    int rowOff = 0,
    const Lambda& postop = IdentityOP()) {
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
    if constexpr (std::is_same<Lambda, IdentityOP>::value) {
      _mm512_storeu_ps(ADDRESS(C, line, col * 16, ldc), vc[i]);
    } else {
      // Apply post op
      vc[i] = postop(vc[i], 0xffff, rowOff + line, col * 16);
      _mm512_storeu_ps(ADDRESS(C, line, col * 16, ldc), vc[i]);
    }
  };

  compile_time_for<LINES * COLS>::op(store);
}

inline void dequant_(
    int8_t* B,
    float* b,
    __m512 float_zero_point,
    __m512 float_scale) {
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
    __m512 float_zero_point,
    __m512 float_scale,
    unsigned short mask) {
  const __m128i b_ = _mm_maskz_loadu_epi8(mask, (const __m128i*)B);
  __m512 vb;
  vb = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b_));
  vb = _mm512_maskz_sub_ps(mask, vb, float_zero_point);
  vb = _mm512_maskz_mul_ps(mask, vb, float_scale);
  _mm512_mask_storeu_ps(b, mask, vb);
}

// per channel
// B is packed and not transposed, shape:[BLOCK_K x BLOCK_N]
template <int BLOCK_K, int BLOCK_N>
void dequant(int8_t* B, float* b, float* zero_point, float* scale) {
  const int COLS = BLOCK_N / 16;
  __m512 float_scale = _mm512_loadu_ps(scale);
  __m512 float_zero_point = _mm512_loadu_ps(zero_point);
  for (int k = 0; k < BLOCK_K; k++) {
    int8_t* src = B;
    float* dst = b;
    int j, idx;
    for (idx = 0, j = 0; j < COLS * 16; j += 16) {
      __m512 float_scale = _mm512_loadu_ps(scale + j);
      __m512 float_zero_point = _mm512_loadu_ps(zero_point + j);
      dequant_(src, dst, float_zero_point, float_scale);
      src += 16;
      dst += 16;
    }
    if (j < BLOCK_N) {
      const int res = BLOCK_N - j;
      unsigned short mask = 0xffff;
      mask = (1 << res) - 1;
      __m512 float_scale = _mm512_maskz_loadu_ps(mask, scale + j);
      __m512 float_zero_point = _mm512_maskz_loadu_ps(mask, zero_point + j);
      dequant_(src, dst, float_zero_point, float_scale, mask);
    }
    B += BLOCK_N;
    b += BLOCK_N;
  }
}

// per channel
// handle edge cases
void dequant(
    int8_t* B,
    float* b,
    int K,
    int N,
    float* zero_point,
    float* scale) {
  const int COLS = N / 16;
  for (int k = 0; k < K; k++) {
    int8_t* src = B;
    float* dst = b;
    int j;
    for (j = 0; j < COLS * 16; j += 16) {
      __m512 float_scale = _mm512_loadu_ps(scale + j);
      __m512 float_zero_point = _mm512_loadu_ps(zero_point + j);
      dequant_(src, dst, float_zero_point, float_scale);
      src += 16;
      dst += 16;
    }
    if (j < N) {
      const int res = N - j;
      unsigned short mask = 0xffff;
      mask = (1 << res) - 1;
      __m512 float_scale = _mm512_maskz_loadu_ps(mask, scale + j);
      __m512 float_zero_point = _mm512_maskz_loadu_ps(mask, zero_point + j);
      dequant_(src, dst, float_zero_point, float_scale, mask);
    }
    B += N;
    b += N;
  }
}

// per tensor
// B is packed and not transposed, shape:[BLOCK_K x BLOCK_N]
template <int BLOCK_K, int BLOCK_N>
void dequant(int8_t* B, float* b, float zero_point, float scale) {
  __m512 float_scale = _mm512_set1_ps(scale);
  __m512 float_zero_point = _mm512_set1_ps(zero_point);
  int COLS = BLOCK_N / 16;
  for (int k = 0; k < BLOCK_K; k++) {
    int8_t* src = B;
    float* dst = b;
    int j;
    for (j = 0; j < COLS * 16; j += 16) {
      dequant_(src, dst, float_zero_point, float_scale);
      src += 16;
      dst += 16;
    }
    if (j < BLOCK_N) { // elements < 16
      const int res = BLOCK_N - j;
      unsigned short mask = 0xffff;
      mask = (1 << res) - 1;
      dequant_(src, dst, float_zero_point, float_scale, mask);
    }
    B += BLOCK_N;
    b += BLOCK_N;
  }
}

// per tensor
// handle edge cases
void dequant(int8_t* B, float* b, int K, int N, float zero_point, float scale) {
  __m512 float_scale = _mm512_set1_ps(scale);
  __m512 float_zero_point = _mm512_set1_ps(zero_point);
  int COLS = N / 16;
  for (int k = 0; k < K; k++) {
    int8_t* src = B;
    float* dst = b;
    int j;
    for (j = 0; j < COLS * 16; j += 16) {
      dequant_(src, dst, float_zero_point, float_scale);
      src += 16;
      dst += 16;
    }
    if (j < N) { // elements < 16
      const int res = N - j;
      unsigned short mask = 0xffff;
      mask = (1 << res) - 1;
      dequant_(src, dst, float_zero_point, float_scale, mask);
    }
    B += N;
    b += N;
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
void dequant(int8_t* B, float* b, float* zero_point, float* scale) {
  AT_ASSERTM(false, "Unable to support AVX512!");
}

void dequant(
    int8_t* B,
    float* b,
    int K,
    int N,
    float* zero_point,
    float* scale) {
  AT_ASSERTM(false, "Unable to support AVX512!");
}

// per tensor
template <int BLOCK_N, int BLOCK_K>
void dequant(int8_t* B, float* b, float zero_point, float scale) {
  AT_ASSERTM(false, "Unable to support AVX512!");
}

void dequant(int8_t* B, float* b, int K, int N, float zero_point, float scale) {
  AT_ASSERTM(false, "Unable to support AVX512!");
}

void add_bias(float* C, float* bias, int M, int N, int ldc) {
  AT_ASSERTM(false, "Unable to support AVX512!");
}

class IdentityOP {
 public:
  int operator()(int col) const {
    AT_ASSERTM(false, "Unable to support AVX512!");
  }
};

template <
    int LINES,
    int N,
    int PREFETCH_K_DIST,
    bool ACC,
    typename Lambda = IdentityOP>
void small_gemm_smallm(
    const float* A,
    const int8_t* B,
    float* C,
    int lda,
    int ldb,
    int ldc,
    int actualN,
    int K,
    float* zero_point,
    float* scale,
    int rowOff = 0,
    const Lambda& postop = IdentityOP()) {
  AT_ASSERTM(false, "Unable to support AVX512!");
}
#endif

const int BLOCK_N = 64, BLOCK_K = 64, PREFETCH_K = 64;

struct DotMicroKernelKey {
  bool trans_a;
  bool trans_b;
  int lda;
  int ldb;
  int ldc;

  DotMicroKernelKey(bool trans_a, bool trans_b, int lda, int ldb, int ldc)
      : trans_a(trans_a), trans_b(trans_b), lda(lda), ldb(ldb), ldc(ldc) {}

  bool operator==(const DotMicroKernelKey& other) const {
    return trans_a == other.trans_a && trans_b == other.trans_b &&
        lda == other.lda && ldb == other.ldb && ldc == other.ldc;
  }
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
class DotMicroKernel {
 public:
  DotMicroKernel(bool trans_a, bool trans_b, int lda, int ldb, int ldc) {
    libxsmm_gemm_shape brshape = libxsmm_create_gemm_shape(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        lda,
        ldb,
        ldc,
        /*type A*/ LIBXSMM_DATATYPE_F32,
        /*type B*/ LIBXSMM_DATATYPE_F32,
        /*type C*/ LIBXSMM_DATATYPE_F32,
        /*acctype*/ LIBXSMM_DATATYPE_F32);
    libxsmm_bitfield brflags =
        (trans_a ? LIBXSMM_GEMM_FLAG_TRANS_A : LIBXSMM_GEMM_FLAG_NONE) |
        (trans_b ? LIBXSMM_GEMM_FLAG_TRANS_B : LIBXSMM_GEMM_FLAG_NONE);
    libxsmm_gemm_batch_reduce_config brconfig;
    memset(&brconfig, 0, sizeof(libxsmm_gemm_batch_reduce_config));
    brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_NONE;

    kernel_func_ = libxsmm_dispatch_brgemm_v2(
        brshape, brflags, /*prefetch_flags=*/0, brconfig);
    memset(&gemm_param_, 0, sizeof(libxsmm_gemm_param));
  }

  void operator()(void* A, void* B, void* C) {
    gemm_param_.a.primary = (void*)A;
    gemm_param_.b.primary = (void*)B;
    gemm_param_.c.primary = (void*)C;
    kernel_func_(&gemm_param_);
  }

 private:
  libxsmm_gemmfunction kernel_func_;
  libxsmm_gemm_param gemm_param_;
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
using DotMicroKernelRef =
    std::shared_ptr<DotMicroKernel<BLOCK_M, BLOCK_N, BLOCK_K>>;

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
DotMicroKernelRef<BLOCK_M, BLOCK_N, BLOCK_K> create_or_get_dot_microkernel(
    bool trans_a,
    bool trans_b,
    int lda,
    int ldb,
    int ldc) {
  thread_local std::unordered_map<
      DotMicroKernelKey,
      DotMicroKernelRef<BLOCK_M, BLOCK_N, BLOCK_K>>
      cache;
  DotMicroKernelKey key(trans_a, trans_b, lda, ldb, ldc);
  auto search = cache.find(key);
  if (search != cache.end()) {
    return search->second;
  } else {
    cache.insert(
        {key,
         std::make_shared<DotMicroKernel<BLOCK_M, BLOCK_N, BLOCK_K>>(
             trans_a, trans_b, lda, ldb, ldc)}); //
    return cache[key];
  }
}

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

void zero_fill(float* C, int M, int N, int stride) {
  for (int m = 0; m < M; m++) {
    memset(C + m * stride, 0, sizeof(float) * N);
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
    float* zero_point,
    float* scale,
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
        if (m_bs == BLOCK_M && n_bs == BLOCK_N) {
          small_gemm_smallm<BLOCK_M, BLOCK_N, PREFETCH_K, true>(
              A_offset,
              B_offset,
              C_offset,
              lda,
              n_bs,
              ldc,
              BLOCK_N,
              BLOCK_K,
              zero_point + nb_start,
              scale + nb_start);
        } else { // edge case
          dequant(
              B_offset,
              bi_offset,
              k_bs,
              n_bs,
              zero_point + nb_start,
              scale + nb_start);
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
// for small M
template <bool has_bias>
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
    float* zero_point,
    float* scale,
    float* bias = NULL) {
#define PTR_OFFSET(base, offset0, offset1, stride0) \
  (base) + (offset0) * (stride0) + (offset1)

  const int NB = (N + BLOCK_N - 1) / BLOCK_N, KB = (K + BLOCK_K - 1) / BLOCK_K;

#pragma omp parallel for collapse(1)
  for (int nb = 0; nb < NB; nb++) {
    int mb_start = 0;
    int m_bs = M;
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
          zero_point + nb_start,
          scale + nb_start);
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
    if constexpr (has_bias) {
      add_bias(C_offset, bias + nb_start, m_bs, n_bs, ldc);
    }
    free(bi_offset);
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
    float* zero_point,
    float* scale,
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
            zero_point + nb_start,
            scale + nb_start);
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
    float zero_point,
    float scale,
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
        dequant(B_offset, bi_offset, k_bs, n_bs, zero_point, scale);
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

void woq_gemm_kernel_impl(
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& zero_points_float,
    const at::Tensor& scales_float,
    const at::Tensor& bias,
    at::Tensor& output) {
#if defined(CPU_CAPABILITY_AVX512)
  auto self_ = self.is_contiguous() ? self : self.contiguous();
  const int64_t dim = self.dim();
  auto self_reshaped =
      dim == 2 ? self_ : self_.reshape({-1, self.size(self.dim() - 1)});
  auto M = self_reshaped.size(0);
  auto K = self_reshaped.size(1);

  auto in_ptr = self_.data_ptr<float>();
  auto weight_ptr = weight.data_ptr<int8_t>();
  auto out_ptr = output.data_ptr<float>();
  auto zero_points_float_ptr = zero_points_float.data_ptr<float>();
  auto scales_float_ptr = scales_float.data_ptr<float>();
  auto N = weight.size(0);
  const auto qtype = weight.qscheme();

  // TODO: per-tensor block size tuning
  if (qtype == c10::kPerTensorAffine) { // per tensor
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
          zero_points_float_ptr[0],
          scales_float_ptr[0],
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
          zero_points_float_ptr[0],
          scales_float_ptr[0]);
    }
  } else if (qtype == c10::kPerChannelAffine) { // per channel
    // TODO: per-channel block size tuning
    if (bias.defined()) { // case with bias
      auto bias_ = bias.is_contiguous() ? bias : bias.contiguous();
      auto bias_ptr = bias_.data_ptr<float>();
      if (M <= 4) {
        switch (M) {
          case 1:
            return woq_gemm_intrinsic<true, 1>(
                in_ptr,
                weight_ptr,
                out_ptr,
                M,
                N,
                K,
                K,
                K,
                N,
                zero_points_float_ptr,
                scales_float_ptr,
                bias_ptr);
            break;
          case 2:
            return woq_gemm_intrinsic<true, 2>(
                in_ptr,
                weight_ptr,
                out_ptr,
                M,
                N,
                K,
                K,
                K,
                N,
                zero_points_float_ptr,
                scales_float_ptr,
                bias_ptr);
            break;
          case 3:
            return woq_gemm_intrinsic<true, 3>(
                in_ptr,
                weight_ptr,
                out_ptr,
                M,
                N,
                K,
                K,
                K,
                N,
                zero_points_float_ptr,
                scales_float_ptr,
                bias_ptr);
            break;
          case 4:
            return woq_gemm_intrinsic<true, 4>(
                in_ptr,
                weight_ptr,
                out_ptr,
                M,
                N,
                K,
                K,
                K,
                N,
                zero_points_float_ptr,
                scales_float_ptr,
                bias_ptr);
            break;
        }
      } else if (M < 196) { // small M
        return woq_gemm_brgemm<true>(
            in_ptr,
            weight_ptr,
            out_ptr,
            M,
            N,
            K,
            K,
            K,
            N,
            zero_points_float_ptr,
            scales_float_ptr,
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
            zero_points_float_ptr,
            scales_float_ptr,
            bias_ptr);
      }
    } else { // case without bias
      if (M <= 4) {
        switch (M) {
          case 1:
            return woq_gemm_intrinsic<false, 1>(
                in_ptr,
                weight_ptr,
                out_ptr,
                M,
                N,
                K,
                K,
                K,
                N,
                zero_points_float_ptr,
                scales_float_ptr);
            break;
          case 2:
            return woq_gemm_intrinsic<false, 2>(
                in_ptr,
                weight_ptr,
                out_ptr,
                M,
                N,
                K,
                K,
                K,
                N,
                zero_points_float_ptr,
                scales_float_ptr);
            break;
          case 3:
            return woq_gemm_intrinsic<false, 3>(
                in_ptr,
                weight_ptr,
                out_ptr,
                M,
                N,
                K,
                K,
                K,
                N,
                zero_points_float_ptr,
                scales_float_ptr);
            break;
          case 4:
            return woq_gemm_intrinsic<false, 4>(
                in_ptr,
                weight_ptr,
                out_ptr,
                M,
                N,
                K,
                K,
                K,
                N,
                zero_points_float_ptr,
                scales_float_ptr);
            break;
        }
      } else if (M < 196) {
        return woq_gemm_brgemm<false>(
            in_ptr,
            weight_ptr,
            out_ptr,
            M,
            N,
            K,
            K,
            K,
            N,
            zero_points_float_ptr,
            scales_float_ptr);
      } else {
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
            zero_points_float_ptr,
            scales_float_ptr);
      }
    }
  }
#else
  auto w = weight.dequantize();
  if (bias.defined()) {
    at::linear_out(output, self, w, bias.detach());
  } else {
    at::linear_out(output, self, w);
  }

#endif
}

at::Tensor woq_linear_packB_impl(
    const at::Tensor& weight,
    const at::Tensor& zero_points,
    const at::Tensor& scales) {
#if defined(CPU_CAPABILITY_AVX512)
  auto N = weight.size(0);
  auto K = weight.size(1);
  auto weight_size = weight.sizes().vec();
  auto weight_packed = at::_empty_per_channel_affine_quantized(
      weight_size,
      scales,
      zero_points,
      1,
      device(c10::kCPU).dtype(c10::kQInt8));
  auto weight_contig = weight.contiguous();

  auto weight_ptr = weight_contig.data_ptr<int8_t>();
  auto weightpacked_ptr = reinterpret_cast<int8_t*>(weight_packed.data_ptr());
  pack(weight_ptr, weightpacked_ptr, K, N, K, true);

  return weight_packed;
#else
  return weight;
#endif
}

at::Tensor woq_linear_unpackB_impl(const at::Tensor& weight) {
#if defined(CPU_CAPABILITY_AVX512)
  auto N = weight.size(0);
  auto K = weight.size(1);
  std::vector<int32_t> weight_zero_points_int32(1, 0);
  const auto qtype = weight.qscheme();
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

  auto weight_size = weight.sizes().vec();
  auto weight_unpacked = at::_empty_per_channel_affine_quantized(
      weight_size,
      scales,
      zero_points,
      1,
      device(c10::kCPU).dtype(c10::kQInt8));

  auto weight_contig = weight.contiguous();

  auto weight_ptr = weight_contig.data_ptr<int8_t>();
  auto weight_unpacked_ptr =
      reinterpret_cast<int8_t*>(weight_unpacked.data_ptr());
  unpack(weight_ptr, weight_unpacked_ptr, K, N, K, true);
  return weight_unpacked;
#else
  return weight;
#endif
}
} // anonymous namespace

REGISTER_DISPATCH(woq_gemm_kernel_stub, &woq_gemm_kernel_impl);
REGISTER_DISPATCH(woq_linear_unpackB_stub, &woq_linear_unpackB_impl);
REGISTER_DISPATCH(woq_linear_packB_stub, &woq_linear_packB_impl);
} // namespace cpu
} // namespace torch_ipex

namespace std {
template <>
struct hash<torch_ipex::cpu::DotMicroKernelKey> {
  std::size_t operator()(const torch_ipex::cpu::DotMicroKernelKey& key) const {
    std::size_t h = std::hash<bool>()(key.trans_a);
    h = std::hash<bool>()(key.trans_b) ^ (h << 1);
    h = std::hash<int>()(key.lda) ^ (h << 1);
    h = std::hash<int>()(key.ldb) ^ (h << 1);
    h = std::hash<int>()(key.ldc) ^ (h << 1);
    return h;
  }
};
} // namespace std