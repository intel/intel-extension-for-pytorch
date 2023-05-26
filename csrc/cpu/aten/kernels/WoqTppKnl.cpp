// weight-only quantization gemm kernel (int8, int4 etc.)
// #include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <aten/Linear.h>
#include "csrc/cpu/tpp/woq/tla.h"

namespace torch_ipex {
namespace cpu {
namespace {

using namespace tpp;
using TensorList = std::vector<at::Tensor>;

#define SMALL_BATCH_THRESHOLD 32
#define PARALLEL_M_THRESHOLD 128
constexpr long PREFETCH_K_DIST = 64; // TODO(jgong5): do not hard-code
constexpr long LOOP_K_UNROLL = 4; // TODO(jgong5): do not hard-code

template <long N_GROUP_SIZE, typename VAT, typename LUT>
inline VAT load_dequant_zp_only_int4(uint8_t *p, VAT vzps, LUT lut) {
  TLA_ASSERT(false, "not implemented");
}

template <long N_GROUP_SIZE, typename VAT>
inline VAT load_dequant_zp_only_int8(uint8_t *p, VAT vzps) {
  TLA_ASSERT(false, "not implemented");
}

// TODO(jgong5): further simplify the dequant intrinsics below with VecOps
#ifdef __AVX512F__
template <>
inline std::array<__m512, 4> load_dequant_zp_only_int8<64>(
  uint8_t *p, std::array<__m512, 4> vzps
) {
  using T = float;
  using VA = VecArray<64, T>;
  using VAT = typename VA::type;
  constexpr long COLS = VA::num_vec;
  auto packed = _mm512_loadu_si512((__m512i*)p);
  VAT vbs;
  compile_time_for<COLS>::op(
    [&](auto i) {
      constexpr long imm = i;
      auto int8 = _mm512_extracti32x4_epi32(packed, imm);
      vbs[i] = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(int8));
      vbs[i] = _mm512_sub_ps(vbs[i], vzps[i]);
    }
  );
  return vbs;
}

template <>
inline std::array<__m512, 2> load_dequant_zp_only_int8<32>(
  uint8_t *p, std::array<__m512, 2> vzps
) {
  using T = float;
  using VA = VecArray<32, T>;
  using VAT = typename VA::type;
  constexpr long COLS = VA::num_vec;
  auto packed = _mm256_loadu_si256((__m256i*)p);
  VAT vbs;
  compile_time_for<COLS>::op(
    [&](auto i) {
      constexpr long imm = i;
      auto int8 = _mm256_extracti128_si256(packed, imm);
      vbs[i] = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(int8));
      vbs[i] = _mm512_sub_ps(vbs[i], vzps[i]);
    }
  );
  return vbs;
}

template <>
inline std::array<__m512, 1> load_dequant_zp_only_int8<16>(
  uint8_t *p, std::array<__m512, 1> vzps
) {
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
#endif

#ifdef __AVX512FP16__
template<>
inline std::array<__m512h, 2> load_dequant_zp_only_int8<64>(uint8_t* p, std::array<__m512h, 2> vzps) {
  using T = tpp::half;
  using VA = VecArray<64, T>;
  using VAT = typename VA::type;
  constexpr long COLS = VA::num_vec;
  auto packed = _mm512_loadu_si512((__m512i*)p);
  VAT vbs;
  compile_time_for<COLS>::op(
    [&](auto i) {
      constexpr long imm = i;
      auto int8 = _mm512_extracti64x4_epi64(packed, imm);
      vbs[i] = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(int8));
      vbs[i] = _mm512_sub_ph(vbs[i], vzps[i]);
    }
  );
  return vbs;
}

template<>
inline std::array<__m512h, 1> load_dequant_zp_only_int8<32>(
  uint8_t* p, std::array<__m512h, 1> vzps
) {
  using T = tpp::half;
  using VA = VecArray<32, T>;
  using VAT = typename VA::type;
  constexpr long COLS = VA::num_vec;
  auto packed = _mm256_loadu_si256((__m256i*)p);
  VAT vbs;
  compile_time_for<COLS>::op(
    [&](auto i) {
      constexpr long imm = i;
      vbs[i] = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(packed));
      vbs[i] = _mm512_sub_ph(vbs[i], vzps[i]);
    }
  );
  return vbs;
}
#endif

template<long N, typename T>
struct load_dequant_int4 {
  using VT = typename VecType<T>::type;
  using V = VecOps<VT>;
  using VA = VecArray<N, T>;
  using VAT = typename VA::type;
  constexpr static long COLS = VA::num_vec;

  static inline VAT call(uint8_t *p, VAT vscales, VAT vzps, VT lut) {
    auto vbs = load_dequant_zp_only_int4<N>(p, vzps, lut);
    compile_time_for<COLS>::op(
      [&](auto idx) {
        vbs[idx] = V::mul(vbs[idx], vscales[idx]);
      }
    );
    return vbs;
  }
};

template<long N, typename T>
struct load_dequant_int8 {
  using VT = typename VecType<T>::type;
  using V = VecOps<VT>;
  using VA = VecArray<N, T>;
  using VAT = typename VA::type;
  constexpr static long COLS = VA::num_vec;

  static inline VAT call(uint8_t *p, VAT vscales, VAT vzps) {
    auto vbs = load_dequant_zp_only_int8<N>(p, vzps);
    compile_time_for<COLS>::op(
      [&](auto idx) {
        vbs[idx] = V::mul(vbs[idx], vscales[idx]);
      }
    );
    return vbs;
  }
};

constexpr int get_n_group_size(int N) {
  return N == 16 ? 16 : (N == 32 ? 32 : 64);
}

// TODO(jgong5): move to tpp.h
// TODO(jgong5): add pre/post op fusion
template <
  typename T, long M, long N, bool transA=false, bool ACC=false, long PREFETCH_K_DIST=0, typename Enabled=void
>
struct GemmMicroKernel {
  template <bool is_int4>
  static inline void call(long K, T* A, long lda, uint8_t* B, long ldb, T* C, long ldc, T* scales, T* zps) {
    TLA_ASSERT(false, "Not implemented");
  }
};

template <
  typename T, long M, long N, bool transA, bool ACC, long PREFETCH_K_DIST
>
struct GemmMicroKernel<
  T, M, N, transA, ACC, PREFETCH_K_DIST,
  typename std::enable_if_t<std::is_same<T, float>::value || std::is_same<T, half>::value>
> {
  // TODO(jgong5): generalize this with pre/post op handlers
  template <bool is_int4>
  static inline void call(long K, T* A, long lda, uint8_t* B, long ldb, T* C, long ldc, T* scales, T* zps) {
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
    VT va;
    VArrayT vb[CNBLOCKS];
    VT vc[M * COLS];
    VArrayT vscales[CNBLOCKS];
    VArrayT vzps[CNBLOCKS];

    // Load scales and zps
    compile_time_for<CNBLOCKS>::op(
      [&](auto i) {
        constexpr const int col = i * CBLOCK;
        vscales[i] = VArray::load1d(scales + col * V::VLEN);
        vzps[i] = VArray::load1d(zps + col * V::VLEN);
      }
    );

    // NB: For fp16 in int8 woq, we do not delay the scale to the post-op but leave it
    // to the dequant otherwise the weight value might be too large to overflow
    // fp16 range.
    constexpr bool scale_as_post_op = !std::is_same<T, half>() || is_int4;

    compile_time_for<M * COLS>::op(
      [&](auto i) { vc[i] = V::setzero(); }
    );

    auto compute = [&](auto i, int k) {
      constexpr const int row = i / CNBLOCKS;
      constexpr const int cbidx = i % CNBLOCKS;

      if constexpr (cbidx == 0) {
        if constexpr (transA) {
          va = V::set1(*(ST*)ADDRESS(A, k, row, lda));
        } else {
          va = V::set1(*(ST*)ADDRESS(A, row, k, lda));
        }
      }

      if constexpr (row == 0) {
        constexpr const int col = cbidx * CBLOCK;
        if constexpr (scale_as_post_op) {
          if constexpr (is_int4) {
            TLA_ASSERT(false, "Not implemented");
          } else {
            vb[cbidx] = load_dequant_zp_only_int8<N_GROUP_SIZE>(ADDRESS(B, k, col * V::VLEN, ldb), vzps[cbidx]);
          }
        } else {
          if constexpr (is_int4) {
            TLA_ASSERT(false, "Not implemented");
          } else {
            vb[cbidx] = load_dequant_int8<N_GROUP_SIZE, T>::call(ADDRESS(B, k, col * V::VLEN, ldb), vscales[cbidx], vzps[cbidx]);
          }
        }
        if constexpr (PREFETCH_K_DIST > 0) {
          if constexpr (is_int4) {
            TLA_ASSERT(false, "Not implemented");
          } else {
            _mm_prefetch(ADDRESS(B, k + PREFETCH_K_DIST, col * V::VLEN, ldb), _MM_HINT_T0);
          }
        }
      }

      compile_time_for<CBLOCK>::op(
        [&](auto col) {
          constexpr const int idx = INDEX(row, INDEX(cbidx, col, CBLOCK), COLS);
          vc[idx] = V::fmadd(va, vb[cbidx][col], vc[idx]);
        }
      );

    };

    // Accumulate along k
    // Do not unroll for half since no performance benefit is observed
    constexpr const int unroll = std::is_same<T, half>::value ? 1 : LOOP_K_UNROLL;
    int k = 0;
    for (; k < K / unroll; k++) {
      compile_time_for<unroll>::op(
        [&](auto i) {
          compile_time_for<M * CNBLOCKS>::op(compute, k*unroll + i);
        }
      );
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
          vc[i] = V::fmadd(vscales[col/CBLOCK][col%CBLOCK], vc[i], vc_old);
        } else {
          vc[i] = V::fmadd(V::set1(1.0f), vc[i], vc_old);
        }
      } else if constexpr (scale_as_post_op) {
        vc[i] = V::mul(vscales[col/CBLOCK][col%CBLOCK], vc[i]);
      }
      V::storeu(ADDRESS(C, row, col * V::VLEN, ldc), vc[i]);
    };

    compile_time_for<M * COLS>::op(store);
  }
};

// a dequant function the requires N to be a multiple of N_GROUP_SIZE
template <
  typename Tin, long N_GROUP_SIZE, bool is_int4
>
struct dequant_n_grouped {
  template <typename Lambda1, typename Lambda2, typename Lambda3>
  static inline void call(
    uint8_t* qB, long K, long N, long ldb, Tin* scales, Tin* zps, Tin* B,
    const Lambda1& load_qparam,
    const Lambda2& load_qint_as_fp,
    const Lambda3& store
  ) {
    for (int n = 0; n < N; n+=N_GROUP_SIZE) {
      // load scales and zps
      auto vscales = load_qparam(scales + n);
      auto vzps = load_qparam(zps + n);
      for (int k = 0; k < K; k++) {
        // load and dequant qB to vb
        auto vbs = load_qint_as_fp(is_int4 ? &qB[k*ldb/2 + n/2] : &qB[k*ldb + n], vscales, vzps);
        // store vb to B
        store(B + k*N + n, vbs);
      }
    }
  }
};

#ifdef __AVX512F__
template <
  long N_GROUP_SIZE, bool is_int4
>
struct dequant_n_grouped<bfloat16, N_GROUP_SIZE, is_int4> {
  template <typename Lambda1, typename Lambda2, typename Lambda3>
  static inline void call(
    uint8_t* qB, long K, long N, long ldb, bfloat16* scales, bfloat16* zps, bfloat16* B,
    const Lambda1& load_qparam,
    const Lambda2& load_qint_as_fp,
    const Lambda3& store
  ) {
    #define ADDRESS(p, x, y, ld) ((p) + (x) * (ld) + (y))

    using VA = VecArray<N_GROUP_SIZE, float>;
    constexpr long COLS = VA::num_vec;

    for (int n = 0; n < N; n+=N_GROUP_SIZE) {
      // load scales and zps
      auto vscales = load_qparam(scales + n);
      auto vzps = load_qparam(zps + n);
      // convert to vnni: [K/2, N, 2]
      for (int k = 0; k < K; k+=2) {
        auto interleave = [](__m512 v0, __m512 v1) {
          __m512i idx_low = _mm512_set_epi32(
            0x17, 0x07, 0x16, 0x06,
            0x15, 0x05, 0x14, 0x04,
            0x13, 0x03, 0x12, 0x02,
            0x11, 0x01, 0x10, 0x00
          );
          __m512i idx_high = _mm512_set_epi32(
            0x1f, 0x0f, 0x1e, 0x0e,
            0x1d, 0x0d, 0x1c, 0x0c,
            0x1b, 0x0b, 0x1a, 0x0a,
            0x19, 0x09, 0x18, 0x08
          );
          return std::array<__m512, 2>(
            {_mm512_permutex2var_ps(v0, idx_low, v1), _mm512_permutex2var_ps(v0, idx_high, v1)}
          );
        };
        // load and dequant qB to vb
        auto vbs_k0 = load_qint_as_fp(is_int4 ? &qB[k*ldb/2 + n/2] : &qB[k*ldb + n], vscales, vzps);
        auto vbs_k1 = load_qint_as_fp(is_int4 ? &qB[(k+1)*ldb/2 + n/2] : &qB[(k+1)*ldb + n], vscales, vzps);
        typename VA::type vbs[2];
        compile_time_for<COLS>::op(
          [&](auto i) {
            auto [low, high] = interleave(vbs_k0[i], vbs_k1[i]);
            vbs[i*2/COLS][i*2%COLS] = low;
            vbs[(i*2+1)/COLS][(i*2+1)%COLS] = high;
          }
        );
        // store vb to B: low: [k + n*2 / N, n*2 % N], high: [k + (n*2+N_GROUP_SIZE) / N, (n*2+N_GROUP_SIZE) % N]
        store(ADDRESS(B, k + (n*2) / N, (n*2) % N, N), vbs[0]);
        store(ADDRESS(B, k + (n*2 + N_GROUP_SIZE) / N, (n*2 + N_GROUP_SIZE) % N, N), vbs[1]);
      }
    }
  }
};
#endif

template<typename Tin, long N_GROUP_SIZE, bool is_int4>
struct Dequantize {
  static void call(uint8_t* qB, long K, long N, long ldb, Tin* scales, Tin* zps, Tin* B);
};

template<long N_GROUP_SIZE, bool is_int4>
struct Dequantize<float, N_GROUP_SIZE, is_int4> {
  static inline void call(uint8_t* qB, long K, long N, long ldb, float* scales, float* zps, float* B) {
#if defined(__AVX512F__)
    using T = float;
    using VA = VecArray<N_GROUP_SIZE, T>;
    constexpr int VLEN = VA::vec_ops::VLEN;
    constexpr long COLS = VA::num_vec;

    dequant_n_grouped<float, N_GROUP_SIZE, is_int4>::call(
      qB, K, N, ldb, scales, zps, B,
      [&](float* p) {
        return VA::load1d(p);
      },
      [&](uint8_t* p, auto vscales, auto vzps) {
        if constexpr (is_int4) {
          TLA_ASSERT(false, "Not implemented");
          // For type induction
          return load_dequant_int8<N_GROUP_SIZE, T>::call(p, vscales, vzps);
        } else {
          return load_dequant_int8<N_GROUP_SIZE, T>::call(p, vscales, vzps);
        }
      },
      [&](auto p, auto vbs) {
        compile_time_for<COLS>::op(
          [&](auto idx) {
            _mm512_storeu_ps(p + idx*VLEN, vbs[idx]);
          }
        );
      }
    );
#else
    TLA_ASSERT(false, "not implemented");
#endif
  }
};

template<long N_GROUP_SIZE, bool is_int4>
struct Dequantize<bfloat16, N_GROUP_SIZE, is_int4> {
  static inline void call(uint8_t* qB, long K, long N, long ldb, bfloat16* scales, bfloat16* zps, bfloat16* B) {
#ifdef __AVX512F__
    using T = bfloat16;
    using VA = VecArray<N_GROUP_SIZE, T>;
    constexpr long COLS = VA::num_vec;

    dequant_n_grouped<bfloat16, N_GROUP_SIZE, is_int4>::call(
      qB, K, N, ldb, scales, zps, B,
      [&](bfloat16* p) {
        return VA::load1d(p);
      },
      [&](uint8_t* p, auto vscales, auto vzps) {
        if constexpr (is_int4) {
          TLA_ASSERT(false, "Not implemented");
          // For type induction
          return load_dequant_int8<N_GROUP_SIZE, float>::call(p, vscales, vzps);
        } else {
          return load_dequant_int8<N_GROUP_SIZE, float>::call(p, vscales, vzps);
        }
      },
      [&](auto p, auto vbs) {
        compile_time_for<COLS/2>::op(
          [&](auto idx) {
            _vec_store_two_floats_as_bfloat16(p + idx*32, vbs[idx*2], vbs[idx*2+1]);
          }
        );
      }
    );
#else
    TLA_ASSERT(false, "not implemented");
#endif
  }
};

template<long N_GROUP_SIZE, bool is_int4>
struct Dequantize<half, N_GROUP_SIZE, is_int4> {
  static inline void call(uint8_t* qB, long K, long N, long ldb, half* scales, half* zps, half* B) {
#ifdef __AVX512FP16__
    using T = half;
    using VA = VecArray<N_GROUP_SIZE, T>;
    constexpr int VLEN = VA::vec_ops::VLEN;
    constexpr long COLS = VA::num_vec;

    dequant_n_grouped<half, N_GROUP_SIZE, is_int4>::call(
      qB, K, N, ldb, scales, zps, B,
      [&](half* p) {
        return VA::load1d(p);
      },
      [&](uint8_t* p, auto vscales, auto vzps) {
        if constexpr (is_int4) {
          TLA_ASSERT(false, "Not implemented");
          // For type induction
          return load_dequant_int8<N_GROUP_SIZE, T>::call(p, vscales, vzps);
        } else {
          return load_dequant_int8<N_GROUP_SIZE, T>::call(p, vscales, vzps);
        }
      },
      [&](auto p, auto vbs) {
        compile_time_for<COLS>::op(
          [&](auto idx) {
            _mm512_storeu_ph(p + idx*VLEN, vbs[idx]);
          }
        );
      }
    );
#else
    TLA_ASSERT(false, "not implemented");
#endif
  }
};

// TODO(jgong5): move to tpp.h
template <
  typename Tin, typename Tout,
  long BLOCK_M, long N, bool transA, bool ACC, bool is_int4, long PREFETCH_K_DIST=0>
class DequantGemmTPP {
public:
  DequantGemmTPP(
      long M,
      long K,
      long lda,
      long ldb,
      long ldc
  )
  :
  M(M),
  K(K),
  lda(lda),
  ldb(ldb),
  ldc(ldc) {
    static_assert(
      N % 16 == 0,
      "N must be a multiple of 16"
    );
    if (std::is_same<Tin, bfloat16>()) TLA_ASSERT(K % 2 == 0, "Kb must be a multiple of 2 for bfloat16");
    pgemm = std::make_shared<BrgemmTPP<Tin, Tout>>(
      M, N, K, 1, 1, lda, ldb, ldc, ACC ? 1 : 0, transA, 1, /*b_vnni*/std::is_same<Tin, bfloat16>()
    );
  }

  void operator()(Tin* A, uint8_t* qB, Tin* scales, Tin* zps, Tout* C) {
    if (
      M < SMALL_BATCH_THRESHOLD &&
      (
        (std::is_same<Tin, half>() && std::is_same<Tout, half>()) ||
        (std::is_same<Tin, float>() && std::is_same<Tout, float>())
      )
    ) {
      for (long m = 0; m < M; m += BLOCK_M) {
        long block_m = std::min(M - m, BLOCK_M);
        range_dispatcher<long, 1, BLOCK_M>::call(block_m,
          [&](auto i) {
            GemmMicroKernel<Tin, i, N, transA, ACC, PREFETCH_K_DIST>::template call<is_int4>(
              K, transA ? (Tin*)A + m : (Tin*)A + m*lda, lda, qB, ldb, (Tin*)C + m*ldc, ldc, scales, zps
            );
          },
          [&](auto i) {
            failing_fallback();
          }
        );
      }
    } else {
      constexpr const int N_GROUP_SIZE = get_n_group_size(N);
      Tin B[K][N];
      // TODO(jgong5): add prefetch
      Dequantize<Tin, N_GROUP_SIZE, is_int4>::call(qB, K, N, ldb, scales, zps, B[0]);
      (*pgemm)(A, B[0], C, 1);
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
  long ldb;
  long ldc;
};

// If T != TComp
//   T -> TComp -> GEMM -> TComp -> bias/PostOp -> T
// If T == TComp (we can save intermediate output buffer and schedule M/N/K loops together)
//   T -> GEMM -> T -> bias/PostOp -> T
template <typename T, typename TComp, typename TGemmOut>
void qlinear_woq_affine_impl(
    const at::Tensor& x,
    const at::Tensor& qw_packed,
    const at::Tensor& scales, // dtype is TComp
    const at::Tensor& zps, // dtype is TComp
    const at::Tensor& b, // dtype is TComp
    at::Tensor y,
    int k_splits,
    int num_concats) {
  TLA_ASSERT(
    qw_packed.scalar_type() == at::kQUInt4x2 || qw_packed.scalar_type() == at::kQInt8,
    "qlinear_woq_affine only supports qint8 and quint4x2 quantized weight"
  );
  auto is_int4 = qw_packed.scalar_type() == at::kQUInt4x2;
  auto x_sizes = x.sizes();
  auto w_sizes = qw_packed.sizes();
  auto M = x_sizes[0];
  auto Nc = w_sizes[0];
  auto Nb = w_sizes[3];
  auto Kc = w_sizes[1];
  auto Kb = w_sizes[2];
  auto N = Nc * Nb;
  auto K = Kc * Kb;

  TLA_ASSERT(Nb % 16 == 0, "Nb must be a multiple of 16");
  TLA_ASSERT(num_concats <= 1 || Nc % num_concats == 0, "Nc must be a multiple of num_concats");

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

  auto compute_type = c10::CppTypeToScalarType<TComp>::value;

  bool no_y_buf = std::is_same<T, TComp>() && std::is_same<T, TGemmOut>() && k_splits == 1;

  auto ldy = num_concats <= 1 ? N : Nc/num_concats * Nb;
  auto ldc = (no_y_buf || k_splits > 1) ? ldy : Nb;

  auto px = GetVLAPtr<T>(x, {Kc,Kb});
  auto pw = GetVLAPtr<uint8_t>((uint8_t*)qw_packed.data_ptr(), {Kc, Kb * (is_int4 ? Nb/2 : Nb)});
  auto py = GetVLAPtr<T>(y, {Nc,Nb}); /*[M, Nc, Nb]*/
  auto py_concat = GetVLAPtr<T>(y, {M,Nc/num_concats,Nb}); /*[num_concats, M, Nc/num_concats, Nb]*/
  auto pscales = GetVLAPtr<TComp>(scales, {Nb});
  auto pzps = GetVLAPtr<TComp>(zps, {Nb});
  auto pb = GetVLAPtr<TGemmOut>(b, {Nb});

  auto copy_bias_out_tpp = CpyBiasTPP<TGemmOut>(BLOCK_M, Nb, ldy);
  auto copy_bias_buf_tpp = CpyBiasTPP<TGemmOut>(BLOCK_M, Nb, Nb);
  auto copy_bias_out_rem_tpp = CpyBiasTPP<TGemmOut>(BLOCK_M_rem, Nb, ldy);
  auto copy_bias_buf_rem_tpp = CpyBiasTPP<TGemmOut>(BLOCK_M_rem, Nb, Nb);
  auto zero_out_tpp = SetZeroTPP<TGemmOut>(BLOCK_M, Nb, ldy);
  auto zero_buf_tpp = SetZeroTPP<TGemmOut>(BLOCK_M, Nb, Nb);
  auto zero_out_rem_tpp = SetZeroTPP<TGemmOut>(BLOCK_M_rem, Nb, ldy);
  auto zero_buf_rem_tpp = SetZeroTPP<TGemmOut>(BLOCK_M_rem, Nb, Nb);

  constexpr long MICRO_BLOCK_M = 4;
  product_dispatcher<
    std::tuple</*BLOCK_N*/long, /*is_int4*/bool>,
    std::tuple<
      enumerate_dispatcher<long, 16, 32, 64, 128, 256>,
      boolean_dispatcher
    >
  >::call(
    std::make_tuple(Nb, is_int4),
    [&](auto tuple) {
      auto BLOCK_N = std::get<0>(tuple);
      auto is_int4 = std::get<1>(tuple);
      // TODO(jgong5): design API to avoid duplicate code of defining similar kernel object
      auto dequant_gemm_tpp = DequantGemmTPP<
        TComp, TGemmOut, MICRO_BLOCK_M, BLOCK_N, /*transA*/false, /*ACC*/true, is_int4, PREFETCH_K_DIST>(
        /*M*/BLOCK_M, /*K*/Kb,
        /*lda*/no_y_buf ? K : Kb,
        /*ldb*/Nb,
        /*ldc*/ldc
      );
      auto dequant_gemm_no_prefetch_tpp = DequantGemmTPP<
        TComp, TGemmOut, MICRO_BLOCK_M, BLOCK_N, /*transA*/false, /*ACC*/true, is_int4, 0>(
        /*M*/BLOCK_M, /*K*/Kb,
        /*lda*/no_y_buf ? K : Kb,
        /*ldb*/Nb,
        /*ldc*/ldc
      );
      auto dequant_gemm_rem_tpp = DequantGemmTPP<
        TComp, TGemmOut, MICRO_BLOCK_M, BLOCK_N, /*transA*/false, /*ACC*/true, is_int4, PREFETCH_K_DIST>(
        /*M*/BLOCK_M_rem, /*K*/Kb,
        /*lda*/no_y_buf ? K : Kb,
        /*ldb*/Nb,
        /*ldc*/ldc
      );
      auto dequant_gemm_no_prefetch_rem_tpp = DequantGemmTPP<
        TComp, TGemmOut, MICRO_BLOCK_M, BLOCK_N, /*transA*/false, /*ACC*/true, is_int4, 0>(
        /*M*/BLOCK_M_rem, /*K*/Kb,
        /*lda*/no_y_buf ? K : Kb,
        /*ldb*/Nb,
        /*ldc*/ldc
      );

      auto cvt_x_tpp = ConvertTPP<T, TComp>(BLOCK_M, Kb, K, Kb);
      auto cvt_x_rem_tpp = ConvertTPP<T, TComp>(BLOCK_M_rem, Kb, K, Kb);
      auto cvt_y_tpp = ConvertTPP<TGemmOut, T>(BLOCK_M, Nb, Nb, ldy);
      auto cvt_y_rem_tpp = ConvertTPP<TGemmOut, T>(BLOCK_M_rem, Nb, Nb, ldy);
      auto cvt_y_private_tpp = ConvertTPP<TGemmOut, T>(BLOCK_M, Nb, N, N);
      auto add_y_tpp = BinaryTPP(
        BLOCK_M, /*row*/
        Nb, /*col*/
        N, /*ldi0*/
        N, /*ldi1*/
        N, /*ldo*/
        XsmmDtype<TGemmOut>(), /*dt_in0*/
        XsmmDtype<T>(), /*dt_in1*/
        XsmmDtype<T>(), /*dt_out*/
        XsmmDtype<float>(), /*dt_compute*/
        LIBXSMM_MELTW_FLAG_BINARY_NONE,
        LIBXSMM_MELTW_TYPE_BINARY_ADD
      );

      // TODO(jgong5): parallelize over M on large BS
      if (no_y_buf) {
        auto loop_scheme = M >= PARALLEL_M_THRESHOLD ? "ACb" : "aCb";
        auto gemm_loop = ThreadedLoop<3>({{0, M, BLOCK_M, false}, {Kc}, {Nc}}, loop_scheme);
        gemm_loop(
          [&](int *idx) {
            int m = idx[0];
            int kc = idx[1];
            int nc = idx[2];
            bool is_rem = (m + BLOCK_M > M);
            TGemmOut* y_ptr = num_concats <= 1 ? (TGemmOut*)py[m][nc] : (TGemmOut*)py_concat[nc/(Nc/num_concats)][m][nc%(Nc/num_concats)];
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
                dequant_gemm_tpp(x_ptr, pw[nc][kc], pscales[nc], pzps[nc], y_ptr);
              } else {
                dequant_gemm_no_prefetch_tpp(x_ptr, pw[nc][kc], pscales[nc], pzps[nc], y_ptr);
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
                dequant_gemm_rem_tpp(x_ptr, pw[nc][kc], pscales[nc], pzps[nc], y_ptr);
              } else {
                dequant_gemm_no_prefetch_rem_tpp(x_ptr, pw[nc][kc], pscales[nc], pzps[nc], y_ptr);
              }
            }
            // TODO(jgong5): post-op fusion
          },
          [&]() { dequant_gemm_tpp.config(); },
          [&]() { dequant_gemm_tpp.release(); }
        );
      } else {
        auto num_threads = omp_get_max_threads();
        TGemmOut* y_private = nullptr;
        bool* y_private_valid = nullptr;
        if (k_splits > 1) {
          // TODO(jgong5): if we know the thread decomposition, we can allocate a smaller buffer
          y_private = (TGemmOut*)std::aligned_alloc(64, num_threads * M * N * sizeof(TGemmOut));
          y_private_valid = (bool*)std::aligned_alloc(64, num_threads * (M/BLOCK_M) * Nc * sizeof(bool));
          memset(y_private_valid, 0, sizeof(bool) * num_threads * (M/BLOCK_M) * Nc);
        }
        auto y_private_ptr = GetVLAPtr<TGemmOut>(y_private, {M,Nc,Nb});
        auto y_private_valid_ptr = GetVLAPtr<bool>(y_private_valid, {M/BLOCK_M,Nc});
        auto loop_scheme = M >= PARALLEL_M_THRESHOLD ? "CAB" : "ABc";
        auto gemm_loop = ThreadedLoop<3>({{Nc}, {0, Kc, Kc/k_splits, true}, {0, M, BLOCK_M, false}}, loop_scheme);
        gemm_loop(
          [&](int *idx) {
            int my_id = omp_get_thread_num();
            int nc = idx[0];
            int kc_start = idx[1];
            int kc_end = kc_start + Kc/k_splits;
            int m = idx[2];
            bool is_rem = (m + BLOCK_M > M);
            auto y_out_ptr = num_concats <= 1 ? py[m][nc] : py_concat[nc/(Nc/num_concats)][m][nc%(Nc/num_concats)];
            alignas(64) TGemmOut y_buf[BLOCK_M][Nb];
            TGemmOut* y_ptr = y_private_ptr[my_id][m][nc];
            if (k_splits > 1) {
              if (!y_private_valid_ptr[my_id][m/BLOCK_M][nc]) {
                if (kc_start == 0 && b.defined()) {
                  copy_bias_out_tpp(pb[nc], y_ptr);
                } else {
                  zero_out_tpp(y_ptr);
                }
                y_private_valid_ptr[my_id][m/BLOCK_M][nc] = true;
              }
            } else {
              y_ptr = y_buf[0];
              if (b.defined()) {
                if(!is_rem){
                  copy_bias_buf_tpp(pb[nc], y_buf[0]);
                }else{
                  copy_bias_buf_rem_tpp(pb[nc], y_buf[0]);
                }
              } else {
                if(!is_rem){
                  zero_buf_tpp(y_buf[0]);
                }else{
                  zero_buf_rem_tpp(y_buf[0]);
                }
              }
            }
            for (int kc = kc_start; kc < kc_end; kc++) {
              if (!is_rem) {
                alignas(64) TComp x_buf[BLOCK_M][Kb];
                cvt_x_tpp(px[m][kc], x_buf[0]);
                if (kc < Kc - 1) {
                  dequant_gemm_tpp(x_buf[0], pw[nc][kc], pscales[nc], pzps[nc], y_ptr);
                } else {
                  dequant_gemm_no_prefetch_tpp(x_buf[0], pw[nc][kc], pscales[nc], pzps[nc], y_ptr);
                }
              } else {
                alignas(64) TComp x_buf[BLOCK_M][Kb];
                cvt_x_rem_tpp(px[m][kc], x_buf[0]);
                if (kc < Kc - 1) {
                  dequant_gemm_rem_tpp(x_buf[0], pw[nc][kc], pscales[nc], pzps[nc], y_ptr);
                } else {
                  dequant_gemm_no_prefetch_rem_tpp(x_buf[0], pw[nc][kc], pscales[nc], pzps[nc], y_ptr);
                }
              }
            }
            // TODO(jgong5): post-op fusion
            if (k_splits <= 1) {
              if(!is_rem){
                cvt_y_tpp(y_buf[0], y_out_ptr);
              }else{
                cvt_y_rem_tpp(y_buf[0], y_out_ptr);
              }
            }
          },
          [&]() { dequant_gemm_tpp.config(); },
          [&]() { dequant_gemm_tpp.release(); }
        );
        if (k_splits > 1) {
          TLA_ASSERT(M % BLOCK_M == 0, "M must be divisible by BLOCK_M for k_splits > 1");
          auto reduce_loop = ThreadedLoop<2>({{0, M, BLOCK_M, true}, {Nc}}, "AB");
          reduce_loop(
            [&](int *idx) {
              int m = idx[0];
              int nc = idx[1];
              bool init = false;
              for (int id = 0; id < num_threads; id++) {
                if (y_private_valid_ptr[id][m/BLOCK_M][nc]) {
                  if (!init) {
                    cvt_y_private_tpp(y_private_ptr[id][m][nc], py[m][nc]);
                    init = true;
                  } else {
                    add_y_tpp(y_private_ptr[id][m][nc], py[m][nc], py[m][nc]);
                  }
                }
              }
            }
          );
          std::free(y_private);
          std::free(y_private_valid);
        }
      }
    },
    [](auto tuple) {
      failing_fallback();
    }
  );
}

/**
 * @brief pack the weight in quantized format.
 * @param qw quantized weight with shape [N, K]
 * @param block_n block size along N, N % block_n == 0, block_n % 16 == 0
 * @param block_k block size along K, K % block_k == 0. block_k % 2 == 0 for bf16 compute_dtype.
 * false if activation is expected to be float32.
 */
at::Tensor qlinear_woq_pack(const at::Tensor& qw, size_t block_n, size_t block_k) {
  TLA_ASSERT(qw.is_contiguous(), "qw must be contiguous");
  TLA_ASSERT(
    qw.qscheme() == at::kPerChannelAffine || qw.qscheme() == at::kPerChannelAffineFloatQParams,
    "qw must be per channel affine quantized");
  auto sizes = qw.sizes();
  auto N = sizes[0];
  auto K = sizes[1];
  TLA_ASSERT(N % block_n == 0, "N must be multiple of block_n");
  TLA_ASSERT(K % block_k == 0, "K must be multiple of block_k");
  TLA_ASSERT(block_n % 16 == 0, "block_n must be multiple of 16 for int4");
  const int N_GROUP_SIZE = get_n_group_size(block_n);
  const int Nc = N / block_n;
  const int Kc = K / block_k;
  if (qw.scalar_type() == at::kQUInt4x2) {
    TLA_ASSERT(false, "Not implemented");
  } else if (qw.scalar_type() == at::kQInt8) {
    auto result = at::_empty_per_channel_affine_quantized(
      {Nc, Kc, block_k, block_n},
      qw.q_per_channel_scales().clone(at::MemoryFormat::Preserve),
      qw.q_per_channel_zero_points().clone(at::MemoryFormat::Preserve),
      qw.q_per_channel_axis(),
      qw.options()
    );
    // Pack weight in [N,K] to [N/block_n, K/block_k, block_k, block_n]
    int8_t* src_data = (int8_t*)qw.data_ptr();
    int8_t* dst_data = (int8_t*)result.data_ptr();
    auto psrc = GetVLAPtr<int8_t>(src_data, {block_n, Kc, block_k});
    auto pdst = GetVLAPtr<int8_t>(dst_data, {Kc, block_k, block_n});
    auto pack_loop = ThreadedLoop<3>({{Nc}, {Kc}, {0, block_n, N_GROUP_SIZE, false}}, "ABc");
    pack_loop(
      [&](int *idx) {
        int nc = idx[0];
        int kc = idx[1];
        int nb = idx[2];
        for (int i = 0; i < N_GROUP_SIZE; i++) {
          for (int kb = 0; kb < block_k; kb++) {
            pdst[nc][kc][kb][nb+i] = psrc[nc][nb+i][kc][kb];
          }
        }
      }
    );
    return result;
  }
  return qw;
}

at::Tensor qlinear_woq_unpack(const at::Tensor& qw_packed) {
  if (qw_packed.dim() == 4) {
    auto w_sizes = qw_packed.sizes();
    auto Nc = w_sizes[0];
    auto Nb = w_sizes[3];
    auto Kc = w_sizes[1];
    auto Kb = w_sizes[2];
    auto N = Nc * Nb;
    auto K = Kc * Kb;
    const int N_GROUP_SIZE = get_n_group_size(Nb);
    if (qw_packed.scalar_type() == at::kQUInt4x2) {
      TLA_ASSERT(false, "Not implemented");
    } else if (qw_packed.scalar_type() == at::kQInt8) {
      auto result = at::_empty_per_channel_affine_quantized(
        {N, K},
        qw_packed.q_per_channel_scales().clone(at::MemoryFormat::Preserve),
        qw_packed.q_per_channel_zero_points().clone(at::MemoryFormat::Preserve),
        qw_packed.q_per_channel_axis(),
        qw_packed.options()
      );
      int8_t* src_data = (int8_t*)qw_packed.data_ptr();
      int8_t* dst_data = (int8_t*)result.data_ptr();
      auto psrc = GetVLAPtr<int8_t>(src_data, {Kc, Kb, Nb});
      auto pdst = GetVLAPtr<int8_t>(dst_data, {Nb, Kc, Kb});
      auto unpack_loop = ThreadedLoop<3>({{Nc}, {Kc}, {0, Nb, N_GROUP_SIZE, false}}, "ABc");
      unpack_loop(
        [&](int *idx) {
          int nc = idx[0];
          int kc = idx[1];
          int nb = idx[2];
          for (int kb = 0; kb < Kb; kb++) {
            for (int i = 0; i < N_GROUP_SIZE; i++) {
              pdst[nc][nb+i][kc][kb] = psrc[nc][kc][kb][nb+i];
            }
          }
        }
      );
      return result;
    } else {
      TLA_ASSERT(false, "not implemented");
    }
  } else {
    TLA_ASSERT(qw_packed.dim() == 2, "qw_packed must be 2D or 4D");
    return qw_packed;
  }
}

#define LOWP_MODE_NONE 0
#define LOWP_MODE_FP16 1
#define LOWP_MODE_BF16 2

/**
 * @brief quantized linear with weight in affine quantized format (scale + zero-point) but
 * activation in floating point format.
 * TODO(jgong5): support epilogue fusion
 * 
 * @param x input activation in floating point format, 2D plain format [M,K]
 * @param qw weight in affine quantized format, could be 4-bit or 8-bit quantized in
 * 4D blocked format [Nc,Kc,Kb,Nb] or 2D plain format [N,K].
 * @param scales_list a list of fp32/fp16/bf16 scales tensors
 * @param zp_list a list of fp32/fp16/bf16 zero points tensors
 * @param bias_list a list of fp32/fp16/bf16 bias tensors
 * @param lowp_mode decide the compute dtype to use.
 *        LOWP_MODE_NONE: keep activation dtype
 *        LOWP_MODE_FP16: use FP16 or FP32 as compute dtype
 *        LOWP_MODE_BF16: use BF16, FP16 or FP32 as compute dtype
 * @return at::Tensor output activation in same dtype as `x`, 2D plain format [M,N]
 */
at::Tensor qlinear_woq_affine(
    const at::Tensor& x,
    const at::Tensor& qw,
    const TensorList& scales_list,
    const TensorList& zp_list,
    const TensorList& bias_list,
    int64_t lowp_mode,
    // int64_t k_splits,
    int64_t num_concats) {
  const int64_t k_splits = 0;
  constexpr size_t fp32_idx = 0, fp16_idx = 1, bf16_idx = 2;
  auto biases = bias_list.empty() ? TensorList({at::Tensor(), at::Tensor(), at::Tensor()}) : bias_list;
  if (qw.dim() == 4) {
    auto w_sizes = qw.sizes();
    auto K = x.size(-1);
    auto M = x.numel() / K;
    auto N = w_sizes[0] * w_sizes[3];
    auto out_sizes = x.sizes().vec();
    out_sizes.back() = N;
    auto y = at::empty(out_sizes, x.options());
    auto x_reshape = x.reshape({M, K});
    enumerate_dispatcher<at::ScalarType, at::kFloat, at::kBFloat16, at::kHalf>::call(x.scalar_type(),
      [&](auto act_dtype) {
        using act_type = typename c10::impl::ScalarTypeToCPPType<act_dtype>::type;
        auto try_compute_in_half = [&]() {
#ifdef __AVX512FP16__
          qlinear_woq_affine_impl<act_type, half, /*TGemmOut*/half>(
              x_reshape, qw, scales_list[fp16_idx], zp_list[fp16_idx], biases[fp16_idx], y, k_splits, num_concats);
#else
          qlinear_woq_affine_impl<act_type, float, /*TGemmOut*/float>(
              x_reshape, qw, scales_list[fp32_idx], zp_list[fp32_idx], biases[fp32_idx], y, k_splits, num_concats);
#endif
        };
        if (lowp_mode == LOWP_MODE_NONE) {
          if (std::is_same<act_type, half>()) {
            try_compute_in_half();
          } else if (std::is_same<act_type, bfloat16>()) {
            qlinear_woq_affine_impl<bfloat16, bfloat16, /*TGemmOut*/float>(
              x_reshape, qw, scales_list[bf16_idx], zp_list[bf16_idx], biases[fp32_idx], y, k_splits, num_concats
            );
          } else {
            qlinear_woq_affine_impl<float, float, /*TGemmOut*/float>(
              x_reshape, qw, scales_list[fp32_idx], zp_list[fp32_idx], biases[fp32_idx], y, k_splits, num_concats
            );
          }
        } else {
          if (lowp_mode == LOWP_MODE_FP16) {
            try_compute_in_half();
          } else {
            TLA_ASSERT(lowp_mode == LOWP_MODE_BF16, "invalid lowp_mode");
            if (M >= SMALL_BATCH_THRESHOLD) {
              // compute in bfloat16 for large bs
              qlinear_woq_affine_impl<act_type, bfloat16, /*TGemmOut*/float>(
                  x_reshape, qw, scales_list[bf16_idx], zp_list[bf16_idx], biases[fp32_idx], y, k_splits, num_concats);
            } else {
              try_compute_in_half();
            }
          }
        }
      },
      failing_fallback<at::ScalarType>
    );
    return y;
  } else {
    TLA_ASSERT(qw.dim() == 2, "weight must be in 4D blocked format or 2D plain format");
    auto compute_dtype = x.scalar_type();
    if (lowp_mode == LOWP_MODE_FP16) {
      compute_dtype = at::kHalf;
    } else if (lowp_mode == LOWP_MODE_BF16) {
      compute_dtype = at::kBFloat16;
    }
    auto w = qw.dequantize().to(compute_dtype);
    auto x_fp = x.to(compute_dtype);
    auto y = at::linear(x_fp, w);
    if (biases[0].defined()) {
      auto b_index = compute_dtype == at::kFloat ? fp32_idx :
                     compute_dtype == at::kHalf ? fp16_idx : bf16_idx;
      y = at::add(y, biases[b_index]);
    }
    if (num_concats > 1) {
      y = y.view({-1, num_concats, y.size(-1)/num_concats}).transpose(0, 1).contiguous().view({-1, y.size(-1)});
    }
    return y.to(x.scalar_type());
  }
}

} // namespace

REGISTER_DISPATCH(woq_tpp_gemm_kernel_stub, &qlinear_woq_affine);
REGISTER_DISPATCH(woq_tpp_gemm_packB_stub, &qlinear_woq_pack);
REGISTER_DISPATCH(woq_tpp_gemm_unpackB_stub, &qlinear_woq_unpack);

} // namespace cpu
} // namespace torch_ipex