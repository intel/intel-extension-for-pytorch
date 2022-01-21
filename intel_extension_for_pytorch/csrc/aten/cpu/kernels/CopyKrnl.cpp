#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/cpu/zmath.h>
#include <ATen/native/quantized/Copy.h>
#include <ATen/native/quantized/affine_quantizer.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/record_function.h>
#include <c10/util/TypeCast.h>

#ifdef USE_FBGEMM
#include <fbgemm/Fbgemm.h>
#include <fbgemm/FbgemmConvert.h>
#endif

#include <csrc/aten/cpu/Copy.h>

#include "csrc/utils/library.h"

namespace torch_ipex {
namespace cpu {

#if defined(DYN_DISP_BUILD)
namespace {
#endif

// transpose operation of a 8*8 block
template <typename T>
inline void transpose_kernel_8x8(
    const T* src,
    int64_t ld_src,
    T* dst,
    int64_t ld_dst) {
  for (int64_t i = 0; i < 8; i++) {
    for (int64_t j = 0; j < 8; j++) {
      dst[j * ld_dst + i] = src[i * ld_src + j];
    }
  }
}

template <>
inline void transpose_kernel_8x8<float>(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  // inputs:
  //   a = {a0, a1, a2, a3, a4, a5, a6, a7}
  //   b = {b0, b1, b2, b3, b4, b5, b6, b7}
  //   c = {c0, c1, c2, c3, c4, c5, c6, c7}
  //   d = {d0, d1, d2, d3, d4, d5, d6, d7}
  //   e = {e0, e1, e2, e3, e4, e5, e6, e7}
  //   f = {f0, f1, f2, f3, f4, f5, f6, f7}
  //   g = {g0, g1, g2, g3, g4, g5, g6, g7}
  //   h = {h0, h1, h2, h3, h4, h5, h6, h7}
  __m256 a = _mm256_loadu_ps(&src[0 * ld_src]);
  __m256 b = _mm256_loadu_ps(&src[1 * ld_src]);
  __m256 c = _mm256_loadu_ps(&src[2 * ld_src]);
  __m256 d = _mm256_loadu_ps(&src[3 * ld_src]);
  __m256 e = _mm256_loadu_ps(&src[4 * ld_src]);
  __m256 f = _mm256_loadu_ps(&src[5 * ld_src]);
  __m256 g = _mm256_loadu_ps(&src[6 * ld_src]);
  __m256 h = _mm256_loadu_ps(&src[7 * ld_src]);

  // interleave 32 bit:
  //   t0 = {a0, b0, a1, b1, a4, b4, a5, b5}
  //   t1 = {a2, b2, a3, b3, a6, b6, a7, b7}
  //   t2 = {c0, d0, c1, d1, c4, d4, c5, d5}
  //   t3 = {c2, d2, c3, d3, c6, d6, c7, d7}
  //   t4 = {e0, f0, e1, f1, e4, f4, e5, f5}
  //   t5 = {e2, f2, e3, f3, e6, f6, e7, f7}
  //   t6 = {g0, h0, g1, h1, g4, h4, g5, h5}
  //   t7 = {g2, h2, g3, h3, g6, h6, g7, h7}
  __m256 t0 = _mm256_unpacklo_ps(a, b);
  __m256 t1 = _mm256_unpackhi_ps(a, b);
  __m256 t2 = _mm256_unpacklo_ps(c, d);
  __m256 t3 = _mm256_unpackhi_ps(c, d);
  __m256 t4 = _mm256_unpacklo_ps(e, f);
  __m256 t5 = _mm256_unpackhi_ps(e, f);
  __m256 t6 = _mm256_unpacklo_ps(g, h);
  __m256 t7 = _mm256_unpackhi_ps(g, h);

  // shuffle 64 bit:
  //   tt0 = {a0, b0, c0, d0, a4, b4, c4, d4}
  //   tt1 = {a1, b1, c1, d1, a5, b5, c5, d5}
  //   tt2 = {e0, f0, g0, h0, e4, f4, g4, h4}
  //   tt3 = {e1, f1, g1, h1, e5, b5, c5, d5}
  //   tt4 = {a2, b2, c2, d2, a6, b6, c6, d6}
  //   tt5 = {a3, b3, c3, d3, a7, b7, c7, d7}
  //   tt6 = {e2, f2, g2, h2, e6, f6, g6, h6}
  //   tt7 = {e3, f3, g3, h3, e7, f7, g7, h7}
  __m256 tt0 = _mm256_shuffle_ps(t0, t2, 0x44);
  __m256 tt1 = _mm256_shuffle_ps(t0, t2, 0xee);
  __m256 tt2 = _mm256_shuffle_ps(t4, t6, 0x44);
  __m256 tt3 = _mm256_shuffle_ps(t4, t6, 0xee);
  __m256 tt4 = _mm256_shuffle_ps(t1, t3, 0x44);
  __m256 tt5 = _mm256_shuffle_ps(t1, t3, 0xee);
  __m256 tt6 = _mm256_shuffle_ps(t5, t7, 0x44);
  __m256 tt7 = _mm256_shuffle_ps(t5, t7, 0xee);

  // swap 128 bit:
  //   a = {a0, b0, c0, d0, e0, f0, g0, h0}
  //   b = {a1, b1, c1, d1, e1, f1, g1, h1}
  //   c = {a2, b2, c2, d2, e2, f2, g2, h2}
  //   d = {a3, b3, c3, d3, e3, f3, g3, h3}
  //   e = {a4, b4, c4, d4, e4, f4, g4, h4}
  //   f = {a5, b5, c5, d5, e5, f5, g5, h5}
  //   g = {a6, b6, c6, d6, e6, f6, g6, h6}
  //   h = {a7, b7, c7, d7, e7, f7, g7, h7}
  a = _mm256_permute2f128_ps(tt0, tt2, 0x20);
  b = _mm256_permute2f128_ps(tt1, tt3, 0x20);
  c = _mm256_permute2f128_ps(tt4, tt6, 0x20);
  d = _mm256_permute2f128_ps(tt5, tt7, 0x20);
  e = _mm256_permute2f128_ps(tt0, tt2, 0x31);
  f = _mm256_permute2f128_ps(tt1, tt3, 0x31);
  g = _mm256_permute2f128_ps(tt4, tt6, 0x31);
  h = _mm256_permute2f128_ps(tt5, tt7, 0x31);

  _mm256_storeu_ps(&dst[0 * ld_dst], a);
  _mm256_storeu_ps(&dst[1 * ld_dst], b);
  _mm256_storeu_ps(&dst[2 * ld_dst], c);
  _mm256_storeu_ps(&dst[3 * ld_dst], d);
  _mm256_storeu_ps(&dst[4 * ld_dst], e);
  _mm256_storeu_ps(&dst[5 * ld_dst], f);
  _mm256_storeu_ps(&dst[6 * ld_dst], g);
  _mm256_storeu_ps(&dst[7 * ld_dst], h);
}

template <>
inline void transpose_kernel_8x8<at::BFloat16>(
    const at::BFloat16* src,
    int64_t ld_src,
    at::BFloat16* dst,
    int64_t ld_dst) {
  // inputs:
  //   a = {a0, a1, a2, a3, a4, a5, a6, a7}
  //   b = {b0, b1, b2, b3, b4, b5, b6, b7}
  //   c = {c0, c1, c2, c3, c4, c5, c6, c7}
  //   d = {d0, d1, d2, d3, d4, d5, d6, d7}
  //   e = {e0, e1, e2, e3, e4, e5, e6, e7}
  //   f = {f0, f1, f2, f3, f4, f5, f6, f7}
  //   g = {g0, g1, g2, g3, g4, g5, g6, g7}
  //   h = {h0, h1, h2, h3, h4, h5, h6, h7}
  __m128i a =
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[0 * ld_src]));
  __m128i b =
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[1 * ld_src]));
  __m128i c =
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[2 * ld_src]));
  __m128i d =
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[3 * ld_src]));
  __m128i e =
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[4 * ld_src]));
  __m128i f =
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[5 * ld_src]));
  __m128i g =
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[6 * ld_src]));
  __m128i h =
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[7 * ld_src]));

  // interleave 16 bit:
  //   t0 = {a0, b0, a1, b1, a2, b2, a3, b3}
  //   t1 = {a4, b4, a5, b5, a6, b6, a7, b7}
  //   t2 = {c0, d0, c1, d1, c2, d2, c3, d3}
  //   t3 = {c4, d4, c5, d5, c6, d6, c7, d7}
  //   t4 = {e0, f0, e1, f1, e2, f2, e3, f3}
  //   t5 = {e4, f4, e5, f5, e6, f6, e7, f7}
  //   t6 = {g0, h0, g1, h1, g2, h2, g3, h3}
  //   t7 = {g4, h4, g5, h5, g6, h6, g7, h7}
  __m128i t0 = _mm_unpacklo_epi16(a, b);
  __m128i t1 = _mm_unpackhi_epi16(a, b);
  __m128i t2 = _mm_unpacklo_epi16(c, d);
  __m128i t3 = _mm_unpackhi_epi16(c, d);
  __m128i t4 = _mm_unpacklo_epi16(e, f);
  __m128i t5 = _mm_unpackhi_epi16(e, f);
  __m128i t6 = _mm_unpacklo_epi16(g, h);
  __m128i t7 = _mm_unpackhi_epi16(g, h);

  // interleave 32 bit:
  //   tt0 = {a0, b0, c0, d0, a1, b1, c1, d1}
  //   tt1 = {a2, b2, c2, d2, a3, b3, c3, d3}
  //   tt2 = {a4, b4, c4, d4, a5, b5, c5, d5}
  //   tt3 = {a6, b6, c6, d6, a7, b7, c7, d7}
  //   tt4 = {e0, f0, g0, h0, e1, f1, g1, g1}
  //   tt5 = {e2, f2, g2, h2, e3, f3, g3, h3}
  //   tt6 = {e4, f4, g4, h4, e5, f5, g5, h5}
  //   tt7 = {e6, f6, g6, h6, e7, f7, g7, h7}
  __m128i tt0 = _mm_unpacklo_epi32(t0, t2);
  __m128i tt1 = _mm_unpackhi_epi32(t0, t2);
  __m128i tt2 = _mm_unpacklo_epi32(t1, t3);
  __m128i tt3 = _mm_unpackhi_epi32(t1, t3);
  __m128i tt4 = _mm_unpacklo_epi32(t4, t6);
  __m128i tt5 = _mm_unpackhi_epi32(t4, t6);
  __m128i tt6 = _mm_unpacklo_epi32(t5, t7);
  __m128i tt7 = _mm_unpackhi_epi32(t5, t7);

  // interleave 64 bit:
  //   a = {a0, b0, c0, d0, e0, f0, g0, h0}
  //   b = {a1, b1, c1, d1, e1, f1, g1, h1}
  //   c = {a2, b2, c2, d2, e2, f2, g2, h2}
  //   d = {a3, b3, c3, d3, e3, f3, g3, h3}
  //   e = {a4, b4, c4, d4, e4, f4, g4, h4}
  //   f = {a5, b5, c5, d5, e5, f5, g5, h5}
  //   g = {a6, b6, c6, d6, e6, f6, g6, h6}
  //   h = {a7, b7, c7, d7, e7, f7, g7, h7}
  a = _mm_unpacklo_epi64(tt0, tt4);
  b = _mm_unpackhi_epi64(tt0, tt4);
  c = _mm_unpacklo_epi64(tt1, tt5);
  d = _mm_unpackhi_epi64(tt1, tt5);
  e = _mm_unpacklo_epi64(tt2, tt6);
  f = _mm_unpackhi_epi64(tt2, tt6);
  g = _mm_unpacklo_epi64(tt3, tt7);
  h = _mm_unpackhi_epi64(tt3, tt7);

  _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[0 * ld_dst]), a);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[1 * ld_dst]), b);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[2 * ld_dst]), c);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[3 * ld_dst]), d);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[4 * ld_dst]), e);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[5 * ld_dst]), f);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[6 * ld_dst]), g);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[7 * ld_dst]), h);
}

static void copy_kernel(at::TensorIterator& iter, bool non_blocking) {
  at::ScalarType dtype = iter.dtype(0);
  if (dtype == iter.dtype(1)) {
    // TODO: as the majority of these operations can be done treating
    // their datatypes as opaque bit patterns, we don't actually need
    // separate instantiations per dtype; we only need a separate
    // instantiation per dtype size.  This would probably save us a
    // little bit of code size here
    // TODO: not sure if optimizer is able to compile two levels of
    // conditionals into a single jump table.  We should have a
    // single jump table here; might be worth just writing out the
    // dispatch statement by hand instead of using AT_DISPATCH
    if (iter.tensor(0).is_neg() == iter.tensor(1).is_neg()) {
      if (dtype == at::ScalarType::Half) {
        at::native::cpu_kernel(iter, [=](at::Half a) -> at::Half { return a; });
      } else if (dtype == at::ScalarType::ComplexHalf) {
        at::native::cpu_kernel(
            iter, [=](c10::complex<at::Half> a) -> c10::complex<at::Half> {
              return a;
            });
      } else if (isQIntType(dtype)) {
        AT_DISPATCH_QINT_TYPES(dtype, "copy_kernel", [&] {
          at::native::cpu_kernel_vec(
              iter,
              [=](scalar_t a) -> scalar_t { return a; },
              [=](at::vec::Vectorized<scalar_t> a)
                  -> at::vec::Vectorized<scalar_t> { return a; });
        });
      } else if (isComplexType(dtype)) {
        // This case should never actually happen since currently there's no way
        // to get a complex tensor with negative bit.
        if (iter.tensor(0).is_conj() == iter.tensor(1).is_conj()) {
          AT_DISPATCH_COMPLEX_TYPES(dtype, "copy_kernel", [&] {
            at::native::cpu_kernel_vec(
                iter,
                [=](scalar_t a) -> scalar_t { return a; },
                [=](at::vec::Vectorized<scalar_t> a)
                    -> at::vec::Vectorized<scalar_t> { return a; });
          });
        } else {
          AT_DISPATCH_COMPLEX_TYPES(dtype, "conj_kernel", [&] {
            at::native::cpu_kernel_vec(
                iter,
                [=](scalar_t a) -> scalar_t {
                  return at::native::conj_impl(a);
                },
                [=](at::vec::Vectorized<scalar_t> a)
                    -> at::vec::Vectorized<scalar_t> { return a.conj(); });
          });
        }
      } else {
        AT_DISPATCH_ALL_TYPES_AND2(
            at::ScalarType::Bool,
            at::ScalarType::BFloat16,
            dtype,
            "copy_kernel",
            [&] {
              at::native::cpu_kernel_vec(
                  iter,
                  [=](scalar_t a) -> scalar_t { return a; },
                  [=](at::vec::Vectorized<scalar_t> a) { return a; });
            });
      }
    } else {
      if (dtype == at::ScalarType::Half) {
        at::native::cpu_kernel(
            iter, [=](at::Half a) -> at::Half { return -a; });
      } else if (isComplexType(dtype)) {
        if (iter.tensor(0).is_conj() == iter.tensor(1).is_conj()) {
          AT_DISPATCH_COMPLEX_TYPES(dtype, "copy_kernel", [&] {
            at::native::cpu_kernel_vec(
                iter,
                [=](scalar_t a) -> scalar_t { return -a; },
                [=](at::vec::Vectorized<scalar_t> a)
                    -> at::vec::Vectorized<scalar_t> { return a.neg(); });
          });
        } else {
          AT_DISPATCH_COMPLEX_TYPES(dtype, "conj_kernel", [&] {
            at::native::cpu_kernel_vec(
                iter,
                [=](scalar_t a) -> scalar_t {
                  return -1 * at::native::conj_impl(a);
                },
                [=](at::vec::Vectorized<scalar_t> a)
                    -> at::vec::Vectorized<scalar_t> {
                  return a.neg().conj();
                });
          });
        }
      } else {
        AT_DISPATCH_ALL_TYPES_AND2(
            at::ScalarType::Bool,
            at::ScalarType::BFloat16,
            dtype,
            "copy_kernel",
            [&] {
              at::native::cpu_kernel_vec(
                  iter,
                  [=](scalar_t a) -> scalar_t { return -a; },
                  [=](at::vec::Vectorized<scalar_t> a)
                      -> at::vec::Vectorized<scalar_t> { return a.neg(); });
            });
      }
    }
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        at::ScalarType::Half,
        at::ScalarType::Bool,
        at::ScalarType::BFloat16,
        dtype,
        "copy_",
        [&] {
          using dest_t = scalar_t;
          AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
              at::ScalarType::Half,
              at::ScalarType::Bool,
              at::ScalarType::BFloat16,
              iter.dtype(1),
              "copy_",
              [&] {
                // Note (@zasdfgbnm):
                //
                // The code below can not be simplified as
                //    at::native::cpu_kernel(iter,
                //    c10::static_cast_with_inter_type<dest_t,
                //    scalar_t>::apply);
                //
                // because this would force the compiler to instantiate the
                // inline function and generate a function call in the loop
                // instead of inlining it, making all the optimizations like
                // vectorization impossible. You can verify this by looking the
                // the symbols of `libtorch_cpu.so`:
                //
                //    readelf -Ws libtorch_cpu.so | grep
                //    static_cast_with_inter_type
                //
                // If done correctly, the above command should have no output.
                //
                // See: https://github.com/pytorch/pytorch/issues/31271
                at::native::cpu_kernel(iter, [](scalar_t src) -> dest_t {
                  return c10::static_cast_with_inter_type<dest_t, scalar_t>::
                      apply(src);
                });
              });
        });
    if (iter.tensor(0).is_conj() != iter.tensor(1).is_conj()) {
      iter.tensor(0).conj_physical_();
    }
    if (iter.tensor(0).is_neg() != iter.tensor(1).is_neg()) {
      iter.tensor(0).neg_();
    }
  }
}

template <typename scalar_t>
void transpose_copy_kernel_impl(at::Tensor& self, const at::Tensor& src) {
  scalar_t* self_data = self.data_ptr<scalar_t>();
  scalar_t* src_data = src.data_ptr<scalar_t>();

  int64_t M = src.size(0);
  int64_t N = src.size(1);

  constexpr int64_t BLOCK_SIZE = 8;
  int64_t K = at::divup(M, BLOCK_SIZE);

  // parallel on outer most dimension
  // TODO: vectorize the remainder
  int64_t grain_size = at::internal::GRAIN_SIZE / N / BLOCK_SIZE;
  at::parallel_for(0, K, grain_size, [&](int64_t begin, int64_t end) {
    int64_t rbegin = begin * BLOCK_SIZE;
    int64_t rend = std::min(end * BLOCK_SIZE, M);

    int64_t i = rbegin;
    for (; i < rend - (rend % BLOCK_SIZE); i += BLOCK_SIZE) {
      int64_t j = 0;
      for (; j < N - (N % BLOCK_SIZE); j += BLOCK_SIZE) {
        transpose_kernel_8x8<scalar_t>(
            &src_data[j * M + i], M, &self_data[i * N + j], N);
      }
      for (; j < N; j++) {
        for (int64_t k = i; k < i + BLOCK_SIZE; k++) {
          self_data[k * N + j] = src_data[j * M + k];
        }
      }
    }
    for (; i < rend; i++) {
      for (int64_t j = 0; j < N; j++) {
        self_data[i * N + j] = src_data[j * M + i];
      }
    }
  });
}

static void transpose_copy_kernel(at::Tensor& self, const at::Tensor& src) {
  TORCH_CHECK(self.is_contiguous(), "self is not contiguous");
  TORCH_CHECK(src.numel() > 0, "expect src number of elements > 0");
  TORCH_CHECK(
      src.dim() == 2 && self.dim() == 2,
      "expect src and self dims to be 2, self dim: ",
      src.dim(),
      "; self dim: ",
      self.dim());
  TORCH_CHECK(src.stride(0) == 1, "src first dimension is not contiguous");
  TORCH_CHECK(
      src.stride(1) == src.size(0), "expect src.stride(1) == src.size(0)");
  TORCH_CHECK(
      src.scalar_type() == self.scalar_type(),
      "expect same data type for src and self, src data type: ",
      src.scalar_type(),
      "; self data type: ",
      self.scalar_type());

  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      src.scalar_type(),
      "transpose_copy_kernel",
      [&] { transpose_copy_kernel_impl<scalar_t>(self, src); });
}

bool copy_transpose_valid(const at::Tensor& self, const at::Tensor& src) {
  const int MIN_SZ = 60 * 60;
  return self.is_contiguous() && src.numel() != 0 && src.dim() == 2 &&
      src.stride(0) == 1 && src.stride(1) == src.size(0) &&
      self.scalar_type() == src.scalar_type() &&
      self.sizes().equals(src.sizes()) && self.numel() >= MIN_SZ;
}

// special case copy where tensor is contiguous and src is a transposed matrix
// This can be generalized to most copies, but it's trickier
void copy_same_type_transpose_(at::Tensor& self, const at::Tensor& src) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t BLOCK_SZ;
  if (self.scalar_type() == at::kByte) {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    BLOCK_SZ = 120;
  } else {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    BLOCK_SZ = 60;
  }
  at::Tensor buf = at::empty({BLOCK_SZ, BLOCK_SZ}, self.options());

  // The code below is implemented with the assumption that sizes are equal
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.sizes().equals(src.sizes()));

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::kHalf, at::kBool, at::kBFloat16, self.scalar_type(), "copy_", [&] {
        scalar_t* sp = src.data_ptr<scalar_t>();
        scalar_t* rp = self.data_ptr<scalar_t>();
        scalar_t* bp = buf.data_ptr<scalar_t>();

        int64_t NR = src.size(0);
        int64_t NC = src.size(1);
        for (int64_t R = 0; R < NR; R += BLOCK_SZ) {
          for (int64_t C = 0; C < NC; C += BLOCK_SZ) {
            scalar_t* spo = sp + R + C * NR;
            scalar_t* rpo = rp + C + R * NC;

            int nr = std::min(NR - R, BLOCK_SZ);
            int nc = std::min(NC - C, BLOCK_SZ);

            // 1. copy columns from src to buf
            for (int c = 0; c < nc; c++) {
              memcpy(bp + c * BLOCK_SZ, spo + c * NR, nr * sizeof(scalar_t));
            }

            // 2. transpose buf in place
            int rc_max = std::max(nr, nc);
            int rc_min = std::min(nr, nc);
            for (int r = 0; r < rc_max; r++) {
              int end = std::min(r, rc_min);
              for (int c = 0; c < end; c++) {
                scalar_t tmp = bp[r + BLOCK_SZ * c];
                bp[r + BLOCK_SZ * c] = bp[r * BLOCK_SZ + c];
                bp[r * BLOCK_SZ + c] = tmp;
              }
            }

            // 3. copy rows from buf to dst
            for (int r = 0; r < nr; r++) {
              memcpy(rpo + r * NC, bp + r * BLOCK_SZ, nc * sizeof(scalar_t));
            }
          }
        }
      });
}

// Devices directly supported by this copy implementation. Other device types
// (e.g. XLA) may be supported by overriding copy_ and _copy_from.
bool is_supported_device(at::Device device) {
  at::DeviceType device_type = device.type();
  return device_type == at::kCPU || device_type == at::kCUDA ||
      device_type == at::kHIP || device_type == at::kVulkan ||
      device_type == at::kMetal;
}

at::Tensor& quantized_copy_from_float_cpu_(
    at::Tensor& self,
    const at::Tensor& src) {
  TORCH_CHECK(
      src.scalar_type() == at::kFloat,
      "Quantized copy only works with kFloat as source Tensor");
  TORCH_CHECK(
      self.is_contiguous() && src.is_contiguous(),
      "Quantized copy only works with contiguous Tensors");
  TORCH_CHECK(
      self.sizes().equals(src.sizes()),
      "Quantized copy only works with Tensors with the same shape");
  TORCH_CHECK(
      self.device().type() == at::kCPU,
      "Quantized copy only works with QuantizedCPU Tensors");
  AT_DISPATCH_QINT_TYPES(self.scalar_type(), "Copy", [&]() {
    float* src_data = src.data_ptr<float>();
    scalar_t* self_data = self.data_ptr<scalar_t>();
    for (int i = 0; i < self.numel(); ++i) {
      self_data[i] = at::native::quantize_val<scalar_t>(
          self.q_scale(), self.q_zero_point(), src_data[i]);
    }
  });
  return self;
}

at::Tensor& copy_kernel_impl(
    at::Tensor& self,
    const at::Tensor& src,
    bool non_blocking) {
  // TODO: this should be handled during dispatch, but that's missing...
  TORCH_CHECK(self.defined(), "self is undefined");
  TORCH_CHECK(src.defined(), "src is undefined");

// FBGeMM kernel support exists only for the following case,
// 1. Memory Format for source and destination tensors is contiguous.
// 2. Device for both the source and destination tensor is CPU.
// 3. dtype conversion between FP32->FP16 and FP16->FP32.
#ifdef USE_FBGEMM
  if (((self.dtype() == at::kFloat && src.dtype() == at::kHalf) ||
       (self.dtype() == at::kHalf && src.dtype() == at::kFloat)) &&
      (self.device().is_cpu() && src.device().is_cpu()) && !self.is_sparse() &&
      !src.is_sparse() &&
      ((self.is_contiguous() && src.is_contiguous()) ||
       (self.is_non_overlapping_and_dense() &&
        self.strides() == src.strides()))) {
    if (src.dtype() == at::kFloat && self.dtype() == at::kHalf) {
      auto* output_ptr =
          reinterpret_cast<fbgemm::float16*>(self.data_ptr<at::Half>());
      if (self.numel() < at::internal::GRAIN_SIZE) {
        fbgemm::FloatToFloat16_simd(
            src.data_ptr<float>(), output_ptr, self.numel());
      } else {
        at::parallel_for(
            0,
            self.numel(),
            at::internal::GRAIN_SIZE,
            [&](int64_t begin, int64_t end) {
              fbgemm::FloatToFloat16_simd(
                  src.data_ptr<float>() + begin,
                  output_ptr + begin,
                  end - begin);
            });
      }
    } else {
      auto in_data =
          reinterpret_cast<fbgemm::float16*>(src.data_ptr<at::Half>());
      auto* output_ptr = self.data_ptr<float>();
      if (self.numel() < at::internal::GRAIN_SIZE) {
        fbgemm::Float16ToFloat_simd(in_data, output_ptr, self.numel());
      } else {
        at::parallel_for(
            0,
            self.numel(),
            at::internal::GRAIN_SIZE,
            [&](int64_t begin, int64_t end) {
              fbgemm::Float16ToFloat_simd(
                  in_data + begin, output_ptr + begin, end - begin);
            });
      }
    }
    return self;
  }
#endif

  if (self.is_sparse() && src.is_sparse()) {
    return at::copy_sparse_to_sparse_(self, src, non_blocking);
  } else if (self.is_sparse() || src.is_sparse()) {
    AT_ERROR(
        "copy_() between dense and sparse Tensors is not implemented! "
        "Found self type = ",
        self.toString(),
        " and src type = ",
        src.toString());
  }

  if (self.is_same(src)) {
    return self;
  }

  // Copies into meta self are OK and just ignored (similar to inplace)
  if (self.is_meta()) {
    // TODO: need to see if there is extra error checking needed
    return self;
  }

  if (src.is_meta()) {
    TORCH_CHECK_NOT_IMPLEMENTED(
        false, "Cannot copy out of meta tensor; no data!")
  }

  // Re-dispatch copies when either src or self device not implemented here
  // (e.g. XLA). _copy_from has a proper device dispatch setup. This includes:
  //   cpu_tensor.copy_(xla_tensor) => xla_tensor._copy_from(cpu_tensor)
  //   xla_tensor.copy_(cpu_tensor) => cpu_tensor._copy_from(xla_tensor)
  // Both the _copy_from calls above will be dispatched to XLA's _copy_from
  // kernels.
  if (!is_supported_device(src.device()) ||
      !is_supported_device(self.device())) {
    at::_copy_from(src, self, non_blocking);
    return self;
  }

  if (self.is_quantized() && !src.is_quantized()) {
    return quantized_copy_from_float_cpu_(self, src);
  }

  if (self.is_quantized() && src.is_quantized()) {
    TORCH_CHECK(
        self.qscheme() == src.qscheme(),
        "Quantized Copy only works with same qscheme");
    TORCH_CHECK(self.scalar_type() == src.scalar_type());
    set_quantizer_(self, src.quantizer());
  }

  if (!self.is_quantized() && src.is_quantized()) {
    TORCH_CHECK(
        false,
        "Copying from quantized Tensor to non-quantized Tensor "
        "is not allowed, please use dequantize to get a float "
        "Tensor from a quantized Tensor");
  }

  //   if (self.device().type() == at::kVulkan || src.device().type() ==
  //   at::kVulkan) { #ifdef USE_VULKAN_API
  //     return vulkan::ops::copy_(self, src);
  //   #else
  //     return at::vulkan::vulkan_copy_(self, src);
  //   #endif
  //   }

  //   if (self.device().type() == at::kMetal || src.device().type() ==
  //   at::kMetal) {
  //     return at::metal::metal_copy_(self, src);
  //   }

  auto iter = at::TensorIteratorConfig()
                  .add_output(self)
                  .add_input(src)
                  .resize_outputs(false)
                  .check_all_same_dtype(false)
                  .check_all_same_device(false)
                  .build();

  if (iter.numel() == 0) {
    return self;
  }

  at::DeviceType device_type = iter.device_type(0);
  if (iter.device_type(1) == at::kCUDA) {
    device_type = at::kCUDA;
  } else if (iter.device_type(1) == at::kHIP) {
    device_type = at::kHIP;
  }

  // TODO: if we need to, we can also enable this path for quantized tensor
  if (device_type == at::kCPU && copy_transpose_valid(self, src) &&
      !self.is_quantized()) {
    auto st = self.scalar_type();
    if (st == at::ScalarType::Float || st == at::ScalarType::BFloat16) {
      transpose_copy_kernel(self, src);
      return self;
    }
    copy_same_type_transpose_(self, src);
    return self;
  }

  if (!self.is_complex() && src.is_complex()) {
    TORCH_WARN_ONCE(
        "Casting complex values to real discards the imaginary part");
  }
  copy_kernel(iter, non_blocking);
  return self;
}

#if defined(DYN_DISP_BUILD)
} // anonymous namespace

REGISTER_DISPATCH(copy_kernel_stub, &copy_kernel_impl);

#endif

} // namespace cpu
} // namespace torch_ipex
