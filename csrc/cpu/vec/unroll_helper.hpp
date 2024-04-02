#ifndef UNROLL_HELPER_HPP
#define UNROLL_HELPER_HPP

#include <immintrin.h>
#include "aten/utils/utils.h"

// This helper aims to provide a set of lambda function to manully unroll
// vectorized intrisics with compile_time_for
// https://github.com/intel/intel-extension-for-pytorch/blob/05aeaf4b675f15c68fcde5b575b4fd5151971129/csrc/cpu/aten/utils/utils.h#L68
// For example,
// auto load_fp32 = [](auto i, __m512* in_vset, auto* basic_ptr) {
//   in_vset[i] = _mm512_loadu_ps(basic_ptr + 16 * i);
// };
// compile_time_for<4>::op(load_fp32, fp32_vec_set, v.data);
// is equal to:
// fp32_vec_set[0] = _mm512_loadu_ps(v.data);
// fp32_vec_set[1] = _mm512_loadu_ps(v.data + 16);
// fp32_vec_set[2] = _mm512_loadu_ps(v.data + 32);
// fp32_vec_set[3] = _mm512_loadu_ps(v.data + 64);

namespace torch_ipex {
namespace cpu {

#if defined(CPU_CAPABILITY_AVX512)
// set zero
inline auto set_zero = [](auto i, __m512* vset) {
  // set all vector to zero in vset
  vset[i] = _mm512_setzero_ps();
};
inline auto set_zero_2d = [](auto i, __m512** vset) {
  // set all vector to zero in 2d [i, 4] vset
  // TODO: Support arbitrary inner size than 4
  compile_time_for<4>::op(set_zero, vset[i]);
};

// fma
inline auto fma =
    [](auto i, __m512* inout_vset, __m512* a_vet, __m512* b_vset) {
      // inout_vset = a_vet * b_vset + inout_vset
      inout_vset[i] = _mm512_fmadd_ps(a_vet[i], b_vset[i], inout_vset[i]);
    };
inline auto fma_constant_a =
    [](auto i, __m512* inout_vset, __m512 a, __m512* b_vset) {
      // inout_vset = a * b_vset + inout_vset, a is not a set
      inout_vset[i] = _mm512_fmadd_ps(a, b_vset[i], inout_vset[i]);
    };
inline auto bcast_fma = [](auto i,
                           __m512** inout_vset,
                           __m512* m2_vset,
                           __m512 bcast,
                           auto* basic_ptr,
                           int32_t offset) {
  // load value from according to basic_ptr and offset to broadcast
  // and do fma with broadcast vector
  bcast = _mm512_set1_ps(*(basic_ptr + offset * i));
  compile_time_for<4>::op(fma_constant_a, inout_vset[i], bcast, m2_vset);
};
inline auto acc_with_fma =
    [](auto i, __m512* dst, __m512* vset_1, __m512* vset_2) {
      // acc vset1[i] * vset2[i] to dst, dst is not a set
      *dst = _mm512_fmadd_ps(vset_1[i], vset_2[i], *dst);
    };

// load
inline auto load_fp32 = [](auto i, __m512* in_vset, auto* basic_ptr) {
  // load fp32 from basic_ptr
  in_vset[i] = _mm512_loadu_ps(basic_ptr + 16 * i);
};
inline auto load_bf16_cast_fp32 = [](auto i,
                                     __m512i* bf16_vset,
                                     __m512* fp32_vset,
                                     auto* basic_ptr) {
  // load bf16 from basic ptr to bf16_vset, and then convert to fp32 in
  // fp32_vset
  bf16_vset[i] = _mm512_loadu_si512(basic_ptr + 32 * i);
  fp32_vset[i * 2] = _mm512_castsi512_ps(_mm512_slli_epi32(
      _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(bf16_vset[i], 0)), 16));
  fp32_vset[i * 2 + 1] = _mm512_castsi512_ps(_mm512_slli_epi32(
      _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(bf16_vset[i], 1)), 16));
};
inline auto load_fp16_cast_fp32 =
    [](auto i, __m512i* fp16_vset, __m512* fp32_vset, auto* basic_ptr) {
      // load bf16 from basic ptr to fp16_vset, and then convert to fp32 in
      // fp32_vset
      fp16_vset[i] = _mm512_loadu_si512(basic_ptr + 32 * i);
      fp32_vset[i * 2] =
          _mm512_cvtph_ps(_mm512_extracti32x8_epi32(fp16_vset[i], 0));
      fp32_vset[i * 2 + 1] =
          _mm512_cvtph_ps(_mm512_extracti32x8_epi32(fp16_vset[i], 1));
    };
inline auto load_lp = [](auto i, __m512i* lp_vset, auto* basic_ptr) {
  // load fp16/bf16 from basic_ptr
  lp_vset[i] = _mm512_loadu_si512(basic_ptr + 32 * i);
};

// store
inline auto store_fp32 = [](auto i, __m512* out_vset, auto* basic_ptr) {
  // store fp32 to basic_ptr
  _mm512_storeu_ps(basic_ptr + 16 * i, out_vset[i]);
};
inline auto store_2d_with_offset =
    [](auto i, __m512** out_vset, auto* basic_ptr, int32_t offset) {
      // store 2d[i, 4] out_vset according to basic_ptr and offset
      // TODO: Support arbitrary inner size than 4
      compile_time_for<4>::op(store_fp32, out_vset[i], basic_ptr + offset * i);
    };
inline auto cast_fp16_and_store =
    [](auto i, __m512* fp32_vset, auto* basic_ptr) {
      // cast out_vset from fp32 to fp16 and store to basic_ptr
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(basic_ptr + 16 * i),
          reinterpret_cast<__m256i>(_mm512_cvtps_ph(
              fp32_vset[i], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC))));
    };
inline auto store_lp = [](auto i, __m512i* lp_vset, auto* basic_ptr) {
  // store bf16/fp16 to basic_ptr
  _mm512_storeu_si512(
      reinterpret_cast<void*>(basic_ptr + 32 * i),
      reinterpret_cast<__m512i>(lp_vset[i]));
};

// add
inline auto add_fp32 = [](auto i, __m512* inout_vset, __m512* in_vset) {
  // inout_vset = inout_vset + in_vset
  inout_vset[i] = _mm512_add_ps(inout_vset[i], in_vset[i]);
};
inline auto add_fp32_const_b = [](auto i, __m512* inout_vset, __m512 b) {
  // inout_vset = inout_vset + b, b is not a set
  inout_vset[i] = _mm512_add_ps(inout_vset[i], b);
};

// mul
inline auto mul_fp32_constant_b = [](auto i, __m512* inout_vset, __m512 b) {
  // inout_vset = inout_vset * b, b is not a set
  inout_vset[i] = _mm512_mul_ps(inout_vset[i], b);
};

// div
inline auto div_fp32_constant_a =
    [](auto i, __m512* out_vset, __m512 a, __m512* b) {
      // out_vset = a / b, a is not a set
      out_vset[i] = _mm512_div_ps(a, b[i]);
    };

// sqrt
inline auto sqrt_fp32 = [](auto i, __m512* out_vset, __m512* in_vset) {
  // out_vset = sqrt(in_vset)
  out_vset[i] = _mm512_sqrt_ps(in_vset[i]);
};

// bf16 pack/split helper
inline auto pack_to_fp32 = [](auto i,
                              __m512i* top_vset,
                              __m512i* trail_vset,
                              __m512* fp32_vset) {
  // use high 16 bit in top_vset and low 16 bit in trail_vset to pack a fp32
  // vector and store in fp32_vset
  auto top0 =
      _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(__m512i(top_vset[i]), 0));
  auto top1 =
      _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(__m512i(top_vset[i]), 1));
  auto trail0 = _mm512_cvtepu16_epi32(
      _mm512_extracti32x8_epi32(__m512i(trail_vset[i]), 0));
  auto trail1 = _mm512_cvtepu16_epi32(
      _mm512_extracti32x8_epi32(__m512i(trail_vset[i]), 1));
  fp32_vset[2 * i] = _mm512_castsi512_ps(
      _mm512_add_epi32(_mm512_slli_epi32(top0, 16), trail0));
  fp32_vset[2 * i + 1] = _mm512_castsi512_ps(
      _mm512_add_epi32(_mm512_slli_epi32(top1, 16), trail1));
};
inline auto split_from_fp32 =
    [](auto i, __m512i* top_vset, __m512i* trail_vset, __m512* fp32_vset) {
      // split fp32_vset to high 16 bit (store in top_vset) and low 16 bit
      // (store in trail vset)
      __m512i x0 = _mm512_castps_si512(__m512(fp32_vset[2 * i]));
      __m512i x1 = _mm512_castps_si512(__m512(fp32_vset[2 * i + 1]));
      __m512i x0_hi = _mm512_srli_epi32(x0, 16);
      __m512i x1_hi = _mm512_srli_epi32(x1, 16);

      __m512i zeros = _mm512_set1_epi32(0xffff);
      __m512i x0_lo = _mm512_and_si512(x0, zeros);
      __m512i x1_lo = _mm512_and_si512(x1, zeros);

      __m512i idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
      top_vset[i] =
          _mm512_permutexvar_epi64(idx, _mm512_packus_epi32(x0_hi, x1_hi));
      trail_vset[i] =
          _mm512_permutexvar_epi64(idx, _mm512_packus_epi32(x0_lo, x1_lo));
    };

#endif

#if defined(CPU_CAPABILITY_AVX512_BF16)
// cast_and_store
inline auto cast_bf16_and_store =
    [](auto i, __m512* out_vset, auto* basic_ptr) {
      // cast out_vset from fp32 to bf16 and store to basic_ptr
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(basic_ptr + 16 * i),
          reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(out_vset[i])));
    };
inline auto cast_bf16_and_store_2d_with_offset =
    [](auto i, __m512** out_vset, auto* basic_ptr, int32_t offset) {
      // store 2d[i, 4] out_vset from fp32 to bf16
      // and store according to basic_ptr and offset
      // TODO: Support arbitrary inner size than 4
      compile_time_for<4>::op(
          cast_bf16_and_store, out_vset[i], basic_ptr + i * offset);
    };
#endif
} // namespace cpu
} // namespace torch_ipex
#endif