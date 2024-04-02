#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/quantized/Quantizer.h>
#include <aten/MergedEmbCat.h>
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>
#include "vec/vec.h"

namespace torch_ipex {
namespace cpu {

namespace {

inline void scalecopy_dense(
    const int64_t bs_bgein,
    const int64_t bs_end,
    const int64_t num_emb,
    int64_t emb_dim,
    const int8_t* dense,
    const double scale,
    int8_t* result) {
#if defined(CPU_CAPABILITY_AVX512_FP16)
  if (emb_dim == 128) {
    __m512h scale_v = (__m512h)_mm512_broadcast_f32x8((__m256)_mm512_cvtps_ph(
        _mm512_set1_ps(scale), _MM_FROUND_TO_NEAREST_INT));
    for (int64_t b = bs_bgein; b < bs_end; ++b) {
      __m512i x00, x64;
      __m512i y00, y32, y64, y96;
      __m512h h00, h32, h64, h96;
      x00 = _mm512_load_si512(dense);
      x64 = _mm512_load_si512(dense + 64);
      y00 = _mm512_cvtepi8_epi16(_mm512_extracti32x8_epi32(x00, 0));
      y32 = _mm512_cvtepi8_epi16(_mm512_extracti32x8_epi32(x00, 1));
      y64 = _mm512_cvtepi8_epi16(_mm512_extracti32x8_epi32(x64, 0));
      y96 = _mm512_cvtepi8_epi16(_mm512_extracti32x8_epi32(x64, 1));

      h00 = _mm512_cvt_roundepi16_ph(
          y00, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      h32 = _mm512_cvt_roundepi16_ph(
          y32, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      h64 = _mm512_cvt_roundepi16_ph(
          y64, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      h96 = _mm512_cvt_roundepi16_ph(
          y96, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      h00 = _mm512_mul_ph(h00, scale_v);
      h32 = _mm512_mul_ph(h32, scale_v);
      h64 = _mm512_mul_ph(h64, scale_v);
      h96 = _mm512_mul_ph(h96, scale_v);

      y00 = _mm512_cvt_roundph_epi16(
          h00, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      y32 = _mm512_cvt_roundph_epi16(
          h32, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      y64 = _mm512_cvt_roundph_epi16(
          h64, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      y96 = _mm512_cvt_roundph_epi16(
          h96, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      x00 = _mm512_inserti64x4(x00, _mm512_cvtsepi16_epi8(y00), 0);
      x00 = _mm512_inserti64x4(x00, _mm512_cvtsepi16_epi8(y32), 1);
      x64 = _mm512_inserti64x4(x64, _mm512_cvtsepi16_epi8(y64), 0);
      x64 = _mm512_inserti64x4(x64, _mm512_cvtsepi16_epi8(y96), 1);

      _mm512_store_si512(result, x00);
      _mm512_store_si512(result + 64, x64);
      result += (num_emb + 1) * emb_dim;
      dense += emb_dim;
    }
    return;
  }
#endif
#if defined(CPU_CAPABILITY_AVX512)
  if (emb_dim == 128) {
    __m512 scale_v = _mm512_set1_ps(scale);
    for (int64_t b = bs_bgein; b < bs_end; ++b) {
      __m512i x00, x64;
      __m512i y0, y1, y2, y3, y4, y5, y6, y7;
      __m512 f0, f1, f2, f3, f4, f5, f6, f7;
      x00 = _mm512_load_si512(dense);
      x64 = _mm512_load_si512(dense + 64);
      y0 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x00, 0));
      y1 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x00, 1));
      y2 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x00, 2));
      y3 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x00, 3));
      y4 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x64, 0));
      y5 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x64, 1));
      y6 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x64, 2));
      y7 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x64, 3));
      f0 = _mm512_cvt_roundepi32_ps(
          y0, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      f1 = _mm512_cvt_roundepi32_ps(
          y1, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      f2 = _mm512_cvt_roundepi32_ps(
          y2, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      f3 = _mm512_cvt_roundepi32_ps(
          y3, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      f4 = _mm512_cvt_roundepi32_ps(
          y4, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      f5 = _mm512_cvt_roundepi32_ps(
          y5, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      f6 = _mm512_cvt_roundepi32_ps(
          y6, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      f7 = _mm512_cvt_roundepi32_ps(
          y7, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      f0 = _mm512_mul_ps(f0, scale_v);
      f1 = _mm512_mul_ps(f1, scale_v);
      f2 = _mm512_mul_ps(f2, scale_v);
      f3 = _mm512_mul_ps(f3, scale_v);
      f4 = _mm512_mul_ps(f4, scale_v);
      f5 = _mm512_mul_ps(f5, scale_v);
      f6 = _mm512_mul_ps(f6, scale_v);
      f7 = _mm512_mul_ps(f7, scale_v);
      y0 = _mm512_cvt_roundps_epi32(
          f0, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      y1 = _mm512_cvt_roundps_epi32(
          f1, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      y2 = _mm512_cvt_roundps_epi32(
          f2, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      y3 = _mm512_cvt_roundps_epi32(
          f3, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      y4 = _mm512_cvt_roundps_epi32(
          f4, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      y5 = _mm512_cvt_roundps_epi32(
          f5, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      y6 = _mm512_cvt_roundps_epi32(
          f6, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      y7 = _mm512_cvt_roundps_epi32(
          f7, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      x00 = _mm512_inserti32x4(x00, _mm512_cvtsepi32_epi8(y0), 0);
      x00 = _mm512_inserti32x4(x00, _mm512_cvtsepi32_epi8(y1), 1);
      x00 = _mm512_inserti32x4(x00, _mm512_cvtsepi32_epi8(y2), 2);
      x00 = _mm512_inserti32x4(x00, _mm512_cvtsepi32_epi8(y3), 3);
      x64 = _mm512_inserti32x4(x64, _mm512_cvtsepi32_epi8(y4), 0);
      x64 = _mm512_inserti32x4(x64, _mm512_cvtsepi32_epi8(y5), 1);
      x64 = _mm512_inserti32x4(x64, _mm512_cvtsepi32_epi8(y6), 2);
      x64 = _mm512_inserti32x4(x64, _mm512_cvtsepi32_epi8(y7), 3);
      _mm512_store_si512(result, x00);
      _mm512_store_si512(result + 64, x64);
      result += (num_emb + 1) * emb_dim;
      dense += emb_dim;
    }
    return;
  }
#endif
  for (int64_t b = bs_bgein; b < bs_end; ++b) {
    for (int64_t d = 0; d < emb_dim; d++) {
      // TODO: Vec version for AVX2
      int32_t value = int32_t(dense[d]);
      value = kernel::_scale_int32(value, scale);
      result[d] = int8_t(value);
    }
    result += (num_emb + 1) * emb_dim;
    dense += emb_dim;
  }
}

template <typename index_t>
inline void qembeddingbag_kern(
    const int64_t bs_begin,
    const int64_t bs_end,
    const int64_t num_emb,
    const int64_t emb_dim,
    const index_t last_offset,
    const index_t* indices,
    const index_t* offsets,
    const int8_t* weight,
    const double scale,
    int8_t* result) {
#if defined(CPU_CAPABILITY_AVX512_FP16)
  if (emb_dim == 128) {
    __m512h scale_v = (__m512h)_mm512_broadcast_f32x8((__m256)_mm512_cvtps_ph(
        _mm512_set1_ps(scale), _MM_FROUND_TO_NEAREST_INT));
    for (int64_t b = bs_begin; b < bs_end; ++b) {
      __m512i x00, x64;
      __m512i y00, y32, y64, y96;
      __m512h h00, h32, h64, h96;
      int64_t start_idx = offsets[b];
      int64_t end_idx = ((b + 1) == bs_end && last_offset != -1)
          ? last_offset
          : offsets[b + 1];
      int64_t idx = indices[start_idx] * emb_dim;
      x00 = _mm512_load_si512(&weight[idx]);
      x64 = _mm512_load_si512(&weight[idx + 64]);
      y00 = _mm512_cvtepi8_epi16(_mm512_extracti32x8_epi32(x00, 0));
      y32 = _mm512_cvtepi8_epi16(_mm512_extracti32x8_epi32(x00, 1));
      y64 = _mm512_cvtepi8_epi16(_mm512_extracti32x8_epi32(x64, 0));
      y96 = _mm512_cvtepi8_epi16(_mm512_extracti32x8_epi32(x64, 1));
      for (int64_t j = start_idx + 1; j < end_idx; ++j) {
        idx = indices[j] * emb_dim;
        x00 = _mm512_load_si512(&weight[idx]);
        x64 = _mm512_load_si512(&weight[idx + 64]);
        y00 = _mm512_adds_epi16(
            y00, _mm512_cvtepi8_epi16(_mm512_extracti32x8_epi32(x00, 0)));
        y32 = _mm512_adds_epi16(
            y32, _mm512_cvtepi8_epi16(_mm512_extracti32x8_epi32(x00, 1)));
        y64 = _mm512_adds_epi16(
            y64, _mm512_cvtepi8_epi16(_mm512_extracti32x8_epi32(x64, 0)));
        y96 = _mm512_adds_epi16(
            y96, _mm512_cvtepi8_epi16(_mm512_extracti32x8_epi32(x64, 1)));
      }
      h00 = _mm512_cvt_roundepi16_ph(
          y00, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      h32 = _mm512_cvt_roundepi16_ph(
          y32, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      h64 = _mm512_cvt_roundepi16_ph(
          y64, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      h96 = _mm512_cvt_roundepi16_ph(
          y96, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      h00 = _mm512_mul_ph(h00, scale_v);
      h32 = _mm512_mul_ph(h32, scale_v);
      h64 = _mm512_mul_ph(h64, scale_v);
      h96 = _mm512_mul_ph(h96, scale_v);
      y00 = _mm512_cvt_roundph_epi16(
          h00, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      y32 = _mm512_cvt_roundph_epi16(
          h32, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      y64 = _mm512_cvt_roundph_epi16(
          h64, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      y96 = _mm512_cvt_roundph_epi16(
          h96, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      x00 = _mm512_inserti64x4(x00, _mm512_cvtsepi16_epi8(y00), 0);
      x00 = _mm512_inserti64x4(x00, _mm512_cvtsepi16_epi8(y32), 1);
      x64 = _mm512_inserti64x4(x64, _mm512_cvtsepi16_epi8(y64), 0);
      x64 = _mm512_inserti64x4(x64, _mm512_cvtsepi16_epi8(y96), 1);
      _mm512_store_si512(result, x00);
      _mm512_store_si512(result + 64, x64);
      result += (num_emb + 1) * emb_dim;
    }
    return;
  }
#endif
#if defined(CPU_CAPABILITY_AVX512)
  if (emb_dim == 128) {
    __m512 scale_v = _mm512_set1_ps(scale);
    for (int64_t b = bs_begin; b < bs_end; ++b) {
      __m512i x00, x64;
      __m512i y0, y1, y2, y3, y4, y5, y6, y7;
      __m512 f0, f1, f2, f3, f4, f5, f6, f7;
      int64_t start_idx = offsets[b];
      int64_t end_idx = ((b + 1) == bs_end && last_offset != -1)
          ? last_offset
          : offsets[b + 1];
      // load first indices
      int64_t idx = indices[start_idx] * emb_dim;
      x00 = _mm512_load_si512(&weight[idx]);
      x64 = _mm512_load_si512(&weight[idx + 64]);
      y0 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x00, 0));
      y1 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x00, 1));
      y2 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x00, 2));
      y3 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x00, 3));
      y4 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x64, 0));
      y5 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x64, 1));
      y6 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x64, 2));
      y7 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x64, 3));
      for (int64_t j = start_idx + 1; j < end_idx; ++j) {
        idx = indices[j] * emb_dim;
        x00 = _mm512_load_si512(&weight[idx]);
        x64 = _mm512_load_si512(&weight[idx + 64]);
        y0 = _mm512_add_epi32(
            y0, _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x00, 0)));
        y1 = _mm512_add_epi32(
            y1, _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x00, 1)));
        y2 = _mm512_add_epi32(
            y2, _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x00, 2)));
        y3 = _mm512_add_epi32(
            y3, _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x00, 3)));
        y4 = _mm512_add_epi32(
            y4, _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x64, 0)));
        y5 = _mm512_add_epi32(
            y5, _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x64, 1)));
        y6 = _mm512_add_epi32(
            y6, _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x64, 2)));
        y7 = _mm512_add_epi32(
            y7, _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(x64, 3)));
      }
      f0 = _mm512_cvt_roundepi32_ps(
          y0, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      f1 = _mm512_cvt_roundepi32_ps(
          y1, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      f2 = _mm512_cvt_roundepi32_ps(
          y2, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      f3 = _mm512_cvt_roundepi32_ps(
          y3, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      f4 = _mm512_cvt_roundepi32_ps(
          y4, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      f5 = _mm512_cvt_roundepi32_ps(
          y5, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      f6 = _mm512_cvt_roundepi32_ps(
          y6, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      f7 = _mm512_cvt_roundepi32_ps(
          y7, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      f0 = _mm512_mul_ps(f0, scale_v);
      f1 = _mm512_mul_ps(f1, scale_v);
      f2 = _mm512_mul_ps(f2, scale_v);
      f3 = _mm512_mul_ps(f3, scale_v);
      f4 = _mm512_mul_ps(f4, scale_v);
      f5 = _mm512_mul_ps(f5, scale_v);
      f6 = _mm512_mul_ps(f6, scale_v);
      f7 = _mm512_mul_ps(f7, scale_v);
      y0 = _mm512_cvt_roundps_epi32(
          f0, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      y1 = _mm512_cvt_roundps_epi32(
          f1, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      y2 = _mm512_cvt_roundps_epi32(
          f2, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      y3 = _mm512_cvt_roundps_epi32(
          f3, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      y4 = _mm512_cvt_roundps_epi32(
          f4, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      y5 = _mm512_cvt_roundps_epi32(
          f5, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      y6 = _mm512_cvt_roundps_epi32(
          f6, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      y7 = _mm512_cvt_roundps_epi32(
          f7, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      x00 = _mm512_inserti32x4(x00, _mm512_cvtsepi32_epi8(y0), 0);
      x00 = _mm512_inserti32x4(x00, _mm512_cvtsepi32_epi8(y1), 1);
      x00 = _mm512_inserti32x4(x00, _mm512_cvtsepi32_epi8(y2), 2);
      x00 = _mm512_inserti32x4(x00, _mm512_cvtsepi32_epi8(y3), 3);
      x64 = _mm512_inserti32x4(x64, _mm512_cvtsepi32_epi8(y4), 0);
      x64 = _mm512_inserti32x4(x64, _mm512_cvtsepi32_epi8(y5), 1);
      x64 = _mm512_inserti32x4(x64, _mm512_cvtsepi32_epi8(y6), 2);
      x64 = _mm512_inserti32x4(x64, _mm512_cvtsepi32_epi8(y7), 3);
      _mm512_store_si512(result, x00);
      _mm512_store_si512(result + 64, x64);
      result += (num_emb + 1) * emb_dim;
    }
    return;
  }
#endif
  for (int64_t b = bs_begin; b < bs_end; ++b) {
    int64_t start_idx = offsets[b];
    int64_t end_idx =
        ((b + 1) == bs_end && last_offset != -1) ? last_offset : offsets[b + 1];
    for (int32_t d = 0; d < emb_dim; d++) {
      int64_t idx = indices[start_idx] * emb_dim;
      int32_t value = int32_t(weight[idx + d]);
      for (int64_t j = start_idx + 1; j < end_idx; ++j) {
        idx = indices[j] * emb_dim;
        value += int32_t(weight[idx + d]);
      }
      value = kernel::_scale_int32(int32_t(value), scale);
      result[d] = int8_t(value);
    }
    result += (num_emb + 1) * emb_dim;
  }
}

template <typename index_t>
void qembeddingbagcat(
    int8_t* o_ptr,
    int8_t** w_ptr,
    index_t** indices_ptr,
    index_t** offsets_ptr,
    int8_t* d_ptr,
    int64_t num_batch,
    int64_t num_emb,
    int64_t emb_dim,
    std::vector<int64_t> last_offsets,
    std::vector<double> w_scale,
    double d_scale,
    double o_scale) {
  constexpr int64_t b_block = 512;
  const int64_t n_b_blocks = (num_batch - 1) / b_block + 1;
  const double copy_scale = d_scale / o_scale;
  for (double& w_sca : w_scale) {
    w_sca = w_sca / o_scale;
  }
#pragma omp parallel for collapse(2)
  for (int64_t b = 0; b < n_b_blocks; ++b) {
    for (int64_t n = 0; n < (num_emb + 1); ++n) {
      const int64_t bs_begin = b * b_block;
      const int64_t bs_end = std::min(num_batch, (b + 1) * b_block);
      int8_t* r = &o_ptr[b * b_block * (num_emb + 1) * emb_dim + n * emb_dim];
      if (n == 0) {
        scalecopy_dense(
            bs_begin,
            bs_end,
            num_emb,
            emb_dim,
            &d_ptr[b * b_block * emb_dim],
            copy_scale,
            r);
      } else {
        const int64_t m = n - 1;
        // avoid offsets not include last batch
        const index_t last_offset = bs_end == num_batch ? last_offsets[m] : -1;
        qembeddingbag_kern(
            bs_begin,
            bs_end,
            num_emb,
            emb_dim,
            last_offset,
            indices_ptr[m],
            offsets_ptr[m],
            w_ptr[m],
            w_scale[m],
            r);
      }
    }
  }
}

Tensor qmerged_embedding_cat_fw_impl(
    const TensorList& qweights,
    const TensorList& indices,
    const TensorList& offsets,
    const Tensor& qdense,
    double o_scale) {
  RECORD_FUNCTION(__FUNCTION__, c10::ArrayRef<c10::IValue>({}));
  int64_t batch_size = qdense.size(0);
  int64_t emb_dim = qdense.size(1);
  int64_t num_emb = qweights.size();

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(num_emb > 0);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(num_emb == indices.size());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(num_emb == offsets.size());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(qdense.dim() == 2 && qdense.is_contiguous());

  auto index_type = indices[0].scalar_type();
  auto int8_type = qdense.scalar_type();

  std::vector<int64_t> last_offsets(num_emb, -1);
  std::vector<double> w_scale(num_emb, -1);

  for (int i = 0; i < num_emb; i++) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        indices[i].is_contiguous() && indices[i].scalar_type() == index_type);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        offsets[i].is_contiguous() && offsets[i].scalar_type() == index_type);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        qweights[i].is_contiguous() && qweights[i].scalar_type() == int8_type);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        qweights[i].dim() == 2 && qweights[i].size(1) == emb_dim);
    // handle last offsets
    last_offsets[i] = indices[i].numel();
    w_scale[i] = native::q_scale_quant(qweights[i]);
  }

  double dense_scale = native::q_scale_quant(qdense);
  QuantizerPtr output_quantizer =
      make_per_tensor_affine_quantizer(o_scale, /*zp=*/0, kQInt8);
  Tensor output = new_qtensor(
      /*sizes=*/{batch_size, (num_emb + 1) * emb_dim},
      qweights[0].options(),
      output_quantizer);
  AT_DISPATCH_INDEX_TYPES(indices[0].scalar_type(), "embeddingbag_cat", [&] {
    int8_t* dense_ptr = qdense.data_ptr<int8_t>();
    int8_t* qweights_ptr[num_emb];
    index_t* indices_ptr[num_emb];
    index_t* offsets_ptr[num_emb];
    for (int i = 0; i < num_emb; i++) {
      qweights_ptr[i] = qweights[i].data_ptr<int8_t>();
      indices_ptr[i] = indices[i].data_ptr<index_t>();
      offsets_ptr[i] = offsets[i].data_ptr<index_t>();
    }
    int8_t* output_ptr = output.data_ptr<int8_t>();
    qembeddingbagcat<index_t>(
        output_ptr,
        qweights_ptr,
        indices_ptr,
        offsets_ptr,
        dense_ptr,
        batch_size,
        num_emb,
        emb_dim,
        last_offsets,
        w_scale,
        dense_scale,
        o_scale);
  });
  return output;
}

} // anonymous namespace

IPEX_REGISTER_DISPATCH(
    qmerged_embeddingbag_cat_fw_stub,
    &qmerged_embedding_cat_fw_impl);

} // namespace cpu
} // namespace torch_ipex