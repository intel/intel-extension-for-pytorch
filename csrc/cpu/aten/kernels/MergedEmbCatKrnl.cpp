#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <aten/MergedEmbCat.h>
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>

namespace torch_ipex {
namespace cpu {

namespace {

template <typename data_t>
inline void copy_dense(
    const int64_t bs_bgein,
    const int64_t bs_end,
    const int64_t num_emb,
    int64_t emb_dim,
    const data_t* dense,
    data_t* result) {
  for (int64_t b = bs_bgein; b < bs_end; ++b) {
    memcpy(result, dense, emb_dim * sizeof(data_t));
    result += (num_emb + 1) * emb_dim;
    dense += emb_dim;
  }
}

template <typename data_t, typename index_t>
typename std::enable_if<
    std::is_same<data_t, float>::value || std::is_same<data_t, double>::value,
    void>::
    type inline embeddingbag_kern_general(
        const int64_t bs_begin,
        const int64_t bs_end,
        const int64_t num_emb,
        const int64_t emb_dim,
        const index_t last_offset,
        const index_t* indices,
        const index_t* offsets,
        const data_t* weight,
        data_t* result) {
  using Vec = at::vec::Vectorized<data_t>;
  auto vec_size = Vec::size();
  for (int64_t b = bs_begin; b < bs_end; ++b) {
    int64_t start_idx = offsets[b];
    int64_t end_idx =
        ((b + 1) == bs_end && last_offset != -1) ? last_offset : offsets[b + 1];
    // vec
    Vec w_vec;
    int64_t i = 0;
    for (; i + vec_size <= emb_dim; i += vec_size) {
      int64_t idx = indices[start_idx] * emb_dim;
      w_vec = Vec::loadu(&weight[idx + i]);
      for (int64_t j = start_idx + 1; j < end_idx; ++j) {
        idx = indices[j] * emb_dim;
        Vec w_next_vec = Vec::loadu(&weight[idx + i]);
        w_vec += w_next_vec;
      }
      w_vec.store(result + i);
    }
    // scalar tail
    data_t w;
    for (; i < emb_dim; i++) {
      int64_t idx = indices[start_idx] * emb_dim;
      w = weight[idx + i];
      for (int64_t j = start_idx + 1; j < end_idx; ++j) {
        idx = indices[j] * emb_dim;
        data_t w_next = weight[idx + i];
        w += w_next;
      }
      result[i] = w;
    }
    result += (num_emb + 1) * emb_dim;
  }
}

template <typename data_t, typename index_t>
typename std::enable_if<
    std::is_same<data_t, Half>::value || std::is_same<data_t, BFloat16>::value,
    void>::
    type inline embeddingbag_kern_general(
        const int64_t bs_begin,
        const int64_t bs_end,
        const int64_t num_emb,
        const int64_t emb_dim,
        const index_t last_offset,
        const index_t* indices,
        const index_t* offsets,
        const data_t* weight,
        data_t* result) {
  using lpVec = at::vec::Vectorized<data_t>;
  using fVec = at::vec::Vectorized<float>;
  auto vec_size = lpVec::size();
  for (int64_t b = bs_begin; b < bs_end; ++b) {
    int64_t start_idx = offsets[b];
    int64_t end_idx =
        ((b + 1) == bs_end && last_offset != -1) ? last_offset : offsets[b + 1];
    // vec
    fVec f_w_vec1, f_w_vec2;
    int64_t i = 0;
    for (; i + vec_size <= emb_dim; i += vec_size) {
      int64_t idx = indices[start_idx] * emb_dim;
      lpVec lp_w_vec = lpVec::loadu(&weight[idx + i]);
      std::tie(f_w_vec1, f_w_vec2) =
          at::vec::convert_to_float<data_t>(lp_w_vec);
      for (int64_t j = start_idx + 1; j < end_idx; ++j) {
        idx = indices[j] * emb_dim;
        lpVec lp_w_next_vec = lpVec::loadu(&weight[idx + i]);
        fVec f_w_next_vec1, f_w_next_vec2;
        std::tie(f_w_next_vec1, f_w_next_vec2) =
            at::vec::convert_to_float<data_t>(lp_w_next_vec);
        f_w_vec1 += f_w_next_vec1;
        f_w_vec2 += f_w_next_vec2;
      }
      lp_w_vec = at::vec::convert_from_float<data_t>(f_w_vec1, f_w_vec2);
      lp_w_vec.store(result + i);
    }
    // scalar tail
    float w;
    for (; i < emb_dim; i++) {
      int64_t idx = indices[start_idx] * emb_dim;
      w = float(weight[idx + i]);
      for (int64_t j = start_idx + 1; j < end_idx; ++j) {
        idx = indices[j] * emb_dim;
        float w_next = float(weight[idx + i]);
        w += w_next;
      }
      result[i] = data_t(w);
    }
    result += (num_emb + 1) * emb_dim;
  }
}

template <typename data_t, typename index_t>
typename std::enable_if<std::is_same<data_t, float>::value, void>::
    type inline embeddingbag_kern(
        const int64_t bs_begin,
        const int64_t bs_end,
        const int64_t num_emb,
        const int64_t emb_dim,
        const index_t last_offset,
        const index_t* indices,
        const index_t* offsets,
        const data_t* weight,
        data_t* result) {
#if defined(CPU_CAPABILITY_AVX512)
  // FP32 avx512 fast path for emb_dim=128
  // ~7% improvement while benchmarking on SPR 56C/S with 1 S.
  // benchmark config: num_emb=26, emb_dim=128, batch_size=32768
  // num_bags = [3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1] for
  // each table
  if (emb_dim == 128) {
    for (int64_t b = bs_begin; b < bs_end; ++b) {
      __m512 x0, x1, x2, x3, x4, x5, x6, x7;
      int64_t start_idx = offsets[b];
      int64_t end_idx = ((b + 1) == bs_end && last_offset != -1)
          ? last_offset
          : offsets[b + 1];
      // load first indices
      int64_t idx = indices[start_idx] * emb_dim;
      x0 = _mm512_load_ps(&weight[idx]);
      x1 = _mm512_load_ps(&weight[idx + 16]);
      x2 = _mm512_load_ps(&weight[idx + 32]);
      x3 = _mm512_load_ps(&weight[idx + 48]);
      x4 = _mm512_load_ps(&weight[idx + 64]);
      x5 = _mm512_load_ps(&weight[idx + 80]);
      x6 = _mm512_load_ps(&weight[idx + 96]);
      x7 = _mm512_load_ps(&weight[idx + 112]);
      for (int64_t j = start_idx + 1; j < end_idx; ++j) {
        // add following idx
        idx = indices[j] * emb_dim;
        x0 = _mm512_add_ps(x0, _mm512_load_ps(&weight[idx]));
        x1 = _mm512_add_ps(x1, _mm512_load_ps(&weight[idx + 16]));
        x2 = _mm512_add_ps(x2, _mm512_load_ps(&weight[idx + 32]));
        x3 = _mm512_add_ps(x3, _mm512_load_ps(&weight[idx + 48]));
        x4 = _mm512_add_ps(x4, _mm512_load_ps(&weight[idx + 64]));
        x5 = _mm512_add_ps(x5, _mm512_load_ps(&weight[idx + 80]));
        x6 = _mm512_add_ps(x6, _mm512_load_ps(&weight[idx + 96]));
        x7 = _mm512_add_ps(x7, _mm512_load_ps(&weight[idx + 112]));
      }
      // store
      _mm512_store_ps(result, x0);
      _mm512_store_ps(result + 16, x1);
      _mm512_store_ps(result + 32, x2);
      _mm512_store_ps(result + 48, x3);
      _mm512_store_ps(result + 64, x4);
      _mm512_store_ps(result + 80, x5);
      _mm512_store_ps(result + 96, x6);
      _mm512_store_ps(result + 112, x7);
      result += (num_emb + 1) * emb_dim;
    }
    return;
  }
#endif
  embeddingbag_kern_general(
      bs_begin,
      bs_end,
      num_emb,
      emb_dim,
      last_offset,
      indices,
      offsets,
      weight,
      result);
}

template <typename data_t, typename index_t>
typename std::enable_if<std::is_same<data_t, Half>::value, void>::
    type inline embeddingbag_kern(
        const int64_t bs_begin,
        const int64_t bs_end,
        const int64_t num_emb,
        const int64_t emb_dim,
        const index_t last_offset,
        const index_t* indices,
        const index_t* offsets,
        const data_t* weight,
        data_t* result) {
#if defined(CPU_CAPABILITY_AVX512_FP16)
  // FP16 avx512_fp16 fast path for emb_dim=128
  // only ~1.5% improvement while benchmarking on SPR 56C/S with 1 S.
  // benchmark config: num_emb=26, emb_dim=128, batch_size=32768
  // num_bags = [3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1] for
  // each table
  if (emb_dim == 128) {
    for (int64_t b = bs_begin; b < bs_end; ++b) {
      __m256i lo, hi;
      __m512i x00, x32, x64, x96;
      __m512 y00, y16, y32, y48, y64, y80, y96, y112;
      int64_t start_idx = offsets[b];
      int64_t end_idx = ((b + 1) == bs_end && last_offset != -1)
          ? last_offset
          : offsets[b + 1];
      // load first indices
      int64_t idx = indices[start_idx] * emb_dim;
      x00 = _mm512_loadu_si512(&weight[idx]);
      x32 = _mm512_loadu_si512(&weight[idx + 32]);
      x64 = _mm512_loadu_si512(&weight[idx + 64]);
      x96 = _mm512_loadu_si512(&weight[idx + 96]);
      lo = _mm512_extracti32x8_epi32(x00, 0);
      hi = _mm512_extracti32x8_epi32(x00, 1);
      y00 = _mm512_cvtph_ps(lo);
      y16 = _mm512_cvtph_ps(hi);
      lo = _mm512_extracti32x8_epi32(x32, 0);
      hi = _mm512_extracti32x8_epi32(x32, 1);
      y32 = _mm512_cvtph_ps(lo);
      y48 = _mm512_cvtph_ps(hi);
      lo = _mm512_extracti32x8_epi32(x64, 0);
      hi = _mm512_extracti32x8_epi32(x64, 1);
      y64 = _mm512_cvtph_ps(lo);
      y80 = _mm512_cvtph_ps(hi);
      lo = _mm512_extracti32x8_epi32(x96, 0);
      hi = _mm512_extracti32x8_epi32(x96, 1);
      y96 = _mm512_cvtph_ps(lo);
      y112 = _mm512_cvtph_ps(hi);
      for (int64_t j = start_idx + 1; j < end_idx; ++j) {
        // add following idx
        idx = indices[j] * emb_dim;
        x00 = _mm512_loadu_si512(&weight[idx]);
        x32 = _mm512_loadu_si512(&weight[idx + 32]);
        x64 = _mm512_loadu_si512(&weight[idx + 64]);
        x96 = _mm512_loadu_si512(&weight[idx + 96]);
        lo = _mm512_extracti32x8_epi32(x00, 0);
        hi = _mm512_extracti32x8_epi32(x00, 1);
        y00 = _mm512_add_ps(y00, _mm512_cvtph_ps(lo));
        y16 = _mm512_add_ps(y16, _mm512_cvtph_ps(hi));
        lo = _mm512_extracti32x8_epi32(x32, 0);
        hi = _mm512_extracti32x8_epi32(x32, 1);
        y32 = _mm512_add_ps(y32, _mm512_cvtph_ps(lo));
        y48 = _mm512_add_ps(y48, _mm512_cvtph_ps(hi));
        lo = _mm512_extracti32x8_epi32(x64, 0);
        hi = _mm512_extracti32x8_epi32(x64, 1);
        y64 = _mm512_add_ps(y64, _mm512_cvtph_ps(lo));
        y80 = _mm512_add_ps(y80, _mm512_cvtph_ps(hi));
        lo = _mm512_extracti32x8_epi32(x96, 0);
        hi = _mm512_extracti32x8_epi32(x96, 1);
        y96 = _mm512_add_ps(y96, _mm512_cvtph_ps(lo));
        y112 = _mm512_add_ps(y112, _mm512_cvtph_ps(hi));
      }
      // store
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(result),
          reinterpret_cast<__m256i>(_mm512_cvtps_ph(
              y00, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC))));
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(result + 16),
          reinterpret_cast<__m256i>(_mm512_cvtps_ph(
              y16, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC))));
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(result + 32),
          reinterpret_cast<__m256i>(_mm512_cvtps_ph(
              y32, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC))));
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(result + 48),
          reinterpret_cast<__m256i>(_mm512_cvtps_ph(
              y48, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC))));
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(result + 64),
          reinterpret_cast<__m256i>(_mm512_cvtps_ph(
              y64, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC))));
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(result + 80),
          reinterpret_cast<__m256i>(_mm512_cvtps_ph(
              y80, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC))));
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(result + 96),
          reinterpret_cast<__m256i>(_mm512_cvtps_ph(
              y96, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC))));
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(result + 112),
          reinterpret_cast<__m256i>(_mm512_cvtps_ph(
              y112, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC))));
      result += (num_emb + 1) * emb_dim;
    }
    return;
  }
#endif
  embeddingbag_kern_general(
      bs_begin,
      bs_end,
      num_emb,
      emb_dim,
      last_offset,
      indices,
      offsets,
      weight,
      result);
}

template <typename data_t, typename index_t>
typename std::enable_if<std::is_same<data_t, BFloat16>::value, void>::
    type inline embeddingbag_kern(
        const int64_t bs_begin,
        const int64_t bs_end,
        const int64_t num_emb,
        const int64_t emb_dim,
        const index_t last_offset,
        const index_t* indices,
        const index_t* offsets,
        const data_t* weight,
        data_t* result) {
#if defined(CPU_CAPABILITY_AVX512_BF16)
  // BF16 avx512_bf16 fast path for emb_dim=128
  // ~30% improvement while benchmarking on SPR 56C/S with 1 S.
  // benchmark config: num_emb=26, emb_dim=128, batch_size=32768
  // num_bags = [3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1] for
  // each table
  if (emb_dim == 128) {
    for (int64_t b = bs_begin; b < bs_end; ++b) {
      __m256i lo, hi;
      __m512i x00, x32, x64, x96;
      __m512 y00, y16, y32, y48, y64, y80, y96, y112;
      int64_t start_idx = offsets[b];
      int64_t end_idx = ((b + 1) == bs_end && last_offset != -1)
          ? last_offset
          : offsets[b + 1];
      // load first indices
      int64_t idx = indices[start_idx] * emb_dim;
      x00 = _mm512_loadu_si512(&weight[idx]);
      x32 = _mm512_loadu_si512(&weight[idx + 32]);
      x64 = _mm512_loadu_si512(&weight[idx + 64]);
      x96 = _mm512_loadu_si512(&weight[idx + 96]);
      lo = _mm512_extracti32x8_epi32(x00, 0);
      hi = _mm512_extracti32x8_epi32(x00, 1);
      y00 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(lo), 16));
      y16 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(hi), 16));
      lo = _mm512_extracti32x8_epi32(x32, 0);
      hi = _mm512_extracti32x8_epi32(x32, 1);
      y32 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(lo), 16));
      y48 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(hi), 16));
      lo = _mm512_extracti32x8_epi32(x64, 0);
      hi = _mm512_extracti32x8_epi32(x64, 1);
      y64 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(lo), 16));
      y80 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(hi), 16));
      lo = _mm512_extracti32x8_epi32(x96, 0);
      hi = _mm512_extracti32x8_epi32(x96, 1);
      y96 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(lo), 16));
      y112 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(hi), 16));
      for (int64_t j = start_idx + 1; j < end_idx; ++j) {
        // add following idx
        idx = indices[j] * emb_dim;
        x00 = _mm512_loadu_si512(&weight[idx]);
        x32 = _mm512_loadu_si512(&weight[idx + 32]);
        x64 = _mm512_loadu_si512(&weight[idx + 64]);
        x96 = _mm512_loadu_si512(&weight[idx + 96]);
        lo = _mm512_extracti32x8_epi32(x00, 0);
        hi = _mm512_extracti32x8_epi32(x00, 1);
        y00 = _mm512_add_ps(
            y00,
            _mm512_castsi512_ps(
                _mm512_slli_epi32(_mm512_cvtepu16_epi32(lo), 16)));
        y16 = _mm512_add_ps(
            y16,
            _mm512_castsi512_ps(
                _mm512_slli_epi32(_mm512_cvtepu16_epi32(hi), 16)));
        lo = _mm512_extracti32x8_epi32(x32, 0);
        hi = _mm512_extracti32x8_epi32(x32, 1);
        y32 = _mm512_add_ps(
            y32,
            _mm512_castsi512_ps(
                _mm512_slli_epi32(_mm512_cvtepu16_epi32(lo), 16)));
        y48 = _mm512_add_ps(
            y48,
            _mm512_castsi512_ps(
                _mm512_slli_epi32(_mm512_cvtepu16_epi32(hi), 16)));
        lo = _mm512_extracti32x8_epi32(x64, 0);
        hi = _mm512_extracti32x8_epi32(x64, 1);
        y64 = _mm512_add_ps(
            y64,
            _mm512_castsi512_ps(
                _mm512_slli_epi32(_mm512_cvtepu16_epi32(lo), 16)));
        y80 = _mm512_add_ps(
            y80,
            _mm512_castsi512_ps(
                _mm512_slli_epi32(_mm512_cvtepu16_epi32(hi), 16)));
        lo = _mm512_extracti32x8_epi32(x96, 0);
        hi = _mm512_extracti32x8_epi32(x96, 1);
        y96 = _mm512_add_ps(
            y96,
            _mm512_castsi512_ps(
                _mm512_slli_epi32(_mm512_cvtepu16_epi32(lo), 16)));
        y112 = _mm512_add_ps(
            y112,
            _mm512_castsi512_ps(
                _mm512_slli_epi32(_mm512_cvtepu16_epi32(hi), 16)));
      }
      // store
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(result),
          reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(y00)));
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(result + 16),
          reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(y16)));
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(result + 32),
          reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(y32)));
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(result + 48),
          reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(y48)));
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(result + 64),
          reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(y64)));
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(result + 80),
          reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(y80)));
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(result + 96),
          reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(y96)));
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(result + 112),
          reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(y112)));
      result += (num_emb + 1) * emb_dim;
    }
    return;
  }
#endif
  embeddingbag_kern_general(
      bs_begin,
      bs_end,
      num_emb,
      emb_dim,
      last_offset,
      indices,
      offsets,
      weight,
      result);
}

template <typename data_t, typename index_t>
typename std::enable_if<std::is_same<data_t, double>::value, void>::
    type inline embeddingbag_kern(
        const int64_t bs_begin,
        const int64_t bs_end,
        const int64_t num_emb,
        const int64_t emb_dim,
        const index_t last_offset,
        const index_t* indices,
        const index_t* offsets,
        const data_t* weight,
        data_t* result) {
  embeddingbag_kern_general(
      bs_begin,
      bs_end,
      num_emb,
      emb_dim,
      last_offset,
      indices,
      offsets,
      weight,
      result);
}

template <typename data_t, typename index_t>
void embeddingbagcat(
    data_t* o_ptr,
    data_t** w_ptr,
    index_t** indices_ptr,
    index_t** offsets_ptr,
    data_t* d_ptr,
    int64_t num_batch,
    int64_t num_emb,
    int64_t emb_dim,
    std::vector<int64_t> last_offsets) {
  constexpr int64_t b_block = 128;
  const int64_t n_b_blocks = (num_batch - 1) / b_block + 1;
#pragma omp parallel for collapse(2)
  for (int64_t b = 0; b < n_b_blocks; ++b) {
    for (int64_t n = 0; n < (num_emb + 1); ++n) {
      const int64_t bs_begin = b * b_block;
      const int64_t bs_end = std::min(num_batch, (b + 1) * b_block);
      data_t* r = &o_ptr[b * b_block * (num_emb + 1) * emb_dim + n * emb_dim];
      if (n == 0) {
        copy_dense(
            bs_begin,
            bs_end,
            num_emb,
            emb_dim,
            &d_ptr[b * b_block * emb_dim],
            r);
      } else {
        const int64_t m = n - 1;
        // avoid offsets not include last batch
        const index_t last_offset = bs_end == num_batch ? last_offsets[m] : -1;
        embeddingbag_kern(
            bs_begin,
            bs_end,
            num_emb,
            emb_dim,
            last_offset,
            indices_ptr[m],
            offsets_ptr[m],
            w_ptr[m],
            r);
      }
    }
  }
}

Tensor merged_embedding_cat_fw_impl(
    const TensorList& weights,
    const TensorList& indices,
    const TensorList& offsets,
    const Tensor& dense) {
  int64_t batch_size = dense.size(0);
  int64_t emb_dim = dense.size(1);
  int64_t num_emb = weights.size();

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(num_emb > 0);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(num_emb == indices.size());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(num_emb == offsets.size());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dense.dim() == 2 && dense.is_contiguous());

  auto index_type = indices[0].scalar_type();
  auto data_type = dense.scalar_type();

  std::vector<int64_t> last_offsets(num_emb, -1);

  for (int i = 0; i < num_emb; i++) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        indices[i].is_contiguous() && indices[i].scalar_type() == index_type);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        offsets[i].is_contiguous() && offsets[i].scalar_type() == index_type);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        weights[i].is_contiguous() && weights[i].scalar_type() == data_type);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        weights[i].dim() == 2 && weights[i].size(1) == emb_dim);
    // handle last offsets
    last_offsets[i] = indices[i].numel();
  }

  Tensor output = zeros({batch_size, (num_emb + 1) * emb_dim}, dense.options());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, dense.scalar_type(), "embeddingbag_cat", [&] {
        AT_DISPATCH_INDEX_TYPES(
            indices[0].scalar_type(), "embeddingbag_cat", [&] {
              scalar_t* dense_ptr = dense.data_ptr<scalar_t>();
              scalar_t* weights_ptr[num_emb];
              index_t* indices_ptr[num_emb];
              index_t* offsets_ptr[num_emb];
              for (int i = 0; i < num_emb; i++) {
                weights_ptr[i] = weights[i].data_ptr<scalar_t>();
                indices_ptr[i] = indices[i].data_ptr<index_t>();
                offsets_ptr[i] = offsets[i].data_ptr<index_t>();
              }
              scalar_t* output_ptr = output.data_ptr<scalar_t>();
              embeddingbagcat<scalar_t, index_t>(
                  output_ptr,
                  weights_ptr,
                  indices_ptr,
                  offsets_ptr,
                  dense_ptr,
                  batch_size,
                  num_emb,
                  emb_dim,
                  last_offsets);
            });
      });
  return output;
}

} // anonymous namespace

REGISTER_DISPATCH(
    merged_embeddingbag_cat_fw_stub,
    &merged_embedding_cat_fw_impl);

} // namespace cpu
} // namespace torch_ipex