#include <ATen/AccumulateType.h>
#include <ATen/Tensor.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <aten/MergedEmbCat.h>
#include <aten/MergedEmbeddingBag.h>
#include <torch/all.h>
#include "autocast/autocast_mode.h"
#include "vec/merged_emb_utils.hpp"
#include "vec/unroll_helper.hpp"
#include "vec/vec.h"

namespace torch_ipex {
namespace cpu {

namespace {

using namespace at;
using namespace torch_ipex::cpu::kernel;

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
        data_t* result,
        int64_t result_stride,
        int64_t pooling_mode) {
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
      if (pooling_mode == MEAN) {
        w_vec = w_vec / Vec(float((end_idx - start_idx)));
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
      if (pooling_mode == MEAN) {
        w = w / (end_idx - start_idx);
      }
      result[i] = w;
    }
    result += result_stride;
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
        data_t* result,
        int64_t result_stride,
        int64_t pooling_mode) {
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
      if (pooling_mode == MEAN) {
        f_w_vec1 = f_w_vec1 / fVec(float((end_idx - start_idx)));
        f_w_vec2 = f_w_vec2 / fVec(float((end_idx - start_idx)));
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
      if (pooling_mode == MEAN) {
        w = w / (end_idx - start_idx);
      }
      result[i] = data_t(w);
    }
    result += result_stride;
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
        data_t* result,
        const int64_t result_stride,
        const int64_t pooling_mode) {
#if defined(CPU_CAPABILITY_AVX512)
  // FP32 avx512 fast path for emb_dim=128
  // ~7% improvement while benchmarking on SPR 56C/S with 1 S.
  // benchmark config: num_emb=26, emb_dim=128, batch_size=32768
  // num_bags = [3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1] for
  // each table
  if (emb_dim == 128) {
    for (int64_t b = bs_begin; b < bs_end; ++b) {
      __m512 w0[8];
      __m512 wj[8];
      int64_t start_idx = offsets[b];
      int64_t end_idx = ((b + 1) == bs_end && last_offset != -1)
          ? last_offset
          : offsets[b + 1];
      // load first indices
      int64_t idx = indices[start_idx] * emb_dim;
      compile_time_for<8>::op(load_fp32, w0, &weight[idx]);
      for (int64_t j = start_idx + 1; j < end_idx; ++j) {
        // add following idx
        idx = indices[j] * emb_dim;
        compile_time_for<8>::op(load_fp32, wj, &weight[idx]);
        compile_time_for<8>::op(add_fp32, w0, wj);
      }
      if (pooling_mode == MEAN) {
        __m512 vec_l = _mm512_set1_ps(1.0 / (end_idx - start_idx));
        compile_time_for<8>::op(mul_fp32_constant_b, w0, vec_l);
      }
      // store
      compile_time_for<8>::op(store_fp32, w0, result);
      result += result_stride;
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
      result,
      result_stride,
      pooling_mode);
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
        data_t* result,
        const int64_t result_stride,
        const int64_t pooling_mode) {
#if defined(CPU_CAPABILITY_AVX512)
  // FP16 avx512_fp16 fast path for emb_dim=128
  // only ~1.5% improvement while benchmarking on SPR 56C/S with 1 S.
  // benchmark config: num_emb=26, emb_dim=128, batch_size=32768
  // num_bags = [3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1] for
  // each table
  if (emb_dim == 128) {
    for (int64_t b = bs_begin; b < bs_end; ++b) {
      __m512i fp16_w0[4], fp16_wj[4];
      __m512 fp32_w0[8], fp32_wj[8];
      int64_t start_idx = offsets[b];
      int64_t end_idx = ((b + 1) == bs_end && last_offset != -1)
          ? last_offset
          : offsets[b + 1];
      // load first indices
      int64_t idx = indices[start_idx] * emb_dim;
      compile_time_for<4>::op(
          load_fp16_cast_fp32, fp16_w0, fp32_w0, &weight[idx]);
      for (int64_t j = start_idx + 1; j < end_idx; ++j) {
        // add following idx
        idx = indices[j] * emb_dim;
        compile_time_for<4>::op(
            load_fp16_cast_fp32, fp16_wj, fp32_wj, &weight[idx]);
        compile_time_for<8>::op(add_fp32, fp32_w0, fp32_wj);
      }
      if (pooling_mode == MEAN) {
        __m512 vec_l = _mm512_set1_ps(1.0 / (end_idx - start_idx));
        compile_time_for<8>::op(mul_fp32_constant_b, fp32_w0, vec_l);
      }
      // store
      compile_time_for<8>::op(cast_fp16_and_store, fp32_w0, result);
      result += result_stride;
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
      result,
      result_stride,
      pooling_mode);
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
        data_t* result,
        const int64_t result_stride,
        const int64_t pooling_mode) {
#if defined(CPU_CAPABILITY_AVX512_BF16)
  // BF16 avx512_bf16 fast path for emb_dim=128
  // ~30% improvement while benchmarking on SPR 56C/S with 1 S.
  // benchmark config: num_emb=26, emb_dim=128, batch_size=32768
  // num_bags = [3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1] for
  // each table
  if (emb_dim == 128) {
    for (int64_t b = bs_begin; b < bs_end; ++b) {
      __m512i bf16_w0[4], bf16_wj[4];
      __m512 fp32_w0[8], fp32_wj[8];
      int64_t start_idx = offsets[b];
      int64_t end_idx = ((b + 1) == bs_end && last_offset != -1)
          ? last_offset
          : offsets[b + 1];
      // load first indices
      int64_t idx = indices[start_idx] * emb_dim;
      compile_time_for<4>::op(
          load_bf16_cast_fp32, bf16_w0, fp32_w0, &weight[idx]);
      for (int64_t j = start_idx + 1; j < end_idx; ++j) {
        // add following idx
        idx = indices[j] * emb_dim;
        compile_time_for<4>::op(
            load_bf16_cast_fp32, bf16_wj, fp32_wj, &weight[idx]);
        compile_time_for<8>::op(add_fp32, fp32_w0, fp32_wj);
      }
      if (pooling_mode == MEAN) {
        __m512 vec_l = _mm512_set1_ps(1.0 / (end_idx - start_idx));
        compile_time_for<8>::op(mul_fp32_constant_b, fp32_w0, vec_l);
      }
      // store
      compile_time_for<8>::op(cast_bf16_and_store, fp32_w0, result);
      result += result_stride;
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
      result,
      result_stride,
      pooling_mode);
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
        data_t* result,
        const int64_t result_stride,
        const int64_t pooling_mode) {
  embeddingbag_kern_general(
      bs_begin,
      bs_end,
      num_emb,
      emb_dim,
      last_offset,
      indices,
      offsets,
      weight,
      result,
      result_stride,
      pooling_mode);
}

template <typename data_t, typename index_t>
void merged_embeddingbag_cat(
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
            r,
            /*result_stride=*/(num_emb + 1) * emb_dim,
            SUM);
      }
    }
  }
}

Tensor merged_embedding_cat_fw_impl(
    const TensorList& weights,
    const TensorList& indices,
    const TensorList& offsets,
    const Tensor& dense) {
  RECORD_FUNCTION(__FUNCTION__, c10::ArrayRef<c10::IValue>({}));
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
      at::kBFloat16,
      at::kHalf,
      dense.scalar_type(),
      "merged_embeddingbag_cat",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            indices[0].scalar_type(), "merged_embeddingbag_cat", [&] {
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
              merged_embeddingbag_cat<scalar_t, index_t>(
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

template <typename data_t, typename index_t>
void merged_embeddingbag(
    data_t** o_ptr,
    data_t** w_ptr,
    index_t** indices_ptr,
    index_t** offsets_ptr,
    int64_t num_batch,
    int64_t num_emb,
    int64_t emb_dim,
    std::vector<int64_t> last_offsets,
    int64_t pooling_mode) {
  constexpr int64_t b_block = 128;
  const int64_t n_b_blocks = (num_batch - 1) / b_block + 1;
#pragma omp parallel for collapse(2)
  for (int64_t b = 0; b < n_b_blocks; ++b) {
    for (int64_t m = 0; m < num_emb; ++m) {
      const int64_t bs_begin = b * b_block;
      const int64_t bs_end = std::min(num_batch, (b + 1) * b_block);
      data_t* r = &o_ptr[m][b * b_block * emb_dim];
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
          r,
          /*result_stride=*/emb_dim,
          pooling_mode);
    }
  }
}

std::vector<Tensor> merged_embeddingbag_forward_cpu_kernel_impl(
    const std::vector<Tensor>& weights,
    const TensorList& indices,
    const TensorList& offsets,
    const int64_t pooling_mode,
    const bool include_last_offsets) {
  RECORD_FUNCTION(__FUNCTION__, c10::ArrayRef<c10::IValue>({}));

  int64_t num_emb = weights.size();

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(num_emb > 0);
  int64_t batch_size = offsets[0].size(0);
  if (include_last_offsets) {
    batch_size -= 1;
  }
  int64_t emb_dim = weights[0].size(1);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(num_emb == indices.size());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(num_emb == offsets.size());

  auto index_type = indices[0].scalar_type();
  auto data_type = weights[0].scalar_type();

  std::vector<int64_t> last_offsets(num_emb, -1);
  std::vector<Tensor> outputs;

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
    outputs.emplace_back(empty({batch_size, emb_dim}, weights[i].options()));
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16,
      at::kHalf,
      weights[0].scalar_type(),
      "merged_embeddingbag",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            indices[0].scalar_type(), "merged_embeddingbag", [&] {
              scalar_t* weights_ptr[num_emb];
              scalar_t* outputs_ptr[num_emb];
              index_t* indices_ptr[num_emb];
              index_t* offsets_ptr[num_emb];
              for (int i = 0; i < num_emb; i++) {
                weights_ptr[i] = weights[i].data_ptr<scalar_t>();
                outputs_ptr[i] = outputs[i].data_ptr<scalar_t>();
                indices_ptr[i] = indices[i].data_ptr<index_t>();
                offsets_ptr[i] = offsets[i].data_ptr<index_t>();
              }
              merged_embeddingbag<scalar_t, index_t>(
                  outputs_ptr,
                  weights_ptr,
                  indices_ptr,
                  offsets_ptr,
                  batch_size,
                  num_emb,
                  emb_dim,
                  last_offsets,
                  pooling_mode);
            });
      });

  return outputs;
}

/**
 * Read from embedding table, and write to world_size * num_chk * num_emb's
 *EmbeddingRowCache world_size dimension decide which ranks should this
 *particial look up result sent to num_emb dimmension devide which emb table
 *should this particial look up result belong to num_chk dimension is hard code
 *to 16 here for better parallel scope, list 3 parallel choices:
 *(1) Only parallel on num_emb, this limite the thread nums == num_emb
 *(2) Parallel on num_emb and gbatch. Total tasks = num_emb * gbatch
 *(3) Parallel on num_emb and num_chk. Total tasks = num_emb * num_chk
 *
 *(2) may have more OMP overhead compared with (3) since per
 *task workload is too small. And num_chk=16 is only an empirical value
 *
 *@param cache EmbeddingRowCache List with  world_size * num_chk * num_emb
 *EmbeddingRowCache
 *@param weight_ptr weight (cat all tables together) ptr
 *@param indices_ptr num_emb's indices ptr's ptr
 *@param row_offsets indices offset for different tables
 *@param offsets_ptr num_emb's offsets_ptr ptr's ptr
 *@param gbatch global batch size
 *@param num_chk number of chunks for global batch size
 *@param num_emb number of tables
 *@param emb_dim num of scalers per row in embedding table
 *@param world_size world size
 *@param rank rank id for current device
 *@param last_offsets last indices in case indices_ptr may not include it
 */
template <typename acc_t, typename data_t, typename index_t>
void weight_to_cache_with_chunk(
    std::vector<EmbeddingRowCache<acc_t>>& cache,
    data_t* weight_ptr,
    index_t** indices_ptr,
    std::vector<int64_t> row_offsets,
    index_t** offsets_ptr,
    int64_t gbatch,
    int64_t num_chk,
    int64_t num_emb,
    int64_t emb_dim,
    int64_t world_size,
    int64_t rank,
    std::vector<int64_t> last_offsets) {
  RECORD_FUNCTION(__FUNCTION__, c10::ArrayRef<c10::IValue>({}));
  const int64_t tbatch = (gbatch - 1) / num_chk + 1;
  const int64_t lbatch = gbatch / world_size;
#pragma omp parallel for collapse(2) schedule(guided) shared(cache)
  for (int64_t n = 0; n < num_emb; ++n) {
    for (int64_t nc = 0; nc < num_chk; ++nc) {
      int64_t tb_start = nc * tbatch;
      int64_t tb_end = std::min(tb_start + tbatch, gbatch);
      index_t* index = indices_ptr[n];
      index_t* offsets = offsets_ptr[n];
      int64_t last_offset = last_offsets[n];
      int64_t row_offset = row_offsets[n];
      for (int64_t gb = tb_start; gb < tb_end; ++gb) {
        int64_t start_idx = offsets[gb];
        int64_t end_idx = ((gb + 1) == gbatch && last_offset != -1)
            ? last_offset
            : offsets[gb + 1];
        for (int64_t j = start_idx; j < end_idx; ++j) {
          index_t emb_idx = index[j] + row_offset;
          if ((emb_idx % world_size == rank)) {
            index_t dest = gb / lbatch;
            index_t rowi = (gb % lbatch * num_emb) + n;
            EmbeddingRowCache<acc_t>& emb_cache =
                cache[dest * num_emb * num_chk + nc * num_emb + n];
            index_t local_index = emb_idx / world_size;
            const data_t* wgtPtr = &weight_ptr[local_index * emb_dim];
            auto find = emb_cache.find(rowi);
            if (find == nullptr) {
              find = emb_cache.emplace(rowi, emb_dim);
            }
            // load and add to cache
            add_ker<acc_t, data_t>(find, wgtPtr, emb_dim);
          }
        }
      }
    }
  }
  return;
}

/**
 * reduce the EmbeddingRowCache list from world_size * num_chk * num_emb to
 *world_size * num_emb (on num_chk dimension)
 *@param cache EmbeddingRowCache List with  world_size * num_emb
 *EmbeddingRowCache
 *@param cache_with_chunk EmbeddingRowCache List with  world_size * num_chk *
 *num_emb
 *@param num_chk number of chunks for global batch size
 *@param num_emb number of tables
 *@param emb_dim num of scalers per row in embedding table
 *@param world_size world size
 */
template <typename acc_t, typename index_t>
void accumulate_on_chunks(
    std::vector<EmbeddingRowCache<acc_t>>& cache,
    std::vector<EmbeddingRowCache<acc_t>>& cache_with_chunk,
    int64_t num_chk,
    int64_t num_emb,
    int64_t emb_dim,
    int64_t world_size) {
  RECORD_FUNCTION(__FUNCTION__, c10::ArrayRef<c10::IValue>({}));
#pragma omp parallel for collapse(2) schedule(guided)
  for (int64_t dest = 0; dest < world_size; ++dest) {
    for (int64_t n = 0; n < num_emb; ++n) {
      EmbeddingRowCache<acc_t>& dst_map = cache[dest * num_emb + n];
      for (int64_t nc = 0; nc < num_chk; ++nc) {
        const EmbeddingRowCache<acc_t>& src_map =
            cache_with_chunk[dest * num_emb * num_chk + nc * num_emb + n];
        auto emb_cache = src_map.cache();
        for (const auto& [k, v] : emb_cache) {
          auto find = dst_map.find(k);
          if (find == nullptr) {
            find = dst_map.emplace(k, v, emb_dim);
          } else {
            add_ker<acc_t, acc_t>(find, v, emb_dim);
          }
        }
      }
    }
  }
  return;
}

std::tuple<std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>>
mergedemb_distribute_forward_local_kernel_impl(
    const Tensor& weight,
    const std::vector<int64_t> row_offset,
    const TensorList& indices,
    const TensorList& offsets,
    const int64_t rank,
    const int64_t world_size,
    const bool include_last_offsets) {
  RECORD_FUNCTION(__FUNCTION__, c10::ArrayRef<c10::IValue>({}));

  int64_t num_emb = indices.size();

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(num_emb > 0);
  int64_t global_batch_size = offsets[0].size(0);
  if (include_last_offsets) {
    global_batch_size -= 1;
  }
  int64_t emb_dim = weight.size(1);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(num_emb == indices.size());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(num_emb == offsets.size());

  auto index_type = indices[0].scalar_type();
  auto data_type = weight.scalar_type();
  std::vector<int64_t> last_offsets(num_emb, -1);

  for (int i = 0; i < num_emb; i++) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        indices[i].is_contiguous() && indices[i].scalar_type() == index_type);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        offsets[i].is_contiguous() && offsets[i].scalar_type() == index_type);
    // handle last offsets
    last_offsets[i] = indices[i].numel();
  }

  std::vector<Tensor> idx(world_size);
  std::vector<Tensor> val(world_size);
  std::vector<Tensor> ofs(world_size);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16,
      at::kHalf,
      weight.scalar_type(),
      "mergedemb_distribute_forward_local",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            indices[0].scalar_type(),
            "mergedemb_distribute_forward_local",
            [&] {
              using acc_t = acc_type<scalar_t, true>;
              std::vector<EmbeddingRowCache<acc_t>> cache(world_size * num_emb);
              scalar_t* weight_ptr = weight.data_ptr<scalar_t>();
              index_t* indices_ptr[num_emb];
              index_t* offsets_ptr[num_emb];
              for (int i = 0; i < num_emb; i++) {
                indices_ptr[i] = indices[i].data_ptr<index_t>();
                offsets_ptr[i] = offsets[i].data_ptr<index_t>();
              }
              // read from weight and accumuate in emb cache
              int64_t num_chk = 16;
              std::vector<EmbeddingRowCache<acc_t>> cache_with_num_chk(
                  world_size * num_chk * num_emb);
              weight_to_cache_with_chunk<acc_t, scalar_t, index_t>(
                  cache_with_num_chk,
                  weight_ptr,
                  indices_ptr,
                  row_offset,
                  offsets_ptr,
                  global_batch_size,
                  num_chk,
                  num_emb,
                  emb_dim,
                  world_size,
                  rank,
                  last_offsets);
              accumulate_on_chunks<acc_t, index_t>(
                  cache,
                  cache_with_num_chk,
                  num_chk,
                  num_emb,
                  emb_dim,
                  world_size);
              // read from emb cache and write to the buffer while will be
              // comunicated with other ranks
              prepare_ccl_buffer<acc_t, scalar_t, index_t>(
                  idx,
                  val,
                  ofs,
                  cache,
                  world_size,
                  num_emb,
                  emb_dim,
                  indices[0].options(),
                  weight.options());
            });
      });

  return std::make_tuple(idx, val, ofs);
}

template <typename acc_t, typename data_t, typename index_t>
void mergedemb_distribute_forward_merge(
    const int64_t world_size,
    const int64_t num_emb,
    const int64_t emb_dim,
    index_t** idx_ptr,
    data_t** val_ptr,
    int64_t** ofs_ptr,
    data_t* res_ptr) {
#pragma omp parallel for
  for (int64_t i = 0; i < num_emb; ++i) {
    EmbeddingRowCache<acc_t> cache;
    for (int64_t j = 0; j < world_size; ++j) {
      const int64_t ts = ofs_ptr[j][i];
      const int64_t te = ofs_ptr[j][i + 1];
      for (int64_t k = ts; k < te; ++k) {
        index_t rowi = idx_ptr[j][k];
        const data_t* accPtr = &val_ptr[j][k * emb_dim];
        auto find = cache.find(rowi);
        if (find == nullptr) {
          find = cache.emplace(rowi, emb_dim);
        }
        add_ker<acc_t, data_t>(find, accPtr, emb_dim);
      }
    }
    auto emb_cache = cache.cache();
    for (auto& [key, value] : emb_cache) {
      data_t* dest = &res_ptr[key * emb_dim]; // EMBRES
      move_ker<data_t, acc_t>(dest, value, emb_dim);
    }
  }
}

void mergedemb_distribute_forward_merge_kernel_impl(
    Tensor& output,
    const TensorList& idx,
    const TensorList& val,
    const TensorList& ofs,
    const int64_t num_emb) {
  RECORD_FUNCTION(__FUNCTION__, c10::ArrayRef<c10::IValue>({}));
  int64_t world_size = idx.size();
  int64_t emb_dim = output.size(2);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16,
      at::kHalf,
      output.scalar_type(),
      "mergedemb_distribute_forward_merge",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            idx[0].scalar_type(), "mergedemb_distribute_forward_merge", [&] {
              using acc_t = acc_type<scalar_t, true>;
              scalar_t* res_ptr = output.data_ptr<scalar_t>();
              index_t* idx_ptr[world_size];
              scalar_t* val_ptr[world_size];
              int64_t* ofs_ptr[world_size];
              for (int i = 0; i < world_size; i++) {
                idx_ptr[i] = idx[i].data_ptr<index_t>();
                val_ptr[i] = val[i].data_ptr<scalar_t>();
                ofs_ptr[i] = ofs[i].data_ptr<int64_t>();
              }
              // read from weight and accumuate in emb cache
              mergedemb_distribute_forward_merge<acc_t, scalar_t, index_t>(
                  world_size,
                  num_emb,
                  emb_dim,
                  idx_ptr,
                  val_ptr,
                  ofs_ptr,
                  res_ptr);
            });
      });

  return;
}

} // anonymous namespace

IPEX_REGISTER_DISPATCH(
    merged_embeddingbag_forward_cpu_kernel_stub,
    &merged_embeddingbag_forward_cpu_kernel_impl);
IPEX_REGISTER_DISPATCH(
    merged_embeddingbag_cat_fw_stub,
    &merged_embedding_cat_fw_impl);
IPEX_REGISTER_DISPATCH(
    mergedemb_distribute_forward_local_kernel_stub,
    &mergedemb_distribute_forward_local_kernel_impl);
IPEX_REGISTER_DISPATCH(
    mergedemb_distribute_forward_merge_kernel_stub,
    &mergedemb_distribute_forward_merge_kernel_impl);
} // namespace cpu
} // namespace torch_ipex
