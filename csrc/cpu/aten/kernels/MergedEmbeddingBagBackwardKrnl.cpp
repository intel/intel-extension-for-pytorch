#include <ATen/cpu/vec/functional.h>
#include <aten/MergedEmbeddingBag.h>
#include <c10/core/CPUAllocator.h>
#include <omp.h>
#include "vec/merged_emb_utils.hpp"
#include "vec/unroll_helper.hpp"
#include "vec/vec.h"

namespace torch_ipex {
namespace cpu {

namespace {

using namespace at;
using namespace torch_ipex::cpu::kernel;

template <typename data_t, typename acc_t>
typename std::enable_if<
    std::is_same<data_t, Half>::value || std::is_same<data_t, BFloat16>::value,
    void>::
    type inline copy_from_grad_cache(
        data_t* wgrad,
        const EmbeddingRowCache<acc_t>& ewc,
        int64_t emb_dim) {
#if defined(CPU_CAPABILITY_AVX512_BF16)
  if (emb_dim == 128) {
    __m512 cache_vec[8];
    auto emb_cache = ewc.cache();
    for (auto& [k, v] : emb_cache) {
      compile_time_for<8>::op(load_fp32, cache_vec, v);
      if (std::is_same<data_t, BFloat16>::value)
        compile_time_for<8>::op(
            cast_bf16_and_store, cache_vec, &wgrad[k * emb_dim]);
      else
        compile_time_for<8>::op(
            cast_fp16_and_store, cache_vec, &wgrad[k * emb_dim]);
    }
    return;
  }
#endif
  using lpVec = at::vec::Vectorized<data_t>;
  using fVec = at::vec::Vectorized<float>;
  auto vec_size = lpVec::size();
  auto fvec_size = fVec::size();
  auto emb_cache = ewc.cache();
  for (auto& [k, v] : emb_cache) {
    int64_t i = 0;
    for (; i + vec_size <= emb_dim; i += vec_size) {
      fVec cache_vec1 = fVec::loadu(&v[i]);
      fVec cache_vec2 = fVec::loadu(&v[i + fvec_size]);
      lpVec out_vec =
          at::vec::convert_from_float<data_t>(cache_vec1, cache_vec2);
      out_vec.store(&wgrad[k * emb_dim + i]);
    }
    for (; i < emb_dim; i++) {
      wgrad[k * emb_dim + i] = data_t(v[i]);
    }
  }
}

template <typename data_t, typename acc_t>
typename std::enable_if<
    std::is_same<data_t, float>::value || std::is_same<data_t, double>::value,
    void>::
    type inline copy_from_grad_cache(
        data_t* wgrad,
        const EmbeddingRowCache<acc_t>& ewc,
        int64_t emb_dim) {
  // do nothing but just make compiler happy
  return;
}

template <typename data_t, typename index_t, typename acc_t, bool use_cache>
typename std::enable_if<
    std::is_same<data_t, float>::value || std::is_same<data_t, double>::value,
    void>::
    type inline embeddingbag_bwd_acc_kern(
        const int64_t bs_begin,
        const int64_t bs_end,
        const int64_t num_emb,
        const int64_t emb_dim,
        const int64_t last_offset,
        const index_t* indices,
        const index_t* offsets,
        const data_t* grad,
        data_t* result,
        const int64_t pooling_mode,
        EmbeddingRowCache<acc_t>& ewc) {
  using Vec = at::vec::Vectorized<data_t>;
  const auto vec_size = Vec::size();
  acc_t* find = nullptr;
  // only accumulate wid % numthd == thdidx
  const int32_t thdidx = omp_get_thread_num();
  const int32_t numthd = omp_get_num_threads();

  for (int32_t b = bs_begin; b < bs_end; ++b) {
    int64_t start_idx = offsets[b];
    int64_t end_idx =
        ((b + 1) == bs_end && last_offset != -1) ? last_offset : offsets[b + 1];
    bool grad_loaded = false;
    Vec grad_vec, wgrad_vec;
    for (int64_t j = start_idx; j < end_idx; ++j) {
      int i = 0;
      int64_t wid = indices[j];
      data_t scale = 1.0 / (end_idx - start_idx);
      if (wid % numthd == thdidx) {
        if (use_cache) {
          find = ewc.find(wid);
          if (find == nullptr) {
            find = ewc.emplace(wid, emb_dim);
          }
        }
        for (; i + vec_size <= emb_dim; i += vec_size) {
          if (!grad_loaded) {
            grad_loaded = true;
            grad_vec = Vec::loadu(&grad[b * emb_dim + i]);
            if (pooling_mode == MEAN && (end_idx - start_idx) > 1) {
              grad_vec = grad_vec * Vec(scale);
            }
          }
          auto acc_ptr = use_cache ? &find[i] : &result[wid * emb_dim + i];
          wgrad_vec = Vec::loadu(acc_ptr);
          wgrad_vec += grad_vec;
          wgrad_vec.store(acc_ptr);
        }
        auto acc_ptr = use_cache ? &find[0] : &result[wid * emb_dim];
        for (; i < emb_dim; i++) {
          auto grad_value = grad[b * emb_dim + i];
          if (pooling_mode == MEAN && (end_idx - start_idx) > 1) {
            grad_value = grad_value * scale;
          }
          acc_ptr[i] += grad_value;
        }
      }
    }
  }
}

template <typename data_t, typename index_t, typename acc_t, bool use_cache>
typename std::enable_if<
    std::is_same<data_t, Half>::value || std::is_same<data_t, BFloat16>::value,
    void>::
    type inline embeddingbag_bwd_acc_kern(
        const int64_t bs_begin,
        const int64_t bs_end,
        const int64_t num_emb,
        const int64_t emb_dim,
        const int64_t last_offset,
        const index_t* indices,
        const index_t* offsets,
        const data_t* grad,
        data_t* result,
        const int64_t pooling_mode,
        EmbeddingRowCache<acc_t>& ewc) {
  // Low precision grad have to accumulate to cache with acc_t
  assert(use_cache);
#if defined(CPU_CAPABILITY_AVX512_BF16)
  __m512i lp_grad[4];
  __m512 fp32_grad[8], fp32_acc_buffer[8];
#endif
  using lpVec = at::vec::Vectorized<data_t>;
  using fVec = at::vec::Vectorized<float>;
  auto vec_size = lpVec::size();
  auto fvec_size = fVec::size();
  // only accumulate wid % numthd == thdidx
  const int32_t thdidx = omp_get_thread_num();
  const int32_t numthd = omp_get_num_threads();

  for (int32_t b = bs_begin; b < bs_end; ++b) {
    int64_t start_idx = offsets[b];
    int64_t end_idx =
        ((b + 1) == bs_end && last_offset != -1) ? last_offset : offsets[b + 1];
    bool grad_loaded = false;
    lpVec lpgrad_vec;
    fVec fgrad_vec1, fgrad_vec2, fwgrad_vec1, fwgrad_vec2;
    for (int64_t j = start_idx; j < end_idx; ++j) {
      int i = 0;
      int32_t wid = indices[j];
      float scale = 1.0 / (end_idx - start_idx);
      if (wid % numthd == thdidx) {
        auto find = ewc.find(wid);
        if (find == nullptr) {
          find = ewc.emplace(wid, emb_dim);
        }
#if defined(CPU_CAPABILITY_AVX512_BF16)
        if (emb_dim == 128) {
          __m512 vec_l = _mm512_set1_ps(scale);
          if (!grad_loaded) {
            grad_loaded = true;
            if (std::is_same<data_t, BFloat16>::value)
              compile_time_for<4>::op(
                  load_bf16_cast_fp32, lp_grad, fp32_grad, &grad[b * emb_dim]);
            else
              compile_time_for<4>::op(
                  load_fp16_cast_fp32, lp_grad, fp32_grad, &grad[b * emb_dim]);
            if (pooling_mode == MEAN && (end_idx - start_idx) > 1) {
              compile_time_for<8>::op(mul_fp32_constant_b, fp32_grad, vec_l);
            }
          }
          compile_time_for<8>::op(load_fp32, fp32_acc_buffer, find);
          compile_time_for<8>::op(add_fp32, fp32_acc_buffer, fp32_grad);
          compile_time_for<8>::op(store_fp32, fp32_acc_buffer, find);
          continue;
        }
#endif
        for (; i + vec_size <= emb_dim; i += vec_size) {
          if (!grad_loaded) {
            grad_loaded = true;
            lpgrad_vec = lpVec::loadu(&grad[b * emb_dim + i]);
            std::tie(fgrad_vec1, fgrad_vec2) =
                at::vec::convert_to_float<data_t>(lpgrad_vec);
            if (pooling_mode == MEAN && (end_idx - start_idx) > 1) {
              fgrad_vec1 = fgrad_vec1 * fVec(scale);
              fgrad_vec2 = fgrad_vec2 * fVec(scale);
            }
          }
          fwgrad_vec1 = fVec::loadu(&find[i]);
          fwgrad_vec2 = fVec::loadu(&find[i + fvec_size]);
          fwgrad_vec1 += fgrad_vec1;
          fwgrad_vec2 += fgrad_vec2;
          fwgrad_vec1.store(&find[i]);
          fwgrad_vec2.store(&find[i + fvec_size]);
        }
        for (; i < emb_dim; i++) {
          float grad_value = grad[b * emb_dim + i];
          if (pooling_mode == MEAN && (end_idx - start_idx) > 1) {
            grad_value = grad_value * scale;
          }
          find[i] += grad_value;
        }
      }
    }
  }
}

template <typename data_t, typename index_t>
typename std::enable_if<
    std::is_same<data_t, Half>::value || std::is_same<data_t, BFloat16>::value,
    void>::type
merged_embeddingbag_dense_backward(
    data_t** o_ptr,
    data_t** grads_ptr,
    index_t** indices_ptr,
    index_t** offsets_ptr,
    int64_t num_batch,
    int64_t num_emb,
    int64_t emb_dim,
    std::vector<int64_t> last_offsets,
    int64_t pooling_mode) {
  using acc_t = acc_type<data_t, true>; // if use_cuda = False, float's acc type
                                        // will be double
#pragma omp parallel
  {
    for (int32_t n = 0; n < num_emb; ++n) {
      EmbeddingRowCache<acc_t> ewc;
      embeddingbag_bwd_acc_kern<data_t, index_t, acc_t, /*use_cache=*/true>(
          /*bs_begin=*/0,
          num_batch,
          num_emb,
          emb_dim,
          last_offsets[n],
          indices_ptr[n],
          offsets_ptr[n],
          grads_ptr[n],
          o_ptr[n],
          pooling_mode,
          ewc);
      copy_from_grad_cache(o_ptr[n], ewc, emb_dim);
    }
  }
}

template <typename data_t, typename index_t>
typename std::enable_if<
    std::is_same<data_t, double>::value || std::is_same<data_t, float>::value,
    void>::type
merged_embeddingbag_dense_backward(
    data_t** o_ptr,
    data_t** grads_ptr,
    index_t** indices_ptr,
    index_t** offsets_ptr,
    int64_t num_batch,
    int64_t num_emb,
    int64_t emb_dim,
    std::vector<int64_t> last_offsets,
    int64_t pooling_mode) {
  // For float/double, do not need ewc to accumulate grad
  EmbeddingRowCache<data_t> dummy_ewc;
#pragma omp parallel
  {
    for (int32_t n = 0; n < num_emb; ++n) {
      embeddingbag_bwd_acc_kern<data_t, index_t, data_t, /*use_cache=*/false>(
          /*bs_begin=*/0,
          num_batch,
          num_emb,
          emb_dim,
          last_offsets[n],
          indices_ptr[n],
          offsets_ptr[n],
          grads_ptr[n],
          o_ptr[n],
          pooling_mode,
          dummy_ewc);
    }
  }
}

std::vector<Tensor> merged_embeddingbag_backward_cpu_kernel_impl(
    const TensorList& grad_outs_,
    const TensorList& weights,
    const TensorList& indices,
    const TensorList& offsets,
    const int64_t pooling_mode,
    const bool include_last_offsets) {
  RECORD_FUNCTION(__FUNCTION__, c10::ArrayRef<c10::IValue>({}));

  int64_t num_emb = weights.size();

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(num_emb > 0);
  int64_t batch_size = grad_outs_[0].size(0);
  int64_t emb_dim = weights[0].size(1);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(num_emb == indices.size());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(num_emb == offsets.size());

  auto index_type = indices[0].scalar_type();
  auto data_type = weights[0].scalar_type();

  std::vector<int64_t> last_offsets(num_emb, -1);
  std::vector<Tensor> contiguous_grad;
  std::vector<Tensor> outputs;

  for (int i = 0; i < num_emb; i++) {
    contiguous_grad.emplace_back(grad_outs_[i].contiguous());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        indices[i].is_contiguous() && indices[i].scalar_type() == index_type);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        offsets[i].is_contiguous() && offsets[i].scalar_type() == index_type);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        contiguous_grad[i].is_contiguous() &&
        contiguous_grad[i].scalar_type() == data_type);
    // handle last offsets
    last_offsets[i] = indices[i].numel();
    outputs.emplace_back(zeros_like(weights[i], weights[i].options()));
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16,
      at::kHalf,
      weights[0].scalar_type(),
      "merged_embeddingbag_dense_backward",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            indices[0].scalar_type(),
            "merged_embeddingbag_dense_backward",
            [&] {
              scalar_t* grads_ptr[num_emb];
              scalar_t* outputs_ptr[num_emb];
              index_t* indices_ptr[num_emb];
              index_t* offsets_ptr[num_emb];
              for (int i = 0; i < num_emb; i++) {
                grads_ptr[i] = contiguous_grad[i].data_ptr<scalar_t>();
                outputs_ptr[i] = outputs[i].data_ptr<scalar_t>();
                indices_ptr[i] = indices[i].data_ptr<index_t>();
                offsets_ptr[i] = offsets[i].data_ptr<index_t>();
              }
              merged_embeddingbag_dense_backward<scalar_t, index_t>(
                  outputs_ptr,
                  grads_ptr,
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

template <typename param_t, typename acc_t>
inline void sgd_update(
    param_t* param_ptr,
    at::BFloat16* trail_ptr,
    acc_t* grad_ptr,
    float weight_decay,
    float lr,
    int size) {
  using Vec = at::vec::Vectorized<param_t>;
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec param_vec = Vec::loadu(param_ptr + d);
    Vec grad_vec =
        Vec::loadu(grad_ptr + d) + param_vec * Vec(param_t(weight_decay));

    param_vec -= grad_vec * Vec(param_t(lr));
    param_vec.store(param_ptr + d);
  }
  for (; d < size; d++) {
    param_t grad_val = grad_ptr[d] + param_ptr[d] * weight_decay;
    param_ptr[d] -= grad_val * lr;
  }
}

template <>
inline void sgd_update<at::BFloat16, float>(
    at::BFloat16* param_ptr,
    at::BFloat16* trail_ptr,
    float* grad_ptr,
    float weight_decay,
    float lr,
    int size) {
#if defined(CPU_CAPABILITY_AVX512)
  if (size == 128) {
    __m512i bf16_top[4], trail[4];
    __m512 fp32_param[8], fp32_grad[8];
    __m512 w_decay_v = _mm512_set1_ps(weight_decay);
    __m512 lr_v = _mm512_set1_ps(-lr);
    compile_time_for<8>::op(load_lp, bf16_top, param_ptr);
    compile_time_for<8>::op(load_lp, trail, trail_ptr);
    compile_time_for<4>::op(pack_to_fp32, bf16_top, trail, fp32_param);
    compile_time_for<8>::op(load_fp32, fp32_grad, grad_ptr);
    compile_time_for<8>::op(fma_constant_a, fp32_grad, w_decay_v, fp32_param);
    compile_time_for<8>::op(fma_constant_a, fp32_param, lr_v, fp32_grad);
    compile_time_for<4>::op(split_from_fp32, bf16_top, trail, fp32_param);
    compile_time_for<4>::op(store_lp, bf16_top, param_ptr);
    compile_time_for<4>::op(store_lp, trail, trail_ptr);
    return;
  }
#endif
  using bVec = at::vec::Vectorized<at::BFloat16>;
  using fVec = at::vec::Vectorized<float>;
  int64_t d = 0;
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec param_bvec = bVec::loadu(param_ptr + d);
    bVec trail_bvec = bVec::loadu(trail_ptr + d);
    fVec param_fvec, param_fvec2;
    std::tie(param_fvec, param_fvec2) =
        at::vec::pack_bfloat16_float(param_bvec, trail_bvec);

    fVec grad_fvec = fVec::loadu(grad_ptr + d);
    fVec grad_fvec2 = fVec::loadu(grad_ptr + d + fVec::size());

    grad_fvec = grad_fvec + param_fvec * fVec(weight_decay);
    grad_fvec2 = grad_fvec2 + param_fvec2 * fVec(weight_decay);

    param_fvec -= grad_fvec * fVec(lr);
    param_fvec2 -= grad_fvec2 * fVec(lr);

    std::tie(param_bvec, trail_bvec) =
        at::vec::unpack_float_bfloat16(param_fvec, param_fvec2);
    param_bvec.store(param_ptr + d);
    trail_bvec.store(trail_ptr + d);
  }
  for (; d < size; d++) {
    float param_val = at::vec::pack_bfloat16_float(param_ptr[d], trail_ptr[d]);
    float grad_val = grad_ptr[d] + param_val * weight_decay;
    param_val -= grad_val * lr;
    std::tie(param_ptr[d], trail_ptr[d]) =
        at::vec::unpack_float_bfloat16(param_val);
  }
}

template <typename param_t, typename acc_t>
inline void adagrad_update(
    param_t* param_ptr,
    at::BFloat16* trail_ptr,
    acc_t* hessian_ptr,
    acc_t* grad_ptr,
    float eps,
    float lr,
    int size) {
  // hessian += grad**2
  // weight += grad * lr / (sqrt(hessian) + eps)
  // lr < 0
  using Vec = at::vec::Vectorized<param_t>;
  Vec lr_vec = Vec(param_t(lr));
  Vec eps_vec = Vec(param_t(eps));
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec param_vec = Vec::loadu(param_ptr + d);
    Vec hessian_vec = Vec::loadu(hessian_ptr + d);
    Vec grad_vec = Vec::loadu(grad_ptr + d);
    hessian_vec += (grad_vec * grad_vec);
    hessian_vec.store(hessian_ptr + d);
    param_vec -= ((grad_vec * lr_vec) / (hessian_vec.sqrt() + eps_vec));
    param_vec.store(param_ptr + d);
  }
  for (; d < size; d++) {
    hessian_ptr[d] += (grad_ptr[d] * grad_ptr[d]);
    param_ptr[d] -= ((grad_ptr[d] * lr) / (std::sqrt(hessian_ptr[d] + eps)));
  }
}

template <>
inline void adagrad_update<at::BFloat16, float>(
    at::BFloat16* param_ptr,
    at::BFloat16* trail_ptr,
    float* hessian_ptr,
    float* grad_ptr,
    float eps,
    float lr,
    int size) {
  // hessian += grad**2
  // weight += grad * lr / (sqrt(hessian) + eps)
  // lr > 0
#if defined(CPU_CAPABILITY_AVX512)
  if (size == 128) {
    __m512i bf16_top[4], trail[4];
    __m512 fp32_param[8], fp32_grad[8], fp32_hessian[8];
    __m512 eps_v = _mm512_set1_ps(eps);
    __m512 lr_v = _mm512_set1_ps(-lr);
    compile_time_for<8>::op(load_lp, bf16_top, param_ptr);
    compile_time_for<8>::op(load_lp, trail, trail_ptr);
    compile_time_for<4>::op(pack_to_fp32, bf16_top, trail, fp32_param);
    compile_time_for<8>::op(load_fp32, fp32_grad, grad_ptr);
    compile_time_for<8>::op(load_fp32, fp32_hessian, hessian_ptr);
    compile_time_for<8>::op(fma, fp32_hessian, fp32_grad, fp32_grad);
    compile_time_for<8>::op(store_fp32, fp32_hessian, hessian_ptr);
    // re-use fp32_hessian set
    compile_time_for<8>::op(sqrt_fp32, fp32_hessian, fp32_hessian);
    compile_time_for<8>::op(add_fp32_const_b, fp32_hessian, eps_v);
    compile_time_for<8>::op(
        div_fp32_constant_a, fp32_hessian, lr_v, fp32_hessian);
    compile_time_for<8>::op(fma, fp32_param, fp32_hessian, fp32_grad);
    compile_time_for<4>::op(split_from_fp32, bf16_top, trail, fp32_param);
    compile_time_for<4>::op(store_lp, bf16_top, param_ptr);
    compile_time_for<4>::op(store_lp, trail, trail_ptr);
    return;
  }
#endif
  using bVec = at::vec::Vectorized<at::BFloat16>;
  using fVec = at::vec::Vectorized<float>;
  fVec lr_vec = fVec(lr);
  fVec eps_vec = fVec(eps);
  int64_t d = 0;
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec param_bvec = bVec::loadu(param_ptr + d);
    bVec trail_bvec = bVec::loadu(trail_ptr + d);
    fVec param_fvec, param_fvec2;
    std::tie(param_fvec, param_fvec2) =
        at::vec::pack_bfloat16_float(param_bvec, trail_bvec);
    fVec hessian_fvec = fVec::loadu(hessian_ptr + d);
    fVec hessian_fvec2 = fVec::loadu(hessian_ptr + d + fVec::size());
    fVec grad_fvec = fVec::loadu(grad_ptr + d);
    fVec grad_fvec2 = fVec::loadu(grad_ptr + d + fVec::size());
    hessian_fvec += (grad_fvec * grad_fvec);
    hessian_fvec2 += (grad_fvec2 * grad_fvec2);
    hessian_fvec.store(hessian_ptr + d);
    hessian_fvec2.store(hessian_ptr + d + fVec::size());
    param_fvec -= ((grad_fvec * lr_vec) / (hessian_fvec.sqrt() + eps_vec));
    param_fvec2 -= ((grad_fvec2 * lr_vec) / (hessian_fvec2.sqrt() + eps_vec));
    std::tie(param_bvec, trail_bvec) =
        at::vec::unpack_float_bfloat16(param_fvec, param_fvec2);
    param_bvec.store(param_ptr + d);
    trail_bvec.store(trail_ptr + d);
  }
  for (; d < size; d++) {
    float param_val = at::vec::pack_bfloat16_float(param_ptr[d], trail_ptr[d]);
    hessian_ptr[d] += (grad_ptr[d] * grad_ptr[d]);
    param_val -= ((grad_ptr[d] * lr) / (std::sqrt(hessian_ptr[d] + eps)));
    std::tie(param_ptr[d], trail_ptr[d]) =
        at::vec::unpack_float_bfloat16(param_val);
  }
}

template <typename data_t, typename acc_t>
void inline EmbeddingGradUpdate<data_t, acc_t, SGDArgs>::update(
    data_t* weight,
    const EmbeddingRowCache<acc_t>& ewc,
    const SGDArgs& args,
    const int32_t table_id,
    const int64_t emb_dim) {
  BFloat16* bf16_trail_ptr = args.bf16_trail[table_id].data_ptr<BFloat16>();
  auto emb_cache = ewc.cache();
  for (auto& it : emb_cache) {
    size_t idx = it.first;
    acc_t* grad = it.second;
    sgd_update<data_t, acc_t>(
        &weight[idx * emb_dim],
        &bf16_trail_ptr[idx * emb_dim],
        grad,
        args.weight_decay,
        args.lr,
        emb_dim);
  }
}

template <typename data_t, typename acc_t>
void inline EmbeddingGradUpdate<data_t, acc_t, AdaGradArgs>::update(
    data_t* weight,
    const EmbeddingRowCache<acc_t>& ewc,
    const AdaGradArgs& args,
    const int32_t table_id,
    const int64_t emb_dim) {
  BFloat16* bf16_trail_ptr = args.bf16_trail[table_id].data_ptr<BFloat16>();
  acc_t* hessian_ptr = args.hessian[table_id].data_ptr<acc_t>();
  auto emb_cache = ewc.cache();
  for (auto& it : emb_cache) {
    size_t idx = it.first;
    acc_t* grad = it.second;
    adagrad_update<data_t, acc_t>(
        &weight[idx * emb_dim],
        &bf16_trail_ptr[idx * emb_dim],
        &hessian_ptr[idx * emb_dim],
        grad,
        args.eps,
        args.lr,
        emb_dim);
  }
}

template <typename data_t, typename index_t, typename optimizer_arg_t>
void merged_embeddingbag_backward_update(
    data_t** w_ptr,
    data_t** grads_ptr,
    index_t** indices_ptr,
    index_t** offsets_ptr,
    int64_t num_batch,
    int64_t num_emb,
    int64_t emb_dim,
    std::vector<int64_t> last_offsets,
    int64_t pooling_mode,
    optimizer_arg_t& args) {
  using acc_t =
      acc_type<data_t, /*use_cuda=*/true>; // if use_cuda = False, float's acc
                                           // type will be double
#pragma omp parallel
  {
    for (int32_t n = 0; n < num_emb; ++n) {
      EmbeddingRowCache<acc_t> ewc;
      embeddingbag_bwd_acc_kern<data_t, index_t, acc_t, /*use_cache=*/true>(
          /*bs_begin=*/0,
          num_batch,
          num_emb,
          emb_dim,
          last_offsets[n],
          indices_ptr[n],
          offsets_ptr[n],
          grads_ptr[n],
          /*outout_ptr=*/nullptr,
          pooling_mode,
          ewc);
      EmbeddingGradUpdate<data_t, acc_t, optimizer_arg_t>::update(
          w_ptr[n], ewc, args, n, emb_dim);
    }
  }
}

void merged_embeddingbag_backward_sgd_cpu_kernel_impl(
    const TensorList& grad_outs_,
    const TensorList& weights,
    const TensorList& indices,
    const TensorList& offsets,
    const int64_t pooling_mode,
    const bool include_last_offsets,
    const TensorList& bf16_trail,
    const double weight_decay,
    const double lr) {
  RECORD_FUNCTION(__FUNCTION__, c10::ArrayRef<c10::IValue>({}));

  int64_t num_emb = weights.size();

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(num_emb > 0);
  int64_t batch_size = grad_outs_[0].size(0);
  int64_t emb_dim = weights[0].size(1);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(num_emb == indices.size());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(num_emb == offsets.size());

  auto index_type = indices[0].scalar_type();
  auto data_type = weights[0].scalar_type();

  std::vector<int64_t> last_offsets(num_emb, -1);
  std::vector<Tensor> contiguous_grad;
  std::vector<Tensor> outputs;

  for (int i = 0; i < num_emb; i++) {
    contiguous_grad.emplace_back(grad_outs_[i].contiguous());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        indices[i].is_contiguous() && indices[i].scalar_type() == index_type);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        offsets[i].is_contiguous() && offsets[i].scalar_type() == index_type);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        contiguous_grad[i].is_contiguous() &&
        contiguous_grad[i].scalar_type() == data_type);
    // handle last offsets
    last_offsets[i] = indices[i].numel();
  }

  AT_DISPATCH_FLOATING_TYPES_AND(
      at::kBFloat16,
      weights[0].scalar_type(),
      "merged_embeddingbag_backward_update",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            indices[0].scalar_type(),
            "merged_embeddingbag_backward_update",
            [&] {
              scalar_t* grads_ptr[num_emb];
              scalar_t* weights_ptr[num_emb];
              index_t* indices_ptr[num_emb];
              index_t* offsets_ptr[num_emb];
              for (int i = 0; i < num_emb; i++) {
                weights_ptr[i] = weights[i].data_ptr<scalar_t>();
                grads_ptr[i] = contiguous_grad[i].data_ptr<scalar_t>();
                indices_ptr[i] = indices[i].data_ptr<index_t>();
                offsets_ptr[i] = offsets[i].data_ptr<index_t>();
              }
              SGDArgs args = SGDArgs(bf16_trail, weight_decay, lr);
              merged_embeddingbag_backward_update<scalar_t, index_t, SGDArgs>(
                  weights_ptr,
                  grads_ptr,
                  indices_ptr,
                  offsets_ptr,
                  batch_size,
                  num_emb,
                  emb_dim,
                  last_offsets,
                  pooling_mode,
                  args);
            });
      });
}

void merged_embeddingbag_backward_adagrad_cpu_kernel_impl(
    const TensorList& grad_outs_,
    const TensorList& weights,
    const TensorList& indices,
    const TensorList& offsets,
    const int64_t pooling_mode,
    const bool include_last_offsets,
    const TensorList& hessian,
    const TensorList& bf16_trail,
    const double eps,
    const double lr) {
  RECORD_FUNCTION(__FUNCTION__, c10::ArrayRef<c10::IValue>({}));

  int64_t num_emb = weights.size();

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(num_emb > 0);
  int64_t batch_size = grad_outs_[0].size(0);
  int64_t emb_dim = weights[0].size(1);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(num_emb == indices.size());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(num_emb == offsets.size());

  auto index_type = indices[0].scalar_type();
  auto data_type = weights[0].scalar_type();

  std::vector<int64_t> last_offsets(num_emb, -1);
  std::vector<Tensor> contiguous_grad;
  std::vector<Tensor> outputs;

  for (int i = 0; i < num_emb; i++) {
    contiguous_grad.emplace_back(grad_outs_[i].contiguous());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        indices[i].is_contiguous() && indices[i].scalar_type() == index_type);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        offsets[i].is_contiguous() && offsets[i].scalar_type() == index_type);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        contiguous_grad[i].is_contiguous() &&
        contiguous_grad[i].scalar_type() == data_type);
    // handle last offsets
    last_offsets[i] = indices[i].numel();
  }

  AT_DISPATCH_FLOATING_TYPES_AND(
      at::kBFloat16,
      weights[0].scalar_type(),
      "merged_embeddingbag_backward_update",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            indices[0].scalar_type(),
            "merged_embeddingbag_backward_update",
            [&] {
              scalar_t* grads_ptr[num_emb];
              scalar_t* weights_ptr[num_emb];
              index_t* indices_ptr[num_emb];
              index_t* offsets_ptr[num_emb];
              for (int i = 0; i < num_emb; i++) {
                weights_ptr[i] = weights[i].data_ptr<scalar_t>();
                grads_ptr[i] = contiguous_grad[i].data_ptr<scalar_t>();
                indices_ptr[i] = indices[i].data_ptr<index_t>();
                offsets_ptr[i] = offsets[i].data_ptr<index_t>();
              }
              AdaGradArgs args = AdaGradArgs(bf16_trail, hessian, eps, lr);
              merged_embeddingbag_backward_update<
                  scalar_t,
                  index_t,
                  AdaGradArgs>(
                  weights_ptr,
                  grads_ptr,
                  indices_ptr,
                  offsets_ptr,
                  batch_size,
                  num_emb,
                  emb_dim,
                  last_offsets,
                  pooling_mode,
                  args);
            });
      });
}

template <typename acc_t, typename data_t, typename index_t>
void prepare_emb_bwd_cache(
    std::vector<EmbeddingRowCache<acc_t>>& cache,
    data_t* grad_ptr,
    index_t** indices_ptr,
    std::vector<int64_t> row_offsets,
    index_t** offsets_ptr,
    int64_t gbatch,
    int64_t num_emb,
    int64_t emb_dim,
    int64_t world_size,
    int64_t rank,
    std::vector<int64_t> last_offsets) {
  const int64_t lbatch = gbatch / world_size;
#pragma omp parallel
  {
    const int64_t num_thd = omp_get_num_threads();
    int64_t tid = omp_get_thread_num();
    for (int64_t n = 0; n < num_emb; ++n) {
      index_t* index = indices_ptr[n];
      index_t* offsets = offsets_ptr[n];
      int64_t last_offset = last_offsets[n];
      int64_t row_offset = row_offsets[n];
      for (int64_t b = 0; b < lbatch; ++b) {
        int64_t gb = rank * lbatch + b;
        int64_t start_idx = offsets[gb];
        int64_t end_idx = ((gb + 1) == gbatch && last_offset != -1)
            ? last_offset
            : offsets[gb + 1];
        for (int64_t j = start_idx; j < end_idx; ++j) {
          index_t emb_idx = index[j] + row_offset;
          if (((emb_idx / world_size) % num_thd == tid)) {
            index_t dest = emb_idx % world_size;
            EmbeddingRowCache<acc_t>& emb_cache = cache[dest * num_thd + tid];
            auto find = emb_cache.find(emb_idx);
            if (find == nullptr) {
              find = emb_cache.emplace(emb_idx, emb_dim);
            }
            const data_t* grdPtr =
                &grad_ptr[(b * num_emb + n) * emb_dim]; // EMBGRD
            add_ker<acc_t, data_t>(find, grdPtr, emb_dim);
          }
        }
      }
    }
  }
  return;
}

std::tuple<std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>>
mergedemb_distribute_backward_local_kernel_impl(
    const Tensor& grad,
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
  int64_t emb_dim = grad.size(2);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(num_emb == indices.size());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(num_emb == offsets.size());

  auto index_type = indices[0].scalar_type();
  auto data_type = grad.scalar_type();
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

  const int64_t num_thd = omp_get_max_threads();

  AT_DISPATCH_FLOATING_TYPES_AND(
      at::kBFloat16,
      grad.scalar_type(),
      "mergedemb_distribute_backward_local",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            indices[0].scalar_type(),
            "mergedemb_distribute_backward_local",
            [&] {
              using acc_t = acc_type<scalar_t, true>;
              std::vector<EmbeddingRowCache<acc_t>> cache(world_size * num_thd);
              scalar_t* grad_ptr = grad.data_ptr<scalar_t>();
              index_t* indices_ptr[num_emb];
              index_t* offsets_ptr[num_emb];
              for (int i = 0; i < num_emb; i++) {
                indices_ptr[i] = indices[i].data_ptr<index_t>();
                offsets_ptr[i] = offsets[i].data_ptr<index_t>();
              }
              // read from grad and accumuate in emb grad cache
              prepare_emb_bwd_cache<acc_t, scalar_t, index_t>(
                  cache,
                  grad_ptr,
                  indices_ptr,
                  row_offset,
                  offsets_ptr,
                  global_batch_size,
                  num_emb,
                  emb_dim,
                  world_size,
                  rank,
                  last_offsets);
              // read from emb grad cache and write to the buffer while will be
              // comunicated with other ranks
              prepare_ccl_buffer<acc_t, scalar_t, index_t>(
                  idx,
                  val,
                  ofs,
                  cache,
                  world_size,
                  num_thd,
                  emb_dim,
                  indices[0].options(),
                  grad.options());
            });
      });

  return std::make_tuple(idx, val, ofs);
}

template <typename acc_t, typename data_t, typename index_t>
void mergedemb_distribute_backward_merge(
    std::vector<EmbeddingRowCache<acc_t>>& thdcache,
    const int64_t world_size,
    const int64_t emb_dim,
    index_t** idx_ptr,
    data_t** val_ptr,
    int64_t** ofs_ptr) {
  RECORD_FUNCTION(__FUNCTION__, c10::ArrayRef<c10::IValue>({}));
#pragma omp parallel shared(thdcache)
  {
    const int64_t numthd = omp_get_num_threads();
    const int64_t thdidx = omp_get_thread_num();
    EmbeddingRowCache<acc_t>& cache = thdcache[thdidx];
    for (int64_t i = 0; i < world_size; ++i) {
      int64_t ts = ofs_ptr[i][thdidx];
      int64_t te = ofs_ptr[i][thdidx + 1];
      const index_t* idx = idx_ptr[i];
      const data_t* val = val_ptr[i];
      // j is the index of current gradient being processed in exchange buffer
      // (val_ptr)
      for (int64_t j = ts; j < te; ++j) {
        const index_t emb_idx = idx[j];
        const index_t row_idx = emb_idx / world_size;
        auto find = cache.find(row_idx);
        if (find == nullptr) {
          find = cache.emplace(row_idx, emb_dim);
        }
        const data_t* grad = &val[j * emb_dim];
        add_ker<acc_t, data_t>(find, grad, emb_dim);
      }
    }
  }
}

template <typename acc_t, typename data_t>
void mergedemb_distribute_adagrad_update(
    std::vector<EmbeddingRowCache<acc_t>>& thdcache,
    data_t* weight_ptr,
    int64_t emb_dim,
    AdaGradArgs args) {
  RECORD_FUNCTION(__FUNCTION__, c10::ArrayRef<c10::IValue>({}));
#pragma omp parallel shared(thdcache)
  {
    const int64_t thdidx = omp_get_thread_num();
    EmbeddingRowCache<acc_t>& cache = thdcache[thdidx];
    EmbeddingGradUpdate<data_t, acc_t, AdaGradArgs>::update(
        weight_ptr, cache, args, /*table_id=*/0, emb_dim);
  }
}

void mergedemb_distribute_backward_merge_adagrad_update_kernel_impl(
    const TensorList& idx,
    const TensorList& val,
    const TensorList& ofs,
    Tensor& weight,
    Tensor& weight_trail,
    Tensor& hessian,
    const double lr,
    const double eps) {
  RECORD_FUNCTION(__FUNCTION__, c10::ArrayRef<c10::IValue>({}));
  int64_t world_size = idx.size();
  int64_t emb_dim = weight.size(1);
  const int64_t num_thd = omp_get_max_threads();
  // res = torch.zeros((lbatch, num_emb, emb_dim), dtype=torch.bfloat16)
  AT_DISPATCH_FLOATING_TYPES_AND(
      at::kBFloat16,
      weight.scalar_type(),
      "mergedemb_distribute_backward_merge",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            idx[0].scalar_type(), "mergedemb_distribute_backward_merge", [&] {
              using acc_t = acc_type<scalar_t, true>;
              std::vector<EmbeddingRowCache<acc_t>> cache(num_thd);
              index_t* idx_ptr[world_size];
              scalar_t* val_ptr[world_size];
              int64_t* ofs_ptr[world_size];
              for (int i = 0; i < world_size; i++) {
                idx_ptr[i] = idx[i].data_ptr<index_t>();
                val_ptr[i] = val[i].data_ptr<scalar_t>();
                ofs_ptr[i] = ofs[i].data_ptr<int64_t>();
              }
              // read from weight and accumuate in emb cache
              mergedemb_distribute_backward_merge<acc_t, scalar_t, index_t>(
                  cache, world_size, emb_dim, idx_ptr, val_ptr, ofs_ptr);
              AdaGradArgs args =
                  AdaGradArgs({weight_trail}, {hessian}, eps, lr);
              scalar_t* weight_ptr = weight.data_ptr<scalar_t>();
              mergedemb_distribute_adagrad_update<acc_t, scalar_t>(
                  cache, weight_ptr, emb_dim, args);
            });
      });

  return;
}

} // anonymous namespace

IPEX_REGISTER_DISPATCH(
    merged_embeddingbag_backward_cpu_kernel_stub,
    &merged_embeddingbag_backward_cpu_kernel_impl);

IPEX_REGISTER_DISPATCH(
    merged_embeddingbag_backward_sgd_cpu_kernel_stub,
    &merged_embeddingbag_backward_sgd_cpu_kernel_impl);

IPEX_REGISTER_DISPATCH(
    merged_embeddingbag_backward_adagrad_cpu_kernel_stub,
    &merged_embeddingbag_backward_adagrad_cpu_kernel_impl);

IPEX_REGISTER_DISPATCH(
    mergedemb_distribute_backward_local_kernel_stub,
    &mergedemb_distribute_backward_local_kernel_impl);

IPEX_REGISTER_DISPATCH(
    mergedemb_distribute_backward_merge_adagrad_update_stub,
    &mergedemb_distribute_backward_merge_adagrad_update_kernel_impl);

} // namespace cpu
} // namespace torch_ipex
