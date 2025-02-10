#pragma once

#ifdef USE_LIBXSMM
#include <dyndisp/DispatchStub.h>
#include <torch/all.h>
#include "csrc/cpu/tpp/woq/tla.h"
#include "woq_defines.h"

namespace torch_ipex {
namespace cpu {

using namespace tpp;

#define QUANT_A_THRESHOLD 30720

template <typename scalar_t>
inline scalar_t max_propagate_nan(scalar_t a, scalar_t b) {
  if (at::_isnan(a)) {
    return a;
  }
  return a > b ? a : b;
}

template <typename scalar_t>
inline scalar_t min_propagate_nan(scalar_t a, scalar_t b) {
  if (at::_isnan(a)) {
    return a;
  }
  return a < b ? a : b;
}

template <typename T>
void compute_int8_qparams_per_tensor(
    const at::Tensor& t,
    float* scale,
    int32_t* zp,
    bool is_sym_quant) {
  auto [t_min, t_max] = at::aminmax(t);
  auto min = t_min.item<float>();
  auto max = t_max.item<float>();
  min = std::min(min, 0.0f);
  max = std::max(max, 0.0f);
  *scale = is_sym_quant ? std::max(fabs(max), fabs(min)) / 127.0f
                        : (max - min) / 255.0f;
  *zp = is_sym_quant ? 0 : (int32_t)(-std::nearbyint(min / *scale));
}

template <>
inline void compute_int8_qparams_per_tensor<float>(
    const at::Tensor& t,
    float* scale,
    int32_t* zp,
    bool is_sym_quant) {
  auto in_ptr0 = t.data_ptr<float>();
  auto n = t.numel();
  auto K = t.size(-1);
  auto M = t.numel() / K;
  auto vecsize = at::vec::Vectorized<float>::size();
  auto compute_block = [&](float* in_ptr, int start, int end) {
    float min_val = std::numeric_limits<float>::infinity();
    float max_val = -std::numeric_limits<float>::infinity();
    auto min_vec = at::vec::Vectorized(min_val);
    auto max_vec = at::vec::Vectorized(max_val);
    int i1;
    for (i1 = start; i1 < end / n * n; i1 += vecsize) {
      auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr + i1, vecsize);
      min_vec = at::vec::minimum(min_vec, tmp0);
      max_vec = at::vec::maximum(tmp0, max_vec);
    }
    for (; i1 < end; i1++) {
      auto tmp0 = in_ptr[i1];
      min_val = std::min(min_val, tmp0);
      max_val = std::max(tmp0, max_val);
    }
    min_val = min_propagate_nan(
        min_val,
        at::vec::vec_reduce_all<float>(
            [](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) {
              return at::vec::minimum(x, y);
            },
            min_vec));
    max_val = max_propagate_nan(
        max_val,
        at::vec::vec_reduce_all<float>(
            [](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) {
              return at::vec::maximum(x, y);
            },
            max_vec));
    return std::make_pair(min_val, max_val);
  };
  if (n > QUANT_A_THRESHOLD) {
    int num_threads = omp_get_max_threads();
    int vec_per_thread = std::ceil((float)n / vecsize / num_threads);
    int thread_used = std::ceil((float)n / vecsize / vec_per_thread);
    float min_vals[thread_used];
    float max_vals[thread_used];
#pragma omp parallel for
    for (int i0 = 0; i0 < n; i0 += vec_per_thread * vecsize) {
      auto vec_start = i0;
      auto vec_end = std::min(i0 + vec_per_thread * vecsize, (int)n);
      auto [min_val, max_val] = compute_block(in_ptr0, vec_start, vec_end);
      min_vals[i0 / vec_per_thread / vecsize] = min_val;
      max_vals[i0 / vec_per_thread / vecsize] = max_val;
    }
    auto min_elem_ptr = std::min_element(min_vals, min_vals + thread_used);
    auto max_elem_ptr = std::max_element(max_vals, max_vals + thread_used);
    *scale = is_sym_quant
        ? std::max(fabs(*max_elem_ptr), fabs(*min_elem_ptr)) / 127.0f
        : (*max_elem_ptr - *min_elem_ptr) / 255.0f;
    *zp = is_sym_quant ? 0 : (int32_t)(-std::nearbyint(*min_elem_ptr / *scale));
  } else {
    auto [min_val, max_val] = compute_block(in_ptr0, 0, n);
    *scale = is_sym_quant ? std::max(fabs(max_val), fabs(min_val)) / 127.0f
                          : (max_val - min_val) / 255.0f;
    *zp = is_sym_quant ? 0 : (int32_t)(-std::nearbyint(min_val / *scale));
  }
}

template <>
inline void compute_int8_qparams_per_tensor<bfloat16>(
    const at::Tensor& t,
    float* scale,
    int32_t* zp,
    bool is_sym_quant) {
  auto in_ptr0 = t.data_ptr<at::BFloat16>();
  auto n = t.numel();
  auto K = t.size(-1);
  auto M = t.numel() / K;
  auto vecsize = at::vec::Vectorized<float>::size();
  auto compute_block = [&](at::BFloat16* in_ptr, int start, int end) {
    float min_val = std::numeric_limits<float>::infinity();
    float max_val = -std::numeric_limits<float>::infinity();
    auto min_vec = at::vec::Vectorized(min_val);
    auto max_vec = at::vec::Vectorized(max_val);
    int i1;
    for (i1 = start; i1 < end / n * n; i1 += vecsize) {
      auto tmp0 =
          at::vec::Vectorized<at::BFloat16>::loadu(in_ptr + i1, vecsize);
      at::vec::Vectorized<float> res_vec1(0);
      at::vec::Vectorized<float> res_vec2(0);
      std::tie(res_vec1, res_vec2) = at::vec::convert_bfloat16_float(tmp0);
      min_vec = at::vec::minimum(min_vec, res_vec1);
      max_vec = at::vec::maximum(res_vec1, max_vec);
    }
    for (; i1 < end; i1++) {
      auto tmp0 = in_ptr[i1];
      min_val = std::min(min_val, (float)tmp0);
      max_val = std::max((float)tmp0, max_val);
    }
    min_val = min_propagate_nan(
        min_val,
        at::vec::vec_reduce_all<float>(
            [](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) {
              return at::vec::minimum(x, y);
            },
            min_vec));
    max_val = max_propagate_nan(
        max_val,
        at::vec::vec_reduce_all<float>(
            [](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) {
              return at::vec::maximum(x, y);
            },
            max_vec));
    return std::make_pair(min_val, max_val);
  };
  if (n > QUANT_A_THRESHOLD) {
    int num_threads = omp_get_max_threads();
    int vec_per_thread = std::ceil((float)n / vecsize / num_threads);
    int thread_used = std::ceil((float)n / vecsize / vec_per_thread);
    float min_vals[thread_used];
    float max_vals[thread_used];
#pragma omp parallel for
    for (int i0 = 0; i0 < n; i0 += vec_per_thread * vecsize) {
      auto vec_start = i0;
      auto vec_end = std::min(i0 + vec_per_thread * vecsize, (int)n);
      auto [min_val, max_val] = compute_block(in_ptr0, vec_start, vec_end);
      min_vals[i0 / vec_per_thread / vecsize] = min_val;
      max_vals[i0 / vec_per_thread / vecsize] = max_val;
    }
    auto min_elem_ptr = std::min_element(min_vals, min_vals + thread_used);
    auto max_elem_ptr = std::max_element(max_vals, max_vals + thread_used);
    *scale = is_sym_quant
        ? std::max(fabs(*max_elem_ptr), fabs(*min_elem_ptr)) / 127.0f
        : (*max_elem_ptr - *min_elem_ptr) / 255.0f;
    *zp = is_sym_quant ? 0 : (int32_t)(-std::nearbyint(*min_elem_ptr / *scale));
  } else {
    auto [min_val, max_val] = compute_block(in_ptr0, 0, n);
    *scale = is_sym_quant ? std::max(fabs(max_val), fabs(min_val)) / 127.0f
                          : (max_val - min_val) / 255.0f;
    *zp = is_sym_quant ? 0 : (int32_t)(-std::nearbyint(min_val / *scale));
  }
}

template <typename T>
std::pair<at::Tensor, at::Tensor> compute_int8_qparams_per_block(
    const at::Tensor& t,
    int quant_block_k,
    int quant_a_mode,
    bool is_sym_quant) {
  auto K = t.size(-1);
  auto n = t.numel();
  auto M = n / K;
  auto t_reshape = t.reshape({M, K});
  if (quant_a_mode == QUANT_A_PER_M || quant_a_mode == QUANT_A_PER_M_SYM) {
    auto grouped_min = std::get<0>(t_reshape.min(-1));
    auto grouped_max = std::get<0>(t_reshape.max(-1));
    auto zeros = at::zeros_like(grouped_min);
    auto min = quant_a_mode == QUANT_A_PER_M ? at::minimum(grouped_min, zeros)
                                             : grouped_min;
    auto max = quant_a_mode == QUANT_A_PER_M ? at::maximum(grouped_max, zeros)
                                             : grouped_max;
    auto scales = is_sym_quant
        ? at::maximum(at::absolute(max), at::absolute(min)) / 127.0f
        : (max - min) / 255.0f;
    auto zps = is_sym_quant ? at::Tensor() : -at::round(min / scales);
    return std::make_pair<at::Tensor&&, at::Tensor&&>(
        std::move(scales.to(c10::kFloat)),
        std::move(is_sym_quant ? zps : zps.to(c10::kInt)));
  }
  int k_rem = K % quant_block_k;
  int block_k = quant_block_k;
  auto grouped =
      t_reshape
          .index({at::indexing::Slice(), at::indexing::Slice(0, K - k_rem)})
          .view({M, K / quant_block_k, quant_block_k});
  at::Tensor grouped_min, grouped_max;
  if (quant_a_mode == QUANT_A_PER_K_BLOCK ||
      quant_a_mode == QUANT_A_PER_K_BLOCK_SYM) {
    grouped_min = std::get<0>(std::get<0>(grouped.min(-1)).min(0));
    grouped_max = std::get<0>(std::get<0>(grouped.max(-1)).max(0));
  } else {
    grouped_min = std::get<0>(grouped.min(-1));
    grouped_max = std::get<0>(grouped.max(-1));
  }
  auto zeros = at::zeros_like(grouped_min);
  auto min = at::minimum(grouped_min, zeros);
  auto max = at::maximum(grouped_max, zeros);
  auto scales = is_sym_quant
      ? at::maximum(at::absolute(max), at::absolute(min)) / 127.0f
      : (max - min) / 255.0f;
  auto zps = is_sym_quant ? at::Tensor() : -at::round(min / scales);
  if (k_rem) {
    auto grouped_rem =
        t_reshape
            .index({at::indexing::Slice(), at::indexing::Slice(K - k_rem, K)})
            .view({M, 1, k_rem});
    at::Tensor grouped_rem_min, grouped_rem_max;
    if (quant_a_mode == QUANT_A_PER_K_BLOCK ||
        quant_a_mode == QUANT_A_PER_K_BLOCK_SYM) {
      grouped_rem_min = std::get<0>(std::get<0>(grouped_rem.min(-1)).min(0));
      grouped_rem_max = std::get<0>(std::get<0>(grouped_rem.max(-1)).max(0));
    } else {
      grouped_rem_min = std::get<0>(grouped_rem.min(-1));
      grouped_rem_max = std::get<0>(grouped_rem.max(-1));
    }
    auto min_rem = at::minimum(grouped_rem_min, at::tensor({0}));
    auto max_rem = at::maximum(grouped_rem_max, at::tensor({0}));
    auto scales_rem = is_sym_quant
        ? at::maximum(at::absolute(max_rem), at::absolute(min_rem)) / 127.0f
        : (max_rem - min_rem) / 255.0f;
    auto zps_rem =
        is_sym_quant ? at::Tensor() : -at::round(min_rem / scales_rem);
    scales = at::cat({scales, scales_rem}, -1).contiguous();
    zps =
        is_sym_quant ? at::Tensor() : at::cat({zps, zps_rem}, -1).contiguous();
  }
  return std::make_pair<at::Tensor&&, at::Tensor&&>(
      std::move(scales.to(c10::kFloat)),
      std::move(is_sym_quant ? zps : zps.to(c10::kInt)));
}

template <>
inline std::pair<at::Tensor, at::Tensor> compute_int8_qparams_per_block<
    bfloat16>(
    const at::Tensor& t,
    int quant_block_k,
    int quant_a_mode,
    bool is_sym_quant) {
  auto in_ptr = t.data_ptr<at::BFloat16>();
  int K = t.size(-1);
  int n = t.numel();
  int M = n / K;
  int Kc = (K + quant_block_k - 1) / quant_block_k;
  auto vecsize = at::vec::Vectorized<float>::size();
  at::Tensor scales, zps;
  if (quant_a_mode == QUANT_A_PER_K_BLOCK ||
      quant_a_mode == QUANT_A_PER_K_BLOCK_SYM) {
    scales = at::empty({Kc}, t.options().dtype(at::kFloat));
    zps = at::empty({Kc}, t.options().dtype(at::kInt));
  } else if (
      quant_a_mode == QUANT_A_PER_M || quant_a_mode == QUANT_A_PER_M_SYM) {
    scales = at::empty({M}, t.options().dtype(at::kFloat));
    zps = at::empty({M}, t.options().dtype(at::kInt));
  } else {
    scales = at::empty({M, Kc}, t.options().dtype(at::kFloat));
    zps = at::empty({M, Kc}, t.options().dtype(at::kInt));
  }
  auto scales_ptr = scales.data_ptr<float>();
  auto zps_ptr = is_sym_quant ? nullptr : zps.data_ptr<int32_t>();
  auto compute_minmax = [vecsize, scales_ptr, zps_ptr](
                            at::BFloat16* ptr,
                            int M,
                            int K,
                            int scale_offset,
                            int zp_offset,
                            int ld,
                            bool is_sym_quant) {
    float min_val = std::numeric_limits<float>::infinity();
    float max_val = -std::numeric_limits<float>::infinity();
    auto in_ptr_ = ptr;
    auto min_vec = at::vec::Vectorized(min_val);
    auto max_vec = at::vec::Vectorized(max_val);
    for (int m = 0; m < M; m++) {
      auto in_ptr0 = in_ptr_;
      int k;
      for (k = 0; k < K / vecsize * vecsize; k += vecsize) {
        auto tmp0 = at::vec::Vectorized<at::BFloat16>::loadu(in_ptr0, vecsize);
        at::vec::Vectorized<float> res_vec1(0);
        at::vec::Vectorized<float> res_vec2(0);
        std::tie(res_vec1, res_vec2) = at::vec::convert_bfloat16_float(tmp0);
        auto tmp1 = res_vec1;
        min_vec = at::vec::minimum(min_vec, tmp1);
        max_vec = at::vec::maximum(tmp1, max_vec);
        in_ptr0 += vecsize;
      }
      for (; k < K; k++) {
        auto tmp0 = in_ptr0[k];
        min_val = std::min(min_val, (float)tmp0);
        max_val = std::max(max_val, (float)tmp0);
      }
      in_ptr_ += ld;
    }
    min_val = min_propagate_nan(
        min_val,
        at::vec::vec_reduce_all<float>(
            [](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) {
              return at::vec::minimum(x, y);
            },
            min_vec));
    max_val = max_propagate_nan(
        max_val,
        at::vec::vec_reduce_all<float>(
            [](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) {
              return at::vec::maximum(x, y);
            },
            max_vec));
    scales_ptr[scale_offset] = is_sym_quant
        ? std::max(fabs(max_val), fabs(min_val)) / 128.0f
        : (max_val - min_val) / 255.0f;
    if (!is_sym_quant) {
      zps_ptr[zp_offset] =
          (int32_t)(-std::nearbyint(min_val / scales_ptr[scale_offset]));
    }
  };
  if (quant_a_mode == QUANT_A_PER_K_BLOCK ||
      quant_a_mode == QUANT_A_PER_K_BLOCK_SYM) {
#pragma omp parallel for
    for (int kc = 0; kc < Kc; kc++) {
      int offset = kc * quant_block_k;
      int block_k = std::min(quant_block_k, K - offset);
      compute_minmax(in_ptr + offset, M, block_k, kc, kc, K, is_sym_quant);
    }
  } else if (
      quant_a_mode == QUANT_A_PER_M || quant_a_mode == QUANT_A_PER_M_SYM) {
#pragma omp parallel for
    for (int m = 0; m < M; m++) {
      int offset = m * K;
      compute_minmax(in_ptr + offset, 1, K, m, m, K, is_sym_quant);
    }
  } else {
#pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++) {
      for (int kc = 0; kc < Kc; kc++) {
        auto in_ptr0 = in_ptr + m * K + kc * quant_block_k;
        auto scale_offset = m * Kc + kc;
        auto zp_offset = m * Kc + kc;
        int block_k = std::min(quant_block_k, K - kc * quant_block_k);
        compute_minmax(
            in_ptr0, 1, block_k, scale_offset, zp_offset, K, is_sym_quant);
      }
    }
  }
  return std::make_pair<at::Tensor&&, at::Tensor&&>(
      std::move(scales), std::move(zps));
}

template <>
inline std::pair<at::Tensor, at::Tensor> compute_int8_qparams_per_block<float>(
    const at::Tensor& t,
    int quant_block_k,
    int quant_a_mode,
    bool is_sym_quant) {
  auto in_ptr = t.data_ptr<float>();
  int K = t.size(-1);
  int n = t.numel();
  int M = n / K;
  int Kc = (K + quant_block_k - 1) / quant_block_k;
  auto vecsize = at::vec::Vectorized<float>::size();
  at::Tensor scales, zps;
  if (quant_a_mode == QUANT_A_PER_K_BLOCK ||
      quant_a_mode == QUANT_A_PER_K_BLOCK_SYM) {
    scales = at::empty({Kc}, t.options().dtype(at::kFloat));
    zps = at::empty({Kc}, t.options().dtype(at::kInt));
  } else if (
      quant_a_mode == QUANT_A_PER_M || quant_a_mode == QUANT_A_PER_M_SYM) {
    scales = at::empty({M}, t.options().dtype(at::kFloat));
    zps = at::empty({M}, t.options().dtype(at::kInt));
  } else {
    scales = at::empty({M, Kc}, t.options().dtype(at::kFloat));
    zps = at::empty({M, Kc}, t.options().dtype(at::kInt));
  }
  auto scales_ptr = scales.data_ptr<float>();
  auto zps_ptr = is_sym_quant ? nullptr : zps.data_ptr<int32_t>();
  auto compute_minmax = [vecsize, scales_ptr, zps_ptr](
                            float* ptr,
                            int M,
                            int K,
                            int scale_offset,
                            int zp_offset,
                            int ld,
                            bool is_sym_quant) {
    float min_val = std::numeric_limits<float>::infinity();
    float max_val = -std::numeric_limits<float>::infinity();
    auto in_ptr_ = ptr;
    auto min_vec = at::vec::Vectorized(min_val);
    auto max_vec = at::vec::Vectorized(max_val);
    for (int m = 0; m < M; m++) {
      auto in_ptr0 = in_ptr_;
      int k;
      for (k = 0; k < K / vecsize * vecsize; k += vecsize) {
        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0, vecsize);
        min_vec = at::vec::minimum(min_vec, tmp0);
        max_vec = at::vec::maximum(tmp0, max_vec);
        in_ptr0 += vecsize;
      }
      for (; k < K; k++) {
        auto tmp0 = in_ptr0[k];
        min_val = std::min(min_val, tmp0);
        max_val = std::max(max_val, tmp0);
      }
      in_ptr_ += ld;
    }
    min_val = min_propagate_nan(
        min_val,
        at::vec::vec_reduce_all<float>(
            [](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) {
              return at::vec::minimum(x, y);
            },
            min_vec));
    max_val = max_propagate_nan(
        max_val,
        at::vec::vec_reduce_all<float>(
            [](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) {
              return at::vec::maximum(x, y);
            },
            max_vec));
    scales_ptr[scale_offset] = is_sym_quant
        ? std::max(fabs(max_val), fabs(min_val)) / 128.0f
        : (max_val - min_val) / 255.0f;
    if (!is_sym_quant) {
      zps_ptr[zp_offset] =
          (int32_t)(-std::nearbyint(min_val / scales_ptr[scale_offset]));
    }
  };
  if (quant_a_mode == QUANT_A_PER_K_BLOCK ||
      quant_a_mode == QUANT_A_PER_K_BLOCK_SYM) {
#pragma omp parallel for
    for (int kc = 0; kc < Kc; kc++) {
      int offset = kc * quant_block_k;
      int block_k = std::min(quant_block_k, K - offset);
      compute_minmax(in_ptr + offset, M, block_k, kc, kc, K, is_sym_quant);
    }
  } else if (
      quant_a_mode == QUANT_A_PER_M || quant_a_mode == QUANT_A_PER_M_SYM) {
#pragma omp parallel for
    for (int m = 0; m < M; m++) {
      int offset = m * K;
      compute_minmax(in_ptr + offset, 1, K, m, m, K, is_sym_quant);
    }
  } else {
#pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++) {
      for (int kc = 0; kc < Kc; kc++) {
        auto in_ptr0 = in_ptr + m * K + kc * quant_block_k;
        auto scale_offset = m * Kc + kc;
        auto zp_offset = m * Kc + kc;
        int block_k = std::min(quant_block_k, K - kc * quant_block_k);
        compute_minmax(
            in_ptr0, 1, block_k, scale_offset, zp_offset, K, is_sym_quant);
      }
    }
  }
  return std::make_pair<at::Tensor&&, at::Tensor&&>(
      std::move(scales), std::move(zps));
}

template <typename T>
at::Tensor quantize_per_tensor(
    const at::Tensor& t,
    float scale,
    int32_t zp,
    bool is_sym_quant) {
  // TODO(jgong5): optimize me
  auto t_q = is_sym_quant ? t / scale : t / scale + zp;
  t_q = is_sym_quant ? at::clamp(at::round(t_q), -128, 127)
                     : at::clamp(at::round(t_q), 0, 255);
  return is_sym_quant ? t_q.to(at::kChar) : t_q.to(at::kByte);
}

template <>
inline at::Tensor quantize_per_tensor<float>(
    const at::Tensor& t,
    float scale,
    int32_t zp,
    bool is_sym_quant) {
#if defined(CPU_CAPABILITY_AVX512)
  auto out_dtype = is_sym_quant ? at::kChar : at::kByte;
  at::Tensor out = at::empty_like(t, out_dtype);
  auto in_ptr0 = t.data_ptr<float>();
  uint8_t* out_ptr0 = is_sym_quant ? nullptr : out.data_ptr<uint8_t>();
  int8_t* out_sym_ptr0 = is_sym_quant ? out.data_ptr<int8_t>() : nullptr;
  auto n = t.numel();
  auto K = t.size(-1);
  auto M = t.numel() / K;
  auto vecsize = at::vec::Vectorized<float>::size();
  auto quantize_block =
      [vecsize, scale, zp](
          float* in_ptr, int start, int end, uint8_t* out_ptr) {
        int i1;
        for (i1 = start; i1 < end / vecsize * vecsize; i1 += vecsize) {
          auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr + i1, vecsize);
          auto tmp1 =
              tmp0 / at::vec::Vectorized<float>(static_cast<float>(scale));
          auto tmp2 = tmp1 + at::vec::Vectorized<float>(static_cast<float>(zp));
          auto tmp3 = tmp2.round();
          auto tmp4 = (tmp3);
          auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.0));
          auto tmp6 = at::vec::maximum(tmp4, tmp5);
          auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(255.0));
          auto tmp8 = at::vec::minimum(tmp6, tmp7);
          auto tmp9 = (tmp8);
          auto tmp10 = at::vec::convert_float_to_int8<uint8_t>(tmp9);
          tmp10.store(out_ptr + i1, vecsize);
        }
        for (; i1 < end; i1++) {
          auto tmp0 = in_ptr[i1];
          auto tmp1 = tmp0 / static_cast<float>(scale);
          auto tmp2 = tmp1 + static_cast<float>(zp);
          auto tmp3 = std::nearbyint(tmp2);
          auto tmp4 = static_cast<float>(tmp3);
          auto tmp5 = static_cast<float>(0.0);
          auto tmp6 = 0;
          if (at::_isnan(tmp4)) {
            tmp6 = tmp4;
          }
          tmp6 = tmp4 > tmp5 ? tmp4 : tmp5;
          auto tmp7 = static_cast<float>(255.0);
          auto tmp8 = 0;
          if (at::_isnan(tmp6)) {
            tmp8 = tmp6;
          }
          tmp8 = tmp6 < tmp7 ? tmp6 : tmp7;
          auto tmp9 = static_cast<float>(tmp8);
          auto tmp10 = static_cast<unsigned char>(tmp9);
          out_ptr[i1] = tmp10;
        }
      };
  auto quantize_block_sym =
      [vecsize, scale, zp](float* in_ptr, int start, int end, int8_t* out_ptr) {
        int i1;
        for (i1 = start; i1 < end / vecsize * vecsize; i1 += vecsize) {
          auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr + i1, vecsize);
          auto tmp1 =
              tmp0 / at::vec::Vectorized<float>(static_cast<float>(scale));
          auto tmp2 = tmp1 + at::vec::Vectorized<float>(static_cast<float>(zp));
          auto tmp3 = tmp2.round();
          auto tmp4 = (tmp3);
          auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(-128.0));
          auto tmp6 = at::vec::maximum(tmp4, tmp5);
          auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(127.0));
          auto tmp8 = at::vec::minimum(tmp6, tmp7);
          auto tmp9 = (tmp8);
          auto tmp10 = at::vec::convert_float_to_int8<int8_t>(tmp9);
          tmp10.store(out_ptr + i1, vecsize);
        }
        for (; i1 < end; i1++) {
          auto tmp0 = in_ptr[i1];
          auto tmp1 = tmp0 / static_cast<float>(scale);
          auto tmp2 = tmp1 + static_cast<float>(zp);
          auto tmp3 = std::nearbyint(tmp2);
          auto tmp4 = static_cast<float>(tmp3);
          auto tmp5 = static_cast<float>(-128.0);
          auto tmp6 = 0;
          if (at::_isnan(tmp4)) {
            tmp6 = tmp4;
          }
          tmp6 = tmp4 > tmp5 ? tmp4 : tmp5;
          auto tmp7 = static_cast<float>(127.0);
          auto tmp8 = 0;
          if (at::_isnan(tmp6)) {
            tmp8 = tmp6;
          }
          tmp8 = tmp6 < tmp7 ? tmp6 : tmp7;
          auto tmp9 = static_cast<float>(tmp8);
          auto tmp10 = static_cast<int8_t>(tmp9);
          out_ptr[i1] = tmp10;
        }
      };
  if (n > QUANT_A_THRESHOLD) {
    int num_threads = omp_get_max_threads();
    int vec_per_thread = std::ceil((float)n / vecsize / num_threads);
#pragma omp parallel for
    for (int i0 = 0; i0 < n; i0 += vec_per_thread * vecsize) {
      auto vec_start = i0;
      auto vec_end = std::min(i0 + vec_per_thread * vecsize, (int)n);
      if (is_sym_quant) {
        quantize_block_sym(in_ptr0, vec_start, vec_end, out_sym_ptr0);
      } else {
        quantize_block(in_ptr0, vec_start, vec_end, out_ptr0);
      }
    }
  } else {
    if (is_sym_quant) {
      quantize_block_sym(in_ptr0, 0, n, out_sym_ptr0);
    } else {
      quantize_block(in_ptr0, 0, n, out_ptr0);
    }
  }
  return out;
#else
  return at::quantize_per_tensor(t, scale, zp, c10::kQUInt8);
#endif
}

template <>
inline at::Tensor quantize_per_tensor<bfloat16>(
    const at::Tensor& t,
    float scale,
    int32_t zp,
    bool is_sym_quant) {
#if defined(CPU_CAPABILITY_AVX512)
  auto out_dtype = is_sym_quant ? at::kChar : at::kByte;
  at::Tensor out = at::empty_like(t, out_dtype);
  auto in_ptr0 = t.data_ptr<at::BFloat16>();
  uint8_t* out_ptr0 = is_sym_quant ? nullptr : out.data_ptr<uint8_t>();
  int8_t* out_sym_ptr0 = is_sym_quant ? out.data_ptr<int8_t>() : nullptr;
  auto n = t.numel();
  auto K = t.size(-1);
  auto M = t.numel() / K;
  auto vecsize = at::vec::Vectorized<float>::size();
  auto quantize_block =
      [vecsize, scale, zp](
          at::BFloat16* in_ptr, int start, int end, uint8_t* out_ptr) {
        int i1;
        for (i1 = start; i1 < end / vecsize * vecsize; i1 += vecsize) {
          auto tmp0 =
              at::vec::Vectorized<at::BFloat16>::loadu(in_ptr + i1, vecsize);
          at::vec::Vectorized<float> res_vec1(0);
          at::vec::Vectorized<float> res_vec2(0);
          std::tie(res_vec1, res_vec2) = at::vec::convert_bfloat16_float(tmp0);
          auto tmp1 = res_vec1;
          auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(scale));
          auto tmp3 = tmp1 / tmp2;
          auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(zp));
          auto tmp5 = tmp3 + tmp4;
          auto tmp6 = tmp5.round();
          auto tmp7 = (tmp6);
          auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.0));
          auto tmp9 = at::vec::maximum(tmp7, tmp8);
          auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(255.0));
          auto tmp11 = at::vec::minimum(tmp9, tmp10);
          auto tmp12 = (tmp11);
          auto tmp13 = at::vec::convert_float_to_int8<uint8_t>(tmp12);
          tmp13.store(out_ptr + i1, vecsize);
        }
        for (; i1 < end; i1++) {
          auto tmp0 = in_ptr[i1];
          auto tmp1 = static_cast<float>(tmp0);
          auto tmp2 = static_cast<float>(scale);
          auto tmp3 = tmp1 / tmp2;
          auto tmp4 = static_cast<float>(zp);
          auto tmp5 = tmp3 + tmp4;
          auto tmp6 = std::nearbyint(tmp5);
          auto tmp7 = static_cast<float>(tmp6);
          auto tmp8 = static_cast<float>(0.0);
          auto tmp9 = 0;
          if (at::_isnan(tmp7)) {
            tmp9 = tmp7;
          }
          tmp9 = tmp7 > tmp8 ? tmp7 : tmp8;
          auto tmp10 = static_cast<float>(255.0);
          auto tmp11 = 0;
          if (at::_isnan(tmp9)) {
            tmp11 = tmp9;
          }
          tmp11 = tmp9 < tmp10 ? tmp9 : tmp10;
          auto tmp12 = static_cast<float>(tmp11);
          auto tmp13 = static_cast<unsigned char>(tmp12);
          out_ptr[i1] = tmp13;
        }
      };
  auto quantize_block_sym =
      [vecsize, scale, zp](
          at::BFloat16* in_ptr, int start, int end, int8_t* out_ptr) {
        int i1;
        for (i1 = start; i1 < end / vecsize * vecsize; i1 += vecsize) {
          auto tmp0 =
              at::vec::Vectorized<at::BFloat16>::loadu(in_ptr + i1, vecsize);
          at::vec::Vectorized<float> res_vec1(0);
          at::vec::Vectorized<float> res_vec2(0);
          std::tie(res_vec1, res_vec2) = at::vec::convert_bfloat16_float(tmp0);
          auto tmp1 = res_vec1;
          auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(scale));
          auto tmp3 = tmp1 / tmp2;
          auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(zp));
          auto tmp5 = tmp3 + tmp4;
          auto tmp6 = tmp5.round();
          auto tmp7 = (tmp6);
          auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(-128.0));
          auto tmp9 = at::vec::maximum(tmp7, tmp8);
          auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(127.0));
          auto tmp11 = at::vec::minimum(tmp9, tmp10);
          auto tmp12 = (tmp11);
          auto tmp13 = at::vec::convert_float_to_int8<int8_t>(tmp12);
          tmp13.store(out_ptr + i1, vecsize);
        }
        for (; i1 < end; i1++) {
          auto tmp0 = in_ptr[i1];
          auto tmp1 = static_cast<float>(tmp0);
          auto tmp2 = static_cast<float>(scale);
          auto tmp3 = tmp1 / tmp2;
          auto tmp4 = static_cast<float>(zp);
          auto tmp5 = tmp3 + tmp4;
          auto tmp6 = std::nearbyint(tmp5);
          auto tmp7 = static_cast<float>(tmp6);
          auto tmp8 = static_cast<float>(-128.0);
          auto tmp9 = 0;
          if (at::_isnan(tmp7)) {
            tmp9 = tmp7;
          }
          tmp9 = tmp7 > tmp8 ? tmp7 : tmp8;
          auto tmp10 = static_cast<float>(127.0);
          auto tmp11 = 0;
          if (at::_isnan(tmp9)) {
            tmp11 = tmp9;
          }
          tmp11 = tmp9 < tmp10 ? tmp9 : tmp10;
          auto tmp12 = static_cast<float>(tmp11);
          auto tmp13 = static_cast<int8_t>(tmp12);
          out_ptr[i1] = tmp13;
        }
      };
  if (n > QUANT_A_THRESHOLD) {
    auto num_threads = omp_get_max_threads();
    int vec_per_thread = std::ceil((float)n / vecsize / num_threads);
#pragma omp parallel for
    for (int i0 = 0; i0 < n; i0 += vec_per_thread * vecsize) {
      auto vec_start = i0;
      auto vec_end = std::min(i0 + vec_per_thread * vecsize, (int)n);
      if (is_sym_quant) {
        quantize_block_sym(in_ptr0, vec_start, vec_end, out_sym_ptr0);
      } else {
        quantize_block(in_ptr0, vec_start, vec_end, out_ptr0);
      }
    }
  } else {
    if (is_sym_quant) {
      quantize_block_sym(in_ptr0, 0, n, out_sym_ptr0);
    } else {
      quantize_block(in_ptr0, 0, n, out_ptr0);
    }
  }
  return out;
#else
  return at::quantize_per_tensor(t.to(c10::kFloat), scale, zp, c10::kQUInt8);
#endif
}

template <typename T>
at::Tensor quantize_per_block(
    const at::Tensor& t,
    const at::Tensor& scale,
    const at::Tensor& zp,
    int quant_block_k,
    int quant_a_mode,
    bool is_sym_quant) {
  auto K = t.size(-1);
  auto n = t.numel();
  auto M = n / K;
  auto k_rem = K % quant_block_k;
  bool has_rem = k_rem != 0;
  auto K_padded = has_rem ? K + quant_block_k - k_rem : K;
  auto t_padded = has_rem
      ? at::cat(
            {t.reshape({M, K}),
             at::zeros({M, quant_block_k - k_rem}, t.options())},
            -1)
      : t;
  auto grouped = t_padded.view({-1, K_padded / quant_block_k, quant_block_k});
  at::Tensor out;
  if (quant_a_mode == QUANT_A_PER_K_BLOCK) {
    out = at::clamp(
        at::round(grouped / scale.unsqueeze(1)) + zp.unsqueeze(1), 0, 255);
  } else if (quant_a_mode == QUANT_A_PER_K_BLOCK_SYM) {
    out = at::clamp(at::round(grouped / scale.unsqueeze(1)), -128, 127);
  } else if (quant_a_mode == QUANT_A_PER_M) {
    out = at::clamp(
        at::round(grouped / scale.unsqueeze(1).unsqueeze(2)) +
            zp.unsqueeze(1).unsqueeze(2),
        0,
        255);
  } else if (quant_a_mode == QUANT_A_PER_M_SYM) {
    out = at::clamp(
        at::round(grouped / scale.unsqueeze(1).unsqueeze(2)), -128, 127);
  } else {
    out = is_sym_quant
        ? at::clamp(at::round(grouped / scale.unsqueeze(-1)), -128, 127)
        : at::clamp(
              at::round(grouped / scale.unsqueeze(-1)) + zp.unsqueeze(-1),
              0,
              255);
  }
  out = out.view({-1, K_padded})
            .index({at::indexing::Slice(), at::indexing::Slice(0, K)});
  return is_sym_quant ? out.to(at::kChar).contiguous()
                      : out.to(at::kByte).contiguous();
}

template <>
inline at::Tensor quantize_per_block<bfloat16>(
    const at::Tensor& t,
    const at::Tensor& scale,
    const at::Tensor& zp,
    int quant_block_k,
    int quant_a_mode,
    bool is_sym_quant) {
  int K = t.size(-1);
  int n = t.numel();
  int M = n / K;
  auto out_dtype = is_sym_quant ? at::kChar : at::kByte;
  at::Tensor out = at::empty_like(t, out_dtype);
  uint8_t* out_ptr = is_sym_quant ? nullptr : out.data_ptr<uint8_t>();
  int8_t* out_sym_ptr = is_sym_quant ? out.data_ptr<int8_t>() : nullptr;
  int Kc = (K + quant_block_k - 1) / quant_block_k;
  auto scale_ptr = scale.data_ptr<float>();
  auto zp_ptr = zp.data_ptr<int32_t>();
  auto in_ptr = t.data_ptr<at::BFloat16>();
  auto vecsize = at::vec::Vectorized<float>::size();
  auto quantize_block = [vecsize](
                            at::BFloat16* in_ptr,
                            uint8_t* out_ptr,
                            int block_k,
                            float scale_,
                            int zp_) {
    int k;
    int k_limit = block_k / vecsize * vecsize;
    for (k = 0; k < k_limit; k += vecsize) {
      auto in_ptr0 = in_ptr + k;
      auto out_ptr0 = out_ptr + k;
      auto tmp0 = at::vec::Vectorized<at::BFloat16>::loadu(in_ptr0, vecsize);
      at::vec::Vectorized<float> res_vec1(0);
      at::vec::Vectorized<float> res_vec2(0);
      std::tie(res_vec1, res_vec2) = at::vec::convert_bfloat16_float(tmp0);
      auto tmp1 = res_vec1;
      auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(scale_));
      auto tmp3 = tmp1 / tmp2;
      auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(zp_));
      auto tmp5 = tmp3 + tmp4;
      auto tmp6 = tmp5.round();
      auto tmp7 = (tmp6);
      auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.0));
      auto tmp9 = at::vec::maximum(tmp7, tmp8);
      auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(255.0));
      auto tmp11 = at::vec::minimum(tmp9, tmp10);
      auto tmp12 = (tmp11);
      auto tmp13 = at::vec::convert_float_to_int8<uint8_t>(tmp12);
      tmp13.store(out_ptr0, vecsize);
    }
    for (; k < block_k; k++) {
      auto tmp0 = in_ptr[k];
      auto tmp1 = static_cast<float>(tmp0);
      auto tmp2 = static_cast<float>(scale_);
      auto tmp3 = tmp1 / tmp2;
      auto tmp4 = static_cast<float>(zp_);
      auto tmp5 = tmp3 + tmp4;
      auto tmp6 = std::nearbyint(tmp5);
      auto tmp7 = static_cast<float>(tmp6);
      auto tmp8 = static_cast<float>(0.0);
      auto tmp9 = 0;
      if (at::_isnan(tmp7)) {
        tmp9 = tmp7;
      }
      tmp9 = tmp7 > tmp8 ? tmp7 : tmp8;
      auto tmp10 = static_cast<float>(255.0);
      auto tmp11 = 0;
      if (at::_isnan(tmp9)) {
        tmp11 = tmp9;
      }
      tmp11 = tmp9 < tmp10 ? tmp9 : tmp10;
      auto tmp12 = static_cast<float>(tmp11);
      auto tmp13 = static_cast<unsigned char>(tmp12);
      out_ptr[k] = tmp13;
    }
  };
  auto quantize_block_sym = [vecsize](
                                at::BFloat16* in_ptr,
                                int8_t* out_ptr,
                                int block_k,
                                float scale_,
                                int zp_) {
    int k;
    int k_limit = block_k / vecsize * vecsize;
    for (k = 0; k < k_limit; k += vecsize) {
      auto in_ptr0 = in_ptr + k;
      auto out_ptr0 = out_ptr + k;
      auto tmp0 = at::vec::Vectorized<at::BFloat16>::loadu(in_ptr0, vecsize);
      at::vec::Vectorized<float> res_vec1(0);
      at::vec::Vectorized<float> res_vec2(0);
      std::tie(res_vec1, res_vec2) = at::vec::convert_bfloat16_float(tmp0);
      auto tmp1 = res_vec1;
      auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(scale_));
      auto tmp3 = tmp1 / tmp2;
      auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(zp_));
      auto tmp5 = tmp3 + tmp4;
      auto tmp6 = tmp5.round();
      auto tmp7 = (tmp6);
      auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(-128.0));
      auto tmp9 = at::vec::maximum(tmp7, tmp8);
      auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(127.0));
      auto tmp11 = at::vec::minimum(tmp9, tmp10);
      auto tmp12 = (tmp11);
      auto tmp13 = at::vec::convert_float_to_int8<int8_t>(tmp12);
      tmp13.store(out_ptr0, vecsize);
    }
    for (; k < block_k; k++) {
      auto tmp0 = in_ptr[k];
      auto tmp1 = static_cast<float>(tmp0);
      auto tmp2 = static_cast<float>(scale_);
      auto tmp3 = tmp1 / tmp2;
      auto tmp4 = static_cast<float>(zp_);
      auto tmp5 = tmp3 + tmp4;
      auto tmp6 = std::nearbyint(tmp5);
      auto tmp7 = static_cast<float>(tmp6);
      auto tmp8 = static_cast<float>(-128.0);
      auto tmp9 = 0;
      if (at::_isnan(tmp7)) {
        tmp9 = tmp7;
      }
      tmp9 = tmp7 > tmp8 ? tmp7 : tmp8;
      auto tmp10 = static_cast<float>(127.0);
      auto tmp11 = 0;
      if (at::_isnan(tmp9)) {
        tmp11 = tmp9;
      }
      tmp11 = tmp9 < tmp10 ? tmp9 : tmp10;
      auto tmp12 = static_cast<float>(tmp11);
      auto tmp13 = static_cast<int8_t>(tmp12);
      out_ptr[k] = tmp13;
    }
  };
  if (quant_a_mode == QUANT_A_PER_K_BLOCK ||
      quant_a_mode == QUANT_A_PER_K_BLOCK_SYM) {
#pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++) {
      for (int kc = 0; kc < Kc; kc++) {
        auto in_ptr0 = in_ptr + m * K + kc * quant_block_k;
        auto scale_ = scale_ptr[kc];
        auto zp_ = is_sym_quant ? 0 : zp_ptr[kc];
        int block_k = std::min(quant_block_k, (int)K - kc * quant_block_k);
        if (is_sym_quant) {
          auto out_ptr0 = out_sym_ptr + m * K + kc * quant_block_k;
          quantize_block_sym(in_ptr0, out_ptr0, block_k, scale_, zp_);
        } else {
          auto out_ptr0 = out_ptr + m * K + kc * quant_block_k;
          quantize_block(in_ptr0, out_ptr0, block_k, scale_, zp_);
        }
      }
    }
  } else if (
      quant_a_mode == QUANT_A_PER_M || quant_a_mode == QUANT_A_PER_M_SYM) {
#pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++) {
      for (int kc = 0; kc < Kc; kc++) {
        auto in_ptr0 = in_ptr + m * K + kc * quant_block_k;
        auto scale_ = scale_ptr[m];
        auto zp_ = is_sym_quant ? 0 : zp_ptr[m];
        int block_k = std::min(quant_block_k, (int)K - kc * quant_block_k);
        if (is_sym_quant) {
          auto out_ptr0 = out_sym_ptr + m * K + kc * quant_block_k;
          quantize_block_sym(in_ptr0, out_ptr0, block_k, scale_, zp_);
        } else {
          auto out_ptr0 = out_ptr + m * K + kc * quant_block_k;
          quantize_block(in_ptr0, out_ptr0, block_k, scale_, zp_);
        }
      }
    }
  } else {
#pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++) {
      for (int kc = 0; kc < Kc; kc++) {
        auto in_ptr0 = in_ptr + m * K + kc * quant_block_k;
        auto scale_ = scale_ptr[m * Kc + kc];
        auto zp_ = is_sym_quant ? 0 : zp_ptr[m * Kc + kc];
        int block_k = std::min(quant_block_k, (int)K - kc * quant_block_k);
        if (is_sym_quant) {
          auto out_ptr0 = out_sym_ptr + m * K + kc * quant_block_k;
          quantize_block_sym(in_ptr0, out_ptr0, block_k, scale_, zp_);
        } else {
          auto out_ptr0 = out_ptr + m * K + kc * quant_block_k;
          quantize_block(in_ptr0, out_ptr0, block_k, scale_, zp_);
        }
      }
    }
  }

  return out;
}

template <>
inline at::Tensor quantize_per_block<float>(
    const at::Tensor& t,
    const at::Tensor& scale,
    const at::Tensor& zp,
    int quant_block_k,
    int quant_a_mode,
    bool is_sym_quant) {
  int K = t.size(-1);
  int n = t.numel();
  int M = n / K;
  auto out_dtype = is_sym_quant ? at::kChar : at::kByte;
  at::Tensor out = at::empty_like(t, out_dtype);
  uint8_t* out_ptr = is_sym_quant ? nullptr : out.data_ptr<uint8_t>();
  int8_t* out_sym_ptr = is_sym_quant ? out.data_ptr<int8_t>() : nullptr;
  int Kc = (K + quant_block_k - 1) / quant_block_k;
  auto scale_ptr = scale.data_ptr<float>();
  auto zp_ptr = zp.data_ptr<int32_t>();
  auto in_ptr = t.data_ptr<float>();
  auto vecsize = at::vec::Vectorized<float>::size();
  auto quantize_block =
      [vecsize](
          float* in_ptr, uint8_t* out_ptr, int block_k, float scale_, int zp_) {
        int k;
        for (k = 0; k < block_k / vecsize * vecsize; k += vecsize) {
          auto in_ptr0 = in_ptr + k;
          auto out_ptr0 = out_ptr + k;
          auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0, vecsize);
          auto tmp1 =
              tmp0 / at::vec::Vectorized<float>(static_cast<float>(scale_));
          auto tmp2 =
              tmp1 + at::vec::Vectorized<float>(static_cast<float>(zp_));
          auto tmp3 = tmp2.round();
          auto tmp4 = (tmp3);
          auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.0));
          auto tmp6 = at::vec::maximum(tmp4, tmp5);
          auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(255.0));
          auto tmp8 = at::vec::minimum(tmp6, tmp7);
          auto tmp9 = (tmp8);
          auto tmp10 = at::vec::convert_float_to_int8<uint8_t>(tmp9);
          tmp10.store(out_ptr0, vecsize);
        }
        for (; k < block_k; k++) {
          auto tmp0 = in_ptr[k];
          auto tmp1 = tmp0 / static_cast<float>(scale_);
          auto tmp2 = tmp1 + static_cast<float>(zp_);
          auto tmp3 = std::nearbyint(tmp2);
          auto tmp4 = static_cast<float>(tmp3);
          auto tmp5 = static_cast<float>(0.0);
          auto tmp6 = 0;
          if (at::_isnan(tmp4)) {
            tmp6 = tmp4;
          }
          tmp6 = tmp4 > tmp5 ? tmp4 : tmp5;
          auto tmp7 = static_cast<float>(255.0);
          auto tmp8 = 0;
          if (at::_isnan(tmp6)) {
            tmp8 = tmp6;
          }
          tmp8 = tmp6 < tmp7 ? tmp6 : tmp7;
          auto tmp9 = static_cast<float>(tmp8);
          auto tmp10 = static_cast<unsigned char>(tmp9);
          out_ptr[k] = tmp10;
        }
      };
  auto quantize_block_sym =
      [vecsize](
          float* in_ptr, int8_t* out_ptr, int block_k, float scale_, int zp_) {
        int k;
        for (k = 0; k < block_k / vecsize * vecsize; k += vecsize) {
          auto in_ptr0 = in_ptr + k;
          auto out_ptr0 = out_ptr + k;
          auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0, vecsize);
          auto tmp1 =
              tmp0 / at::vec::Vectorized<float>(static_cast<float>(scale_));
          auto tmp2 =
              tmp1 + at::vec::Vectorized<float>(static_cast<float>(zp_));
          auto tmp3 = tmp2.round();
          auto tmp4 = (tmp3);
          auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(-128.0));
          auto tmp6 = at::vec::maximum(tmp4, tmp5);
          auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(127.0));
          auto tmp8 = at::vec::minimum(tmp6, tmp7);
          auto tmp9 = (tmp8);
          auto tmp10 = at::vec::convert_float_to_int8<int8_t>(tmp9);
          tmp10.store(out_ptr0, vecsize);
        }
        for (; k < block_k; k++) {
          auto tmp0 = in_ptr[k];
          auto tmp1 = tmp0 / static_cast<float>(scale_);
          auto tmp2 = tmp1 + static_cast<float>(zp_);
          auto tmp3 = std::nearbyint(tmp2);
          auto tmp4 = static_cast<float>(tmp3);
          auto tmp5 = static_cast<float>(-128.0);
          auto tmp6 = 0;
          if (at::_isnan(tmp4)) {
            tmp6 = tmp4;
          }
          tmp6 = tmp4 > tmp5 ? tmp4 : tmp5;
          auto tmp7 = static_cast<float>(127.0);
          auto tmp8 = 0;
          if (at::_isnan(tmp6)) {
            tmp8 = tmp6;
          }
          tmp8 = tmp6 < tmp7 ? tmp6 : tmp7;
          auto tmp9 = static_cast<float>(tmp8);
          auto tmp10 = static_cast<int8_t>(tmp9);
          out_ptr[k] = tmp10;
        }
      };
  if (quant_a_mode == QUANT_A_PER_K_BLOCK ||
      quant_a_mode == QUANT_A_PER_K_BLOCK_SYM) {
#pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++) {
      for (int kc = 0; kc < Kc; kc++) {
        auto in_ptr0 = in_ptr + m * K + kc * quant_block_k;
        auto scale_ = scale_ptr[kc];
        auto zp_ = is_sym_quant ? 0 : zp_ptr[kc];
        int block_k = std::min(quant_block_k, (int)K - kc * quant_block_k);
        if (is_sym_quant) {
          auto out_ptr0 = out_sym_ptr + m * K + kc * quant_block_k;
          quantize_block_sym(in_ptr0, out_ptr0, block_k, scale_, zp_);
        } else {
          auto out_ptr0 = out_ptr + m * K + kc * quant_block_k;
          quantize_block(in_ptr0, out_ptr0, block_k, scale_, zp_);
        }
      }
    }
  } else if (
      quant_a_mode == QUANT_A_PER_M || quant_a_mode == QUANT_A_PER_M_SYM) {
#pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++) {
      for (int kc = 0; kc < Kc; kc++) {
        auto in_ptr0 = in_ptr + m * K + kc * quant_block_k;
        auto scale_ = scale_ptr[m];
        auto zp_ = is_sym_quant ? 0 : zp_ptr[m];
        int block_k = std::min(quant_block_k, (int)K - kc * quant_block_k);
        if (is_sym_quant) {
          auto out_ptr0 = out_sym_ptr + m * K + kc * quant_block_k;
          quantize_block_sym(in_ptr0, out_ptr0, block_k, scale_, zp_);
        } else {
          auto out_ptr0 = out_ptr + m * K + kc * quant_block_k;
          quantize_block(in_ptr0, out_ptr0, block_k, scale_, zp_);
        }
      }
    }
  } else {
#pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++) {
      for (int kc = 0; kc < Kc; kc++) {
        auto in_ptr0 = in_ptr + m * K + kc * quant_block_k;
        auto scale_ = scale_ptr[m * Kc + kc];
        auto zp_ = is_sym_quant ? 0 : zp_ptr[m * Kc + kc];
        int block_k = std::min(quant_block_k, (int)K - kc * quant_block_k);
        if (is_sym_quant) {
          auto out_ptr0 = out_sym_ptr + m * K + kc * quant_block_k;
          quantize_block_sym(in_ptr0, out_ptr0, block_k, scale_, zp_);
        } else {
          auto out_ptr0 = out_ptr + m * K + kc * quant_block_k;
          quantize_block(in_ptr0, out_ptr0, block_k, scale_, zp_);
        }
      }
    }
  }
  return out;
}

// Find qparam and quantize tensor per block
// Return: quantized tensor, scales, zero points
template <typename T>
std::tuple<at::Tensor, at::Tensor, at::Tensor> dynamic_quantize_per_block(
    const at::Tensor& t,
    int quant_block_k,
    int quant_a_mode) {
  bool is_sym_quant = !is_asymmetric_quant_a(quant_a_mode);
  auto K = t.size(-1);
  auto M = t.numel() / K;
  auto t_reshape = t.reshape({M, K});
  auto out_dtype = is_sym_quant ? at::kChar : at::kByte;
  if (quant_a_mode == QUANT_A_PER_M || quant_a_mode == QUANT_A_PER_M_SYM) {
    auto grouped_min = std::get<0>(t_reshape.min(-1));
    auto grouped_max = std::get<0>(t_reshape.max(-1));
    auto zeros = at::zeros_like(grouped_min);
    auto min = grouped_min;
    auto max = grouped_max;
    auto scales = is_sym_quant
        ? at::maximum(at::absolute(max), at::absolute(min)) / 127.0f
        : (max - min) / 255.0f;
    auto zps = is_sym_quant ? at::Tensor() : -at::round(min / scales);
    at::Tensor out;
    if (quant_a_mode == QUANT_A_PER_M) {
      out = at::clamp(
          at::round(t_reshape / scales.unsqueeze(-1)) + zps.unsqueeze(-1),
          0,
          255);
    } else if (quant_a_mode == QUANT_A_PER_M_SYM) {
      out = at::clamp(at::round(t_reshape / scales.unsqueeze(-1)), -128, 127);
    }
    out = out.to(out_dtype).view(t.sizes());
    return std::make_tuple<at::Tensor&&, at::Tensor&&, at::Tensor&&>(
        std::move(out),
        std::move(scales.to(c10::kFloat)),
        std::move(is_sym_quant ? zps : zps.to(c10::kInt)));
  }
  int k_rem = K % quant_block_k;
  int block_k = quant_block_k;
  auto grouped =
      t_reshape
          .index({at::indexing::Slice(), at::indexing::Slice(0, K - k_rem)})
          .view({M, K / quant_block_k, quant_block_k});
  at::Tensor grouped_min, grouped_max;
  if (quant_a_mode == QUANT_A_PER_K_BLOCK ||
      quant_a_mode == QUANT_A_PER_K_BLOCK_SYM) {
    grouped_min = std::get<0>(std::get<0>(grouped.min(-1)).min(0));
    grouped_max = std::get<0>(std::get<0>(grouped.max(-1)).max(0));
  } else {
    grouped_min = std::get<0>(grouped.min(-1));
    grouped_max = std::get<0>(grouped.max(-1));
  }
  auto zeros = at::zeros_like(grouped_min);
  auto min = grouped_min;
  auto max = grouped_max;
  auto scales = is_sym_quant
      ? at::maximum(at::absolute(max), at::absolute(min)) / 127.0f
      : (max - min) / 255.0f;
  auto zps = is_sym_quant ? at::Tensor() : -at::round(min / scales);
  at::Tensor out;
  if (quant_a_mode == QUANT_A_PER_K_BLOCK) {
    out = at::clamp(
        at::round(grouped / scales.unsqueeze(1)) + zps.unsqueeze(1), 0, 255);
  } else if (quant_a_mode == QUANT_A_PER_K_BLOCK_SYM) {
    out = at::clamp(at::round(grouped / scales.unsqueeze(1)), -128, 127);
  } else {
    out = is_sym_quant
        ? at::clamp(at::round(grouped / scales.unsqueeze(-1)), -128, 127)
        : at::clamp(
              at::round(grouped / scales.unsqueeze(-1)) + zps.unsqueeze(-1),
              0,
              255);
  }
  if (k_rem) {
    auto grouped_rem =
        t_reshape
            .index({at::indexing::Slice(), at::indexing::Slice(K - k_rem, K)})
            .view({M, 1, k_rem});
    at::Tensor grouped_rem_min, grouped_rem_max;
    if (quant_a_mode == QUANT_A_PER_K_BLOCK ||
        quant_a_mode == QUANT_A_PER_K_BLOCK_SYM) {
      grouped_rem_min = std::get<0>(std::get<0>(grouped_rem.min(-1)).min(0));
      grouped_rem_max = std::get<0>(std::get<0>(grouped_rem.max(-1)).max(0));
    } else {
      grouped_rem_min = std::get<0>(grouped_rem.min(-1));
      grouped_rem_max = std::get<0>(grouped_rem.max(-1));
    }
    auto min_rem = grouped_rem_min;
    auto max_rem = grouped_rem_max;
    auto scales_rem = is_sym_quant
        ? at::maximum(at::absolute(max_rem), at::absolute(min_rem)) / 127.0f
        : (max_rem - min_rem) / 255.0f;
    auto zps_rem =
        is_sym_quant ? at::Tensor() : -at::round(min_rem / scales_rem);
    at::Tensor out_rem;
    if (quant_a_mode == QUANT_A_PER_K_BLOCK) {
      out_rem = at::clamp(
          at::round(grouped_rem / scales_rem.unsqueeze(1)) +
              zps_rem.unsqueeze(1),
          0,
          255);
    } else if (quant_a_mode == QUANT_A_PER_K_BLOCK_SYM) {
      out_rem = at::clamp(
          at::round(grouped_rem / scales_rem.unsqueeze(1)), -128, 127);
    } else {
      out_rem = is_sym_quant
          ? at::clamp(
                at::round(grouped_rem / scales_rem.unsqueeze(-1)), -128, 127)
          : at::clamp(
                at::round(grouped_rem / scales_rem.unsqueeze(-1)) +
                    zps_rem.unsqueeze(-1),
                0,
                255);
    }
    scales = at::cat({scales, scales_rem}, -1).contiguous();
    zps =
        is_sym_quant ? at::Tensor() : at::cat({zps, zps_rem}, -1).contiguous();
    out = at::cat({out.view({M, -1}), out_rem.view({M, -1})}, -1).contiguous();
  }
  out = out.to(out_dtype).view(t.sizes());
  return std::make_tuple<at::Tensor&&, at::Tensor&&, at::Tensor&&>(
      std::move(out),
      std::move(scales.to(c10::kFloat)),
      std::move(is_sym_quant ? zps : zps.to(c10::kInt)));
}

template <>
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> dynamic_quantize_per_block<
    float>(const at::Tensor& t, int quant_block_k, int quant_a_mode) {
  bool is_sym_quant = !is_asymmetric_quant_a(quant_a_mode);
  auto in_ptr = t.data_ptr<float>();
  int K = t.size(-1);
  int n = t.numel();
  int M = n / K;
  int Kc = (K + quant_block_k - 1) / quant_block_k;

  auto out_dtype = is_sym_quant ? at::kChar : at::kByte;
  at::Tensor out = at::empty_like(t, out_dtype);
  uint8_t* out_ptr = is_sym_quant ? nullptr : out.data_ptr<uint8_t>();
  int8_t* out_sym_ptr = is_sym_quant ? out.data_ptr<int8_t>() : nullptr;

  auto vecsize = at::vec::Vectorized<float>::size();
  at::Tensor scales, zps;
  if (quant_a_mode == QUANT_A_PER_K_BLOCK ||
      quant_a_mode == QUANT_A_PER_K_BLOCK_SYM) {
    scales = at::empty({Kc}, t.options().dtype(at::kFloat));
    zps = at::empty({Kc}, t.options().dtype(at::kInt));
  } else if (
      quant_a_mode == QUANT_A_PER_M || quant_a_mode == QUANT_A_PER_M_SYM) {
    scales = at::empty({M}, t.options().dtype(at::kFloat));
    zps = at::empty({M}, t.options().dtype(at::kInt));
  } else {
    scales = at::empty({M, Kc}, t.options().dtype(at::kFloat));
    zps = at::empty({M, Kc}, t.options().dtype(at::kInt));
  }
  auto scales_ptr = scales.data_ptr<float>();
  auto zps_ptr = is_sym_quant ? nullptr : zps.data_ptr<int32_t>();
  auto compute_one_block = [vecsize, scales_ptr, zps_ptr](
                               float* in_ptr,
                               uint8_t* out_ptr,
                               int8_t* out_sym_ptr,
                               int M,
                               int K,
                               int scale_offset,
                               int zp_offset,
                               int ld,
                               bool is_sym_quant) {
    // Find scale and zero points
    float min_val = std::numeric_limits<float>::infinity();
    float max_val = -std::numeric_limits<float>::infinity();
    auto min_vec = at::vec::Vectorized(min_val);
    auto max_vec = at::vec::Vectorized(max_val);
    auto in_ptr_ = in_ptr;
    int k_vec_limit = K / vecsize * vecsize;
    for (int m = 0; m < M; m++) {
      auto in_ptr0 = in_ptr_;
      int k;
      for (k = 0; k < k_vec_limit; k += vecsize) {
        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0, vecsize);
        min_vec = at::vec::minimum(min_vec, tmp0);
        max_vec = at::vec::maximum(tmp0, max_vec);
        in_ptr0 += vecsize;
      }
      for (; k < K; k++) {
        auto tmp0 = in_ptr0[k];
        min_val = std::min(min_val, tmp0);
        max_val = std::max(max_val, tmp0);
      }
      in_ptr_ += ld;
    }
    min_val = min_propagate_nan(
        min_val,
        at::vec::vec_reduce_all<float>(
            [](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) {
              return at::vec::minimum(x, y);
            },
            min_vec));
    max_val = max_propagate_nan(
        max_val,
        at::vec::vec_reduce_all<float>(
            [](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) {
              return at::vec::maximum(x, y);
            },
            max_vec));
    int quant_min = is_sym_quant ? -128 : 0;
    int quant_max = is_sym_quant ? 127 : 255;
    float scale_ = is_sym_quant
        ? std::max(fabs(max_val), fabs(min_val)) / (float)quant_max
        : (max_val - min_val) / (float)quant_max;
    scales_ptr[scale_offset] = scale_;
    int32_t zp_ = is_sym_quant
        ? 0
        : (int32_t)(-std::nearbyint(min_val / scales_ptr[scale_offset]));
    if (!is_sym_quant) {
      zps_ptr[zp_offset] = zp_;
    }

    // Quantize
    in_ptr_ = in_ptr;
    auto out_ptr_ = out_ptr;
    auto out_sym_ptr_ = out_sym_ptr;
    for (int m = 0; m < M; m++) {
      int k;
      for (k = 0; k < k_vec_limit; k += vecsize) {
        auto in_ptr0 = in_ptr_ + k;
        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0, vecsize);
        auto tmp1 =
            tmp0 / at::vec::Vectorized<float>(static_cast<float>(scale_));
        auto tmp2 = tmp1 + at::vec::Vectorized<float>(static_cast<float>(zp_));
        auto tmp3 = tmp2.round();
        auto tmp4 = (tmp3);
        auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(quant_min));
        auto tmp6 = at::vec::maximum(tmp4, tmp5);
        auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(quant_max));
        auto tmp8 = at::vec::minimum(tmp6, tmp7);
        auto tmp9 = (tmp8);
        if (is_sym_quant) {
          auto out_ptr0 = out_sym_ptr_ + k;
          auto tmp10 = at::vec::convert_float_to_int8<int8_t>(tmp9);
          tmp10.store(out_ptr0, vecsize);
        } else {
          auto out_ptr0 = out_ptr_ + k;
          auto tmp10 = at::vec::convert_float_to_int8<uint8_t>(tmp9);
          tmp10.store(out_ptr0, vecsize);
        }
      }
      for (; k < K; k++) {
        auto tmp0 = in_ptr_[k];
        auto tmp1 = tmp0 / static_cast<float>(scale_);
        auto tmp2 = tmp1 + static_cast<float>(zp_);
        auto tmp3 = std::nearbyint(tmp2);
        auto tmp4 = static_cast<float>(tmp3);
        auto tmp5 = static_cast<float>(quant_min);
        auto tmp6 = 0;
        if (at::_isnan(tmp4)) {
          tmp6 = tmp4;
        }
        tmp6 = tmp4 > tmp5 ? tmp4 : tmp5;
        auto tmp7 = static_cast<float>(quant_max);
        auto tmp8 = 0;
        if (at::_isnan(tmp6)) {
          tmp8 = tmp6;
        }
        tmp8 = tmp6 < tmp7 ? tmp6 : tmp7;
        auto tmp9 = static_cast<float>(tmp8);
        if (is_sym_quant) {
          auto tmp10 = static_cast<char>(tmp9);
          out_sym_ptr_[k] = tmp10;
        } else {
          auto tmp10 = static_cast<unsigned char>(tmp9);
          out_ptr_[k] = tmp10;
        }
      }
      in_ptr_ += ld;
      if (is_sym_quant)
        out_sym_ptr_ += ld;
      else
        out_ptr_ += ld;
    }
  };
  if (quant_a_mode == QUANT_A_PER_K_BLOCK ||
      quant_a_mode == QUANT_A_PER_K_BLOCK_SYM) {
#pragma omp parallel for
    for (int kc = 0; kc < Kc; kc++) {
      int offset = kc * quant_block_k;
      int block_k = std::min(quant_block_k, K - offset);
      compute_one_block(
          in_ptr + offset,
          out_ptr + offset,
          out_sym_ptr + offset,
          M,
          block_k,
          kc,
          kc,
          K,
          is_sym_quant);
    }
  } else if (
      quant_a_mode == QUANT_A_PER_M || quant_a_mode == QUANT_A_PER_M_SYM) {
#pragma omp parallel for
    for (int m = 0; m < M; m++) {
      int offset = m * K;
      compute_one_block(
          in_ptr + offset,
          out_ptr + offset,
          out_sym_ptr + offset,
          1,
          K,
          m,
          m,
          K,
          is_sym_quant);
    }
  } else {
#pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++) {
      for (int kc = 0; kc < Kc; kc++) {
        auto in_ptr0 = in_ptr + m * K + kc * quant_block_k;
        auto out_ptr0 = out_ptr + m * K + kc * quant_block_k;
        auto out_sym_ptr0 = out_sym_ptr + m * K + kc * quant_block_k;
        auto scale_offset = m * Kc + kc;
        auto zp_offset = m * Kc + kc;
        int block_k = std::min(quant_block_k, K - kc * quant_block_k);
        compute_one_block(
            in_ptr0,
            out_ptr0,
            out_sym_ptr0,
            1,
            block_k,
            scale_offset,
            zp_offset,
            K,
            is_sym_quant);
      }
    }
  }
  return std::make_tuple<at::Tensor&&, at::Tensor&&, at::Tensor&&>(
      std::move(out), std::move(scales), std::move(zps));
}

template <>
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> dynamic_quantize_per_block<
    bfloat16>(const at::Tensor& t, int quant_block_k, int quant_a_mode) {
  bool is_sym_quant = !is_asymmetric_quant_a(quant_a_mode);
  auto in_ptr = t.data_ptr<at::BFloat16>();
  int K = t.size(-1);
  int n = t.numel();
  int M = n / K;
  int Kc = (K + quant_block_k - 1) / quant_block_k;

  auto out_dtype = is_sym_quant ? at::kChar : at::kByte;
  at::Tensor out = at::empty_like(t, out_dtype);
  uint8_t* out_ptr = is_sym_quant ? nullptr : out.data_ptr<uint8_t>();
  int8_t* out_sym_ptr = is_sym_quant ? out.data_ptr<int8_t>() : nullptr;

  auto vecsize = at::vec::Vectorized<float>::size();
  at::Tensor scales, zps;
  if (quant_a_mode == QUANT_A_PER_K_BLOCK ||
      quant_a_mode == QUANT_A_PER_K_BLOCK_SYM) {
    scales = at::empty({Kc}, t.options().dtype(at::kFloat));
    zps = at::empty({Kc}, t.options().dtype(at::kInt));
  } else if (
      quant_a_mode == QUANT_A_PER_M || quant_a_mode == QUANT_A_PER_M_SYM) {
    scales = at::empty({M}, t.options().dtype(at::kFloat));
    zps = at::empty({M}, t.options().dtype(at::kInt));
  } else {
    scales = at::empty({M, Kc}, t.options().dtype(at::kFloat));
    zps = at::empty({M, Kc}, t.options().dtype(at::kInt));
  }
  auto scales_ptr = scales.data_ptr<float>();
  auto zps_ptr = is_sym_quant ? nullptr : zps.data_ptr<int32_t>();
  auto compute_one_block = [vecsize, scales_ptr, zps_ptr](
                               at::BFloat16* in_ptr,
                               uint8_t* out_ptr,
                               int8_t* out_sym_ptr,
                               int M,
                               int K,
                               int scale_offset,
                               int zp_offset,
                               int ld,
                               bool is_sym_quant) {
    // Find scale and zero points
    float min_val = std::numeric_limits<float>::infinity();
    float max_val = -std::numeric_limits<float>::infinity();
    auto min_vec = at::vec::Vectorized(min_val);
    auto max_vec = at::vec::Vectorized(max_val);
    auto in_ptr_ = in_ptr;
    int k_vec_limit = K / vecsize * vecsize;
    for (int m = 0; m < M; m++) {
      auto in_ptr0 = in_ptr_;
      int k;
      for (k = 0; k < k_vec_limit; k += vecsize) {
        auto tmp0 = at::vec::Vectorized<at::BFloat16>::loadu(in_ptr0, vecsize);
        at::vec::Vectorized<float> res_vec1(0);
        at::vec::Vectorized<float> res_vec2(0);
        std::tie(res_vec1, res_vec2) = at::vec::convert_bfloat16_float(tmp0);
        auto tmp1 = res_vec1;
        min_vec = at::vec::minimum(min_vec, tmp1);
        max_vec = at::vec::maximum(tmp1, max_vec);
        in_ptr0 += vecsize;
      }
      for (; k < K; k++) {
        auto tmp0 = in_ptr0[k];
        min_val = std::min(min_val, (float)tmp0);
        max_val = std::max(max_val, (float)tmp0);
      }
      in_ptr_ += ld;
    }
    min_val = min_propagate_nan(
        min_val,
        at::vec::vec_reduce_all<float>(
            [](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) {
              return at::vec::minimum(x, y);
            },
            min_vec));
    max_val = max_propagate_nan(
        max_val,
        at::vec::vec_reduce_all<float>(
            [](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) {
              return at::vec::maximum(x, y);
            },
            max_vec));
    int quant_min = is_sym_quant ? -128 : 0;
    int quant_max = is_sym_quant ? 127 : 255;
    float scale_ = is_sym_quant
        ? std::max(fabs(max_val), fabs(min_val)) / (float)quant_max
        : (max_val - min_val) / (float)quant_max;
    scales_ptr[scale_offset] = scale_;
    int32_t zp_ = is_sym_quant
        ? 0
        : (int32_t)(-std::nearbyint(min_val / scales_ptr[scale_offset]));
    if (!is_sym_quant) {
      zps_ptr[zp_offset] = zp_;
    }

    // Quantize
    in_ptr_ = in_ptr;
    auto out_ptr_ = out_ptr;
    auto out_sym_ptr_ = out_sym_ptr;
    for (int m = 0; m < M; m++) {
      int k;
      for (k = 0; k < k_vec_limit; k += vecsize) {
        auto in_ptr0 = in_ptr_ + k;
        auto tmp0 = at::vec::Vectorized<at::BFloat16>::loadu(in_ptr0, vecsize);
        at::vec::Vectorized<float> res_vec1(0);
        at::vec::Vectorized<float> res_vec2(0);
        std::tie(res_vec1, res_vec2) = at::vec::convert_bfloat16_float(tmp0);
        auto tmp1 = res_vec1;
        auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(scale_));
        auto tmp3 = tmp1 / tmp2;
        auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(zp_));
        auto tmp5 = tmp3 + tmp4;
        auto tmp6 = tmp5.round();
        auto tmp7 = (tmp6);
        auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(quant_min));
        auto tmp9 = at::vec::maximum(tmp7, tmp8);
        auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(quant_max));
        auto tmp11 = at::vec::minimum(tmp9, tmp10);
        auto tmp12 = (tmp11);
        if (is_sym_quant) {
          auto out_ptr0 = out_sym_ptr_ + k;
          auto tmp13 = at::vec::convert_float_to_int8<int8_t>(tmp12);
          tmp13.store(out_ptr0, vecsize);
        } else {
          auto out_ptr0 = out_ptr_ + k;
          auto tmp13 = at::vec::convert_float_to_int8<uint8_t>(tmp12);
          tmp13.store(out_ptr0, vecsize);
        }
      }
      for (; k < K; k++) {
        auto tmp0 = in_ptr[k];
        auto tmp1 = static_cast<float>(tmp0);
        auto tmp2 = static_cast<float>(scale_);
        auto tmp3 = tmp1 / tmp2;
        auto tmp4 = static_cast<float>(zp_);
        auto tmp5 = tmp3 + tmp4;
        auto tmp6 = std::nearbyint(tmp5);
        auto tmp7 = static_cast<float>(tmp6);
        auto tmp8 = static_cast<float>(quant_min);
        auto tmp9 = 0;
        if (at::_isnan(tmp7)) {
          tmp9 = tmp7;
        }
        tmp9 = tmp7 > tmp8 ? tmp7 : tmp8;
        auto tmp10 = static_cast<float>(quant_max);
        auto tmp11 = 0;
        if (at::_isnan(tmp9)) {
          tmp11 = tmp9;
        }
        tmp11 = tmp9 < tmp10 ? tmp9 : tmp10;
        auto tmp12 = static_cast<float>(tmp11);
        if (is_sym_quant) {
          auto tmp13 = static_cast<char>(tmp12);
          out_sym_ptr_[k] = tmp10;
        } else {
          auto tmp13 = static_cast<unsigned char>(tmp12);
          out_ptr_[k] = tmp10;
        }
      }
      in_ptr_ += ld;
      if (is_sym_quant)
        out_sym_ptr_ += ld;
      else
        out_ptr_ += ld;
    }
  };
  if (quant_a_mode == QUANT_A_PER_K_BLOCK ||
      quant_a_mode == QUANT_A_PER_K_BLOCK_SYM) {
#pragma omp parallel for
    for (int kc = 0; kc < Kc; kc++) {
      int offset = kc * quant_block_k;
      int block_k = std::min(quant_block_k, K - offset);
      compute_one_block(
          in_ptr + offset,
          out_ptr + offset,
          out_sym_ptr + offset,
          M,
          block_k,
          kc,
          kc,
          K,
          is_sym_quant);
    }
  } else if (
      quant_a_mode == QUANT_A_PER_M || quant_a_mode == QUANT_A_PER_M_SYM) {
#pragma omp parallel for
    for (int m = 0; m < M; m++) {
      int offset = m * K;
      compute_one_block(
          in_ptr + offset,
          out_ptr + offset,
          out_sym_ptr + offset,
          1,
          K,
          m,
          m,
          K,
          is_sym_quant);
    }
  } else {
#pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++) {
      for (int kc = 0; kc < Kc; kc++) {
        auto in_ptr0 = in_ptr + m * K + kc * quant_block_k;
        auto out_ptr0 = out_ptr + m * K + kc * quant_block_k;
        auto out_sym_ptr0 = out_sym_ptr + m * K + kc * quant_block_k;
        auto scale_offset = m * Kc + kc;
        auto zp_offset = m * Kc + kc;
        int block_k = std::min(quant_block_k, K - kc * quant_block_k);
        compute_one_block(
            in_ptr0,
            out_ptr0,
            out_sym_ptr0,
            1,
            block_k,
            scale_offset,
            zp_offset,
            K,
            is_sym_quant);
      }
    }
  }
  return std::make_tuple<at::Tensor&&, at::Tensor&&, at::Tensor&&>(
      std::move(out), std::move(scales), std::move(zps));
}

} // namespace cpu
} // namespace torch_ipex

#endif