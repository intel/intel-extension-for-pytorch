#include <aten/MultiHeadAttention.h>
#include "csrc/cpu/tpp/woq/tla.h"
#include "mkl.h"
#include "vec/vec.h"

namespace torch_ipex {
using namespace tpp;
namespace cpu {

namespace {

// `qsplit_ranges` and `qsplit_sizes` are for the segmented q block sizes,
// where `qsplit_ranges` is a group of q lengths and `qsplit_sizes` is a group
// of q block sizes. This segmented configuration is applied for all other
// models (like BERT) except for Stable Diffusion. It is proved that Stable
// Diffusion achieves a better performance with a fixed q block size (32).
const std::vector<int64_t> qsplit_ranges{767, 191, 31};
const std::vector<int64_t> qsplit_sizes{256, 64, 32};
const int64_t qsplit_size = 32;
const int64_t kvsplit_size = 512;

#if defined(CPU_CAPABILITY_AVX512)
using namespace torch_ipex::cpu::kernel;

// Div_Add_SoftMax kernel for BERT MHA Fusion based on the Flash Attention
template <typename scalar_t>
void _mha_div_add_softmax_bf16_kernel(
    float* a,
    scalar_t* b,
    float* dst,
    const scalar_t* rel_kv,
    float* max,
    float* sum,
    const float& scale,
    const int& qsize,
    const int& kvsize,
    const int& headsize,
    const int& idx) {
  auto alpha = _mm512_set1_ps(scale);
  float tmp_max = 0.f, tmp_sum = 0.f, sum_old = 0.f, exp_tmp = 0.f;

  for (int i = 0; i < qsize; ++i) {
    sum_old = sum[i];

    _dil_div_add_reduce_max_fusion_kernel<float, scalar_t>(
        a + i * kvsize, rel_kv, scale, kvsize, a + i * kvsize, tmp_max);
    tmp_max = max[i] > tmp_max ? max[i] : tmp_max;

    tmp_sum = tmp_max;
    _dil_exp_reduce_sum_fusion_kernel(
        a + i * kvsize, kvsize, a + i * kvsize, tmp_sum);
    exp_tmp = exp(max[i] - tmp_max);
    sum[i] = tmp_sum + exp_tmp * sum[i];
    max[i] = tmp_max;

    _dil_normalization_kernel<scalar_t>(
        a + i * kvsize, sum[i], kvsize, b + i * kvsize);

    if (idx) {
      _mha_update_sum_max_kernel(
          dst + i * headsize,
          sum_old,
          sum[i],
          exp_tmp,
          headsize,
          dst + i * headsize);
    }
  }
}

// Mul_SoftMax kernel for Stable-Diffusion MHA Fusion based on
// the Flash Attention
template <typename scalar_t>
void _mha_mul_softmax_bf16_kernel(
    float* a,
    scalar_t* b,
    float* dst,
    float* max,
    float* sum,
    const float& scale,
    const int& qsize,
    const int& kvsize,
    const int& headsize,
    const int& idx) {
  auto alpha = _mm512_set1_ps(scale);
  float tmp_max = 0.f, tmp_sum = 0.f, sum_old = 0.f, exp_tmp = 0.f;

  for (int i = 0; i < qsize; ++i) {
    sum_old = sum[i];

    _dil_mul_reduce_max_fusion_kernel(
        a + i * kvsize, scale, kvsize, a + i * kvsize, tmp_max);
    tmp_max = max[i] > tmp_max ? max[i] : tmp_max;

    tmp_sum = tmp_max;
    _dil_exp_reduce_sum_fusion_kernel(
        a + i * kvsize, kvsize, a + i * kvsize, tmp_sum);
    exp_tmp = exp(max[i] - tmp_max);
    sum[i] = tmp_sum + exp_tmp * sum[i];
    max[i] = tmp_max;

    _dil_normalization_kernel<scalar_t>(
        a + i * kvsize, sum[i], kvsize, b + i * kvsize);

    if (idx) {
      _mha_update_sum_max_kernel(
          dst + i * headsize,
          sum_old,
          sum[i],
          exp_tmp,
          headsize,
          dst + i * headsize);
    }
  }
}

at::Tensor sd_mha_base_kernel(
    at::BFloat16* query,
    at::BFloat16* key,
    at::BFloat16* value,
    const int64_t& qStride,
    const int64_t& kStride,
    const int64_t& vStride,
    const int64_t& batchSize,
    const int64_t& qSize,
    const int64_t& kvSize,
    const int64_t& num_head,
    const int64_t& headSize,
    const int64_t& hiddenSize,
    const double& scale) {
  at::Tensor output = at::empty({batchSize, qSize, hiddenSize}, at::kBFloat16);

  int64_t qSplitSize = qSize >= qsplit_size ? qsplit_size : qSize;
  int64_t kvSplitSize = kvSize >= kvsplit_size ? kvsplit_size : kvSize;

  int64_t qSlice = (qSize - 1) / qSplitSize + 1;
  int64_t qTail = (qSize - 1) % qSplitSize + 1;
  int64_t kvSlice = (kvSize - 1) / kvSplitSize + 1;
  int64_t kvTail = (kvSize - 1) % kvSplitSize + 1;

  int64_t num_thread = omp_get_max_threads();

  at::Tensor qk_fp32 =
      at::empty({num_thread, qSplitSize, kvSplitSize}, at::kFloat);
  at::Tensor qk_bf16 =
      at::empty({num_thread, qSplitSize, kvSplitSize}, at::kBFloat16);
  at::Tensor qk_max = at::empty({num_thread, qSplitSize}, at::kFloat);
  at::Tensor qk_sum = at::empty({num_thread, qSplitSize}, at::kFloat);
  at::Tensor dst_fp32 =
      at::empty({num_thread, qSplitSize, headSize}, at::kFloat);

  auto output_ptr = output.data_ptr<at::BFloat16>();
  auto qk_fp32_ptr = qk_fp32.data_ptr<float>();
  auto qk_bf16_ptr = qk_bf16.data_ptr<at::BFloat16>();
  auto qk_max_ptr = qk_max.data_ptr<float>();
  auto qk_sum_ptr = qk_sum.data_ptr<float>();
  auto dst_fp32_ptr = dst_fp32.data_ptr<float>();

  // Create tpp kernels for Query @ Key
  int qk_gemm_K = headSize % 2 == 0
      ? headSize
      : 2; // If K of Gemm is not even, use mkl gemm instead of tpp
  // [qSplitSize,headSize] x [headSize,kvSplitSize] -> [qSplitSize,kvSplitSize]
  auto qk_gemm_tpp = SCOPEITGEMM((BrgemmTPP<at::BFloat16, float>(
      /*M*/ qSplitSize,
      /*N*/ kvSplitSize,
      /*K*/ qk_gemm_K,
      /*str_a*/ 1,
      /*str_b*/ 1,
      /*lda*/ qStride,
      /*ldb*/ kvSplitSize,
      /*ldc*/ kvSplitSize,
      /*beta*/ 0.0,
      /*a_trans*/ 0,
      /*unroll_hint*/ 1)));
  auto qk_gemm_tpp_tail = SCOPEITGEMM((BrgemmTPP<at::BFloat16, float>(
      /*M*/ qSplitSize,
      /*N*/ kvTail,
      /*K*/ qk_gemm_K,
      /*str_a*/ 1,
      /*str_b*/ 1,
      /*lda*/ qStride,
      /*ldb*/ kvTail,
      /*ldc*/ kvTail,
      /*beta*/ 0.0,
      /*a_trans*/ 0,
      /*unroll_hint*/ 1)));

  // Create tpp transforms for Key
  auto xform_k = XformTPP::XFORM_XPOSE_N2V_TPP;
  auto xform_tpp_k = // [kvSplitSize,headSize] -> [headSize,kvSplitSize] with
                     // vnni
      SCOPEIT(
          XformExtTPP<at::BFloat16>(
              kvSplitSize,
              qk_gemm_K,
              qk_gemm_K,
              kvSplitSize,
              kStride,
              kvSplitSize,
              xform_k,
              true),
          XPOSE);
  auto xform_tpp_k_tail = // [kvTail,headSize] -> [headSize,kvTail] with vnni
      SCOPEIT(
          XformExtTPP<at::BFloat16>(
              kvTail,
              qk_gemm_K,
              qk_gemm_K,
              kvTail,
              kStride,
              kvTail,
              xform_k,
              true),
          XPOSE);

  // Create tpp kernels for Attention @ Value
  int av_gemm_K = kvSplitSize % 2 == 0
      ? kvSplitSize
      : 2; // If K of Gemm is not even, use mkl gemm instead of tpp
  int av_gemm_K_tail = kvTail % 2 == 0
      ? kvTail
      : 2; // If K of Gemm is not even, use mkl gemm instead of tpp
  // [qSplitSize,kvSplitSize] x [kvSplitSize,headSize] -> [qSplitSize,headSize]
  auto av_gemm_tpp = SCOPEITGEMM((BrgemmTPP<at::BFloat16, float>(
      /*M*/ qSplitSize,
      /*N*/ headSize,
      /*K*/ av_gemm_K,
      /*str_a*/ 1,
      /*str_b*/ 1,
      /*lda*/ av_gemm_K,
      /*ldb*/ headSize,
      /*ldc*/ headSize,
      /*beta*/ 1.0,
      /*a_trans*/ 0,
      /*unroll_hint*/ 1)));
  auto av_gemm_tpp_tail = SCOPEITGEMM((BrgemmTPP<at::BFloat16, float>(
      /*M*/ qSplitSize,
      /*N*/ headSize,
      /*K*/ av_gemm_K_tail,
      /*str_a*/ 1,
      /*str_b*/ 1,
      /*lda*/ av_gemm_K_tail,
      /*ldb*/ headSize,
      /*ldc*/ headSize,
      /*beta*/ 1.0,
      /*a_trans*/ 0,
      /*unroll_hint*/ 1)));
  auto av_gemm_tpp_nobias = SCOPEITGEMM((BrgemmTPP<at::BFloat16, float>(
      /*M*/ qSplitSize,
      /*N*/ headSize,
      /*K*/ av_gemm_K,
      /*str_a*/ 1,
      /*str_b*/ 1,
      /*lda*/ av_gemm_K,
      /*ldb*/ headSize,
      /*ldc*/ headSize,
      /*beta*/ 0.0,
      /*a_trans*/ 0,
      /*unroll_hint*/ 1)));
  auto av_gemm_tpp_nobias_tail = SCOPEITGEMM((BrgemmTPP<at::BFloat16, float>(
      /*M*/ qSplitSize,
      /*N*/ headSize,
      /*K*/ av_gemm_K_tail,
      /*str_a*/ 1,
      /*str_b*/ 1,
      /*lda*/ av_gemm_K_tail,
      /*ldb*/ headSize,
      /*ldc*/ headSize,
      /*beta*/ 0.0,
      /*a_trans*/ 0,
      /*unroll_hint*/ 1)));

  // Create tpp transforms for Value
  auto xform_v = XformTPP::XFORM_N2V_TPP;
  auto xform_tpp_v = // [kvSplitSize,headSize] -> [kvSplitSize, headSize]
      SCOPEIT(
          XformExtTPP<at::BFloat16>(
              av_gemm_K,
              headSize,
              av_gemm_K,
              headSize,
              kStride,
              headSize,
              xform_v,
              true),
          XPOSE);
  auto xform_tpp_v_tail = // [kvTail,headSize] -> [kvTail, headSize]
      SCOPEIT(
          XformExtTPP<at::BFloat16>(
              av_gemm_K_tail,
              headSize,
              av_gemm_K_tail,
              headSize,
              kStride,
              headSize,
              xform_v,
              true),
          XPOSE);

  // Buffer to store Key and Value after transforms
  at::Tensor key_t_reorder =
      at::empty({batchSize, num_head, headSize, kvSize}, at::kBFloat16);
  auto key_reorder_ptr = key_t_reorder.data_ptr<at::BFloat16>();
  at::Tensor value_t_reorder =
      at::empty({batchSize, num_head, kvSize, headSize}, at::kBFloat16);
  auto value_reorder_ptr = value_t_reorder.data_ptr<at::BFloat16>();

  // Reorder K, V
#pragma omp parallel for collapse(3)
  for (int i = 0; i < batchSize; ++i) {
    for (int j = 0; j < num_head; ++j) {
      for (int l = 0; l < kvSlice; ++l) {
        if (l != kvSlice - 1) {
          // main
          if (headSize % 2 == 0) {
            xform_tpp_k(
                key + i * kvSize * kStride + headSize * j +
                    l * kvSplitSize * kStride,
                key_reorder_ptr + i * num_head * headSize * kvSize +
                    j * headSize * kvSize + l * kvSplitSize * headSize);
          }
          if (kvSplitSize % 2 == 0) {
            xform_tpp_v(
                value + i * kvSize * vStride + headSize * j +
                    l * kvSplitSize * vStride,
                value_reorder_ptr + i * num_head * kvSize * headSize +
                    j * kvSize * headSize + l * kvSplitSize * headSize);
          }
        } else {
          // Tail
          if (headSize % 2 == 0) {
            xform_tpp_k_tail(
                key + i * kvSize * kStride + headSize * j +
                    l * kvSplitSize * kStride,
                key_reorder_ptr + i * num_head * headSize * kvSize +
                    j * headSize * kvSize + l * kvSplitSize * headSize);
          }
          if (kvTail % 2 == 0) {
            xform_tpp_v_tail(
                value + i * kvSize * vStride + headSize * j +
                    l * kvSplitSize * vStride,
                value_reorder_ptr + i * num_head * kvSize * headSize +
                    j * kvSize * headSize + l * kvSplitSize * headSize);
          }
        }
      }
    }
  }

#pragma omp parallel for collapse(3)
  for (int i = 0; i < batchSize; ++i) {
    for (int j = 0; j < num_head; ++j) {
      for (int k = 0; k < qSlice; ++k) {
        int qBlockSize = (k == qSlice - 1) ? qTail : qSplitSize;
        int ompIdx = omp_get_thread_num();
        _init_mha_buffer_kernel(
            qk_max_ptr + ompIdx * qSplitSize,
            qk_sum_ptr + ompIdx * qSplitSize,
            qBlockSize);

        for (int l = 0; l < kvSlice; ++l) {
          int kvBlockSize = (l == kvSlice - 1) ? kvTail : kvSplitSize;
          if (headSize % 2 != 0) {
            // If K of Gemm is not even, use mkl gemm instead of tpp
            cblas_gemm_bf16bf16f32(
                CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                qBlockSize,
                kvBlockSize,
                headSize,
                1.f,
                (const MKL_BF16*)(query + i * qSize * qStride + headSize * j + k * qSplitSize * qStride),
                qStride,
                (const MKL_BF16*)(key + i * kvSize * kStride + headSize * j + l * kvSplitSize * kStride),
                kStride,
                0.f,
                qk_fp32_ptr + ompIdx * qSplitSize * kvSplitSize,
                kvBlockSize);
          } else if (l != kvSlice - 1) {
            qk_gemm_tpp(
                query + i * qSize * qStride + headSize * j +
                    k * qSplitSize * qStride,
                key_reorder_ptr + i * num_head * headSize * kvSize +
                    j * headSize * kvSize + l * kvSplitSize * headSize,
                qk_fp32_ptr + ompIdx * qSplitSize * kvSplitSize,
                1); // [qSplitSize,headSize] x [headSize,kvSplitSize] ->
                    // [qSplitSize,kvSplitSize]
          } else {
            // Tail
            qk_gemm_tpp_tail(
                query + i * qSize * qStride + headSize * j +
                    k * qSplitSize * qStride,
                key_reorder_ptr + i * num_head * headSize * kvSize +
                    j * headSize * kvSize + l * kvSplitSize * headSize,
                qk_fp32_ptr + ompIdx * qSplitSize * kvSplitSize,
                1); // [qSplitSize,headSize] x [headSize,kvSplitSize] ->
                    // [qSplitSize,kvSplitSize]
          }

          _mha_mul_softmax_bf16_kernel<at::BFloat16>(
              qk_fp32_ptr + ompIdx * qSplitSize * kvSplitSize,
              qk_bf16_ptr + ompIdx * qSplitSize * kvSplitSize,
              dst_fp32_ptr + ompIdx * qSplitSize * headSize,
              qk_max_ptr + ompIdx * qSplitSize,
              qk_sum_ptr + ompIdx * qSplitSize,
              scale,
              qBlockSize,
              kvBlockSize,
              headSize,
              l);

          if ((l != kvSlice - 1 && kvSplitSize % 2 != 0) ||
              (l == kvSlice - 1 && kvTail % 2 != 0)) {
            // If K of Gemm is not even, use mkl gemm instead of tpp
            cblas_gemm_bf16bf16f32(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                qBlockSize,
                headSize,
                kvBlockSize,
                1.f,
                (const MKL_BF16*)(qk_bf16_ptr + ompIdx * qSplitSize * kvSplitSize),
                kvBlockSize,
                (const MKL_BF16*)(value + i * kvSize * vStride + headSize * j + l * kvSplitSize * vStride),
                vStride,
                l == 0 ? 0.f : 1.f,
                dst_fp32_ptr + ompIdx * qSplitSize * headSize,
                headSize);
          } else if (l != kvSlice - 1) {
            if (l != 0) {
              av_gemm_tpp(
                  qk_bf16_ptr + ompIdx * qSplitSize * kvSplitSize,
                  value_reorder_ptr + i * num_head * kvSize * headSize +
                      j * kvSize * headSize + l * kvSplitSize * headSize,
                  dst_fp32_ptr + ompIdx * qSplitSize * headSize,
                  1);
            } else {
              av_gemm_tpp_nobias(
                  qk_bf16_ptr + ompIdx * qSplitSize * kvSplitSize,
                  value_reorder_ptr + i * num_head * kvSize * headSize +
                      j * kvSize * headSize + l * kvSplitSize * headSize,
                  dst_fp32_ptr + ompIdx * qSplitSize * headSize,
                  1);
            }
          } else {
            // Tail
            if (l != 0) {
              av_gemm_tpp_tail(
                  qk_bf16_ptr + ompIdx * qSplitSize * kvSplitSize,
                  value_reorder_ptr + i * num_head * kvSize * headSize +
                      j * kvSize * headSize + l * kvSplitSize * headSize,
                  dst_fp32_ptr + ompIdx * qSplitSize * headSize,
                  1);
            } else {
              av_gemm_tpp_nobias_tail(
                  qk_bf16_ptr + ompIdx * qSplitSize * kvSplitSize,
                  value_reorder_ptr + i * num_head * kvSize * headSize +
                      j * kvSize * headSize + l * kvSplitSize * headSize,
                  dst_fp32_ptr + ompIdx * qSplitSize * headSize,
                  1);
            }
          }
        }
        _reorder_mha_output_kernel<at::BFloat16>(
            dst_fp32_ptr + ompIdx * qSplitSize * headSize,
            output_ptr + i * qSize * hiddenSize + headSize * j +
                k * qSplitSize * hiddenSize,
            qBlockSize,
            headSize,
            hiddenSize);
      }
    }
  }
  return output;
}
#endif

at::Tensor bert_mha_kernel_impl(
    const at::Tensor& qkv,
    const at::Tensor& rel_kv,
    const int64_t& num_head,
    const int64_t& headSize,
    const double& dim_per_head) {
  TORCH_CHECK(
      qkv.dtype() == at::kBFloat16,
      "Currently the BERT MHA fusion only supports BF16 data type.");

  int64_t batchSize = qkv.dim() > 2 ? qkv.size(0) : 1;
  int64_t sequenceSize = qkv.dim() > 2 ? qkv.size(1) : qkv.size(0);
  int64_t hiddenSize = num_head * headSize;
  int64_t qkvColSize = hiddenSize * 3;
  at::Tensor output =
      at::empty({batchSize, sequenceSize, num_head, headSize}, at::kBFloat16);

#if defined(CPU_CAPABILITY_AVX512)
  int64_t qSplitSize = sequenceSize;
  for (int i = 0; i < qsplit_ranges.size(); ++i) {
    if (sequenceSize > qsplit_ranges[i]) {
      qSplitSize = qsplit_sizes[i];
      break;
    }
  }
  int64_t kvSplitSize =
      sequenceSize >= kvsplit_size ? kvsplit_size : sequenceSize;

  int64_t qSlice = (sequenceSize - 1) / qSplitSize + 1;
  int64_t qTail = (sequenceSize - 1) % qSplitSize + 1;
  int64_t kvSlice = (sequenceSize - 1) / kvSplitSize + 1;
  int64_t kvTail = (sequenceSize - 1) % kvSplitSize + 1;

  int64_t num_thread = omp_get_max_threads();

  at::Tensor qk_fp32 =
      at::empty({num_thread, qSplitSize, kvSplitSize}, at::kFloat);
  at::Tensor qk_bf16 =
      at::empty({num_thread, qSplitSize, kvSplitSize}, at::kBFloat16);
  at::Tensor qk_max = at::empty({num_thread, qSplitSize}, at::kFloat);
  at::Tensor qk_sum = at::empty({num_thread, qSplitSize}, at::kFloat);
  at::Tensor dst_fp32 =
      at::empty({num_thread, qSplitSize, headSize}, at::kFloat);

#pragma omp parallel for collapse(3)
  for (int i = 0; i < batchSize; ++i) {
    for (int j = 0; j < num_head; ++j) {
      for (int k = 0; k < qSlice; ++k) {
        int qBlockSize = (k == qSlice - 1) ? qTail : qSplitSize;
        int ompIdx = omp_get_thread_num();
        _init_mha_buffer_kernel(
            qk_max.data_ptr<float>() + ompIdx * qSplitSize,
            qk_sum.data_ptr<float>() + ompIdx * qSplitSize,
            qBlockSize);

        for (int l = 0; l < kvSlice; ++l) {
          int kvBlockSize = (l == kvSlice - 1) ? kvTail : kvSplitSize;
          cblas_gemm_bf16bf16f32(
              CblasRowMajor,
              CblasNoTrans,
              CblasTrans,
              qBlockSize,
              kvBlockSize,
              headSize,
              1.f,
              (const MKL_BF16*)(qkv.data_ptr<at::BFloat16>() + i * sequenceSize * qkvColSize + headSize * j + k * qSplitSize * qkvColSize),
              qkvColSize,
              (const MKL_BF16*)(qkv.data_ptr<at::BFloat16>() + i * sequenceSize * qkvColSize + hiddenSize + headSize * j + l * kvSplitSize * qkvColSize),
              qkvColSize,
              0.f,
              qk_fp32.data_ptr<float>() + ompIdx * qSplitSize * kvSplitSize,
              kvBlockSize);

          _mha_div_add_softmax_bf16_kernel<at::BFloat16>(
              qk_fp32.data_ptr<float>() + ompIdx * qSplitSize * kvSplitSize,
              qk_bf16.data_ptr<at::BFloat16>() +
                  ompIdx * qSplitSize * kvSplitSize,
              dst_fp32.data_ptr<float>() + ompIdx * qSplitSize * headSize,
              rel_kv.data_ptr<at::BFloat16>() + i * sequenceSize +
                  l * qSplitSize,
              qk_max.data_ptr<float>() + ompIdx * qSplitSize,
              qk_sum.data_ptr<float>() + ompIdx * qSplitSize,
              dim_per_head,
              qBlockSize,
              kvBlockSize,
              headSize,
              l);

          cblas_gemm_bf16bf16f32(
              CblasRowMajor,
              CblasNoTrans,
              CblasNoTrans,
              qBlockSize,
              headSize,
              kvBlockSize,
              1.f,
              (const MKL_BF16*)(qk_bf16.data_ptr<at::BFloat16>() + ompIdx * qSplitSize * kvSplitSize),
              kvBlockSize,
              (const MKL_BF16*)(qkv.data_ptr<at::BFloat16>() + i * sequenceSize * qkvColSize + hiddenSize * 2 + headSize * j + l * kvSplitSize * qkvColSize),
              qkvColSize,
              l == 0 ? 0.f : 1.f,
              dst_fp32.data_ptr<float>() + ompIdx * qSplitSize * headSize,
              headSize);
        }
        _reorder_mha_output_kernel<at::BFloat16>(
            dst_fp32.data_ptr<float>() + ompIdx * qSplitSize * headSize,
            output.data_ptr<at::BFloat16>() + i * sequenceSize * hiddenSize +
                headSize * j + k * qSplitSize * hiddenSize,
            qBlockSize,
            headSize,
            hiddenSize);
      }
    }
  }
  return output;
#endif
  auto qkv_mat = dil_mat_split<at::BFloat16>(
      qkv, at::IntArrayRef({hiddenSize, hiddenSize, hiddenSize}));
  auto query = qkv_mat[0];
  auto key = qkv_mat[1];
  auto value = qkv_mat[2];
  query.resize_({batchSize, sequenceSize, num_head, headSize}).transpose_(1, 2);
  key.resize_({batchSize, sequenceSize, num_head, headSize})
      .transpose_(1, 2)
      .transpose_(2, 3);
  value.resize_({batchSize, sequenceSize, num_head, headSize}).transpose_(1, 2);

  auto qk = at::div(at::matmul(query, key), dim_per_head);
  auto qk_sm = at::softmax(at::add(qk, rel_kv, 1.f), -1);
  output = at::matmul(qk_sm, value);

  output = output.transpose_(1, 2).contiguous();
  return output;
}

at::Tensor sd_mha_kernel_v1_impl(
    const at::Tensor& qkv,
    const int64_t& num_head,
    const int64_t& headSize,
    const double& scale) {
  TORCH_CHECK(
      qkv.dtype() == at::kBFloat16,
      "Currently the Stable-Diffusion MHA fusion only supports BF16 data type.");

  int64_t qkvOffset = num_head * headSize;
  int64_t qkvStride = qkv.size(-1);
  int64_t batchSize = qkv.size(0);
  int64_t sequenceSize = qkv.size(1);
  int64_t hiddenSize = num_head * headSize;
#if defined(CPU_CAPABILITY_AVX512)
  return sd_mha_base_kernel(
      qkv.data_ptr<at::BFloat16>(),
      qkv.data_ptr<at::BFloat16>() + qkvOffset,
      qkv.data_ptr<at::BFloat16>() + qkvOffset * 2,
      qkvStride,
      qkvStride,
      qkvStride,
      batchSize,
      sequenceSize,
      sequenceSize,
      num_head,
      headSize,
      hiddenSize,
      scale);
#endif
  auto qkv_mat = dil_mat_split<at::BFloat16>(
      qkv, at::IntArrayRef({hiddenSize, hiddenSize, hiddenSize}));
  auto query = qkv_mat[0];
  auto key = qkv_mat[1];
  auto value = qkv_mat[2];
  query.resize_({batchSize, sequenceSize, num_head, headSize}).transpose_(1, 2);
  key.resize_({batchSize, sequenceSize, num_head, headSize})
      .transpose_(1, 2)
      .transpose_(2, 3);
  value.resize_({batchSize, sequenceSize, num_head, headSize}).transpose_(1, 2);

  auto qk = at::matmul(query, key);
  qk = at::softmax(at::mul(qk, scale), -1);
  auto output = at::matmul(qk, value);

  output = output.transpose_(1, 2).contiguous().resize_(
      {batchSize, sequenceSize, hiddenSize});
  return output;
}

at::Tensor sd_mha_kernel_v2_impl(
    const at::Tensor& _query,
    const at::Tensor& _key,
    const at::Tensor& _value,
    const int64_t& num_head,
    const int64_t& headSize,
    const double& scale) {
  auto query = _query.contiguous();
  auto key = _key.contiguous();
  auto value = _value.contiguous();
  TORCH_CHECK(
      (query.dtype() == at::kBFloat16 && key.dtype() == at::kBFloat16 &&
       value.dtype() == at::kBFloat16),
      "Currently the Stable-Diffusion MHA fusion only supports BF16 data type.");

  int64_t batchSize = query.size(0);
  int64_t qStride = query.size(-1);
  int64_t kStride = key.size(-1);
  int64_t vStride = value.size(-1);
  int64_t qSize = query.size(1);
  int64_t kvSize = value.size(1);
  int64_t hiddenSize = num_head * headSize;
#if defined(CPU_CAPABILITY_AVX512)
  return sd_mha_base_kernel(
      query.data_ptr<at::BFloat16>(),
      key.data_ptr<at::BFloat16>(),
      value.data_ptr<at::BFloat16>(),
      qStride,
      kStride,
      vStride,
      batchSize,
      qSize,
      kvSize,
      num_head,
      headSize,
      hiddenSize,
      scale);
#endif
  query.resize_({batchSize, qSize, num_head, headSize}).transpose_(1, 2);
  key.resize_({batchSize, kvSize, num_head, headSize})
      .transpose_(1, 2)
      .transpose_(2, 3);
  value.resize_({batchSize, kvSize, num_head, headSize}).transpose_(1, 2);

  auto qk = at::matmul(query, key);
  qk = at::softmax(at::mul(qk, scale), -1);
  auto output = at::matmul(qk, value);

  output = output.transpose_(1, 2).contiguous().resize_(
      {batchSize, qSize, hiddenSize});
  return output;
}

} // anonymous namespace

IPEX_REGISTER_DISPATCH(bert_mha_kernel_stub, &bert_mha_kernel_impl);
IPEX_REGISTER_DISPATCH(sd_mha_kernel_v1_stub, &sd_mha_kernel_v1_impl);
IPEX_REGISTER_DISPATCH(sd_mha_kernel_v2_stub, &sd_mha_kernel_v2_impl);

} // namespace cpu
} // namespace torch_ipex