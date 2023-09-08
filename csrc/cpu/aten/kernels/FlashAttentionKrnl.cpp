#include <ATen/Tensor.h>
#include <aten/FlashAttention.h>
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>
#include <limits>
#include "mkl.h"
#include "vec/vec.h"

namespace torch_ipex {
namespace cpu {

namespace {

const int64_t qsplit_size = 384;
const int64_t kvsplit_size = 512;

#if defined(CPU_CAPABILITY_AVX512)
using namespace torch_ipex::cpu::kernel;

template <typename scalar_t>
void _mha_mul_softmax_bf16_kernel(
    float* a,
    scalar_t* b,
    float* dst,
    float* max,
    float* sum,
    const int& qsize,
    const int& kvsize,
    const int& headsize,
    const int& idx) {
  float tmp_max = 0.f, tmp_sum = 0.f, sum_old = 0.f, exp_tmp = 0.f;

  for (int i = 0; i < qsize; ++i) {
    sum_old = sum[i];

    _dil_reduce_max_fusion_kernel(
        a + i * kvsize, kvsize, a + i * kvsize, tmp_max);
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

at::Tensor flash_base_kernel(
    at::BFloat16* query,
    at::BFloat16* key,
    at::BFloat16* value,
    at::BFloat16* attn_mask,
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
              float(1.f / scale),
              (const MKL_BF16*)(query + i * qSize * qStride + headSize * j + k * qSplitSize * qStride),
              qStride,
              (const MKL_BF16*)(key + i * kvSize * kStride + headSize * j + l * kvSplitSize * kStride),
              kStride,
              0.f,
              qk_fp32.data_ptr<float>() + ompIdx * qSplitSize * kvSplitSize,
              kvBlockSize);

          // update attention weights with attention mask
          for (int r = 0; r < qBlockSize; r++) {
            _dil_add_kernel<at::BFloat16>(
                attn_mask + i * qSize * kvSize + (k * qSplitSize + r) * kvSize +
                    l * kvSplitSize,
                qk_fp32.data_ptr<float>() + ompIdx * qSplitSize * kvSplitSize +
                    r * kvBlockSize,
                kvBlockSize);
          }

          _mha_mul_softmax_bf16_kernel<at::BFloat16>(
              qk_fp32.data_ptr<float>() + ompIdx * qSplitSize * kvSplitSize,
              qk_bf16.data_ptr<at::BFloat16>() +
                  ompIdx * qSplitSize * kvSplitSize,
              dst_fp32.data_ptr<float>() + ompIdx * qSplitSize * headSize,
              qk_max.data_ptr<float>() + ompIdx * qSplitSize,
              qk_sum.data_ptr<float>() + ompIdx * qSplitSize,
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
              (const MKL_BF16*)(value + i * kvSize * vStride + headSize * j + l * kvSplitSize * vStride),
              vStride,
              l == 0 ? 0.f : 1.f,
              dst_fp32.data_ptr<float>() + ompIdx * qSplitSize * headSize,
              headSize);
        }
        _reorder_mha_output_kernel<at::BFloat16>(
            dst_fp32.data_ptr<float>() + ompIdx * qSplitSize * headSize,
            output.data_ptr<at::BFloat16>() + i * qSize * hiddenSize +
                headSize * j + k * qSplitSize * hiddenSize,
            qBlockSize,
            headSize,
            hiddenSize);
      }
    }
  }
  return output;
}
#endif

at::Tensor flash_attention_kernel_impl(
    at::Tensor query,
    at::Tensor key,
    at::Tensor value,
    const double scale_attn,
    at::Tensor attention_mask) {
  TORCH_CHECK(
      query.scalar_type() == at::kBFloat16 && query.dtype() == key.dtype() &&
          query.dtype() == attention_mask.dtype(),
      "Q/K/V/AttnMask must be BF16 to use ipex::flash_attention_kernel_impl");
  TORCH_CHECK(
      query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
      "Q/K/V must be 4D for ipex::flash_attention_kernel_impl");
  TORCH_CHECK(
      attention_mask.size(1) == 1,
      "Attetntion mask size(1) != 1 for ipex::flash_attention_kernel_imp");

#if defined(CPU_CAPABILITY_AVX512)
  int64_t batchSize = query.size(0);
  int64_t qSize = query.size(1);
  int64_t kvSize = value.size(1);
  int64_t num_head = query.size(2);
  int64_t headSize = query.size(3);
  int64_t hiddenSize = num_head * headSize;

  int64_t qStride = query.stride(1);
  int64_t kStride = key.stride(1);
  int64_t vStride = value.stride(1);
  auto attn_outputs = flash_base_kernel(
      query.data_ptr<at::BFloat16>(),
      key.data_ptr<at::BFloat16>(),
      value.data_ptr<at::BFloat16>(),
      attention_mask.data_ptr<at::BFloat16>(),
      qStride,
      kStride,
      vStride,
      batchSize,
      qSize,
      kvSize,
      num_head,
      headSize,
      hiddenSize,
      scale_attn);
  return attn_outputs.resize_({batchSize, qSize, num_head, headSize})
      .transpose_(1, 2);
#else
  key = key.permute({0, 2, 1, 3});
  query = query.permute({0, 2, 1, 3});
  value = value.permute({0, 2, 1, 3});
  auto attn_weights = query.matmul(key.transpose(-1, -2));
  attn_weights = attn_weights.div(scale_attn);
  attn_weights = attn_weights + attention_mask;
  attn_weights = attn_weights.softmax(-1);
  attn_weights = attn_weights.to(value.dtype());
  auto out = attn_weights.matmul(value);
  out = out.transpose_(1, 2).contiguous().transpose_(1, 2);
  return out;
#endif
}
} // anonymous namespace

REGISTER_DISPATCH(flash_attention_kernel_stub, &flash_attention_kernel_impl);

} // namespace cpu
} // namespace torch_ipex
