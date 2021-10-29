#include "cpu/bf16/vec/bf16_vec_kernel.h"
#include "optimizer.h"

#include <torch/csrc/autograd/function.h>
#include <torch/extension.h>

namespace torch_ipex {
namespace cpu {

void packed_add(at::Tensor &top_half_, at::Tensor &bot_half_,
                const at::Tensor &grad_, double alpha) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(grad_.scalar_type() ==
                                   at::ScalarType::BFloat16);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(top_half_.scalar_type() ==
                                   at::ScalarType::BFloat16);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(bot_half_.scalar_type() ==
                                   at::ScalarType::BFloat16);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(top_half_.sizes() == bot_half_.sizes());

  at::Tensor top_half = top_half_.contiguous();
  at::Tensor bot_half = bot_half_.contiguous();
  at::Tensor grad = grad_.is_sparse() ? grad_ : grad_.contiguous();

#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("packed_add", std::vector<c10::IValue>({}));
#endif

  float alpha_ = static_cast<float>(alpha);
  if (grad.is_sparse()) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(top_half.dim() == 2);
    auto sparse_nnz = grad._nnz();
    auto sparse_dim = grad.sparse_dim();
    auto values = grad._values();
    auto indices = grad._indices();
    auto entry_range = top_half.size(0);
    auto feature_size = values.stride(0);
    auto indices_accessor = indices.accessor<int64_t, 2>();

    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(values.is_contiguous());
    auto value_ptr = values.data_ptr<at::BFloat16>();
    auto top_half_ptr = top_half.data_ptr<at::BFloat16>();
    auto bot_half_ptr = bot_half.data_ptr<at::BFloat16>();

    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(value_ptr != nullptr);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(top_half_ptr != nullptr);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(bot_half_ptr != nullptr);

    std::vector<int64_t> sparse_stride(sparse_dim);
    for (int64_t d = 0; d < sparse_dim; d++) {
      sparse_stride[d] = top_half.stride(d);
    }

    int32_t max_threads = at::get_num_threads();
    max_threads = (entry_range < max_threads) ? entry_range : max_threads;
    int64_t avg_size = entry_range / max_threads;
    int64_t tail_size = entry_range % max_threads;
    std::vector<int64_t> chunk_size(max_threads, avg_size);
    std::transform(chunk_size.begin(), chunk_size.begin() + tail_size,
                   chunk_size.begin(),
                   [](int64_t a) -> int64_t { return a + 1; });
    std::vector<int64_t> acc_chunk_size(max_threads + 1);
    for (int64_t i = 1; i < max_threads + 1; i++) {
      acc_chunk_size[i] = acc_chunk_size[i - 1] + chunk_size[i - 1];
    }

    at::parallel_for(0, max_threads, 0, [&](int64_t start, int64_t end) {
      for (int64_t c = start; c < end; c++) {
        int64_t chunk_begin = acc_chunk_size[c];
        int64_t chunk_end = acc_chunk_size[c + 1];
        for (int64_t n = 0; n < sparse_nnz; n++) {
          int64_t chunk_offset = indices_accessor[0][n];
          if (chunk_offset >= chunk_begin && chunk_offset < chunk_end) {
            int64_t table_offset = 0;
            for (int64_t d = 0; d < sparse_dim; d++) {
              table_offset += sparse_stride[d] * indices_accessor[d][n];
            }
            auto value_index = value_ptr + n * feature_size;
            auto top_half_index = top_half_ptr + table_offset;
            auto bot_half_index = bot_half_ptr + table_offset;
            packed_bf16_add_ker(top_half_index, bot_half_index, value_index,
                                feature_size, alpha_);
          }
        }
      }
    });
  } else {
    // TODO: vector implementation basing on vector size
    union packed_bf16 {
      unsigned short s[2];
      float f;
    };

    auto len = top_half.numel();
    auto value_ptr = grad.data_ptr<at::BFloat16>();
    auto top_half_ptr = (unsigned short *)top_half.data_ptr();
    auto bot_half_ptr = (unsigned short *)bot_half.data_ptr();

    at::parallel_for(0, len, 64, [&](int64_t start, int64_t end) {
      int64_t i = start;
#if defined(CPU_AVX512)
      auto alpha_vec = _mm512_set1_ps(alpha_);
      for (; i < end - 31; i+=32) {
        auto bot0 = _mm256_loadu_si256((__m256i *)(bot_half_ptr + i));
        auto bot1 = _mm256_loadu_si256((__m256i *)(bot_half_ptr + i + 16));
        auto top0 = _mm256_loadu_si256((__m256i *)(top_half_ptr + i));
        auto top1 = _mm256_loadu_si256((__m256i *)(top_half_ptr + i + 16));
        auto value0 = _mm256_loadu_si256((__m256i *)(value_ptr + i));
        auto value1 = _mm256_loadu_si256((__m256i *)(value_ptr + i + 16));
        auto pack0_fp32 = pack_bf16_to_fp32(top0, bot0);
        auto pack1_fp32 = pack_bf16_to_fp32(top1, bot1);
        auto value0_fp32 = cvt_bf16_to_fp32(value0);
        auto value1_fp32 = cvt_bf16_to_fp32(value1);
        auto result0 = _mm512_fmadd_ps(alpha_vec, value0_fp32, pack0_fp32);
        auto result1 = _mm512_fmadd_ps(alpha_vec, value1_fp32, pack1_fp32);
        _mm256_storeu_si256((__m256i *)(top_half_ptr + i), trunc_fp32_to_bf16(result0));
        _mm256_storeu_si256((__m256i *)(top_half_ptr + i + 16), trunc_fp32_to_bf16(result1));
        _mm256_storeu_si256((__m256i *)(bot_half_ptr + i), _mm512_cvtepi32_epi16(_mm512_castps_si512(result0)));
        _mm256_storeu_si256((__m256i *)(bot_half_ptr + i + 16), _mm512_cvtepi32_epi16(_mm512_castps_si512(result1)));
      }
      for (; i < end; i++) {
        packed_bf16 p16;
        p16.s[0] = bot_half_ptr[i];
        p16.s[1] = top_half_ptr[i];
        p16.f += alpha_ * (float)(value_ptr[i]);
        bot_half_ptr[i] = p16.s[0];
        top_half_ptr[i] = p16.s[1];
      }
#else
      for (; i < end; i++) {
        packed_bf16 p16 = {};
        p16.s[0] = bot_half_ptr[i];
        p16.s[1] = top_half_ptr[i];
        p16.f += alpha_ * (float)(value_ptr[i]);
        bot_half_ptr[i] = p16.s[0];
        top_half_ptr[i] = p16.s[1];
      }
#endif
    });
  }

  if (!top_half_.is_contiguous()) {
    top_half_.copy_(top_half);
  }
  if (!bot_half_.is_contiguous()) {
    bot_half_.copy_(bot_half);
  }
}

}  // namespace cpu
}  // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def("packed_add(Tensor top_half, Tensor bot_half, Tensor grad, float alpha) -> ()", torch_ipex::cpu::packed_add);
}

}
