#include <ATen/Tensor.h>
#include <ATen/native/CPUBlas.h>
#include <aten/Decode.h>
#include <aten/FlashAttention.h>
#include <aten/Gemm.h>
#include <aten/MaskedMultiHeadAttention.h>
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>
#include <limits>
#include "../../utils/isa_utils.h"
#include "vec/vec.h"
namespace torch_ipex {
namespace cpu {

namespace {

inline bool is_first_token_optimizable(at::Tensor key) {
  return key.scalar_type() == at::kFloat ||
      key.scalar_type() == at::kBFloat16 ||
      (key.scalar_type() == at::kHalf && utils::isa_has_avx512_fp16_support());
}

template <typename T, typename KT, typename CT>
inline void reduce_head(
    const T* q_ptr_start,
    const KT* k_ptr_start,
    float* attn_w_pos,
    int64_t head_size,
    bool store_key,
    CT* k_cache_start) {
  for (auto hsi = 0; hsi < head_size; hsi++) {
    if (store_key) {
      k_cache_start[hsi] =
          static_cast<CT>(k_ptr_start[hsi]); // cat the key into the key_cache.
    }
    attn_w_pos[0] += static_cast<float>(q_ptr_start[hsi]) *
        static_cast<float>(k_ptr_start[hsi]);
  }
}
template <typename T, typename KT, typename CT>
inline void reduce_head(
    const T* q_ptr_start,
    const KT* k_ptr_start,
    CT* attn_w_pos,
    int64_t head_size) {
  float sum = 0;
  for (auto hsi = 0; hsi < head_size; hsi++) {
    sum += static_cast<float>(q_ptr_start[hsi]) *
        static_cast<float>(k_ptr_start[hsi]);
  }
  attn_w_pos[0] = static_cast<CT>(sum);
}
#if defined(CPU_CAPABILITY_AVX512)
template <>
inline void reduce_head(
    const float* q_ptr_start,
    const float* k_ptr_start,
    float* attn_w_pos,
    int64_t head_size,
    bool store_key,
    float* k_cache_start) {
  auto hsi = 0;
  auto vec_size = 16; // 512/32
  auto qk_sum_vec = _mm512_setzero_ps();
  for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
    auto q_vec = _mm512_loadu_ps(q_ptr_start + hsi);
    auto k_vec = _mm512_loadu_ps(k_ptr_start + hsi);
    if (store_key) {
      _mm512_storeu_ps(k_cache_start + hsi, k_vec);
    }
    qk_sum_vec = _mm512_fmadd_ps(q_vec, k_vec, qk_sum_vec);
  }
  attn_w_pos[0] += _mm512_reduce_add_ps(qk_sum_vec);
  for (; hsi < head_size; hsi++) {
    if (store_key) {
      k_cache_start[hsi] = k_ptr_start[hsi]; // cat the key into the key_cache.
    }
    attn_w_pos[0] += q_ptr_start[hsi] * k_ptr_start[hsi];
  }
  return;
}

template <>
inline void reduce_head(
    const at::BFloat16* q_ptr_start,
    const at::BFloat16* k_ptr_start,
    float* attn_w_pos,
    int64_t head_size,
    bool store_key,
    at::BFloat16* k_cache_start) {
  auto hsi = 0;
  auto vec_size = 16; // 512/32
  auto qk_sum_vec = _mm512_setzero_ps();
  for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
    // load 16 bfloat16 query from q_ptr_start and convert to 16 float32 values
    auto q_vec_bf16 = _mm256_loadu_si256((__m256i*)(q_ptr_start + hsi));
    auto q_vec_fp32 = torch_ipex::cpu::kernel::convert_bf16_to_fp32(q_vec_bf16);
    // load 16 bfloat16 key from k_ptr_start and convert to 16 float32 values
    auto k_vec_bf16 = _mm256_loadu_si256((__m256i*)(k_ptr_start + hsi));
    auto k_vec_fp32 = torch_ipex::cpu::kernel::convert_bf16_to_fp32(k_vec_bf16);
    if (store_key) {
      _mm256_storeu_si256((__m256i*)(k_cache_start + hsi), k_vec_bf16);
    }
    qk_sum_vec = _mm512_fmadd_ps(q_vec_fp32, k_vec_fp32, qk_sum_vec);
  }
  attn_w_pos[0] += _mm512_reduce_add_ps(qk_sum_vec);
  for (; hsi < head_size; hsi++) {
    if (store_key) {
      k_cache_start[hsi] = k_ptr_start[hsi]; // cat the key into the key_cache.
    }
    attn_w_pos[0] += q_ptr_start[hsi] * k_ptr_start[hsi];
  }
  return;
}

template <>
inline void reduce_head(
    const at::Half* q_ptr_start,
    const at::Half* k_ptr_start,
    float* attn_w_pos,
    int64_t head_size,
    bool store_key,
    at::Half* k_cache_start) {
  auto hsi = 0;
  auto vec_size = 16; // 512/32
  auto qk_sum_vec = _mm512_setzero_ps();
  for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
    // load 16 float16 query from q_ptr_start and convert to 16 float32 values
    auto q_vec_fp16 = _mm256_loadu_si256((__m256i*)(q_ptr_start + hsi));
    auto q_vec_fp32 = cvt_fp16_to_fp32(q_vec_fp16);
    // load 16 float16 key from k_ptr_start and convert to 16 float32 values
    auto k_vec_fp16 = _mm256_loadu_si256((__m256i*)(k_ptr_start + hsi));
    auto k_vec_fp32 = cvt_fp16_to_fp32(k_vec_fp16);
    if (store_key) {
      _mm256_storeu_si256((__m256i*)(k_cache_start + hsi), k_vec_fp16);
    }
    qk_sum_vec = _mm512_fmadd_ps(q_vec_fp32, k_vec_fp32, qk_sum_vec);
  }
  attn_w_pos[0] += _mm512_reduce_add_ps(qk_sum_vec);
  for (; hsi < head_size; hsi++) {
    if (store_key) {
      k_cache_start[hsi] = k_ptr_start[hsi]; // cat the key into the key_cache.
    }
    attn_w_pos[0] += q_ptr_start[hsi] * k_ptr_start[hsi];
  }
  return;
}

template <>
inline void reduce_head(
    const float* q_ptr_start,
    const float* k_ptr_start,
    float* attn_w_pos,
    int64_t head_size) {
  auto hsi = 0;
  auto vec_size = 16; // 512/32
  auto qk_sum_vec = _mm512_setzero_ps();
  for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
    auto q_vec = _mm512_loadu_ps(q_ptr_start + hsi);
    auto k_vec = _mm512_loadu_ps(k_ptr_start + hsi);
    qk_sum_vec = _mm512_fmadd_ps(q_vec, k_vec, qk_sum_vec);
  }
  attn_w_pos[0] += _mm512_reduce_add_ps(qk_sum_vec);
  for (; hsi < head_size; hsi++) {
    attn_w_pos[0] += q_ptr_start[hsi] * k_ptr_start[hsi];
  }
  return;
}

template <>
inline void reduce_head(
    const at::BFloat16* q_ptr_start,
    const at::BFloat16* k_ptr_start,
    at::BFloat16* attn_w_pos,
    int64_t head_size) {
  auto hsi = 0;
  auto vec_size = 16; // 512/32
  auto qk_sum_vec = _mm512_setzero_ps();
  float sum = 0;
  for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
    // load 16 bfloat16 query from q_ptr_start and convert to 16 float32 values
    auto q_vec_bf16 = _mm256_loadu_si256((__m256i*)(q_ptr_start + hsi));
    auto q_vec_fp32 = torch_ipex::cpu::kernel::convert_bf16_to_fp32(q_vec_bf16);
    // load 16 bfloat16 key from k_ptr_start and convert to 16 float32 values
    auto k_vec_bf16 = _mm256_loadu_si256((__m256i*)(k_ptr_start + hsi));
    auto k_vec_fp32 = torch_ipex::cpu::kernel::convert_bf16_to_fp32(k_vec_bf16);
    qk_sum_vec = _mm512_fmadd_ps(q_vec_fp32, k_vec_fp32, qk_sum_vec);
  }
  sum += _mm512_reduce_add_ps(qk_sum_vec);
  for (; hsi < head_size; hsi++) {
    sum += q_ptr_start[hsi] * k_ptr_start[hsi];
  }
  attn_w_pos[0] = static_cast<at::BFloat16>(sum);
  return;
}

template <>
inline void reduce_head(
    const at::Half* q_ptr_start,
    const at::Half* k_ptr_start,
    at::Half* attn_w_pos,
    int64_t head_size) {
  auto hsi = 0;
  auto vec_size = 16; // 512/32
  auto qk_sum_vec = _mm512_setzero_ps();
  for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
    // load 16 float16 query from q_ptr_start and convert to 16 float32 values
    auto q_vec_fp16 = _mm256_loadu_si256((__m256i*)(q_ptr_start + hsi));
    auto q_vec_fp32 = cvt_fp16_to_fp32(q_vec_fp16);
    // load 16 float16 key from k_ptr_start and convert to 16 float32 values
    auto k_vec_fp16 = _mm256_loadu_si256((__m256i*)(k_ptr_start + hsi));
    auto k_vec_fp32 = cvt_fp16_to_fp32(k_vec_fp16);
    qk_sum_vec = _mm512_fmadd_ps(q_vec_fp32, k_vec_fp32, qk_sum_vec);
  }
  attn_w_pos[0] += _mm512_reduce_add_ps(qk_sum_vec);
  for (; hsi < head_size; hsi++) {
    attn_w_pos[0] += q_ptr_start[hsi] * k_ptr_start[hsi];
  }
  return;
}

#endif

#if defined(CPU_CAPABILITY_AVX512_FP16)
inline void reduce_head_half(
    const at::Half* q_ptr_start,
    const at::Half* k_ptr_start,
    at::Half* attn_w_pos,
    int64_t head_size,
    bool store_key,
    at::Half* k_cache_start) {
  auto hsi = 0;
  auto vec_size = 32; // 512/16
  auto qk_sum_vec = _mm512_setzero_ph();
  for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
    auto q_vec = _mm512_loadu_ph(q_ptr_start + hsi);
    auto k_vec = _mm512_loadu_ph(k_ptr_start + hsi);
    if (store_key) {
      _mm512_storeu_ph(k_cache_start + hsi, k_vec);
    }
    qk_sum_vec = _mm512_fmadd_ph(q_vec, k_vec, qk_sum_vec);
  }
  attn_w_pos[0] += _mm512_reduce_add_ph(qk_sum_vec);
  for (; hsi < head_size; hsi++) {
    if (store_key) {
      k_cache_start[hsi] = k_ptr_start[hsi]; // cat the key into the key_cache.
    }
    attn_w_pos[0] += q_ptr_start[hsi] * k_ptr_start[hsi];
  }
}

inline void reduce_head_half(
    const at::Half* q_ptr_start,
    int64_t kv_head_group_size,
    const at::Half* k_ptr_start,
    at::Half* attn_w_pos,
    int attn_w_stride,
    int64_t head_size,
    bool store_key,
    at::Half* k_cache_start) {
  for (auto i = 0; i < kv_head_group_size; i++) {
    attn_w_pos[i * attn_w_stride] = 0;
    reduce_head_half(
        q_ptr_start + i * head_size,
        k_ptr_start,
        attn_w_pos + i * attn_w_stride,
        head_size,
        store_key,
        k_cache_start);
  }
}

inline void reduce_head_half(
    const at::Half* q_ptr_start,
    int qStrideB,
    int64_t kv_head_group_size,
    const at::Half* k_ptr_start,
    at::Half* attn_w_pos,
    int attn_w_stride,
    int64_t head_size,
    int64_t beam_size) {
  for (auto i = 0; i < kv_head_group_size; i++) {
    for (auto b = 0; b < beam_size; b++) {
      attn_w_pos[i * attn_w_stride + b] = 0;
      reduce_head_half(
          q_ptr_start + i * head_size + b * qStrideB,
          k_ptr_start,
          attn_w_pos + i * attn_w_stride + b,
          head_size,
          false,
          nullptr);
    }
  }
}
#endif

template <>
inline void reduce_head(
    const at::BFloat16* q_ptr_start,
    const at::BFloat16* k_ptr_start,
    float* attn_w_pos,
    int64_t head_size,
    bool store_key,
    at::Float8_e5m2* k_cache_start) {
  auto hsi = 0;
#if defined(CPU_CAPABILITY_AVX512_FP16)
  auto vec_size = 32; // 512/16
  auto qk_sum_vec = _mm512_setzero_ps();
  const __m512i vnaninf = _mm512_set1_epi16(0x7c00);
  const __m512i vrneadd = _mm512_set1_epi16(0x007f);
  const __m512i vfixup = _mm512_set1_epi16(0x0001);
  const __m512i vfixupmask = _mm512_set1_epi16(0x0100);
  for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
    // load 16 bfloat16 query from q_ptr_start and convert to 16 float32 values
    auto q0_vec_bf16 = _mm256_loadu_si256((__m256i*)(q_ptr_start + hsi));
    auto q0_vec_fp32 =
        torch_ipex::cpu::kernel::convert_bf16_to_fp32(q0_vec_bf16);
    // load 16 bfloat16 key from k_ptr_start and convert to 16 float32 values
    auto k0_vec_bf16 = _mm256_loadu_si256((__m256i*)(k_ptr_start + hsi));
    auto k0_vec_fp32 =
        torch_ipex::cpu::kernel::convert_bf16_to_fp32(k0_vec_bf16);
    // load 16 bfloat16 query from q_ptr_start and convert to 16 float32 values
    auto q1_vec_bf16 = _mm256_loadu_si256((__m256i*)(q_ptr_start + hsi + 16));
    auto q1_vec_fp32 =
        torch_ipex::cpu::kernel::convert_bf16_to_fp32(q1_vec_bf16);
    // load 16 bfloat16 key from k_ptr_start and convert to 16 float32 values
    auto k1_vec_bf16 = _mm256_loadu_si256((__m256i*)(k_ptr_start + hsi + 16));
    auto k1_vec_fp32 =
        torch_ipex::cpu::kernel::convert_bf16_to_fp32(k1_vec_bf16);
    if (store_key) {
      __m512 b = k0_vec_fp32;
      __m512 a = k1_vec_fp32;
      __m256i ah_ =
          _mm512_cvtps_ph(a, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      __m256i bh_ =
          _mm512_cvtps_ph(b, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      const __m512i a_ = _mm512_inserti64x4(
          _mm512_inserti64x4(_mm512_setzero_si512(), bh_, 0), ah_, 1);
      const __mmask32 maska1_ = _mm512_cmp_epi16_mask(
          _mm512_and_si512(a_, vnaninf), vnaninf, _MM_CMPINT_NE);
      const __mmask32 maska2_ = _mm512_cmp_epi16_mask(
          _mm512_and_si512(a_, vfixupmask), vfixupmask, _MM_CMPINT_EQ);
      __m512i a_rne_ = _mm512_mask_add_epi16(
          a_,
          maska1_,
          a_,
          _mm512_mask_add_epi16(vrneadd, maska2_, vrneadd, vfixup));
      a_rne_ = _mm512_srli_epi16(a_rne_, 8);
      _mm256_storeu_epi8(&k_cache_start[hsi], _mm512_cvtepi16_epi8(a_rne_));
    }
    qk_sum_vec = _mm512_fmadd_ps(q0_vec_fp32, k0_vec_fp32, qk_sum_vec);
    qk_sum_vec = _mm512_fmadd_ps(q1_vec_fp32, k1_vec_fp32, qk_sum_vec);
  }
  attn_w_pos[0] += _mm512_reduce_add_ps(qk_sum_vec);
#endif
  for (; hsi < head_size; hsi++) {
    if (store_key) {
      k_cache_start[hsi] = static_cast<at::Float8_e5m2>(
          k_ptr_start[hsi]); // cat the key into the key_cache.
    }
    attn_w_pos[0] += q_ptr_start[hsi] * k_ptr_start[hsi];
  }
  return;
}
template <>
inline void reduce_head(
    const at::BFloat16* q_ptr_start,
    const at::Float8_e5m2* k_ptr_start,
    float* attn_w_pos,
    int64_t head_size,
    bool store_key,
    at::Float8_e5m2* k_cache_start) {
  TORCH_CHECK(store_key == false, "not support store_key");
  auto hsi = 0;
#if defined(CPU_CAPABILITY_AVX512_FP16)
  auto vec_size = 32; // 512/16
  auto qk_sum_vec = _mm512_setzero_ps();
  for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
    // load 16 bfloat16 query from q_ptr_start and convert to 16 float32 values
    auto q0_vec_bf16 = _mm256_loadu_si256((__m256i*)(q_ptr_start + hsi));
    auto q0_vec_fp32 =
        torch_ipex::cpu::kernel::convert_bf16_to_fp32(q0_vec_bf16);
    // load 16 bfloat16 query from q_ptr_start and convert to 16 float32 values
    auto q1_vec_bf16 = _mm256_loadu_si256((__m256i*)(q_ptr_start + hsi + 16));
    auto q1_vec_fp32 =
        torch_ipex::cpu::kernel::convert_bf16_to_fp32(q1_vec_bf16);
    // load 32 e5m2 key from k_ptr_start and convert to 2 x 16 float32 values
    auto k_vec_ = torch_ipex::cpu::kernel::_mm512_cvte5m2_fp16(
        _mm256_loadu_si256((__m256i*)&k_ptr_start[hsi]));
    auto k0_vec_fp32 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(k_vec_, 0));
    auto k1_vec_fp32 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(k_vec_, 1));
    qk_sum_vec = _mm512_fmadd_ps(q0_vec_fp32, k0_vec_fp32, qk_sum_vec);
    qk_sum_vec = _mm512_fmadd_ps(q1_vec_fp32, k1_vec_fp32, qk_sum_vec);
  }
  attn_w_pos[0] += _mm512_reduce_add_ps(qk_sum_vec);
#endif
  for (; hsi < head_size; hsi++) {
    attn_w_pos[0] += q_ptr_start[hsi] * static_cast<float>(k_ptr_start[hsi]);
  }
  return;
}

template <typename T, typename KT, typename CT>
inline void reduce_head(
    const T* q_ptr_start,
    int64_t kv_head_group_size,
    const KT* k_ptr_start,
    float* attn_w_pos,
    int attn_w_stride,
    int64_t head_size,
    bool store_key,
    CT* k_cache_start) {
  for (auto i = 0; i < kv_head_group_size; i++) {
    attn_w_pos[i * attn_w_stride] = 0;
    reduce_head<T, KT, CT>(
        q_ptr_start + i * head_size,
        k_ptr_start,
        attn_w_pos + i * attn_w_stride,
        head_size,
        store_key,
        k_cache_start);
  }
}

template <typename T, typename KT>
inline void reduce_head(
    const T* q_ptr_start,
    int qStrideB,
    int64_t kv_head_group_size,
    const KT* k_ptr_start,
    float* attn_w_pos,
    int attn_w_stride,
    int64_t head_size,
    int64_t beam_size) {
  for (auto i = 0; i < kv_head_group_size; i++) {
    for (auto b = 0; b < beam_size; b++) {
      attn_w_pos[i * attn_w_stride + b] = 0;
      reduce_head<T, KT, KT>(
          q_ptr_start + i * head_size + b * qStrideB,
          k_ptr_start,
          attn_w_pos + i * attn_w_stride + b,
          head_size,
          false,
          nullptr);
    }
  }
}
/*
 *reduce the attention_weights with the value embedding by the dimension of
 *head_size for every head
 */
template <typename T, typename T1, typename CT>
inline void mul_attenion_weights_and_value_of_head(
    float& attn_w,
    const T* v_ptr_start,
    T1* attn_out_start,
    int64_t head_size,
    bool store_value,
    CT* v_cache_start,
    bool accumulate) {
  for (auto hsi = 0; hsi < head_size; hsi++) {
    if (accumulate) {
      attn_out_start[hsi] += attn_w * static_cast<float>(v_ptr_start[hsi]);
    } else {
      attn_out_start[hsi] = attn_w * static_cast<float>(v_ptr_start[hsi]);
    }
    if (store_value) {
      v_cache_start[hsi] = static_cast<CT>(v_ptr_start[hsi]);
    }
  }
}

template <typename T, typename T1, typename CT>
inline void mul_attenion_weights_and_value_of_head(
    float* attn_w,
    int attn_w_stride,
    const T* v_ptr_start,
    T1* attn_out_start,
    int attn_out_strideH,
    int kv_head_group_size,
    int64_t head_size,
    bool store_value,
    CT* v_cache_start,
    uint8_t* flag_access) {
  for (auto i = 0; i < kv_head_group_size; i++) {
    mul_attenion_weights_and_value_of_head<T, T1, CT>(
        attn_w[i * attn_w_stride],
        v_ptr_start,
        attn_out_start + i * attn_out_strideH,
        head_size,
        store_value,
        v_cache_start,
        flag_access[i]);
    if (flag_access[i] == 0)
      flag_access[i] = 1;
  }
}

template <typename T, typename T1>
inline void mul_attenion_weights_and_value_of_head(
    float* attn_w,
    int attn_w_stride,
    const T* v_ptr_start,
    T1* attn_out_start,
    int attn_out_strideB,
    int attn_out_strideH,
    int kv_head_group_size,
    int64_t head_size,
    bool store_value,
    T* v_cache_start,
    uint8_t* flag_access,
    int flag_access_stride,
    int64_t beam_size) {
  for (auto i = 0; i < kv_head_group_size; i++) {
    for (auto b = 0; b < beam_size; b++) {
      mul_attenion_weights_and_value_of_head<T, T1, T>(
          attn_w[i * attn_w_stride + b],
          v_ptr_start,
          attn_out_start + i * attn_out_strideH + b * attn_out_strideB,
          head_size,
          store_value,
          v_cache_start,
          flag_access[b * flag_access_stride + i]);
      if (flag_access[b * flag_access_stride + i] == 0)
        flag_access[b * flag_access_stride + i] = 1;
    }
  }
}

#if defined(CPU_CAPABILITY_AVX512)
template <>
inline void mul_attenion_weights_and_value_of_head(
    float& attn_w,
    const float* v_ptr_start,
    float* attn_out_start,
    int64_t head_size,
    bool store_value,
    float* v_cache_start,
    bool accumulate) {
  auto hsi = 0;
  auto vec_size = 16; // 512/32
  for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
    auto attn_w_vec = _mm512_set1_ps(attn_w);
    auto v_vec = _mm512_loadu_ps(v_ptr_start + hsi);
    if (accumulate) {
      auto attn_out_vec = _mm512_loadu_ps(attn_out_start + hsi);
      auto attn_out_vec_new = _mm512_fmadd_ps(attn_w_vec, v_vec, attn_out_vec);
      _mm512_storeu_ps(attn_out_start + hsi, attn_out_vec_new);
    } else {
      auto attn_out_vec_new = _mm512_mul_ps(attn_w_vec, v_vec);
      _mm512_storeu_ps(attn_out_start + hsi, attn_out_vec_new);
    }
    if (store_value) {
      _mm512_storeu_ps(v_cache_start + hsi, v_vec);
    }
  }
  for (; hsi < head_size; hsi++) {
    if (accumulate) {
      attn_out_start[hsi] += attn_w * v_ptr_start[hsi];
    } else {
      attn_out_start[hsi] = attn_w * v_ptr_start[hsi];
    }
    if (store_value) {
      v_cache_start[hsi] = v_ptr_start[hsi];
    }
  }
  return;
}

template <>
inline void mul_attenion_weights_and_value_of_head(
    float& attn_w,
    const at::BFloat16* v_ptr_start,
    at::BFloat16* attn_out_start,
    int64_t head_size,
    bool store_value,
    at::BFloat16* v_cache_start,
    bool accumulate) {
  auto hsi = 0;
  auto vec_size = 16; // 512/32
  for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
    // get 1 bfloat16 values from attn_w_ptr_start and broadcast to 16 float32
    // values
    auto attn_w_vec_fp32 = _mm512_set1_ps(attn_w);
    // load 16 bfloat16 values from v_ptr_start and convert to 16 float32 values
    auto v_vec_bf16 = _mm256_loadu_si256((__m256i*)(v_ptr_start + hsi));
    auto v_vec_fp32 = torch_ipex::cpu::kernel::convert_bf16_to_fp32(v_vec_bf16);
    if (accumulate) {
      // load 16 bfloat16 values from attn_out_start and convert to 16 float32
      // values
      auto attn_out_vec_fp32 = torch_ipex::cpu::kernel::convert_bf16_to_fp32(
          _mm256_loadu_si256((__m256i*)(attn_out_start + hsi)));
      // calculate the new attn_out_vec_fp32 and convert to bfloat16
      auto attn_out_vec_new =
          _mm512_fmadd_ps(attn_w_vec_fp32, v_vec_fp32, attn_out_vec_fp32);
      auto attn_out_vec_new_bf16 = cvt_fp32_to_bf16(attn_out_vec_new); //_m256i
      // store the new attn_out_vec_new_bf16 to attn_outs
      _mm256_storeu_si256(
          (__m256i*)(attn_out_start + hsi), attn_out_vec_new_bf16);
    } else {
      // calculate the new attn_out_vec_fp32 and convert to bfloat16
      auto attn_out_vec_new = _mm512_mul_ps(attn_w_vec_fp32, v_vec_fp32);
      auto attn_out_vec_new_bf16 = cvt_fp32_to_bf16(attn_out_vec_new); //_m256i
      // store the new attn_out_vec_new_bf16 to attn_outs
      _mm256_storeu_si256(
          (__m256i*)(attn_out_start + hsi), attn_out_vec_new_bf16);
    }
    // store the v_vec_bf16 to v_cache
    if (store_value) {
      _mm256_storeu_si256((__m256i*)(v_cache_start + hsi), v_vec_bf16);
    }
  }
  for (; hsi < head_size; hsi++) {
    if (accumulate) {
      attn_out_start[hsi] += attn_w * v_ptr_start[hsi];
    } else {
      attn_out_start[hsi] = attn_w * v_ptr_start[hsi];
    }
    if (store_value) {
      v_cache_start[hsi] = v_ptr_start[hsi];
    }
  }
  return;
}
template <>
inline void mul_attenion_weights_and_value_of_head(
    float& attn_w,
    const at::BFloat16* v_ptr_start,
    float* attn_out_start,
    int64_t head_size,
    bool store_value,
    at::BFloat16* v_cache_start,
    bool accumulate) {
  auto hsi = 0;
  auto vec_size = 16; // 512/32
  for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
    // get 1 bfloat16 values from attn_w_ptr_start and broadcast to 16 float32
    // values
    auto attn_w_vec_fp32 = _mm512_set1_ps(attn_w);
    // load 16 bfloat16 values from v_ptr_start and convert to 16 float32 values
    auto v_vec_bf16 = _mm256_loadu_si256((__m256i*)(v_ptr_start + hsi));
    auto v_vec_fp32 = torch_ipex::cpu::kernel::convert_bf16_to_fp32(v_vec_bf16);
    if (accumulate) {
      auto attn_out_vec_fp32 = _mm512_loadu_ps(attn_out_start + hsi);
      // calculate the new attn_out_vec_fp32 and convert to bfloat16
      auto attn_out_vec_new =
          _mm512_fmadd_ps(attn_w_vec_fp32, v_vec_fp32, attn_out_vec_fp32);
      _mm512_storeu_ps(attn_out_start + hsi, attn_out_vec_new);
    } else {
      // calculate the new attn_out_vec_fp32 and convert to bfloat16
      auto attn_out_vec_new = _mm512_mul_ps(attn_w_vec_fp32, v_vec_fp32);
      _mm512_storeu_ps(attn_out_start + hsi, attn_out_vec_new);
    }
    // store the v_vec_bf16 to v_cache
    if (store_value) {
      _mm256_storeu_si256((__m256i*)(v_cache_start + hsi), v_vec_bf16);
    }
  }
  for (; hsi < head_size; hsi++) {
    if (accumulate) {
      attn_out_start[hsi] += attn_w * v_ptr_start[hsi];
    } else {
      attn_out_start[hsi] = attn_w * v_ptr_start[hsi];
    }
    if (store_value) {
      v_cache_start[hsi] = v_ptr_start[hsi];
    }
  }
  return;
}

template <>
inline void mul_attenion_weights_and_value_of_head(
    float& attn_w,
    const at::Half* v_ptr_start,
    at::Half* attn_out_start,
    int64_t head_size,
    bool store_value,
    at::Half* v_cache_start,
    bool accumulate) {
  auto hsi = 0;
  auto vec_size = 16; // 512/32
  for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
    // get 1 float16 values from attn_w_ptr_start and broadcast to 16 float32
    // values
    auto attn_w_vec_fp32 = _mm512_set1_ps(attn_w);
    // load 16 float16 values from v_ptr_start and convert to 16 float32 values
    auto v_vec_fp16 = _mm256_loadu_si256((__m256i*)(v_ptr_start + hsi));
    auto v_vec_fp32 = cvt_fp16_to_fp32(v_vec_fp16);
    if (accumulate) {
      // load 16 float16 values from attn_out_start and convert to 16 float32
      // values
      auto attn_out_vec_fp32 = cvt_fp16_to_fp32(
          _mm256_loadu_si256((__m256i*)(attn_out_start + hsi)));
      // calculate the new attn_out_vec_fp32 and convert to float16
      auto attn_out_vec_new =
          _mm512_fmadd_ps(attn_w_vec_fp32, v_vec_fp32, attn_out_vec_fp32);
      auto attn_out_vec_new_fp16 = cvt_fp32_to_fp16(attn_out_vec_new); //_m256i
      // store the new attn_out_vec_new_fp16 to attn_outs
      _mm256_storeu_si256(
          (__m256i*)(attn_out_start + hsi), attn_out_vec_new_fp16);
    } else {
      // calculate the new attn_out_vec_fp32 and convert to float16
      auto attn_out_vec_new = _mm512_mul_ps(attn_w_vec_fp32, v_vec_fp32);
      auto attn_out_vec_new_fp16 = cvt_fp32_to_fp16(attn_out_vec_new); //_m256i
      // store the new attn_out_vec_new_fp16 to attn_outs
      _mm256_storeu_si256(
          (__m256i*)(attn_out_start + hsi), attn_out_vec_new_fp16);
    }
    // store the v_vec_fp16 to v_cache
    if (store_value) {
      _mm256_storeu_si256((__m256i*)(v_cache_start + hsi), v_vec_fp16);
    }
  }
  for (; hsi < head_size; hsi++) {
    if (accumulate) {
      attn_out_start[hsi] += attn_w * v_ptr_start[hsi];
    } else {
      attn_out_start[hsi] = attn_w * v_ptr_start[hsi];
    }
    if (store_value) {
      v_cache_start[hsi] = v_ptr_start[hsi];
    }
  }
  return;
}
template <>
inline void mul_attenion_weights_and_value_of_head(
    float& attn_w,
    const at::Half* v_ptr_start,
    float* attn_out_start,
    int64_t head_size,
    bool store_value,
    at::Half* v_cache_start,
    bool accumulate) {
  auto hsi = 0;
  auto vec_size = 16; // 512/32
  for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
    // get 1 float16 values from attn_w_ptr_start and broadcast to 16 float32
    // values
    auto attn_w_vec_fp32 = _mm512_set1_ps(attn_w);
    // load 16 float16 values from v_ptr_start and convert to 16 float32 values
    auto v_vec_fp16 = _mm256_loadu_si256((__m256i*)(v_ptr_start + hsi));
    auto v_vec_fp32 = cvt_fp16_to_fp32(v_vec_fp16);
    if (accumulate) {
      auto attn_out_vec_fp32 = _mm512_loadu_ps(attn_out_start + hsi);
      // calculate the new attn_out_vec_fp32 and convert to float16
      auto attn_out_vec_new =
          _mm512_fmadd_ps(attn_w_vec_fp32, v_vec_fp32, attn_out_vec_fp32);
      _mm512_storeu_ps(attn_out_start + hsi, attn_out_vec_new);
    } else {
      // calculate the new attn_out_vec_fp32 and convert to float16
      auto attn_out_vec_new = _mm512_mul_ps(attn_w_vec_fp32, v_vec_fp32);
      _mm512_storeu_ps(attn_out_start + hsi, attn_out_vec_new);
    }
    // store the v_vec_fp16 to v_cache
    if (store_value) {
      _mm256_storeu_si256((__m256i*)(v_cache_start + hsi), v_vec_fp16);
    }
  }
  for (; hsi < head_size; hsi++) {
    if (accumulate) {
      attn_out_start[hsi] += attn_w * v_ptr_start[hsi];
    } else {
      attn_out_start[hsi] = attn_w * v_ptr_start[hsi];
    }
    if (store_value) {
      v_cache_start[hsi] = v_ptr_start[hsi];
    }
  }
  return;
}
#endif

#if defined(CPU_CAPABILITY_AVX512_FP16)
inline void mul_attenion_weights_and_value_of_head_half(
    at::Half& attn_w,
    const at::Half* v_ptr_start,
    at::Half* attn_out_start,
    int64_t head_size,
    bool store_value,
    at::Half* v_cache_start,
    bool accumulate) {
  auto hsi = 0;
  auto vec_size = 32; // 512/16
  for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
    auto attn_w_vec = _mm512_set1_ph(*(_Float16*)&attn_w);
    auto v_vec = _mm512_loadu_ph(v_ptr_start + hsi);
    if (accumulate) {
      auto attn_out_vec = _mm512_loadu_ph(attn_out_start + hsi);
      auto attn_out_vec_new = _mm512_fmadd_ph(attn_w_vec, v_vec, attn_out_vec);
      _mm512_storeu_ph(attn_out_start + hsi, attn_out_vec_new);
    } else {
      auto attn_out_vec_new = _mm512_mul_ph(attn_w_vec, v_vec);
      _mm512_storeu_ph(attn_out_start + hsi, attn_out_vec_new);
    }
    // store the v_vec to v_cache
    if (store_value) {
      _mm512_storeu_ph(v_cache_start + hsi, v_vec);
    }
  }
  for (; hsi < head_size; hsi++) {
    if (accumulate) {
      attn_out_start[hsi] += attn_w * v_ptr_start[hsi];
    } else {
      attn_out_start[hsi] = attn_w * v_ptr_start[hsi];
    }
    if (store_value) {
      v_cache_start[hsi] = v_ptr_start[hsi];
    }
  }
}

inline void mul_attenion_weights_and_value_of_head_half(
    at::Half* attn_w,
    int attn_w_stride,
    const at::Half* v_ptr_start,
    at::Half* attn_out_start,
    int attn_out_strideH,
    int kv_head_group_size,
    int64_t head_size,
    bool store_value,
    at::Half* v_cache_start,
    uint8_t* flag_access) {
  for (auto i = 0; i < kv_head_group_size; i++) {
    mul_attenion_weights_and_value_of_head_half(
        attn_w[i * attn_w_stride],
        v_ptr_start,
        attn_out_start + i * attn_out_strideH,
        head_size,
        store_value,
        v_cache_start,
        flag_access[i]);
    if (flag_access[i] == 0)
      flag_access[i] = 1;
  }
}
inline void mul_attenion_weights_and_value_of_head_half(
    at::Half* attn_w,
    int attn_w_stride,
    const at::Half* v_ptr_start,
    at::Half* attn_out_start,
    int attn_out_strideB,
    int attn_out_strideH,
    int kv_head_group_size,
    int64_t head_size,
    bool store_value,
    at::Half* v_cache_start,
    uint8_t* flag_access,
    int flag_access_stride,
    int64_t beam_size) {
  for (auto i = 0; i < kv_head_group_size; i++) {
    for (auto b = 0; b < beam_size; b++) {
      mul_attenion_weights_and_value_of_head_half(
          attn_w[i * attn_w_stride + b],
          v_ptr_start,
          attn_out_start + i * attn_out_strideH + b * attn_out_strideB,
          head_size,
          store_value,
          v_cache_start,
          flag_access[b * flag_access_stride + i]);
      if (flag_access[b * flag_access_stride + i] == 0)
        flag_access[b * flag_access_stride + i] = 1;
    }
  }
}
#endif

template <>
inline void mul_attenion_weights_and_value_of_head(
    float& attn_w,
    const at::BFloat16* v_ptr_start,
    float* attn_out_start,
    int64_t head_size,
    bool store_value,
    at::Float8_e5m2* v_cache_start,
    bool accumulate) {
  auto hsi = 0;
#if defined(CPU_CAPABILITY_AVX512_FP16)
  auto vec_size = 32; // 512/32
  const __m512i vnaninf = _mm512_set1_epi16(0x7c00);
  const __m512i vrneadd = _mm512_set1_epi16(0x007f);
  const __m512i vfixup = _mm512_set1_epi16(0x0001);
  const __m512i vfixupmask = _mm512_set1_epi16(0x0100);
  for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
    // get 1 bfloat16 values from attn_w_ptr_start and broadcast to 16 float32
    // values
    auto attn_w_vec_fp32 = _mm512_set1_ps(attn_w);
    // load 16 bfloat16 values from v_ptr_start and convert to 16 float32 values
    auto v0_vec_bf16 = _mm256_loadu_si256((__m256i*)(v_ptr_start + hsi));
    auto v0_vec_fp32 =
        torch_ipex::cpu::kernel::convert_bf16_to_fp32(v0_vec_bf16);
    auto v1_vec_bf16 = _mm256_loadu_si256((__m256i*)(v_ptr_start + hsi + 16));
    auto v1_vec_fp32 =
        torch_ipex::cpu::kernel::convert_bf16_to_fp32(v1_vec_bf16);
    if (accumulate) {
      auto attn_out0_vec_fp32 = _mm512_loadu_ps(attn_out_start + hsi);
      auto attn_out1_vec_fp32 = _mm512_loadu_ps(attn_out_start + hsi + 16);
      // calculate the new attn_out_vec_fp32 and convert to bfloat16
      auto attn_out0_vec_new =
          _mm512_fmadd_ps(attn_w_vec_fp32, v0_vec_fp32, attn_out0_vec_fp32);
      auto attn_out1_vec_new =
          _mm512_fmadd_ps(attn_w_vec_fp32, v1_vec_fp32, attn_out1_vec_fp32);
      _mm512_storeu_ps(attn_out_start + hsi, attn_out0_vec_new);
      _mm512_storeu_ps(attn_out_start + hsi + 16, attn_out1_vec_new);
    } else {
      // calculate the new attn_out_vec_fp32 and convert to bfloat16
      auto attn_out0_vec_new = _mm512_mul_ps(attn_w_vec_fp32, v0_vec_fp32);
      auto attn_out1_vec_new = _mm512_mul_ps(attn_w_vec_fp32, v1_vec_fp32);
      _mm512_storeu_ps(attn_out_start + hsi, attn_out0_vec_new);
      _mm512_storeu_ps(attn_out_start + hsi + 16, attn_out1_vec_new);
    }
    // store the v_vec_bf16 to v_cache
    if (store_value) {
      __m512 b = v0_vec_fp32;
      __m512 a = v1_vec_fp32;
      __m256i ah_ =
          _mm512_cvtps_ph(a, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      __m256i bh_ =
          _mm512_cvtps_ph(b, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      const __m512i a_ = _mm512_inserti64x4(
          _mm512_inserti64x4(_mm512_setzero_si512(), bh_, 0), ah_, 1);
      const __mmask32 maska1_ = _mm512_cmp_epi16_mask(
          _mm512_and_si512(a_, vnaninf), vnaninf, _MM_CMPINT_NE);
      const __mmask32 maska2_ = _mm512_cmp_epi16_mask(
          _mm512_and_si512(a_, vfixupmask), vfixupmask, _MM_CMPINT_EQ);
      __m512i a_rne_ = _mm512_mask_add_epi16(
          a_,
          maska1_,
          a_,
          _mm512_mask_add_epi16(vrneadd, maska2_, vrneadd, vfixup));
      a_rne_ = _mm512_srli_epi16(a_rne_, 8);
      _mm256_storeu_epi8(&v_cache_start[hsi], _mm512_cvtepi16_epi8(a_rne_));
    }
  }
#endif
  for (; hsi < head_size; hsi++) {
    if (accumulate) {
      attn_out_start[hsi] += attn_w * v_ptr_start[hsi];
    } else {
      attn_out_start[hsi] = attn_w * v_ptr_start[hsi];
    }
    if (store_value) {
      v_cache_start[hsi] = static_cast<at::Float8_e5m2>(v_ptr_start[hsi]);
    }
  }
  return;
}

template <>
inline void mul_attenion_weights_and_value_of_head<
    at::Float8_e5m2,
    float,
    at::Float8_e5m2>(
    float& attn_w,
    const at::Float8_e5m2* v_ptr_start,
    float* attn_out_start,
    int64_t head_size,
    bool store_value,
    at::Float8_e5m2* v_cache_start,
    bool accumulate) {
  TORCH_CHECK(store_value == false, "not support store_value");
  auto hsi = 0;
#if defined(CPU_CAPABILITY_AVX512_FP16)
  auto vec_size = 32; // 512/32
  for (hsi = 0; hsi <= head_size - vec_size; hsi += vec_size) {
    // get 1 bfloat16 values from attn_w_ptr_start and broadcast to 16 float32
    // values
    auto attn_w_vec_fp32 = _mm512_set1_ps(attn_w);
    // load 16 bfloat16 values from v_ptr_start and convert to 16 float32 values
    auto v_vec_ = torch_ipex::cpu::kernel::_mm512_cvte5m2_fp16(
        _mm256_loadu_si256((__m256i*)&v_ptr_start[hsi]));
    auto v0_vec_fp32 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(v_vec_, 0));
    auto v1_vec_fp32 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(v_vec_, 1));
    if (accumulate) {
      auto attn_out0_vec_fp32 = _mm512_loadu_ps(attn_out_start + hsi);
      auto attn_out1_vec_fp32 = _mm512_loadu_ps(attn_out_start + hsi + 16);
      // calculate the new attn_out_vec_fp32 and convert to bfloat16
      auto attn_out0_vec_new =
          _mm512_fmadd_ps(attn_w_vec_fp32, v0_vec_fp32, attn_out0_vec_fp32);
      auto attn_out1_vec_new =
          _mm512_fmadd_ps(attn_w_vec_fp32, v1_vec_fp32, attn_out1_vec_fp32);
      _mm512_storeu_ps(attn_out_start + hsi, attn_out0_vec_new);
      _mm512_storeu_ps(attn_out_start + hsi + 16, attn_out1_vec_new);
    } else {
      // calculate the new attn_out_vec_fp32 and convert to bfloat16
      auto attn_out0_vec_new = _mm512_mul_ps(attn_w_vec_fp32, v0_vec_fp32);
      auto attn_out1_vec_new = _mm512_mul_ps(attn_w_vec_fp32, v1_vec_fp32);
      _mm512_storeu_ps(attn_out_start + hsi, attn_out0_vec_new);
      _mm512_storeu_ps(attn_out_start + hsi + 16, attn_out1_vec_new);
    }
  }
#endif
  for (; hsi < head_size; hsi++) {
    if (accumulate) {
      attn_out_start[hsi] += attn_w * static_cast<float>(v_ptr_start[hsi]);
    } else {
      attn_out_start[hsi] = attn_w * static_cast<float>(v_ptr_start[hsi]);
    }
  }
  return;
}

template <typename T>
inline void move_and_convert(T* src_ptr, T* cache_ptr, int64_t len) {
  torch_ipex::cpu::kernel::move_ker<T, T>(cache_ptr, src_ptr, len);
}
inline void move_and_convert(
    at::BFloat16* src_ptr,
    at::Float8_e5m2* dst_ptr,
    int64_t len) {
#if defined(CPU_CAPABILITY_AVX512)
  torch_ipex::cpu::kernel::cvt_bf16_e5m2_rne_intrinsic(src_ptr, dst_ptr, len);
#else
  for (size_t i = 0; i < len; i++) {
    dst_ptr[i] = static_cast<at::Float8_e5m2>(src_ptr[i]);
  }
#endif
}
inline void move_and_convert(
    float* src_ptr,
    at::Float8_e5m2* dst_ptr,
    int64_t len) {
#if defined(CPU_CAPABILITY_AVX512)
  torch_ipex::cpu::kernel::cvt_fp32_e5m2_rne_intrinsic(src_ptr, dst_ptr, len);
#else
  for (size_t i = 0; i < len; i++) {
    dst_ptr[i] = static_cast<at::Float8_e5m2>(src_ptr[i]);
  }
#endif
}
template <typename T, typename CT>
inline void copy_key_value(
    at::Tensor key_cache,
    const at::Tensor key,
    at::Tensor value_cache,
    const at::Tensor value,
    int beam_batch) {
  RECORD_FUNCTION("ipex::copy_key_value", c10::ArrayRef<c10::IValue>({}));
  auto bs = key.size(0);
  auto seq_len = key.size(1); // only process cur_len==1
  auto head_num = key.size(2);
  auto head_size = key.size(3);
  auto hidden_size = head_num * head_size;
  auto key_cache_ptr = key_cache.data_ptr<CT>();
  auto key_ptr = key.data_ptr<T>();
  auto value_cache_ptr = value_cache.data_ptr<CT>();
  auto value_ptr = value.data_ptr<T>();
  auto token_stride = beam_batch * hidden_size;
  auto beam_size = beam_batch / bs;
#pragma omp parallel for collapse(2)
  for (auto si = 0; si < seq_len; si++) {
    for (auto bi = 0; bi < bs; bi++) {
      auto cache_stride = si * token_stride + bi * beam_size * hidden_size;
      auto state_stride = (bi * seq_len + si) * hidden_size;
      auto key_cache_start = key_cache_ptr + cache_stride;
      auto key_ptr_start = key_ptr + state_stride;
      move_and_convert(key_ptr_start, key_cache_start, hidden_size);
      auto value_cache_ptr_start = value_cache_ptr + cache_stride;
      auto value_ptr_start = value_ptr + state_stride;
      move_and_convert(value_ptr_start, value_cache_ptr_start, hidden_size);
    }
  }
}

template <typename T, typename CT>
inline void copy_key_value(
    at::Tensor& kv_cache,
    const at::Tensor& kv,
    const at::Tensor& k_pe,
    int beam_batch,
    long offset) {
  RECORD_FUNCTION("ipex::copy_key_value", c10::ArrayRef<c10::IValue>({}));
  auto bs = kv.size(0);
  auto seq_len = kv.size(1);
  auto head_num = k_pe.size(2);
  auto head_size = kv.size(-1);
  auto k_pe_head_size = k_pe.size(-1);
  auto k_pe_stride_b = k_pe.stride(0);
  auto k_pe_stride_s = k_pe.stride(1);
  auto kv_hidden_size = head_num * head_size;
  auto k_pe_hidden_size = head_num * k_pe_head_size;
  auto hidden_size = kv_hidden_size + k_pe_hidden_size;
  auto kv_cache_ptr = kv_cache.data_ptr<CT>();
  auto kv_ptr = kv.data_ptr<T>();
  auto k_pe_ptr = k_pe.data_ptr<T>();
  auto token_stride = beam_batch * hidden_size;
  auto beam_size = beam_batch / bs;
  if (offset == 0) {
#pragma omp parallel for collapse(2)
    for (auto si = 0; si < seq_len; si++) {
      for (auto bi = 0; bi < bs; bi++) {
        auto cache_stride = si * token_stride + bi * beam_size * hidden_size;
        auto state_stride = (bi * seq_len + si) * kv_hidden_size;
        auto kv_cache_start = kv_cache_ptr + cache_stride;
        auto kv_ptr_start = kv_ptr + state_stride;
        move_and_convert(kv_ptr_start, kv_cache_start, kv_hidden_size);
        auto k_pe_start = k_pe_ptr + bi * k_pe_stride_b + si * k_pe_stride_s;
        auto k_pe_cache_start = kv_cache_start + kv_hidden_size;
        move_and_convert(k_pe_start, k_pe_cache_start, k_pe_hidden_size);
      }
    }
  } else {
#pragma omp parallel for collapse(2)
    for (auto si = 0; si < seq_len; si++) {
      for (auto bi = 0; bi < bs; bi++) {
        auto cache_stride = (si + offset) * token_stride + bi * hidden_size;
        auto state_stride = (bi * seq_len + si) * kv_hidden_size;
        auto kv_cache_start = kv_cache_ptr + cache_stride;
        auto kv_ptr_start = kv_ptr + state_stride;
        move_and_convert(kv_ptr_start, kv_cache_start, kv_hidden_size);
        auto k_pe_start = k_pe_ptr + bi * k_pe_stride_b + si * k_pe_stride_s;
        auto k_pe_cache_start = kv_cache_start + kv_hidden_size;
        move_and_convert(k_pe_start, k_pe_cache_start, k_pe_hidden_size);
      }
    }
  }
}

/*
 *The scale-dot product for indirect access kv chache and fuse
 *matmul+div+add+softmax to improve data reuse
 *@param  query Query embeeding with the of [beam_size*batch, cur_len, head_num,
 *head_size]
 *@param  key Key embeeding with the of [beam_size*batch, cur_len, head_num,
 *head_size]
 *@param  value Key embeeding with the of [beam_size*batch, cur_len, head_num,
 *head_size]
 *@param  key_cache Cache past key embeeding with the of [max_len,
 *beam_size*batch, head_num, head_size]
 *@param  value_chache Cache past value embeeding with the of [max_len,
 *beam_size*batch, head_num, head_size]
 *@param  beam_idx Beam info for every token [max_len, beam_size*batch]
 *@param  offset  The length of decoded(past) token.
 *@param  scale_factor the sqrt(head_dim).
 *@param  head_mask Which is not used by our kernel now.
 *@param  attention_mask Which is combined mask for padding mask and casual
 *mask.
 *@return attn_outs, None, key_cache, value_cache, beam_idx
 */
template <typename QT, typename VT, typename KCT, typename VCT>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
scale_dot_product_for_indirect_access_kv_cache(
    at::Tensor query,
    at::Tensor key,
    at::Tensor value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    at::Tensor& beam_idx,
    const int64_t offset,
    const double scale_factor,
    at::Tensor& attention_mask) {
  RECORD_FUNCTION(
      "ipex::scale_dot_product_for_indirect_access_kv_cache",
      c10::ArrayRef<c10::IValue>({}));
  int beam_batch = beam_idx.size(1);
  auto bs = query.size(0);
  auto cur_len = query.size(1); // only process cur_len==1
  auto head_num = query.size(2);
  auto head_size = query.size(3);
  auto b_ptr = beam_idx.data_ptr<long>();
  auto max_cache_size = beam_idx.size(0);
  long new_beam_idx[beam_batch][offset + query.size(1) + 1] = {};
  auto prompt_len = b_ptr[(max_cache_size - 2) * beam_batch];
  auto prompt_bs = b_ptr[(max_cache_size - 1) * beam_batch];
  auto beam_size = 1;
  if (prompt_bs != 0) {
    beam_size = beam_batch / prompt_bs;
  }
  auto need_update_beam_idx = offset > 0 and beam_size > 1;
  auto kv_head = key.size(2);
  auto group_size = head_num / kv_head;
  auto seq_len = offset + cur_len;
  auto kc_token_stride = beam_batch * kv_head * head_size;
  at::Tensor attn_weights, attn_weights2;
  bool chg_attn_w_layout = false;
  auto target_bs = bs;
  if (beam_size > 1 && prompt_len <= 2048 && prompt_bs > 20 &&
      group_size == 1) {
    chg_attn_w_layout = true;
    attn_weights = at::empty(
        {prompt_bs, head_num, cur_len, seq_len, beam_size}, at::kFloat);
    attn_weights2 = at::empty(
        {prompt_bs, head_num, cur_len, beam_size, seq_len}, at::kFloat);
    target_bs = prompt_bs;
  } else {
    attn_weights = at::empty({bs, head_num, cur_len, seq_len}, at::kFloat);
    attn_weights2 = attn_weights;
  }
  query = query.contiguous();
  key = key.contiguous();
  auto q_ptr = query.data_ptr<QT>();
  auto k_ptr = key.data_ptr<QT>();
  auto k_cache_ptr = key_cache.data_ptr<KCT>();
  auto mask_ptr = attention_mask.data_ptr<QT>();
  auto mask_head_num = attention_mask.size(1);
  auto mask_dim2 = attention_mask.size(2);
  auto mask_bs_stride = mask_head_num * mask_dim2 * seq_len;
  // value realted
  value = value.contiguous();
  auto attn_outs =
      at::empty({bs, head_num, cur_len, head_size}, value.options());
  auto v_ptr = value.data_ptr<VT>();
  auto v_cache_ptr = value_cache.data_ptr<VCT>();
  auto attn_out_ptr = attn_outs.data_ptr<VT>();
  auto attn_w_ptr = attn_weights.data_ptr<float>();
  auto attn_w_ptr2 = attn_weights2.data_ptr<float>();

  // stride information
  auto qStrideB = query.stride(0);
  auto qStrideS = query.stride(1);
  auto qStrideH = query.stride(2);

  auto kStrideB = key.stride(0);
  auto kStrideS = key.stride(1);
  auto kStrideH = key.stride(2);

  auto kcStrideB = key_cache.stride(1);
  auto kcStrideS = key_cache.stride(0);
  auto kcStrideH = key_cache.stride(2);

  auto vStrideB = value.stride(0);
  auto vStrideS = value.stride(1);
  auto vStrideH = value.stride(2);

  auto vcStrideB = value_cache.stride(1);
  auto vcStrideS = value_cache.stride(0);
  auto vcStrideH = value_cache.stride(2);

  auto attn_w_strideH = attn_weights.stride(1);

  auto thread_numbers = omp_get_max_threads();
  auto max_parallel_parts = thread_numbers * 4;

  auto target_block_size = 32L;
  if (target_bs <= 32 and seq_len < 65536) {
    target_block_size = 8L;
  }
  auto kv_block_size = target_bs * head_num >= max_parallel_parts
      ? seq_len
      : std::max(seq_len / max_parallel_parts, 1L);
  kv_block_size = std::min(kv_block_size, target_block_size);
  auto kv_block_count = (seq_len + kv_block_size - 1) / kv_block_size;
  if (need_update_beam_idx) {
    // according to last decoded token to get the target beam for the past
    for (int i = 0; i < bs; i++) {
      new_beam_idx[i][offset - 1] = b_ptr[(offset - 1) * bs + i];
      // for the token of input, the target beam is alwarys bi - bi%beam_size
      for (int j = offset - 2; j >= prompt_len; j--) {
        new_beam_idx[i][j] = b_ptr[j * bs + new_beam_idx[i][j + 1]];
      }
    }
  }
  {
    RECORD_FUNCTION(
        "ipex::iakv_sdp::matmul(query, key)", c10::ArrayRef<c10::IValue>({}));
#pragma omp parallel for collapse(3)
    for (auto block_id = 0; block_id < kv_block_count; block_id++) {
      for (auto bsi = 0; bsi < prompt_bs; bsi++) {
        for (auto head_group_start = 0; head_group_start < head_num;
             head_group_start += group_size) {
          auto k_start = block_id * kv_block_size;
          auto block_size = std::min(kv_block_size, seq_len - k_start);
          auto query_ti = 0;
          // maping the query head to key/value head to support MGA/MQA
          auto kv_hi = head_group_start / group_size;
          if (chg_attn_w_layout) {
            auto attn_w_stride =
                (bsi * head_num + head_group_start) * attn_w_strideH;
            for (auto ti = k_start; ti < k_start + block_size; ti++) {
              // caculate the innerproduct for the current token and store the
              // key
              if (ti == query_ti + offset) {
                for (auto bbi = 0; bbi < beam_size; bbi++) {
                  auto bi = bsi * beam_size + bbi;
                  auto q_ptr_start =
                      q_ptr + bi * qStrideB + head_group_start * qStrideH;
                  auto attn_w_pos = attn_w_ptr + attn_w_stride +
                      query_ti * seq_len + ti * beam_size + bbi;
                  auto kc_head_start = k_cache_ptr + ti * kcStrideS +
                      bi * kcStrideB + kv_hi * kcStrideH;
                  auto k_ptr_start = k_ptr + bi * kStrideB + kv_hi * kStrideH;
                  reduce_head<QT, QT, KCT>(
                      q_ptr_start,
                      group_size,
                      k_ptr_start,
                      attn_w_pos,
                      attn_w_strideH,
                      head_size,
                      true,
                      kc_head_start);
                }
              } else { // caculate the innerproduct for the past token
                auto bi = bsi * beam_size;
                auto q_ptr_start =
                    q_ptr + bi * qStrideB + head_group_start * qStrideH;
                auto attn_w_pos = attn_w_ptr + attn_w_stride +
                    query_ti * seq_len + ti * beam_size;
                if (need_update_beam_idx && ti >= prompt_len) {
                  for (auto bbi = 0; bbi < beam_size; bbi++) {
                    auto beam = new_beam_idx[bi + bbi][ti];
                    auto kc_head_start = k_cache_ptr + ti * kcStrideS +
                        beam * kcStrideB + kv_hi * kcStrideH;
                    reduce_head<QT, KCT, KCT>(
                        q_ptr_start + bbi * qStrideB,
                        group_size,
                        kc_head_start,
                        attn_w_pos + bbi,
                        attn_w_strideH,
                        head_size,
                        false,
                        nullptr);
                  }
                } else {
                  auto kc_head_start = k_cache_ptr + ti * kcStrideS +
                      bi * kcStrideB + kv_hi * kcStrideH;
                  reduce_head<QT, KCT>(
                      q_ptr_start,
                      qStrideB,
                      group_size,
                      kc_head_start,
                      attn_w_pos,
                      attn_w_strideH,
                      head_size,
                      beam_size);
                }
              }
            }
          } else {
            for (auto bbi = 0; bbi < beam_size; bbi++) {
              auto bi = bsi * beam_size + bbi;
              for (auto ti = k_start; ti < k_start + block_size; ti++) {
                auto q_ptr_start =
                    q_ptr + bi * qStrideB + head_group_start * qStrideH;
                auto attn_w_stride =
                    (bi * head_num + head_group_start) * attn_w_strideH;
                auto attn_w_pos =
                    attn_w_ptr + attn_w_stride + query_ti * seq_len + ti;
                attn_w_pos[0] = 0.0f;
                auto beam = need_update_beam_idx && ti >= prompt_len
                    ? new_beam_idx[bi][ti]
                    : bsi * beam_size;
                // caculate the innerproduct for the current token and store the
                // key
                if (ti == query_ti + offset) {
                  auto kc_head_start = k_cache_ptr + ti * kcStrideS +
                      bi * kcStrideB + kv_hi * kcStrideH;
                  auto k_ptr_start = k_ptr + bi * kStrideB + kv_hi * kStrideH;
                  reduce_head<QT, QT, KCT>(
                      q_ptr_start,
                      group_size,
                      k_ptr_start,
                      attn_w_pos,
                      attn_w_strideH,
                      head_size,
                      true,
                      kc_head_start);
                } else { // caculate the innerproduct for the past token
                  auto kc_head_start = k_cache_ptr + ti * kcStrideS +
                      beam * kcStrideB + kv_hi * kcStrideH;
                  reduce_head<QT, KCT, KCT>(
                      q_ptr_start,
                      group_size,
                      kc_head_start,
                      attn_w_pos,
                      attn_w_strideH,
                      head_size,
                      false,
                      nullptr);
                }
              }
            }
          }
        }
      }
    }
  }
  {
    RECORD_FUNCTION(
        "ipex::iakv_sdp::div_add_softmax", c10::ArrayRef<c10::IValue>({}));
#pragma omp parallel for collapse(2)
    for (auto bsi = 0; bsi < prompt_bs; bsi++) {
      for (auto hi = 0; hi < head_num; hi++) {
        for (auto query_ti = 0; query_ti < cur_len; query_ti++) {
          for (auto bbi = 0; bbi < beam_size; bbi++) {
            auto bi = bsi * beam_size + bbi;
            auto mask_ptr_start = mask_ptr + bi * mask_bs_stride +
                (hi % mask_head_num) * mask_dim2 * seq_len;
// div+add+softmax
#if defined(CPU_CAPABILITY_AVX512)
            auto max_val = -100000.0f;
            if (chg_attn_w_layout) {
              auto attn_w_stride =
                  (bsi * head_num + hi) * cur_len * seq_len * beam_size;
              auto attn_w_query_start = attn_w_ptr + attn_w_stride +
                  query_ti * seq_len * beam_size + bbi;
              auto attn_w_query_start2 = attn_w_ptr2 + attn_w_stride +
                  query_ti * beam_size * seq_len + bbi * seq_len;
              __m512i decrement_sequence = _mm512_set_epi32(
                  15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
              __m512i beam_size_vector = _mm512_set1_epi32(beam_size);
              int ti = 0;
              for (ti = 0; ti <= seq_len - 16; ti += 16) {
                __m512i ti_vector = _mm512_set1_epi32(ti);
                __m512i index_sequence =
                    _mm512_add_epi32(decrement_sequence, ti_vector);
                __m512i index =
                    _mm512_mullo_epi32(index_sequence, beam_size_vector);

                __m512 data = _mm512_i32gather_ps(
                    index, attn_w_query_start, sizeof(float));
                _mm512_storeu_ps(attn_w_query_start2 + ti, data);
              }

              for (; ti < seq_len; ti++) {
                attn_w_query_start2[ti] = attn_w_query_start[ti * beam_size];
              }
              torch_ipex::cpu::kernel::
                  _dil_div_add_reduce_max_fusion_kernel<float, QT>(
                      attn_w_query_start2,
                      mask_ptr_start + (query_ti % mask_dim2) * seq_len,
                      scale_factor,
                      seq_len,
                      attn_w_query_start2,
                      max_val);
              torch_ipex::cpu::kernel::_dil_exp_reduce_sum_fusion_kernel(
                  attn_w_query_start2, seq_len, attn_w_query_start2, max_val);
              torch_ipex::cpu::kernel::_dil_normalization_kernel<float>(
                  attn_w_query_start2, max_val, seq_len, attn_w_query_start2);
              for (ti = 0; ti <= seq_len - 16; ti += 16) {
                __m512i ti_vector = _mm512_set1_epi32(ti);
                __m512i index_sequence =
                    _mm512_add_epi32(decrement_sequence, ti_vector);
                __m512i index =
                    _mm512_mullo_epi32(index_sequence, beam_size_vector);
                __m512 data = _mm512_loadu_ps(attn_w_query_start2 + ti);
                _mm512_i32scatter_ps(
                    attn_w_query_start, index, data, sizeof(float));
              }
              for (; ti < seq_len; ti++) {
                attn_w_query_start[ti * beam_size] = attn_w_query_start2[ti];
              }
            } else {
              auto attn_w_stride = (bi * head_num + hi) * cur_len * seq_len;
              auto attn_w_query_start =
                  attn_w_ptr + attn_w_stride + query_ti * seq_len;
              torch_ipex::cpu::kernel::
                  _dil_div_add_reduce_max_fusion_kernel<float, QT>(
                      attn_w_query_start,
                      mask_ptr_start + (query_ti % mask_dim2) * seq_len,
                      scale_factor,
                      seq_len,
                      attn_w_query_start,
                      max_val);
              torch_ipex::cpu::kernel::_dil_exp_reduce_sum_fusion_kernel(
                  attn_w_query_start, seq_len, attn_w_query_start, max_val);
              torch_ipex::cpu::kernel::_dil_normalization_kernel<float>(
                  attn_w_query_start, max_val, seq_len, attn_w_query_start);
            }
#else
            auto max_val = -100000.0f;
            if (chg_attn_w_layout) {
              auto attn_w_stride =
                  (bsi * head_num + hi) * cur_len * seq_len * beam_size;
              auto attn_w_query_start = attn_w_ptr + attn_w_stride +
                  query_ti * seq_len * beam_size + bbi;
              auto total_len = seq_len * beam_size;
              // div+add and find max
              for (auto si = 0; si < total_len; si += beam_size) {
                attn_w_query_start[si] = attn_w_query_start[si] / scale_factor +
                    mask_ptr_start[(query_ti % mask_dim2) * seq_len +
                                   si / beam_size];
                if (attn_w_query_start[si] > max_val) {
                  max_val = attn_w_query_start[si];
                }
              }
              // softmax
              float sum = 0.0f;
              // exp and sum
              for (auto si = 0; si < total_len; si += beam_size) {
                attn_w_query_start[si] = exp(attn_w_query_start[si] - max_val);
                sum += attn_w_query_start[si];
              }
              // normalization
              for (auto si = 0; si < total_len; si += beam_size) {
                attn_w_query_start[si] = attn_w_query_start[si] / sum;
              }
            } else {
              auto attn_w_stride = (bi * head_num + hi) * cur_len * seq_len;
              auto attn_w_query_start =
                  attn_w_ptr + attn_w_stride + query_ti * seq_len;
              // div+add and find max
              for (auto si = 0; si < seq_len; si++) {
                attn_w_query_start[si] = attn_w_query_start[si] / scale_factor +
                    mask_ptr_start[(query_ti % mask_dim2) * seq_len + si];
                if (attn_w_query_start[si] > max_val) {
                  max_val = attn_w_query_start[si];
                }
              }
              // softmax
              float sum = 0.0f;
              // exp and sum
              for (auto si = 0; si < seq_len; si++) {
                attn_w_query_start[si] = exp(attn_w_query_start[si] - max_val);
                sum += attn_w_query_start[si];
              }
              // normalization
              for (auto si = 0; si < seq_len; si++) {
                attn_w_query_start[si] = attn_w_query_start[si] / sum;
              }
            }
#endif
          }
        }
      }
    }
  }
  auto private_attn_outs =
      at::empty({thread_numbers, bs, head_num, cur_len, head_size}, at::kFloat);
  auto private_attn_out_flag =
      at::zeros({thread_numbers, bs, head_num}, at::kByte);
  auto flag_access = private_attn_out_flag.accessor<uint8_t, 3>();
  uint8_t* flag_access_ptr = flag_access.data();
  auto private_attn_out_ptr = private_attn_outs.data_ptr<float>();
  // private_attn_outs.numel());
  auto attn_outs_stride_privT = private_attn_outs.stride(0);
  auto attn_outs_stride_privB = private_attn_outs.stride(1);
  auto attn_outs_stride_privH = private_attn_outs.stride(2);

  {
    RECORD_FUNCTION(
        "ipex::iakv_sdp::matmul(attn_w, value)",
        c10::ArrayRef<c10::IValue>({}));
#pragma omp parallel for collapse(3)
    for (auto block_id = 0; block_id < kv_block_count; block_id++) {
      for (auto bsi = 0; bsi < prompt_bs; bsi++) {
        for (auto hi = 0; hi < head_num; hi += group_size) {
          auto thread_id = 0;
          if (kv_block_size < seq_len)
            thread_id = omp_get_thread_num();
          auto v_start = block_id * kv_block_size;
          auto block_size = std::min(kv_block_size, seq_len - v_start);
          auto query_ti = 0;
          // maping the query head to key/value head to support MGA/MQA
          auto kv_hi = hi / group_size;
          if (chg_attn_w_layout) {
            auto attn_w_stride = (bsi * head_num + hi) * attn_w_strideH;
            for (auto vi = v_start; vi < v_start + block_size; vi++) {
              if (vi == offset) {
                for (auto bbi = 0; bbi < beam_size; bbi++) {
                  auto bi = bsi * beam_size + bbi;
                  auto attn_w_query_start = attn_w_ptr + attn_w_stride +
                      query_ti * seq_len + vi * beam_size + bbi;
                  // calculate weighted value and store the result to
                  // attn_outs[bs, head_num, cur_len, head_size]
                  auto attn_out_start = private_attn_out_ptr +
                      thread_id * attn_outs_stride_privT +
                      bi * attn_outs_stride_privB + hi * attn_outs_stride_privH;
                  auto flag_access_start = flag_access_ptr +
                      head_num * bs * thread_id + head_num * bi + hi;
                  auto v_cache_head_start = v_cache_ptr + vi * vcStrideS +
                      bi * vcStrideB + kv_hi * vcStrideH;
                  auto v_ptr_start = v_ptr + bi * vStrideB + kv_hi * vStrideH;
                  mul_attenion_weights_and_value_of_head<VT, float, VCT>(
                      attn_w_query_start,
                      attn_w_strideH,
                      v_ptr_start,
                      attn_out_start,
                      head_size,
                      group_size,
                      head_size,
                      true,
                      v_cache_head_start,
                      flag_access_start);
                }
              } else {
                // caculate the innerproduct for the past token
                if (need_update_beam_idx && vi >= prompt_len) {
                  for (auto bbi = 0; bbi < beam_size; bbi++) {
                    auto bi = bsi * beam_size + bbi;
                    auto attn_w_query_start = attn_w_ptr + attn_w_stride +
                        query_ti * seq_len + vi * beam_size + bbi;
                    // calculate weighted value and store the result to
                    // attn_outs[bs, head_num, cur_len, head_size]
                    auto attn_out_start = private_attn_out_ptr +
                        thread_id * attn_outs_stride_privT +
                        bi * attn_outs_stride_privB +
                        hi * attn_outs_stride_privH;
                    auto flag_access_start = flag_access_ptr +
                        head_num * bs * thread_id + head_num * bi + hi;
                    auto v_ptr_start = v_ptr + bi * vStrideB + kv_hi * vStrideH;
                    auto beam = new_beam_idx[bi][vi];
                    auto v_cache_head_start = v_cache_ptr + vi * vcStrideS +
                        beam * vcStrideB + kv_hi * vcStrideH;
                    mul_attenion_weights_and_value_of_head<VCT, float, VCT>(
                        attn_w_query_start,
                        attn_w_strideH,
                        v_cache_head_start,
                        attn_out_start,
                        head_size,
                        group_size,
                        head_size,
                        false,
                        nullptr,
                        flag_access_start);
                  }
                } else {
                  auto bi = bsi * beam_size;
                  auto attn_w_query_start = attn_w_ptr + attn_w_stride +
                      query_ti * seq_len + vi * beam_size;
                  // calculate weighted value and store the result to
                  // attn_outs[bs, head_num, cur_len, head_size]
                  auto attn_out_start = private_attn_out_ptr +
                      thread_id * attn_outs_stride_privT +
                      bi * attn_outs_stride_privB + hi * attn_outs_stride_privH;
                  auto flag_access_start = flag_access_ptr +
                      head_num * bs * thread_id + head_num * bi + hi;
                  auto v_cache_head_start = v_cache_ptr + vi * vcStrideS +
                      bi * vcStrideB + kv_hi * vcStrideH;
                  mul_attenion_weights_and_value_of_head<VCT, float>(
                      attn_w_query_start,
                      attn_w_strideH,
                      v_cache_head_start,
                      attn_out_start,
                      attn_outs_stride_privB,
                      head_size,
                      group_size,
                      head_size,
                      false,
                      nullptr,
                      flag_access_start,
                      head_num,
                      beam_size);
                }
              }
            }
          } else {
            for (auto bbi = 0; bbi < beam_size; bbi++) {
              auto bi = bsi * beam_size + bbi;
              for (auto vi = v_start; vi < v_start + block_size; vi++) {
                auto attn_w_stride = (bi * head_num + hi) * attn_w_strideH;
                auto attn_w_query_start =
                    attn_w_ptr + attn_w_stride + query_ti * seq_len + vi;
                // calculate weighted value and store the result to
                // attn_outs[bs, head_num, cur_len, head_size]
                auto attn_out_start = private_attn_out_ptr +
                    thread_id * attn_outs_stride_privT +
                    bi * attn_outs_stride_privB + hi * attn_outs_stride_privH;
                auto flag_access_start = flag_access_ptr +
                    head_num * bs * thread_id + head_num * bi + hi;

                auto beam = need_update_beam_idx && vi >= prompt_len
                    ? new_beam_idx[bi][vi]
                    : bsi * beam_size;
                // caculate the innerproduct for the current token and store the
                // key
                if (vi == offset) {
                  auto v_cache_head_start = v_cache_ptr + vi * vcStrideS +
                      bi * vcStrideB + kv_hi * vcStrideH;
                  auto v_ptr_start = v_ptr + bi * vStrideB + kv_hi * vStrideH;
                  mul_attenion_weights_and_value_of_head<VT, float, VCT>(
                      attn_w_query_start,
                      attn_w_strideH,
                      v_ptr_start,
                      attn_out_start,
                      head_size,
                      group_size,
                      head_size,
                      true,
                      v_cache_head_start,
                      flag_access_start);
                } else {
                  // caculate the innerproduct for the past token
                  auto v_cache_head_start = v_cache_ptr + vi * vcStrideS +
                      beam * vcStrideB + kv_hi * vcStrideH;
                  mul_attenion_weights_and_value_of_head<VCT, float, VCT>(
                      attn_w_query_start,
                      attn_w_strideH,
                      v_cache_head_start,
                      attn_out_start,
                      head_size,
                      group_size,
                      head_size,
                      false,
                      nullptr,
                      flag_access_start);
                }
              }
            }
          }
        }
      }
    }
  }
  {
    RECORD_FUNCTION(
        "ipex::iakv_sdp::reduction_private_result",
        c10::ArrayRef<c10::IValue>({}));
#pragma omp parallel for collapse(3)
    for (auto bi = 0; bi < bs; bi++) {
      for (auto hi = 0; hi < head_num; hi++) {
        for (auto qi = 0; qi < cur_len; qi++) {
          auto thr0_head_start = private_attn_out_ptr +
              bi * attn_outs_stride_privB + hi * attn_outs_stride_privH;
          if (flag_access[0][bi][hi] == 0) {
            torch_ipex::cpu::kernel::zero_ker(thr0_head_start, head_size);
          }
          if (kv_block_size < seq_len) {
            for (auto thread_id = 1; thread_id < thread_numbers; thread_id++) {
              if (flag_access[thread_id][bi][hi] == 0) {
                continue;
              }
              auto private_attn_out_start = private_attn_out_ptr +
                  thread_id * attn_outs_stride_privT +
                  bi * attn_outs_stride_privB + hi * attn_outs_stride_privH;
              torch_ipex::cpu::kernel::add_ker<float, float>(
                  thr0_head_start, private_attn_out_start, head_size);
            }
          }

          auto attn_outs_start = attn_out_ptr +
              (bi * head_num + hi) * cur_len * head_size + qi * head_size;
          torch_ipex::cpu::kernel::move_ker<VT, float>(
              attn_outs_start, thr0_head_start, head_size);
        }
      }
    }
  }

  return std::make_tuple(
      attn_outs, at::Tensor(), key_cache, value_cache, beam_idx);
}

#if defined(CPU_CAPABILITY_AVX512_FP16)
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
scale_dot_product_for_indirect_access_kv_cache_half(
    at::Tensor query,
    at::Tensor key,
    at::Tensor value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    at::Tensor& beam_idx,
    const int64_t offset,
    const double scale_factor,
    at::Tensor& attention_mask) {
  RECORD_FUNCTION(
      "ipex::scale_dot_product_for_indirect_access_kv_cache_half",
      c10::ArrayRef<c10::IValue>({}));
  int beam_batch = beam_idx.size(1);
  auto bs = query.size(0);
  auto cur_len = query.size(1); // only process cur_len==1
  auto head_num = query.size(2);
  auto head_size = query.size(3);
  auto b_ptr = beam_idx.data_ptr<long>();
  auto max_cache_size = beam_idx.size(0);
  long new_beam_idx[beam_batch][offset + query.size(1) + 1] = {};
  auto prompt_len = b_ptr[(max_cache_size - 2) * beam_batch];
  auto prompt_bs = b_ptr[(max_cache_size - 1) * beam_batch];
  auto beam_size = 1;
  if (prompt_bs != 0) {
    beam_size = beam_batch / prompt_bs;
  }
  auto need_update_beam_idx = offset > 0 and beam_size > 1;
  auto kv_head = key.size(2);
  auto group_size = head_num / kv_head;
  auto seq_len = offset + cur_len;
  auto kc_token_stride = beam_batch * kv_head * head_size;
  at::Tensor attn_weights, attn_weights2;
  bool chg_attn_w_layout = false;
  auto target_bs = bs;
  if (beam_size > 1 && prompt_len <= 2048 && prompt_bs > 20 &&
      group_size == 1) {
    chg_attn_w_layout = true;
    attn_weights = at::empty(
        {prompt_bs, head_num, cur_len, seq_len, beam_size}, key.options());
    attn_weights2 = at::empty(
        {prompt_bs, head_num, cur_len, beam_size, seq_len}, key.options());
    target_bs = prompt_bs;
  } else {
    attn_weights = at::empty({bs, head_num, cur_len, seq_len}, key.options());
    attn_weights2 = attn_weights;
  }
  query = query.contiguous();
  key = key.contiguous();
  auto q_ptr = query.data_ptr<at::Half>();
  auto k_ptr = key.data_ptr<at::Half>();
  auto k_cache_ptr = key_cache.data_ptr<at::Half>();
  auto mask_ptr = attention_mask.data_ptr<at::Half>();
  auto mask_head_num = attention_mask.size(1);
  auto mask_dim2 = attention_mask.size(2);
  auto mask_bs_stride = mask_head_num * mask_dim2 * seq_len;
  // value realted
  value = value.contiguous();
  auto attn_outs =
      at::empty({bs, head_num, cur_len, head_size}, value.options());
  auto v_ptr = value.data_ptr<at::Half>();
  auto v_cache_ptr = value_cache.data_ptr<at::Half>();
  auto attn_out_ptr = attn_outs.data_ptr<at::Half>();
  auto attn_w_ptr = attn_weights.data_ptr<at::Half>();
  auto attn_w_ptr2 = attn_weights2.data_ptr<at::Half>();

  // stride information
  auto qStrideB = query.stride(0);
  auto qStrideS = query.stride(1);
  auto qStrideH = query.stride(2);

  auto kStrideB = key.stride(0);
  auto kStrideS = key.stride(1);
  auto kStrideH = key.stride(2);

  auto kcStrideB = key_cache.stride(1);
  auto kcStrideS = key_cache.stride(0);
  auto kcStrideH = key_cache.stride(2);

  auto vStrideB = value.stride(0);
  auto vStrideS = value.stride(1);
  auto vStrideH = value.stride(2);

  auto vcStrideB = value_cache.stride(1);
  auto vcStrideS = value_cache.stride(0);
  auto vcStrideH = value_cache.stride(2);

  auto attn_w_strideH = attn_weights.stride(1);

  auto thread_numbers = omp_get_max_threads();
  auto max_parallel_parts = thread_numbers * 4;

  auto target_block_size = 32L;
  if (target_bs <= 32 and seq_len < 65536) {
    target_block_size = 8L;
  }
  auto kv_block_size = target_bs * head_num >= max_parallel_parts
      ? seq_len
      : std::max(seq_len / max_parallel_parts, 1L);
  kv_block_size = std::min(kv_block_size, target_block_size);
  auto kv_block_count = (seq_len + kv_block_size - 1) / kv_block_size;
  if (need_update_beam_idx) {
    // according to the last decoded token to get the target beam for the past
    // token
    for (int i = 0; i < bs; i++) {
      new_beam_idx[i][offset - 1] = b_ptr[(offset - 1) * bs + i];
      // for the token of input, the target beam is alwarys bi - bi%beam_size
      for (int j = offset - 2; j >= prompt_len; j--) {
        new_beam_idx[i][j] = b_ptr[j * bs + new_beam_idx[i][j + 1]];
      }
    }
  }
  {
    RECORD_FUNCTION(
        "ipex::iakv_sdp::matmul(query, key)", c10::ArrayRef<c10::IValue>({}));
#pragma omp parallel for collapse(3)
    for (auto block_id = 0; block_id < kv_block_count; block_id++) {
      for (auto bsi = 0; bsi < prompt_bs; bsi++) {
        for (auto head_group_start = 0; head_group_start < head_num;
             head_group_start += group_size) {
          auto k_start = block_id * kv_block_size;
          auto block_size = std::min(kv_block_size, seq_len - k_start);
          auto query_ti = 0;
          // maping the query head to key/value head to support MGA/MQA
          auto kv_hi = head_group_start / group_size;
          if (chg_attn_w_layout) {
            auto attn_w_stride =
                (bsi * head_num + head_group_start) * attn_w_strideH;
            for (auto ti = k_start; ti < k_start + block_size; ti++) {
              // caculate the innerproduct for the current token and store the
              // key
              if (ti == query_ti + offset) {
                for (auto bbi = 0; bbi < beam_size; bbi++) {
                  auto bi = bsi * beam_size + bbi;
                  auto q_ptr_start =
                      q_ptr + bi * qStrideB + head_group_start * qStrideH;
                  auto attn_w_pos = attn_w_ptr + attn_w_stride +
                      query_ti * seq_len + ti * beam_size + bbi;
                  auto kc_head_start = k_cache_ptr + ti * kcStrideS +
                      bi * kcStrideB + kv_hi * kcStrideH;
                  auto k_ptr_start = k_ptr + bi * kStrideB + kv_hi * kStrideH;
                  reduce_head_half(
                      q_ptr_start,
                      group_size,
                      k_ptr_start,
                      attn_w_pos,
                      attn_w_strideH,
                      head_size,
                      true,
                      kc_head_start);
                }
              } else { // caculate the innerproduct for the past token
                auto bi = bsi * beam_size;
                auto q_ptr_start =
                    q_ptr + bi * qStrideB + head_group_start * qStrideH;
                auto attn_w_pos = attn_w_ptr + attn_w_stride +
                    query_ti * seq_len + ti * beam_size;
                if (need_update_beam_idx && ti >= prompt_len) {
                  for (auto bbi = 0; bbi < beam_size; bbi++) {
                    auto beam = new_beam_idx[bi + bbi][ti];
                    auto kc_head_start = k_cache_ptr + ti * kcStrideS +
                        beam * kcStrideB + kv_hi * kcStrideH;
                    reduce_head_half(
                        q_ptr_start + bbi * qStrideB,
                        group_size,
                        kc_head_start,
                        attn_w_pos + bbi,
                        attn_w_strideH,
                        head_size,
                        false,
                        nullptr);
                  }
                } else {
                  auto kc_head_start = k_cache_ptr + ti * kcStrideS +
                      bi * kcStrideB + kv_hi * kcStrideH;
                  reduce_head_half(
                      q_ptr_start,
                      qStrideB,
                      group_size,
                      kc_head_start,
                      attn_w_pos,
                      attn_w_strideH,
                      head_size,
                      beam_size);
                }
              }
            }
          } else {
            for (auto bbi = 0; bbi < beam_size; bbi++) {
              auto bi = bsi * beam_size + bbi;
              for (auto ti = k_start; ti < k_start + block_size; ti++) {
                auto q_ptr_start =
                    q_ptr + bi * qStrideB + head_group_start * qStrideH;
                auto attn_w_stride =
                    (bi * head_num + head_group_start) * attn_w_strideH;
                auto attn_w_pos =
                    attn_w_ptr + attn_w_stride + query_ti * seq_len + ti;
                attn_w_pos[0] = 0.0f;
                auto beam = need_update_beam_idx && ti >= prompt_len
                    ? new_beam_idx[bi][ti]
                    : bsi * beam_size;
                // caculate the innerproduct for the current token and store the
                // key
                if (ti == query_ti + offset) {
                  auto kc_head_start = k_cache_ptr + ti * kcStrideS +
                      bi * kcStrideB + kv_hi * kcStrideH;
                  auto k_ptr_start = k_ptr + bi * kStrideB + kv_hi * kStrideH;
                  reduce_head_half(
                      q_ptr_start,
                      group_size,
                      k_ptr_start,
                      attn_w_pos,
                      attn_w_strideH,
                      head_size,
                      true,
                      kc_head_start);
                } else { // caculate the innerproduct for the past token
                  auto kc_head_start = k_cache_ptr + ti * kcStrideS +
                      beam * kcStrideB + kv_hi * kcStrideH;
                  reduce_head_half(
                      q_ptr_start,
                      group_size,
                      kc_head_start,
                      attn_w_pos,
                      attn_w_strideH,
                      head_size,
                      false,
                      nullptr);
                }
              }
            }
          }
        }
      }
    }
  }
  {
    RECORD_FUNCTION(
        "ipex::iakv_sdp::div_add_softmax", c10::ArrayRef<c10::IValue>({}));
#pragma omp parallel for collapse(2)
    for (auto bsi = 0; bsi < prompt_bs; bsi++) {
      for (auto hi = 0; hi < head_num; hi++) {
        for (auto query_ti = 0; query_ti < cur_len; query_ti++) {
          for (auto bbi = 0; bbi < beam_size; bbi++) {
            auto bi = bsi * beam_size + bbi;
            auto mask_ptr_start = mask_ptr + bi * mask_bs_stride +
                (hi % mask_head_num) * mask_dim2 * seq_len;

            // div+add+softmax
            at::Half max_val = -100000.0f;
            if (chg_attn_w_layout) {
              auto attn_w_stride =
                  (bsi * head_num + hi) * cur_len * seq_len * beam_size;
              auto attn_w_query_start = attn_w_ptr + attn_w_stride +
                  query_ti * seq_len * beam_size + bbi;
              auto attn_w_query_start2 = attn_w_ptr2 + attn_w_stride +
                  query_ti * beam_size * seq_len + bbi * seq_len;
              for (auto ti = 0; ti < seq_len; ti++) {
                attn_w_query_start2[ti] = attn_w_query_start[ti * beam_size];
              }
              torch_ipex::cpu::kernel::
                  _dil_div_add_reduce_max_fusion_kernel_half(
                      attn_w_query_start2,
                      mask_ptr_start + (query_ti % mask_dim2) * seq_len,
                      scale_factor,
                      seq_len,
                      attn_w_query_start2,
                      max_val);

              torch_ipex::cpu::kernel::_dil_exp_reduce_sum_fusion_kernel_half(
                  attn_w_query_start2, seq_len, attn_w_query_start2, max_val);
              torch_ipex::cpu::kernel::_dil_normalization_kernel_half(
                  attn_w_query_start2, max_val, seq_len, attn_w_query_start2);
              for (auto ti = 0; ti < seq_len; ti++) {
                attn_w_query_start[ti * beam_size] = attn_w_query_start2[ti];
              }
            } else {
              auto attn_w_stride = (bi * head_num + hi) * cur_len * seq_len;
              auto attn_w_query_start =
                  attn_w_ptr + attn_w_stride + query_ti * seq_len;
              torch_ipex::cpu::kernel::
                  _dil_div_add_reduce_max_fusion_kernel_half(
                      attn_w_query_start,
                      mask_ptr_start + (query_ti % mask_dim2) * seq_len,
                      scale_factor,
                      seq_len,
                      attn_w_query_start,
                      max_val);
              torch_ipex::cpu::kernel::_dil_exp_reduce_sum_fusion_kernel_half(
                  attn_w_query_start, seq_len, attn_w_query_start, max_val);
              torch_ipex::cpu::kernel::_dil_normalization_kernel_half(
                  attn_w_query_start, max_val, seq_len, attn_w_query_start);
            }
          }
        }
      }
    }
  }
  auto private_attn_outs = at::empty(
      {thread_numbers, bs, head_num, cur_len, head_size}, key.options());
  auto private_attn_out_flag =
      at::zeros({thread_numbers, bs, head_num}, at::kByte);
  auto flag_access = private_attn_out_flag.accessor<uint8_t, 3>();
  uint8_t* flag_access_ptr = flag_access.data();
  auto private_attn_out_ptr = private_attn_outs.data_ptr<at::Half>();
  // private_attn_outs.numel());
  auto attn_outs_stride_privT = private_attn_outs.stride(0);
  auto attn_outs_stride_privB = private_attn_outs.stride(1);
  auto attn_outs_stride_privH = private_attn_outs.stride(2);

  {
    RECORD_FUNCTION(
        "ipex::iakv_sdp::matmul(attn_w, value)",
        c10::ArrayRef<c10::IValue>({}));
#pragma omp parallel for collapse(3)
    for (auto block_id = 0; block_id < kv_block_count; block_id++) {
      for (auto bsi = 0; bsi < prompt_bs; bsi++) {
        for (auto hi = 0; hi < head_num; hi += group_size) {
          auto thread_id = 0;
          if (kv_block_size < seq_len)
            thread_id = omp_get_thread_num();
          auto v_start = block_id * kv_block_size;
          auto block_size = std::min(kv_block_size, seq_len - v_start);
          auto query_ti = 0;
          // maping the query head to key/value head to support MGA/MQA
          auto kv_hi = hi / group_size;
          if (chg_attn_w_layout) {
            auto attn_w_stride = (bsi * head_num + hi) * attn_w_strideH;
            for (auto vi = v_start; vi < v_start + block_size; vi++) {
              // caculate the attention values for the current token
              if (vi == offset) {
                for (auto bbi = 0; bbi < beam_size; bbi++) {
                  auto bi = bsi * beam_size + bbi;
                  auto attn_w_query_start = attn_w_ptr + attn_w_stride +
                      query_ti * seq_len + vi * beam_size + bbi;
                  // calculate weighted value and store the result to
                  // attn_outs[bs, head_num, cur_len, head_size]
                  auto attn_out_start = private_attn_out_ptr +
                      thread_id * attn_outs_stride_privT +
                      bi * attn_outs_stride_privB + hi * attn_outs_stride_privH;
                  auto flag_access_start = flag_access_ptr +
                      head_num * bs * thread_id + head_num * bi + hi;
                  auto v_cache_head_start = v_cache_ptr + vi * vcStrideS +
                      bi * vcStrideB + kv_hi * vcStrideH;
                  auto v_ptr_start = v_ptr + bi * vStrideB + kv_hi * vStrideH;
                  mul_attenion_weights_and_value_of_head_half(
                      attn_w_query_start,
                      attn_w_strideH,
                      v_ptr_start,
                      attn_out_start,
                      head_size,
                      group_size,
                      head_size,
                      true,
                      v_cache_head_start,
                      flag_access_start);
                }
              } else {
                // caculate the innerproduct for the past token
                if (need_update_beam_idx && vi >= prompt_len) {
                  for (auto bbi = 0; bbi < beam_size; bbi++) {
                    auto bi = bsi * beam_size + bbi;
                    auto attn_w_query_start = attn_w_ptr + attn_w_stride +
                        query_ti * seq_len + vi * beam_size + bbi;
                    // calculate weighted value and store the result to
                    // attn_outs[bs, head_num, cur_len, head_size]
                    auto attn_out_start = private_attn_out_ptr +
                        thread_id * attn_outs_stride_privT +
                        bi * attn_outs_stride_privB +
                        hi * attn_outs_stride_privH;
                    auto flag_access_start = flag_access_ptr +
                        head_num * bs * thread_id + head_num * bi + hi;
                    auto v_ptr_start = v_ptr + bi * vStrideB + kv_hi * vStrideH;
                    auto beam = new_beam_idx[bi][vi];
                    auto v_cache_head_start = v_cache_ptr + vi * vcStrideS +
                        beam * vcStrideB + kv_hi * vcStrideH;
                    mul_attenion_weights_and_value_of_head_half(
                        attn_w_query_start,
                        attn_w_strideH,
                        v_cache_head_start,
                        attn_out_start,
                        head_size,
                        group_size,
                        head_size,
                        false,
                        nullptr,
                        flag_access_start);
                  }
                } else {
                  auto bi = bsi * beam_size;
                  auto attn_w_query_start = attn_w_ptr + attn_w_stride +
                      query_ti * seq_len + vi * beam_size;
                  // calculate weighted value and store the result to
                  // attn_outs[bs, head_num, cur_len, head_size]
                  auto attn_out_start = private_attn_out_ptr +
                      thread_id * attn_outs_stride_privT +
                      bi * attn_outs_stride_privB + hi * attn_outs_stride_privH;
                  auto flag_access_start = flag_access_ptr +
                      head_num * bs * thread_id + head_num * bi + hi;
                  auto v_cache_head_start = v_cache_ptr + vi * vcStrideS +
                      bi * vcStrideB + kv_hi * vcStrideH;
                  mul_attenion_weights_and_value_of_head_half(
                      attn_w_query_start,
                      attn_w_strideH,
                      v_cache_head_start,
                      attn_out_start,
                      attn_outs_stride_privB,
                      head_size,
                      group_size,
                      head_size,
                      false,
                      nullptr,
                      flag_access_start,
                      head_num,
                      beam_size);
                }
              }
            }
          } else {
            for (auto bbi = 0; bbi < beam_size; bbi++) {
              auto bi = bsi * beam_size + bbi;
              for (auto vi = v_start; vi < v_start + block_size; vi++) {
                auto attn_w_stride = (bi * head_num + hi) * attn_w_strideH;
                auto attn_w_query_start =
                    attn_w_ptr + attn_w_stride + query_ti * seq_len + vi;
                // calculate weighted value and store the result to
                // attn_outs[bs, head_num, cur_len, head_size]
                auto attn_out_start = private_attn_out_ptr +
                    thread_id * attn_outs_stride_privT +
                    bi * attn_outs_stride_privB + hi * attn_outs_stride_privH;
                auto flag_access_start = flag_access_ptr +
                    head_num * bs * thread_id + head_num * bi + hi;

                auto beam = need_update_beam_idx && vi >= prompt_len
                    ? new_beam_idx[bi][vi]
                    : bsi * beam_size;
                // caculate the attention values for the current token
                if (vi == offset) {
                  auto v_cache_head_start = v_cache_ptr + vi * vcStrideS +
                      bi * vcStrideB + kv_hi * vcStrideH;
                  auto v_ptr_start = v_ptr + bi * vStrideB + kv_hi * vStrideH;
                  mul_attenion_weights_and_value_of_head_half(
                      attn_w_query_start,
                      attn_w_strideH,
                      v_ptr_start,
                      attn_out_start,
                      head_size,
                      group_size,
                      head_size,
                      true,
                      v_cache_head_start,
                      flag_access_start);
                } else {
                  // caculate the innerproduct for the past token
                  auto v_cache_head_start = v_cache_ptr + vi * vcStrideS +
                      beam * vcStrideB + kv_hi * vcStrideH;
                  mul_attenion_weights_and_value_of_head_half(
                      attn_w_query_start,
                      attn_w_strideH,
                      v_cache_head_start,
                      attn_out_start,
                      head_size,
                      group_size,
                      head_size,
                      false,
                      nullptr,
                      flag_access_start);
                }
              }
            }
          }
        }
      }
    }
  }
  {
    RECORD_FUNCTION(
        "ipex::iakv_sdp::reduction_private_result",
        c10::ArrayRef<c10::IValue>({}));
#pragma omp parallel for collapse(3)
    for (auto bi = 0; bi < bs; bi++) {
      for (auto hi = 0; hi < head_num; hi++) {
        for (auto qi = 0; qi < cur_len; qi++) {
          auto thr0_head_start = private_attn_out_ptr +
              bi * attn_outs_stride_privB + hi * attn_outs_stride_privH;
          if (flag_access[0][bi][hi] == 0) {
            torch_ipex::cpu::kernel::zero_ker(thr0_head_start, head_size);
          }
          if (kv_block_size < seq_len) {
            for (auto thread_id = 1; thread_id < thread_numbers; thread_id++) {
              if (flag_access[thread_id][bi][hi] == 0) {
                continue;
              }
              auto private_attn_out_start = private_attn_out_ptr +
                  thread_id * attn_outs_stride_privT +
                  bi * attn_outs_stride_privB + hi * attn_outs_stride_privH;
              torch_ipex::cpu::kernel::add_ker<at::Half, at::Half>(
                  thr0_head_start, private_attn_out_start, head_size);
            }
          }

          auto attn_outs_start = attn_out_ptr +
              (bi * head_num + hi) * cur_len * head_size + qi * head_size;
          torch_ipex::cpu::kernel::move_ker<at::Half, at::Half>(
              attn_outs_start, thr0_head_start, head_size);
        }
      }
    }
  }
  return std::make_tuple(
      attn_outs, at::Tensor(), key_cache, value_cache, beam_idx);
}
#endif

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
zero_copy_kv_cache_masked_multihead_self_attention_kernel_impl(
    at::Tensor query,
    at::Tensor key,
    at::Tensor value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    at::Tensor& beam_idx,
    const int64_t offset,
    const double scale_attn,
    at::Tensor& attention_mask) {
  assert(
      key.scalar_type() == at::kBFloat16 || key.scalar_type() == at::kFloat ||
      key.scalar_type() == at::kHalf);
  if (key_cache.scalar_type() == at::ScalarType::Float8_e5m2 &&
      query.scalar_type() == at::kBFloat16 &&
      value.scalar_type() == at::kBFloat16) {
    return scale_dot_product_for_indirect_access_kv_cache<
        at::BFloat16,
        at::BFloat16,
        at::Float8_e5m2,
        at::Float8_e5m2>(
        query,
        key,
        value,
        key_cache,
        value_cache,
        beam_idx,
        offset,
        scale_attn,
        attention_mask);
  } else if (
      query.scalar_type() == at::kFloat && value.scalar_type() == at::kFloat) {
    return scale_dot_product_for_indirect_access_kv_cache<
        float,
        float,
        float,
        float>(
        query,
        key,
        value,
        key_cache,
        value_cache,
        beam_idx,
        offset,
        scale_attn,
        attention_mask);
  } else if (
      query.scalar_type() == at::kFloat &&
      value.scalar_type() == at::kBFloat16) {
    return scale_dot_product_for_indirect_access_kv_cache<
        float,
        at::BFloat16,
        float,
        at::BFloat16>(
        query,
        key,
        value,
        key_cache,
        value_cache,
        beam_idx,
        offset,
        scale_attn,
        attention_mask);
  } else if (
      key.scalar_type() == at::kBFloat16 && value.scalar_type() == at::kFloat) {
    return scale_dot_product_for_indirect_access_kv_cache<
        at::BFloat16,
        float,
        at::BFloat16,
        float>(
        query,
        key,
        value,
        key_cache,
        value_cache,
        beam_idx,
        offset,
        scale_attn,
        attention_mask);
  } else if (
      query.scalar_type() == at::kHalf && value.scalar_type() == at::kHalf) {
#if defined(CPU_CAPABILITY_AVX512_FP16)
    return scale_dot_product_for_indirect_access_kv_cache_half(
        query,
        key,
        value,
        key_cache,
        value_cache,
        beam_idx,
        offset,
        scale_attn,
        attention_mask);
#else
    return scale_dot_product_for_indirect_access_kv_cache<
        at::Half,
        at::Half,
        at::Half,
        at::Half>(
        query,
        key,
        value,
        key_cache,
        value_cache,
        beam_idx,
        offset,
        scale_attn,
        attention_mask);
#endif
  } else if (
      query.scalar_type() == at::kFloat && value.scalar_type() == at::kHalf) {
    return scale_dot_product_for_indirect_access_kv_cache<
        float,
        at::Half,
        float,
        at::Half>(
        query,
        key,
        value,
        key_cache,
        value_cache,
        beam_idx,
        offset,
        scale_attn,
        attention_mask);
  } else if (
      query.scalar_type() == at::kHalf && value.scalar_type() == at::kFloat) {
    return scale_dot_product_for_indirect_access_kv_cache<
        at::Half,
        float,
        at::Half,
        float>(
        query,
        key,
        value,
        key_cache,
        value_cache,
        beam_idx,
        offset,
        scale_attn,
        attention_mask);
  }
  return scale_dot_product_for_indirect_access_kv_cache<
      at::BFloat16,
      at::BFloat16,
      at::BFloat16,
      at::BFloat16>(
      query,
      key,
      value,
      key_cache,
      value_cache,
      beam_idx,
      offset,
      scale_attn,
      attention_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
first_token_masked_mha(
    at::Tensor query,
    at::Tensor key,
    at::Tensor value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    at::Tensor& beam_idx,
    const int64_t beam_batch,
    const double scale_attn,
    at::Tensor attention_mask,
    bool add_casual_mask = true) {
  auto origin_type = query.scalar_type();
  auto bs = query.size(0);
  auto query_length = query.size(1);
  auto key_lenght = key.size(1);
  auto kv_head_num = key.size(2);
  auto head_size = key.size(3);
  if (key.scalar_type() != at::kBFloat16 && key.scalar_type() != at::kFloat &&
      key.scalar_type() != at::kHalf) {
    TORCH_CHECK(
        false,
        "key and value must be float, float16 or bfloat16 to use ipex::masked_multihead_self_attention_kernel_impl");
  }
  if (key_cache.scalar_type() == at::ScalarType::Float8_e5m2 &&
      key.scalar_type() == at::ScalarType::BFloat16) {
    copy_key_value<at::BFloat16, at::Float8_e5m2>(
        key_cache, key, value_cache, value, beam_batch);
  } else if (key.scalar_type() == at::kFloat) {
    copy_key_value<float, float>(
        key_cache, key, value_cache, value, beam_batch);
  } else if (key.scalar_type() == at::kBFloat16) {
    copy_key_value<at::BFloat16, at::BFloat16>(
        key_cache, key, value_cache, value, beam_batch);
  } else {
    copy_key_value<at::Half, at::Half>(
        key_cache, key, value_cache, value, beam_batch);
  }
  // support MGQ/MQA
  // expand the head dimensiopn of key/value to be same to the query
  if (query.size(2) != key.size(2)) {
    auto n_req = query.size(2) / key.size(2);
    key = key.repeat_interleave(n_req, 2);
    value = value.repeat_interleave(n_req, 2);
  }
  auto attn_outputs = at::Tensor();
  auto attn_weights = at::Tensor();
  if (is_first_token_optimizable(key) && attention_mask.stride(-1) == 1) {
    query = query.transpose(1, 2);
    key = key.transpose(1, 2);
    value = value.transpose(1, 2);
    attn_outputs = std::get<0>(torch_ipex::cpu::flash_attention_kernel_stub(
        kCPU,
        query,
        key,
        value,
        /* dropout */ 0.0,
        add_casual_mask,
        attention_mask,
        1. / scale_attn));
  } else {
    if (origin_type == at::kHalf) {
      key = key.to(at::kFloat);
      query = query.to(at::kFloat);
      value = value.to(at::kFloat);
    }
    key = key.permute({0, 2, 1, 3});
    query = query.permute({0, 2, 1, 3});
    value = value.permute({0, 2, 1, 3});
    attn_weights = query.matmul(key.transpose(-1, -2));
    attn_weights = attn_weights.div(scale_attn);
    attn_weights = attn_weights + attention_mask;
    attn_weights = attn_weights.softmax(-1);
    attn_weights = attn_weights.to(value.dtype());
    attn_outputs = attn_weights.matmul(value);
    if (origin_type == at::kHalf) {
      attn_weights = attn_weights.to(origin_type);
      attn_outputs = attn_outputs.to(origin_type);
    }
  }
  return std::make_tuple(
      attn_outputs, attn_weights, key_cache, value_cache, beam_idx);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
first_token_deepseekv2_mla(
    at::Tensor query,
    at::Tensor key,
    at::Tensor value,
    at::Tensor& kv_cache,
    at::Tensor& beam_idx,
    const int64_t beam_batch,
    const double scale_attn,
    at::Tensor attention_mask,
    int64_t v_head_size,
    bool add_casual_mask = true) {
  auto origin_type = query.scalar_type();
  auto bs = query.size(0);
  auto query_length = query.size(1);
  auto key_lenght = key.size(1);
  auto kv_head_num = key.size(2);
  auto head_size = key.size(3);
  if (key.scalar_type() != at::kBFloat16 && key.scalar_type() != at::kFloat &&
      key.scalar_type() != at::kHalf) {
    TORCH_CHECK(
        false,
        "key and value must be float, float16 or bfloat16 to use ipex::masked_multihead_self_attention_kernel_impl");
  }

  // support MGQ/MQA
  // expand the head dimensiopn of key/value to be same to the query
  if (query.size(2) != key.size(2)) {
    auto n_req = query.size(2) / key.size(2);
    key = key.repeat_interleave(n_req, 2);
    value = value.repeat_interleave(n_req, 2);
  }
  auto attn_outputs = at::Tensor();
  auto attn_weights = at::Tensor();
  if (is_first_token_optimizable(key) && attention_mask.stride(-1) == 1) {
    query = query.transpose(1, 2);
    key = key.transpose(1, 2);
    value = value.transpose(1, 2);
    attn_outputs = std::get<0>(torch_ipex::cpu::flash_attention_kernel_stub(
        kCPU,
        query,
        key,
        value,
        /* dropout */ 0.0,
        add_casual_mask,
        attention_mask,
        1. / scale_attn));
  } else {
    if (origin_type == at::kHalf) {
      key = key.to(at::kFloat);
      query = query.to(at::kFloat);
      value = value.to(at::kFloat);
    }
    key = key.permute({0, 2, 1, 3});
    query = query.permute({0, 2, 1, 3});
    value = value.permute({0, 2, 1, 3});
    attn_weights = query.matmul(key.transpose(-1, -2));
    attn_weights = attn_weights.div(scale_attn);
    attn_weights = attn_weights + attention_mask;
    attn_weights = attn_weights.softmax(-1);
    attn_weights = attn_weights.to(value.dtype());
    attn_outputs = attn_weights.matmul(value);
    if (origin_type == at::kHalf) {
      attn_weights = attn_weights.to(origin_type);
      attn_outputs = attn_outputs.to(origin_type);
    }
  }
  if (v_head_size != head_size) {
    attn_outputs = attn_outputs.slice(-1, 0, v_head_size);
  }
  return std::make_tuple(attn_outputs, attn_weights, kv_cache, beam_idx);
}

inline std::optional<at::Tensor> convert_boolean_attn_mask(
    const std::optional<at::Tensor>& attn_mask,
    caffe2::TypeMeta dtype) {
  // Pass through
  if (!attn_mask.has_value()) {
    return c10::nullopt;
  }
  // Convert boolean mask to additive mask
  if (attn_mask->dtype() == at::kBool) {
    auto new_attn_mask = at::zeros_like(attn_mask.value(), dtype);
    new_attn_mask.masked_fill_(
        attn_mask->logical_not(), -std::numeric_limits<double>::infinity());
    return new_attn_mask;
  }
  // Otherwise, attn_mask represents an additive attention tensor
  return attn_mask;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
masked_multihead_self_attention_kernel_impl(
    at::Tensor& query,
    at::Tensor& key,
    at::Tensor& value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    at::Tensor& beam_idx,
    at::Tensor seq_info,
    const double scale_attn,
    int64_t max_positions,
    const c10::optional<at::Tensor>& head_mask /* optional */,
    const c10::optional<at::Tensor>& attention_mask /* optional */,
    c10::optional<bool> add_casual_mask /* optional */) {
  TORCH_CHECK(
      attention_mask.has_value(),
      "Attention mask is necessary for ipex::masked_multihead_self_attention_kernel_impl");
  TORCH_CHECK(
      attention_mask.value().dim() == 4,
      "Attention mask must be 4D for ipex::masked_multihead_self_attention_kernel_impl");

  TORCH_CHECK(
      head_mask.has_value() != true,
      "Head mask is not supported in ipex::masked_multihead_self_attention_kernel_impl");
  TORCH_CHECK(
      query.dtype() == key.dtype(),
      "query and key must have the same data type to use ipex::masked_multihead_self_attention_kernel_impl");

  query = query.contiguous();
  key = key.contiguous();
  value = value.contiguous();
  std::optional<at::Tensor> attn_mask = attention_mask;
  if (!is_first_token_optimizable(key)) {
    attn_mask = convert_boolean_attn_mask(attention_mask, query.dtype());
  }
  auto attention_mask_v = attn_mask.value().contiguous();
  attention_mask_v = attention_mask_v.to(query.dtype());
  auto beam_batch = beam_idx.size(1); // need to prepare the fake beam_idx as
                                      // (max_position, bs) for the first token
  auto offset = seq_info.data_ptr<long>()[0];
  auto cache_size = key_cache.size(0);
  auto cur_len = query.size(1);
  if (offset == 0) {
    max_positions =
        max_positions > cur_len ? max_positions : max_positions + cur_len;
    if (key_cache.scalar_type() == at::ScalarType::Float8_e5m2) {
      key_cache = at::empty(
          {max_positions, beam_batch, key.size(2), key.size(3)},
          key.options().dtype(at::kFloat8_e5m2));
      value_cache = at::empty(
          {max_positions, beam_batch, value.size(2), value.size(3)},
          value.options().dtype(at::kFloat8_e5m2));
    } else {
      key_cache = at::empty(
          {max_positions, beam_batch, key.size(2), key.size(3)}, key.options());
      value_cache = at::empty(
          {max_positions, beam_batch, value.size(2), value.size(3)},
          value.options());
    }
    beam_idx = at::zeros({max_positions + 2, beam_batch}, beam_idx.options());
    auto beam_idx_access = beam_idx.accessor<long, 2>();
#pragma omp parallel for collapse(2)
    for (auto i = 0; i < max_positions; i++) {
      for (auto j = 0; j < beam_batch; j++) {
        if (key.size(0) == beam_batch) {
          beam_idx_access[i][j] = j;
        } else {
          auto beam_size = beam_batch / key.size(0);
          beam_idx_access[i][j] = j / beam_size * beam_size;
        }
      }
    }
    beam_idx_access[max_positions][0] = cur_len; // record the prompt token len
    beam_idx_access[max_positions + 1][0] =
        query.size(0); // record the promt bs info

  } else if (offset > 0 && offset + cur_len > cache_size) {
    auto new_cache_size = cache_size * 2;
    auto new_key_cache = at::empty(
        {new_cache_size, beam_batch, key.size(2), key.size(3)},
        key_cache.options());
    auto new_value_cache = at::empty(
        {new_cache_size, beam_batch, value.size(2), value.size(3)},
        value_cache.options());
    auto new_beam_idx =
        at::zeros({new_cache_size + 2, beam_batch}, beam_idx.options());
    new_key_cache.slice(0, 0, cache_size).copy_(key_cache);
    new_value_cache.slice(0, 0, cache_size).copy_(value_cache);
    new_beam_idx.slice(0, 0, cache_size + 2).copy_(beam_idx);
    auto new_beam_idx_access = new_beam_idx.accessor<long, 2>();
    auto beam_idx_access = beam_idx.accessor<long, 2>();
    for (auto i = offset; i < new_cache_size; i++) {
      for (auto j = 0; j < beam_batch; j++) {
        new_beam_idx_access[i][j] = beam_idx_access[0][j];
      }
    }
    new_beam_idx_access[new_cache_size][0] = beam_idx_access[cache_size][0];
    new_beam_idx_access[new_cache_size + 1][0] =
        beam_idx_access[cache_size + 1][0];
    key_cache = new_key_cache;
    value_cache = new_value_cache;
    beam_idx = new_beam_idx;
  }
  if (offset != 0) {
    auto cur_len = query.size(1);
    if (cur_len == 1)
      return zero_copy_kv_cache_masked_multihead_self_attention_kernel_impl(
          query,
          key,
          value,
          key_cache,
          value_cache,
          beam_idx,
          offset,
          scale_attn,
          attention_mask_v);
    // just a  funcationality path,need to optimize
    auto tokens_outs = std::vector<at::Tensor>(cur_len);
    for (auto i = 0; i < cur_len; i++) {
      auto query_i = query.select(1, i).unsqueeze(1);
      ;
      auto key_i = key.select(1, i).unsqueeze(1);
      ;
      auto value_i = value.select(1, i).unsqueeze(1);
      ;
      auto next_outs =
          zero_copy_kv_cache_masked_multihead_self_attention_kernel_impl(
              query_i,
              key_i,
              value_i,
              key_cache,
              value_cache,
              beam_idx,
              offset + i,
              scale_attn,
              attention_mask_v);
      tokens_outs[i] = std::get<0>(next_outs);
    }
    auto attn_outs = at::cat(tokens_outs, 2);
    return std::make_tuple(
        attn_outs, at::Tensor(), key_cache, value_cache, beam_idx);
  } else {
    return first_token_masked_mha(
        query,
        key,
        value,
        key_cache,
        value_cache,
        beam_idx,
        beam_batch,
        scale_attn,
        attention_mask_v,
        add_casual_mask.value_or(true));
  }
}

template <typename T>
std::tuple<at::Tensor, at::Tensor> get_key_value(
    const at::Tensor& kv,
    const at::Tensor& k_pe,
    const int64_t q_head_num,
    const int64_t q_head_dim,
    const int64_t v_head_dim) {
  RECORD_FUNCTION("get_key_value", c10::ArrayRef<c10::IValue>({}));
  auto kv_bs = kv.size(0);
  auto cur_len = kv.size(1);
  auto kv_head_dim = kv.size(-1);
  auto qk_rope_head_dim = k_pe.size(-1);
  auto kv_head_num = k_pe.size(2);
  auto qk_nope_head_dim = q_head_dim - qk_rope_head_dim;
  auto head_dim = qk_nope_head_dim + qk_rope_head_dim;

  auto key = at::empty({kv_bs, cur_len, q_head_num, head_dim}, kv.options());
  auto value = at::empty({kv_bs, cur_len, q_head_num, head_dim}, kv.options());
  auto kv_ptr = kv.data_ptr<T>();
  auto value_ptr = value.data_ptr<T>();
  auto key_ptr = key.data_ptr<T>();
  auto k_pe_ptr = k_pe.data_ptr<T>();
  auto kv_stride0 = kv.stride(0);
  auto kv_stride1 = kv.stride(1);
  auto kv_stride2 = kv.stride(2);
  auto key_stride0 = key.stride(0);
  auto key_stride1 = key.stride(1);
  auto key_stride2 = key.stride(2);
  auto value_stride0 = value.stride(0);
  auto value_stride1 = value.stride(1);
  auto value_stride2 = value.stride(2);
  auto k_pe_stride0 = k_pe.stride(0);
  auto k_pe_stride1 = k_pe.stride(1);
  auto k_pe_stride2 = k_pe.stride(2);
#pragma omp parallel for collapse(3)
  for (auto bi = 0; bi < kv_bs; bi++) {
    for (auto si = 0; si < cur_len; si++) {
      for (auto hi = 0; hi < q_head_num; hi++) {
        auto key_ptr_start =
            key_ptr + bi * key_stride0 + si * key_stride1 + hi * key_stride2;
        auto value_ptr_start = value_ptr + bi * value_stride0 +
            si * value_stride1 + hi * value_stride2;
        auto k_pe_ptr_start = k_pe_ptr + bi * k_pe_stride0 + si * k_pe_stride1;
        auto kv_ptr_start =
            kv_ptr + bi * kv_stride0 + si * kv_stride1 + hi * kv_stride2;
        torch_ipex::cpu::kernel::move_ker<T, T>(
            key_ptr_start, kv_ptr_start, qk_nope_head_dim);
        torch_ipex::cpu::kernel::move_ker<T, T>(
            value_ptr_start, kv_ptr_start + qk_nope_head_dim, v_head_dim);
        torch_ipex::cpu::kernel::zero_ker<T>(
            value_ptr_start + v_head_dim, qk_rope_head_dim);
        torch_ipex::cpu::kernel::move_ker<T, T>(
            key_ptr_start + qk_nope_head_dim, k_pe_ptr_start, qk_rope_head_dim);
      }
    }
  }
  return std::make_tuple(key, value);
}

template <typename T>
at::Tensor get_query(
    const at::Tensor& q_out,
    const at::Tensor& query,
    const int64_t qk_nope_head_dim,
    const int64_t qk_rope_head_dim,
    const int64_t kv_lora_rank) {
  RECORD_FUNCTION("get_query", c10::ArrayRef<c10::IValue>({}));
  auto bs = query.size(0);
  auto q_head_num = query.size(1);
  auto query_ptr = query.data_ptr<T>();
  auto query_stride0 = query.stride(0);
  auto query_stride1 = query.stride(1);
  auto q_ptr = q_out.data_ptr<T>();
  auto q_stride0 = q_out.stride(0);
  auto q_stride1 = q_out.stride(1);

#pragma omp parallel for collapse(2)
  for (auto bi = 0; bi < bs; bi++) {
    for (auto hi = 0; hi < q_head_num; hi++) {
      auto q_rope_ptr_start = query_ptr + bi * query_stride0 +
          +hi * query_stride1 + qk_nope_head_dim;
      auto q_ptr_start = q_ptr + bi * q_stride0 + hi * q_stride1 + kv_lora_rank;
      torch_ipex::cpu::kernel::move_ker<T, T>(
          q_ptr_start, q_rope_ptr_start, qk_rope_head_dim);
    }
  }
  return q_out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
deepseekv2_mla_kernel_impl(
    at::Tensor& query,
    at::Tensor& kv,
    at::Tensor& k_pe,
    at::Tensor& kv_cache,
    at::Tensor& kv_b_weight,
    at::Tensor& w_kc,
    at::Tensor& w_vc,
    at::Tensor& beam_idx,
    at::Tensor seq_info,
    const double scale_attn,
    int64_t max_positions,
    int64_t v_head_dim,
    const c10::optional<at::Tensor>& head_mask /* optional */,
    const c10::optional<at::Tensor>& attention_mask /* optional */,
    const c10::optional<at::Tensor>& w_scale /* optional */,
    c10::optional<bool> add_casual_mask /* optional */) {
  TORCH_CHECK(
      attention_mask.has_value(),
      "Attention mask is necessary for ipex::deepseekv2_mla_kernel_impl");
  TORCH_CHECK(
      attention_mask.value().dim() == 4,
      "Attention mask must be 4D for ipex::deepseekv2_mla_kernel_impl");

  TORCH_CHECK(
      head_mask.has_value() != true,
      "Head mask is not supported in ipex::deepseekv2_mla_kernel_impl");
  TORCH_CHECK(
      query.dtype() == kv.dtype(),
      "query and kv must have the same data type to use ipex::deepseekv2_mla_kernel_impl");
  TORCH_CHECK(
      query.dtype() == k_pe.dtype(),
      "query and k_pe must have the same data type to use ipex::deepseekv2_mla_kernel_impl");
  auto offset = seq_info.data_ptr<long>()[0];
  auto beam_batch = beam_idx.size(1); // need to prepare the fake beam_idx as
                                      // (max_position, bs) for the first token
  auto cache_size = kv_cache.size(0);
  auto cur_len = query.size(1);
  auto kv_bs = kv.size(0);
  auto kv_lora_rank = kv.size(-1);
  auto qk_rope_head_dim = k_pe.size(-1);
  auto kv_head_num = k_pe.size(2);
  auto q_head_num = query.size(2);
  auto q_head_dim = query.size(-1);
  auto kv_head_size = qk_rope_head_dim + kv_lora_rank;
  auto qk_nope_head_dim = q_head_dim - qk_rope_head_dim;
  std::optional<at::Tensor> attn_mask = attention_mask;
  if (!is_first_token_optimizable(kv)) {
    attn_mask = convert_boolean_attn_mask(attention_mask, query.dtype());
  }
  auto attention_mask_v = attn_mask.value().contiguous();
  attention_mask_v = attention_mask_v.to(query.dtype());
  if (offset == 0) {
    max_positions =
        max_positions > cur_len ? max_positions : max_positions + cur_len;
    if (kv_cache.scalar_type() == at::ScalarType::Float8_e5m2) {
      kv_cache = at::empty(
          {max_positions, beam_batch, kv_head_num, kv_head_size},
          kv.options().dtype(at::kFloat8_e5m2));
    } else {
      kv_cache = at::empty(
          {max_positions, beam_batch, kv_head_num, kv_head_size}, kv.options());
    }
    beam_idx = at::zeros({max_positions + 2, beam_batch}, beam_idx.options());
    auto beam_idx_access = beam_idx.accessor<long, 2>();
#pragma omp parallel for collapse(2)
    for (auto i = 0; i < max_positions; i++) {
      for (auto j = 0; j < beam_batch; j++) {
        if (kv_bs == beam_batch) {
          beam_idx_access[i][j] = j;
        } else {
          auto beam_size = beam_batch / kv_bs;
          beam_idx_access[i][j] = j / beam_size * beam_size;
        }
      }
    }
    beam_idx_access[max_positions][0] = cur_len; // record the prompt token len
    beam_idx_access[max_positions + 1][0] =
        query.size(0); // record the promt bs info

  } else if (offset > 0 && offset + cur_len > cache_size) {
    auto new_cache_size = cache_size * 2;
    auto new_kv_cache = at::empty(
        {new_cache_size, beam_batch, kv_head_num, kv_head_size},
        kv_cache.options());
    auto new_beam_idx =
        at::zeros({new_cache_size + 2, beam_batch}, beam_idx.options());
    new_kv_cache.slice(0, 0, cache_size).copy_(kv_cache);
    new_beam_idx.slice(0, 0, cache_size + 2).copy_(beam_idx);
    auto new_beam_idx_access = new_beam_idx.accessor<long, 2>();
    auto beam_idx_access = beam_idx.accessor<long, 2>();
    for (auto i = offset; i < new_cache_size; i++) {
      for (auto j = 0; j < beam_batch; j++) {
        new_beam_idx_access[i][j] = beam_idx_access[0][j];
      }
    }
    new_beam_idx_access[new_cache_size][0] = beam_idx_access[cache_size][0];
    new_beam_idx_access[new_cache_size + 1][0] =
        beam_idx_access[cache_size + 1][0];
    kv_cache = new_kv_cache;
    beam_idx = new_beam_idx;
  }

  if (kv_cache.scalar_type() == at::ScalarType::Float8_e5m2 &&
      kv.scalar_type() == at::ScalarType::BFloat16) {
    copy_key_value<at::BFloat16, at::Float8_e5m2>(
        kv_cache, kv, k_pe, beam_batch, offset);
  } else if (kv.scalar_type() == at::kFloat) {
    copy_key_value<float, float>(kv_cache, kv, k_pe, beam_batch, offset);
  } else if (kv.scalar_type() == at::kBFloat16) {
    copy_key_value<at::BFloat16, at::BFloat16>(
        kv_cache, kv, k_pe, beam_batch, offset);
  } else {
    copy_key_value<at::Half, at::Half>(kv_cache, kv, k_pe, beam_batch, offset);
  }

  at::Tensor key, value;

  if (offset == 0) {
    kv = kv.matmul(kv_b_weight).view({kv_bs, kv.size(1), q_head_num, -1});
    if (kv.scalar_type() == at::kFloat) {
      std::tie(key, value) =
          get_key_value<float>(kv, k_pe, q_head_num, q_head_dim, v_head_dim);
    } else if (kv.scalar_type() == at::kBFloat16) {
      std::tie(key, value) = get_key_value<at::BFloat16>(
          kv, k_pe, q_head_num, q_head_dim, v_head_dim);
    } else {
      std::tie(key, value) =
          get_key_value<at::Half>(kv, k_pe, q_head_num, q_head_dim, v_head_dim);
    }
    query = query.contiguous();
    return first_token_deepseekv2_mla(
        query,
        key,
        value,
        kv_cache,
        beam_idx,
        beam_batch,
        scale_attn,
        attention_mask_v,
        v_head_dim,
        add_casual_mask.value_or(true));
  }
  auto q =
      at::empty({kv_bs * cur_len, q_head_num, kv_head_size}, query.options());
  auto bmm_out = q.narrow(-1, 0, kv_lora_rank).transpose_(0, 1);
  auto q_tmp = query.reshape({-1, q_head_num, q_head_dim});
  auto mat1 = q_tmp.narrow(2, 0, qk_nope_head_dim).transpose_(0, 1);
  torch_ipex::cpu::bmm_kernel_stub(
      kCPU, bmm_out, mat1, w_kc, true, w_scale, false, false, block_size_n());
  if (query.scalar_type() == at::kFloat) {
    q = get_query<float>(
        q, q_tmp, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank);
  } else if (query.scalar_type() == at::kBFloat16) {
    q = get_query<at::BFloat16>(
        q, q_tmp, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank);
  } else if (query.scalar_type() == at::kHalf) {
    q = get_query<at::Half>(
        q, q_tmp, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank);
  } else {
    q = at::cat(
        {bmm_out.transpose_(0, 1),
         q_tmp.narrow(-1, qk_nope_head_dim, qk_rope_head_dim)},
        -1);
    q = q.view({-1, q_head_num, kv_head_size});
  }

  at::Tensor attn_output;
  auto num_kv_splits = 8;
  auto attn_outs =
      at::empty({kv_bs * cur_len, q_head_num, kv.size(-1)}, q.options());
  auto attn_weights =
      at::empty({kv_bs, q_head_num, num_kv_splits, kv.size(-1) + 1});

  auto b_ptr = beam_idx.data_ptr<long>();
  auto max_cache_size = beam_idx.size(0);
  auto prompt_len = b_ptr[(max_cache_size - 2) * beam_batch];
  auto prompt_bs = b_ptr[(max_cache_size - 1) * beam_batch];
  auto beam_size = 1;
  if (prompt_bs != 0) {
    beam_size = beam_batch / prompt_bs;
  }
  if (beam_size > 1) {
    auto new_beam_idx = at::empty({beam_batch, offset + 1}, at::kInt);
    auto new_b_ptr = new_beam_idx.data_ptr<int>();
    auto new_b_stride0 = new_beam_idx.stride(0);
// according to last decoded token to get the target beam for the past
#pragma omp parallel for
    for (int i = 0; i < kv_bs; i++) {
      new_b_ptr[i * new_b_stride0 + offset] = i + offset * beam_batch;
      new_b_ptr[i * new_b_stride0 + offset - 1] =
          b_ptr[(offset - 1) * kv_bs + i] + (offset - 1) * beam_batch;
      for (int j = offset - 2; j >= prompt_len; j--) {
        new_b_ptr[i * new_b_stride0 + j] =
            b_ptr
                [j * kv_bs + new_b_ptr[i * new_b_stride0 + j + 1] -
                 (j + 1) * beam_batch] +
            j * beam_batch;
      }
    }
#pragma omp parallel for collapse(2)
    for (int i = 0; i < kv_bs; i++) {
      for (int j = prompt_len - 1; j >= 0; j--) {
        new_b_ptr[i * new_b_stride0 + j] =
            b_ptr[j * kv_bs + i - i % beam_size] + j * beam_batch;
      }
    }
    torch_ipex::cpu::decode_attention_kernel_stub(
        kCPU,
        q,
        attn_outs,
        kv_cache,
        new_beam_idx,
        attn_weights,
        1 / scale_attn,
        0,
        offset);
  } else {
    torch_ipex::cpu::decode_attention_opt_kernel_stub(
        kCPU, q, attn_outs, kv_cache, attn_weights, 1 / scale_attn, 0, offset);
  }
  attn_outs.transpose_(0, 1);
  attn_output = at::empty(
      {attn_outs.size(0), attn_outs.size(1), w_vc.size(1)},
      attn_outs.options());
  torch_ipex::cpu::bmm_kernel_stub(
      kCPU,
      attn_output,
      attn_outs,
      w_vc,
      true,
      w_scale,
      false,
      false,
      block_size_n());
  attn_output.transpose_(0, 1).unsqueeze_(2);

  return std::make_tuple(attn_output, attn_weights, kv_cache, beam_idx);
}

template <typename T>
void attention_mask_2d_to_4d(
    const T* attention_mask_ptr,
    T* causal_4d_mask_ptr,
    at::Tensor& finfo_min,
    int64_t batch_size,
    int64_t seq_length,
    int64_t src_length,
    int64_t past_key_value_length,
    int64_t length,
    int64_t diagonal) {
  T finfo_min_val = finfo_min.item<T>();

  for (int64_t b = 0; b < batch_size; ++b) {
    for (int64_t l = 0; l < seq_length; ++l) {
      for (int64_t c = 0; c < length; ++c) {
        int64_t idx = b * seq_length * length + l * length + c;
        int64_t mask_idx = l * length + c;
        T value = finfo_min_val;
        if (l + diagonal <= c && l + past_key_value_length >= c) {
          value = 0;
        }
        if (c < src_length) {
          T inverted_mask_value = 1.0 - attention_mask_ptr[b * src_length + c];
          if (inverted_mask_value != 0) {
            value = finfo_min_val;
          }
        }
        causal_4d_mask_ptr[idx] = value;
      }
    }
  }
}

at::Tensor prepare_4d_causal_attention_mask_kernel_impl(
    at::Tensor& attention_mask,
    at::Tensor& inputs_embeds,
    at::Tensor& past_kv_len,
    at::Tensor& finfo_min,
    int64_t sliding_window) {
  auto dtype = inputs_embeds.scalar_type();
  int64_t batch_size = inputs_embeds.size(0);
  int64_t seq_length = inputs_embeds.size(1);
  int64_t src_length = attention_mask.size(-1);
  int64_t past_key_value_length = past_kv_len.item<int64_t>();
  int64_t length = seq_length + past_key_value_length;
  int64_t diagonal = past_key_value_length - sliding_window;

  at::Tensor causal_4d_mask = torch::empty(
      {batch_size, 1, seq_length, length}, inputs_embeds.options());
  attention_mask = attention_mask.to(inputs_embeds.dtype());

  if (dtype == at::kFloat) {
    float* attention_mask_ptr = attention_mask.data_ptr<float>();
    float* causal_4d_mask_ptr = causal_4d_mask.data_ptr<float>();
    attention_mask_2d_to_4d<float>(
        attention_mask_ptr,
        causal_4d_mask_ptr,
        finfo_min,
        batch_size,
        seq_length,
        src_length,
        past_key_value_length,
        length,
        diagonal);
  } else if (dtype == at::kBFloat16) {
    at::BFloat16* attention_mask_ptr = attention_mask.data_ptr<at::BFloat16>();
    at::BFloat16* causal_4d_mask_ptr = causal_4d_mask.data_ptr<at::BFloat16>();
    attention_mask_2d_to_4d<at::BFloat16>(
        attention_mask_ptr,
        causal_4d_mask_ptr,
        finfo_min,
        batch_size,
        seq_length,
        src_length,
        past_key_value_length,
        length,
        diagonal);
  } else if (dtype == at::kHalf) {
    at::Half* attention_mask_ptr = attention_mask.data_ptr<at::Half>();
    at::Half* causal_4d_mask_ptr = causal_4d_mask.data_ptr<at::Half>();
    attention_mask_2d_to_4d<at::Half>(
        attention_mask_ptr,
        causal_4d_mask_ptr,
        finfo_min,
        batch_size,
        seq_length,
        src_length,
        past_key_value_length,
        length,
        diagonal);
  } else {
    AT_ASSERT(
        0, "TPP does not support current dtype %s:%d\n", __FILE__, __LINE__);
  }

  return causal_4d_mask;
}
} // anonymous namespace

IPEX_REGISTER_DISPATCH(
    masked_multihead_self_attention_kernel_stub,
    &masked_multihead_self_attention_kernel_impl);
IPEX_REGISTER_DISPATCH(deepseekv2_mla_kernel_stub, &deepseekv2_mla_kernel_impl);

IPEX_REGISTER_DISPATCH(
    prepare_4d_causal_attention_mask_kernel_stub,
    &prepare_4d_causal_attention_mask_kernel_impl);

} // namespace cpu
} // namespace torch_ipex
