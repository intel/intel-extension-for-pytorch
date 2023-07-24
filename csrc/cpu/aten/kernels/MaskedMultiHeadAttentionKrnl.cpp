#include <ATen/Tensor.h>
#include <aten/MaskedMultiHeadAttention.h>
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>
#include <limits>
#include "vec/vec.h"

namespace torch_ipex {
namespace cpu {

namespace {

template<typename T>
void reduce_head(
    const T* q_ptr_start,
    const T* k_ptr_start,
    float* attn_w_pos,
    int64_t head_size,
    bool store_key,
    T* k_cache_start) {
    for(auto hsi = 0; hsi < head_size; hsi++){
        if(store_key){
            k_cache_start[hsi]=k_ptr_start[hsi];//cat the key into the key_cache.
        } 
        attn_w_pos[0] += q_ptr_start[hsi]*k_ptr_start[hsi];
    }  
}

template<>
void reduce_head(
    const float* q_ptr_start,
    const float* k_ptr_start,
    float* attn_w_pos,
    int64_t head_size,
    bool store_key,
    float* k_cache_start) {
    auto hsi = 0;   
    #if defined(CPU_CAPABILITY_AVX512)
    auto vec_size= 16; // 512/32
    auto qk_sum_vec = _mm512_setzero_ps();
    for(hsi=0; hsi <= head_size - vec_size; hsi+=vec_size){
        auto q_vec = _mm512_loadu_ps(q_ptr_start + hsi);
        auto k_vec = _mm512_loadu_ps(k_ptr_start + hsi);
        if(store_key) {
            _mm512_storeu_ps(k_cache_start + hsi, k_vec);
        }
        qk_sum_vec = _mm512_fmadd_ps(q_vec, k_vec, qk_sum_vec);        
    }
    attn_w_pos[0] += _mm512_reduce_add_ps(qk_sum_vec);
    for(; hsi < head_size; hsi++){
        k_cache_start[hsi]=k_ptr_start[hsi];//cat the key into the key_cache.
        attn_w_pos[0] += q_ptr_start[hsi]*k_ptr_start[hsi]; 
    }  
    return;    
    #endif
    for(hsi=0; hsi < head_size; hsi++){
        k_cache_start[hsi]=k_ptr_start[hsi];//cat the key into the key_cache.
        attn_w_pos[0] += q_ptr_start[hsi]*k_ptr_start[hsi]; 
    }
}

template<>
void reduce_head(
    const at::BFloat16* q_ptr_start,
    const at::BFloat16* k_ptr_start,
    float* attn_w_pos,
    int64_t head_size,
    bool store_key,
    at::BFloat16* k_cache_start) {
    auto hsi = 0;   
    #if defined(CPU_CAPABILITY_AVX512)
    auto vec_size= 16; // 512/32
    auto qk_sum_vec = _mm512_setzero_ps();
    for(hsi=0; hsi <= head_size - vec_size; hsi+=vec_size){
        //load 16 bfloat16 query from q_ptr_start and convert to 16 float32 values
        auto q_vec_bf16 = _mm256_loadu_si256((__m256i*)(q_ptr_start + hsi));
        auto q_vec_fp32 = torch_ipex::cpu::kernel::convert_bf16_to_fp32(q_vec_bf16);        
        //load 16 bfloat16 key from k_ptr_start and convert to 16 float32 values
        auto k_vec_bf16 = _mm256_loadu_si256((__m256i*)(k_ptr_start + hsi));
        auto k_vec_fp32 = torch_ipex::cpu::kernel::convert_bf16_to_fp32(k_vec_bf16);
        if(store_key) {
            _mm256_storeu_si256((__m256i*)(k_cache_start + hsi), k_vec_bf16);
        }
        qk_sum_vec = _mm512_fmadd_ps(q_vec_fp32, k_vec_fp32, qk_sum_vec);
    }
    attn_w_pos[0] += (at::BFloat16)_mm512_reduce_add_ps(qk_sum_vec);
    for(; hsi < head_size; hsi++){
        k_cache_start[hsi]=k_ptr_start[hsi];//cat the key into the key_cache.
        attn_w_pos[0] += q_ptr_start[hsi]*k_ptr_start[hsi]; 
    }
    return;
    #endif
    for(hsi=0; hsi < head_size; hsi++){
        k_cache_start[hsi]=k_ptr_start[hsi];//cat the key into the key_cache.
        attn_w_pos[0] += q_ptr_start[hsi]*k_ptr_start[hsi]; 
    }
}


/* 
*reduce the attnetion_weights with the value embeeding by the dimension of head_size  for every head 
*/
template<typename T>
void mul_attenion_weights_and_value_of_head(
    float& attn_w,
    const T* v_ptr_start,
    T* attn_out_start,
    int64_t head_size,
    bool store_value,
    T* v_cache_start) {
    for(auto hsi = 0; hsi < head_size; hsi++){
        attn_out_start[hsi] += attn_w * v_ptr_start[hsi];
        if(store_value){
            v_cache_start[hsi] = v_ptr_start[hsi];
        }
    }  
}

template<>
void mul_attenion_weights_and_value_of_head(
    float& attn_w,
    const float* v_ptr_start,
    float* attn_out_start,
    int64_t head_size,
    bool store_value,
    float* v_cache_start) {
    auto hsi = 0;
    #if defined(CPU_CAPABILITY_AVX512)
    auto vec_size= 16; // 512/32
    for(hsi=0; hsi <= head_size - vec_size; hsi+=vec_size){
        auto attn_w_vec = _mm512_set1_ps(attn_w);
        auto v_vec = _mm512_loadu_ps(v_ptr_start + hsi);
        auto attn_out_vec = _mm512_loadu_ps(attn_out_start + hsi);
        auto attn_out_vec_new = _mm512_fmadd_ps(attn_w_vec, v_vec, attn_out_vec);
        _mm512_storeu_ps(attn_out_start + hsi, attn_out_vec_new);
        if(store_value){
            _mm512_storeu_ps(v_cache_start + hsi, v_vec);
        }
    }
    for(; hsi < head_size; hsi++){
        attn_out_start[hsi] += attn_w * v_ptr_start[hsi];
        if(store_value){
            v_cache_start[hsi] = v_ptr_start[hsi];
        }
    }
    return;
    #endif
    for(hsi=0; hsi < head_size; hsi++){
        attn_out_start[hsi] += attn_w * v_ptr_start[hsi];
        if(store_value){
            v_cache_start[hsi] = v_ptr_start[hsi];
        }
    }
}

template<>
void mul_attenion_weights_and_value_of_head(
    float& attn_w,
    const at::BFloat16* v_ptr_start,
    at::BFloat16* attn_out_start,
    int64_t head_size,
    bool store_value,
    at::BFloat16* v_cache_start) {
    auto hsi = 0;
    #if defined(CPU_CAPABILITY_AVX512)
    auto vec_size= 16; // 512/32
    for(hsi=0; hsi <= head_size - vec_size; hsi+=vec_size){
        //get 1 bfloat16 values from attn_w_ptr_start and broadcast to 16 float32 values
        auto attn_w_vec_fp32 = _mm512_set1_ps(attn_w);
        //load 16 bfloat16 values from v_ptr_start and convert to 16 float32 values
        auto v_vec_bf16 = _mm256_loadu_si256((__m256i*)(v_ptr_start + hsi));
        auto v_vec_fp32 = torch_ipex::cpu::kernel::convert_bf16_to_fp32(v_vec_bf16);        
        //load 16 bfloat16 values from attn_out_start and convert to 16 float32 values
        auto attn_out_vec_fp32 = torch_ipex::cpu::kernel::convert_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(attn_out_start + hsi)));
        //calculate the new attn_out_vec_fp32 and convert to bfloat16
        auto attn_out_vec_new = _mm512_fmadd_ps(attn_w_vec_fp32, v_vec_fp32, attn_out_vec_fp32);
        auto attn_out_vec_new_bf16 = cvt_fp32_to_bf16(attn_out_vec_new);//_m256i
        //store the new attn_out_vec_new_bf16 to attn_outs
        _mm256_storeu_si256((__m256i*)(attn_out_start + hsi), attn_out_vec_new_bf16);
        //store the v_vec_bf16 to v_cache
        if(store_value){
            _mm256_storeu_si256((__m256i*)(v_cache_start + hsi), v_vec_bf16);
        }
    }
    for(; hsi < head_size; hsi++){
        attn_out_start[hsi] += attn_w * v_ptr_start[hsi];
        if(store_value){
            v_cache_start[hsi] = v_ptr_start[hsi];
        }
    }
    return;
    #endif
    for(hsi=0; hsi < head_size; hsi++){
        attn_out_start[hsi] += attn_w * v_ptr_start[hsi];
        if(store_value){
            v_cache_start[hsi] = v_ptr_start[hsi];
        }
    }

}

template <typename T>
void copy_key_value(
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
  auto key_cache_ptr = key_cache.data_ptr<T>();
  auto key_ptr = key.data_ptr<T>();
  auto value_cache_ptr = value_cache.data_ptr<T>();
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
      torch_ipex::cpu::kernel::move_ker<T, T>(key_cache_start, key_ptr_start, hidden_size);
      auto value_cache_ptr_start = value_cache_ptr + cache_stride;
      auto value_ptr_start = value_ptr + state_stride;
      torch_ipex::cpu::kernel::move_ker<T, T>(value_cache_ptr_start, value_ptr_start, hidden_size);
    }
  }
}

/* 
*The scale-dot product for indirect access kv chache and fuse matmul+div+add+softmax to improve data reuse
*@param  query Query embeeding with the of [beam_size*batch, cur_len, head_num, head_size]
*@param  key Key embeeding with the of [beam_size*batch, cur_len, head_num, head_size]
*@param  value Key embeeding with the of [beam_size*batch, cur_len, head_num, head_size]
*@param  key_cache Cache past key embeeding with the of [max_len, beam_size*batch, head_num, head_size]
*@param  value_chache Cache past value embeeding with the of [max_len, beam_size*batch, head_num, head_size]
*@param  beam_idx Beam info for every token [max_len, beam_size*batch]
*@param  offset  The length of decoded(past) token. 
*@param  scale_factor the sqrt(head_dim).
*@param  head_mask Which is not used by our kernel now. 
*@param  attention_mask Which is combined mask for padding mask and casual mask. 
*@return attn_outs, None, key_cache, value_cache, beam_idx
*/
template <typename QT, typename VT>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>  scale_dot_product_for_indirect_access_kv_cache(
    at::Tensor query,
    at::Tensor key,
    at::Tensor value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    at::Tensor&  beam_idx,    
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
  auto kv_head = key.size(2);
  auto group_size = head_num / kv_head;
  auto head_size = query.size(3);
  auto seq_len = offset + cur_len;
  auto kc_token_stride = beam_batch * kv_head * head_size;
  auto attn_weights =
      at::empty({bs, head_num, cur_len, seq_len}, key.options());
  query = query.contiguous();
  key = key.contiguous();
  auto q_ptr = query.data_ptr<QT>();
  auto k_ptr = key.data_ptr<QT>();
  auto k_cache_ptr = key_cache.data_ptr<QT>();
  auto mask_ptr = attention_mask.data_ptr<QT>();
  auto mask_head_num = attention_mask.size(1);
  auto mask_dim2 = attention_mask.size(2);
  auto mask_bs_stride = mask_head_num * mask_dim2 * seq_len;
  // value realted
  value = value.contiguous();
  auto attn_outs =
      at::empty({bs, head_num, cur_len, head_size}, value.options());
  auto v_ptr = value.data_ptr<VT>();
  auto v_cache_ptr = value_cache.data_ptr<VT>();
  auto attn_out_ptr = attn_outs.data_ptr<VT>();

  // beam_idx: [beam_size, offset] for every decoded token, the beam_idx is the
  // target beam idx for the past token the target beam_idx for the input tokens
  // are always 0 compute the offset info for the past token std::cout <<
  // "beam_idx:" << beam_idx << std::endl;  
  // the targe beam for the past token
  auto new_beam_idx = std::vector<std::vector<long>>(
      beam_batch, std::vector<long>(offset + query.size(1), 0));
  auto b_ptr = beam_idx.data_ptr<long>();
  if (offset > 0) {
    // according to the last decoded token to get the target beam for the past
    // token
    for (int i = 0; i < bs; i++) {
      new_beam_idx[i][offset - 1] = b_ptr[(offset - 1) * bs + i];
      for (int j = offset - 2; j >= 0;
           j--) { // for the token of input, the target beam is alwarys 0
        new_beam_idx[i][j] = b_ptr[j * bs + new_beam_idx[i][j + 1]];
      }
    }
  }
// query Query embeeding with the of [beam_size*batch, cur_len, head_num,
// head_size] key Key embeeding with the of [beam_size*batch, cur_len, head_num,
// head_size] key_cache Cache past key embeeding with the of [past_len,
// beam_size*batch, cur_len, head_num, head_size] Try to reshape the query to
// [beam_size*batch, cur_len, head_size, head_num]
#pragma omp parallel for collapse(3)
  for (auto bi = 0; bi < bs; bi++) {
    for (auto hi = 0; hi < head_num; hi++) {
      // auto kv_hi = hi / group_size;
      // e.g.,cur_len = 2, offset=3
      // query:            t4 t5
      // key:  t0 t1 t2 t3|t4 t5
      // output shape (2, 5)
      //[qk_t0 qk_t1 qk_t2 qk_t3 qk_t4 -10000.0]
      //[qk_t0 qk_t1 qk_t2 qk_t3 qk_t4 qk_t5   ]
      // fused div+add+softmax
      for (auto query_ti = 0; query_ti < cur_len; query_ti++) {
        float p[1][seq_len];
        auto kv_hi = hi / group_size; // maping the query head to key/value head
                                      // to support MGA/MQA
        //printf("beam_batch: %d bi/bs: %d/%d group_size:%d hi:%d kv_hi:%d kv_head:%d \n", beam_batch, bi, bs, group_size, hi, kv_hi, kv_head); fflush(stdout);
        auto mask_ptr_start = mask_ptr + bi * mask_bs_stride;
        auto q_ptr_start = q_ptr +
            (bi * cur_len + query_ti) * head_num * head_size + hi * head_size;
        for (auto ti = 0; ti < seq_len; ti++) {
          p[0][ti] = 0.0f;
          auto kc_token_start = ti * kc_token_stride;
          auto kc_t_beam_start = kc_token_start;
          if (ti > query_ti + offset) { // only caculate the innerproduct for
                                        // the past token and current token
            p[0][ti] = -10000.0f;
          } else if (ti == query_ti + offset) { // caculate the innerproduct for
                                                // the current token and store
                                                // the key
            if (cur_len > 1) { // this may occur for processing the promt
              auto beam_size = beam_batch / bs;
              // need to store key accross beam
              kc_t_beam_start =
                  kc_t_beam_start + bi * beam_size * kv_head * head_size;
            } else {
              kc_t_beam_start = kc_t_beam_start + bi * kv_head * head_size;
            }
            auto kc_head_start =
                k_cache_ptr + kc_t_beam_start + kv_hi * head_size;
            auto k_ptr_start = k_ptr +
                (bi * cur_len + ti - offset) * kv_head * head_size +
                kv_hi * head_size;
            reduce_head<QT>(
                q_ptr_start,
                k_ptr_start,
                &p[0][ti],
                head_size,
                true,
                kc_head_start);
          } else { // caculate the innerproduct for the past token
            if (ti >= offset) {
              auto k_ptr_start = k_ptr +
                  (bi * cur_len + ti - offset) * kv_head * head_size +
                  kv_hi * head_size;
              reduce_head<QT>(
                  q_ptr_start,
                  k_ptr_start,
                  &p[0][ti],
                  head_size,
                  false,
                  nullptr);
            } else {
              kc_t_beam_start =
                  kc_t_beam_start + new_beam_idx[bi][ti] * kv_head * head_size;
              if (cur_len > 1) {
                auto beam_size = beam_batch / bs;
                kc_t_beam_start =
                    kc_t_beam_start + bi * beam_size * kv_head * head_size;
              }
              //printf("new_beam_idx[bi][ti]:%d \n", new_beam_idx[bi][ti]);
              auto kc_head_start =
                  k_cache_ptr + kc_t_beam_start + kv_hi * head_size;
              reduce_head<QT>(
                  q_ptr_start,
                  kc_head_start,
                  &p[0][ti],
                  head_size,
                  false,
                  nullptr);
            }
          }
          // std::cout << " " << p[0][ti];
        }
// std::cout << std::endl;
// div+add+softmax
#if defined(CPU_CAPABILITY_AVX512)
        for (auto qi = 0; qi < 1; qi++) {
          auto max_val = -100000.0f;
          torch_ipex::cpu::kernel::
              _dil_div_add_reduce_max_fusion_kernel<float, QT>(
                  &p[qi][0],
                  mask_ptr_start + (query_ti % mask_dim2) * seq_len,
                  scale_factor,
                  seq_len,
                  &p[qi][0],
                  max_val);
          torch_ipex::cpu::kernel::_dil_exp_reduce_sum_fusion_kernel(
              &p[qi][0], seq_len, &p[qi][0], max_val);
          torch_ipex::cpu::kernel::_dil_normalization_kernel<float>(
              &p[qi][0], max_val, seq_len, &p[qi][0]);
        }
#else
        assert(
            false &&
            "AVX512 is required in ipex::scale_dot_product_for_indirect_access_kv_cache");
#endif
        // calculate weighted value and store the result to attn_outs[bs,
        // head_num, cur_len, head_size]
        auto attn_out_head_stride = (bi * head_num + hi) * cur_len * head_size;
        for (auto qi = 0; qi < 1; qi++) {
          auto attn_out_start =
              attn_out_ptr + attn_out_head_stride + (query_ti + qi) * head_size;
          for (auto i = 0; i < head_size; i++) {
            attn_out_start[i] = 0.0f;
          }
          for (auto vi = 0; vi < seq_len; vi++) {
            auto vc_token_start = vi * kc_token_stride;
            if (vi == qi + query_ti + offset) { // caculate the attention values
                                                // for the current token
              auto vc_t_beam_start = vc_token_start;
              if (cur_len > 1) { // this may occur for processing the promt
                auto beam_size = beam_batch / bs;
                // removed the redundant computation, need to store key accross
                // beam
                vc_t_beam_start =
                    vc_t_beam_start + bi * beam_size * kv_head * head_size;
              } else {
                vc_t_beam_start = vc_t_beam_start + bi * kv_head * head_size;
              }
              auto v_cache_head_start =
                  v_cache_ptr + vc_t_beam_start + kv_hi * head_size;
              auto v_ptr_start = v_ptr +
                  (bi * cur_len + vi - offset) * kv_head * head_size +
                  kv_hi * head_size;
              mul_attenion_weights_and_value_of_head<VT>(
                  p[qi][vi],
                  v_ptr_start,
                  attn_out_start,
                  head_size,
                  true,
                  v_cache_head_start);
            } else if (vi < qi + query_ti + offset) { // caculate attention
                                                      // values for the past
                                                      // token
              if (vi >= offset) {
                auto v_ptr_start = v_ptr +
                    (bi * cur_len + vi - offset) * kv_head * head_size +
                    kv_hi * head_size;
                mul_attenion_weights_and_value_of_head<VT>(
                    p[qi][vi],
                    v_ptr_start,
                    attn_out_start,
                    head_size,
                    false,
                    nullptr);
              } else {
                //printf("new_beam_idx[bi][vi]:%d \n", new_beam_idx[bi][vi]);
                auto vc_t_beam_start =
                    vc_token_start + new_beam_idx[bi][vi] * kv_head * head_size;
                if (cur_len > 1) {
                  auto beam_size = beam_batch / bs;
                  // printf("beam_size:%d, kv_head: %d, head_size: %d \n",
                  // beam_size, kv_head, head_size); fflush(stdout);
                  vc_t_beam_start =
                      vc_t_beam_start + bi * beam_size * kv_head * head_size;
                }
                auto v_cache_head_start =
                    v_cache_ptr + vc_t_beam_start + kv_hi * head_size;
                mul_attenion_weights_and_value_of_head<VT>(
                    p[qi][vi],
                    v_cache_head_start,
                    attn_out_start,
                    head_size,
                    false,
                    nullptr);
              }
            }
            // std::cout << " " << p[qi][vi];
          }
          // std::cout << std::endl;
        }
      }
      // std::cout << "p:" << p << std::endl;
    }
  }
  return std::make_tuple(attn_outs, at::Tensor(), key_cache, value_cache, beam_idx);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>  zero_copy_kv_cache_masked_multihead_self_attention_kernel_impl(
    at::Tensor query,
    at::Tensor key,
    at::Tensor value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    at::Tensor&  beam_idx,    
    const int64_t offset,
    const double scale_attn,
    const int64_t max_positions,
    at::Tensor& attention_mask) {      
    assert(key.scalar_type()==at::kBFloat16 || key.scalar_type()==at::kFloat);
    if (query.scalar_type() == at::kFloat && value.scalar_type() == at::kFloat) {
        return scale_dot_product_for_indirect_access_kv_cache<float, float>(query, key, value, key_cache, value_cache, beam_idx, offset, scale_attn, attention_mask);
    }else if(query.scalar_type() == at::kFloat && value.scalar_type() == at::kBFloat16){
        return scale_dot_product_for_indirect_access_kv_cache<float, at::BFloat16>(query, key, value, key_cache, value_cache, beam_idx, offset, scale_attn, attention_mask);
    }else if(key.scalar_type() == at::kBFloat16 && value.scalar_type() == at::kFloat){
        return scale_dot_product_for_indirect_access_kv_cache<at::BFloat16, float>(query, key, value, key_cache, value_cache, beam_idx, offset, scale_attn, attention_mask);
    }
    return scale_dot_product_for_indirect_access_kv_cache<at::BFloat16, at::BFloat16>(query, key, value, key_cache, value_cache, beam_idx, offset, scale_attn, attention_mask);  
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> first_token_masked_mha(
    at::Tensor query,
    at::Tensor key,
    at::Tensor value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    at::Tensor& beam_idx,
    const int64_t beam_batch,
    const double scale_attn,
    int64_t max_positions,
    at::Tensor attention_mask
) {
    
    auto query_length = query.size(1);
    auto key_lenght = key.size(1);
    auto kv_head_num = key.size(2);
    auto head_size = key.size(3);
    auto casual_mask = at::full({query_length, key_lenght}, -1e6, query.options());
    casual_mask = at::triu(casual_mask, 1);    
    casual_mask = casual_mask.unsqueeze(0).unsqueeze(0);
    attention_mask = attention_mask + casual_mask;
    if(max_positions < query_length){
        max_positions = query_length + max_positions;
    }
    if(key.scalar_type() != at::kBFloat16 && key.scalar_type() != at::kFloat){
        TORCH_CHECK(false, "key and value must be float or bfloat16 to use ipex::masked_multihead_self_attention_kernel_impl");
    }
    if (key.scalar_type() == at::kFloat) {
      copy_key_value<float>(key_cache, key, value_cache, value, beam_batch);
    } else {
      copy_key_value<at::BFloat16>(
          key_cache, key, value_cache, value, beam_batch);
    }    
    //surpport MGQ/MQA
    //expand the head dimensiopn of key/value to be same to the query
    if(query.size(2) != key.size(2)){
        auto n_req = query.size(2) / key.size(2);
        key = key.repeat_interleave(n_req, 2);
        value = value.repeat_interleave(n_req, 2);
    }    
    key = key.permute({0, 2, 1, 3});
    query = query.permute({0, 2, 1, 3});
    value = value.permute({0, 2, 1, 3});
    auto attn_weights = query.matmul(key.transpose(-1, -2));
    attn_weights = attn_weights.div(scale_attn);
    attn_weights = attn_weights + attention_mask;
    attn_weights = attn_weights.softmax(-1);
    attn_weights = attn_weights.to(value.dtype());
    auto attn_outputs = attn_weights.matmul(value);
    return std::make_tuple(attn_outputs, attn_weights, key_cache, value_cache, beam_idx);
}
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>  masked_multihead_self_attention_kernel_impl(
    at::Tensor query,
    at::Tensor key,
    at::Tensor value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    at::Tensor&  beam_idx,    
    at::Tensor seq_info,
    const double scale_attn,
    int64_t max_positions,
    const c10::optional<at::Tensor>& head_mask/* optional */,
    const c10::optional<at::Tensor>& attention_mask/* optional */) {  
    if(attention_mask.has_value() == false){
        TORCH_CHECK(false, "Attention mask is neccessary for ipex::masked_multihead_self_attention_kernel_impl");
    }
    if(attention_mask.value().dim() != 4){
        TORCH_CHECK(false, "Attention mask must be 4D for ipex::masked_multihead_self_attention_kernel_impl");
    }
    if(head_mask.has_value() == true){
        TORCH_CHECK(false, "Head mask is not supported in ipex::masked_multihead_self_attention_kernel_impl");
    }
    if (query.dtype() != key.dtype()) {
        TORCH_CHECK(false, "query and key must have the same data type to use ipex::masked_multihead_self_attention_kernel_impl");
    }
    query = query.contiguous();
    key = key.contiguous();
    value = value.contiguous();
    auto attention_mask_v = attention_mask.value().contiguous();
    attention_mask_v = attention_mask_v.to(query.dtype());
    auto beam_batch = beam_idx.size(1);//need to prepare the fake beam_idx as (max_position, bs) for the first token      
    auto offset = seq_info.data_ptr<long>()[0];
    auto cache_size = key_cache.size(0);
    auto cur_len = query.size(1);
    if (offset == 0) {
      max_positions =
          max_positions > cur_len ? max_positions : max_positions + cur_len;
      key_cache = at::empty(
          {max_positions, beam_batch, key.size(2), key.size(3)}, key.options());
      value_cache = at::empty(
          {max_positions, beam_batch, value.size(2), value.size(3)}, value.options());
      beam_idx = at::zeros({max_positions, beam_batch}, beam_idx.options());
      for (auto i = 0; i < max_positions; i++){
          for (auto j = 0; j < beam_batch; j++){
              if(key.size(0) == beam_batch){
                 beam_idx[i][j] = j;
              }else{
                 auto beam_size = beam_batch / key.size(0);
                 beam_idx[i][j] = j / beam_size * beam_size;
              }
            }
       }
    } else if (offset > 0 && offset + cur_len > cache_size) {
      auto new_cache_size = cache_size * 2;
      auto new_key_cache = at::zeros(
          {new_cache_size, beam_batch, key.size(2), key.size(3)}, key.options());
      auto new_value_cache = at::zeros(
          {new_cache_size, beam_batch, value.size(2), value.size(3)}, value.options());
      auto new_beam_idx = at::zeros({new_cache_size, beam_batch}, beam_idx.options());
      new_key_cache.slice(0, 0, cache_size).copy_(key_cache);
      new_value_cache.slice(0, 0, cache_size).copy_(value_cache);
      new_beam_idx.slice(0, 0, cache_size).copy_(beam_idx);
      for (auto i = offset; i < new_cache_size; i++){
          for (auto j = 0; j < beam_batch; j++){
              if(key.size(0) == beam_batch){
                 new_beam_idx[i][j] = j;
              }else{
                 auto beam_size = beam_batch / key.size(0);
                 new_beam_idx[i][j] = j / beam_size * beam_size;
              }
            }
      }
      key_cache = new_key_cache;
      value_cache = new_value_cache;
      beam_idx = new_beam_idx;
    }
    if(offset > 0){
      return zero_copy_kv_cache_masked_multihead_self_attention_kernel_impl(
          query,
          key,
          value,
          key_cache,
          value_cache,
          beam_idx,
          offset,
          scale_attn,
          max_positions,
          attention_mask_v);
    }else{
      return first_token_masked_mha(
          query,
          key,
          value,
          key_cache,
          value_cache,
          beam_idx,
          beam_batch,
          scale_attn,
          max_positions,
          attention_mask_v);
    }
    
}
} // anonymous namespace

REGISTER_DISPATCH(masked_multihead_self_attention_kernel_stub, &masked_multihead_self_attention_kernel_impl);

} // namespace cpu
} // namespace torch_ipex