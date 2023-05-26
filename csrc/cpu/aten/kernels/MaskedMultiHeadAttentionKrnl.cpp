#include <aten/MaskedMultiHeadAttention.h>
#include <torch/csrc/autograd/function.h>
#include <torch/all.h>
#include <ATen/Tensor.h>
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

/* 
*The scale-dot product for indirect access kv chache and fuse matmul+div+add+softmax to improve data reuse
*@param  query Query embeeding with the of [beam_size*batch, cur_len, head_num, head_size]
*@param  key Key embeeding with the of [beam_size*batch, cur_len, head_num, head_size]
*@param  beam_idx Beam info for every token [beam_size, offset]
*@param  key_cache Cache past key embeeding with the of [max_len, beam_size*batch, cur_len, head_num, head_size]
*@param  offset  The length of decoded(past) token. 
*@param  scale_factor the sqrt(head_dim).
*@param  attention_mask Which is combined mask for padding mask and casual mask. 
*@param  value The vaule for current tokens. 
*@param  value_chache Cache past value embeeding with the of [max_len, beam_size*batch, cur_len, head_num, head_size]
*@return attn_outs With shape of [beam*bs, head_num, 1, head_size]
*/
template<typename T>
at::Tensor scale_dot_product_for_indirect_access_kv_cache(at::Tensor query, at::Tensor key, const std::vector<std::vector<long>> beam_idx, at::Tensor 
&key_cache, int offset, float scale_factor, at::Tensor attention_mask, at::Tensor value, at::Tensor &value_cache){
    RECORD_FUNCTION("ipex::scale_dot_product_for_indirect_access_kv_cache", c10::ArrayRef<c10::IValue>({}));
    auto bs = query.size(0);//beam_size * batch_size
    auto cur_len = query.size(1);// only process cur_len==1
    auto head_num = query.size(2);
    auto kv_head = key.size(2);
    auto group_size = head_num / kv_head;    
    auto head_size = query.size(3);  
    auto seq_len = offset + cur_len;
    auto kc_token_stride = bs * kv_head * head_size;
    auto attn_weights = at::empty({bs, head_num, cur_len, seq_len}, key.options());   
    query = query.contiguous();
    key = key.contiguous();
    auto q_ptr = query.data_ptr<T>();
    auto k_ptr = key.data_ptr<T>();
    auto k_cache_ptr = key_cache.data_ptr<T>();
    auto attn_w_ptr = attn_weights.data_ptr<T>();    
    auto mask_ptr = attention_mask.data_ptr<T>();
    auto mask_head_num = attention_mask.size(1);
    auto mask_token_stride = mask_head_num * cur_len * seq_len;    
    //value realted 
    value = value.contiguous();
    auto attn_outs = at::empty({bs, head_num, cur_len, head_size}, value.options());
    auto v_ptr = value.data_ptr<T>();
    auto v_cache_ptr = value_cache.data_ptr<T>();
    auto attn_out_ptr = attn_outs.data_ptr<T>();    

    //query Query embeeding with the of [beam_size*batch, cur_len, head_num, head_size]
    //key Key embeeding with the of [beam_size*batch, cur_len, head_num, head_size]
    //key_cache Cache past key embeeding with the of [past_len, beam_size*batch, cur_len, head_num, head_size]
    //Try to reshape the query to [beam_size*batch, cur_len, head_size, head_num]    
    #pragma omp parallel for collapse(2)
    for(auto bi = 0; bi < bs; bi++){
        for (auto hi = 0; hi < head_num; hi++){
            auto kv_hi = hi / group_size;//maping the query head to key/value head to support MGA/MQA
           //printf("group_size:%d hi:%d kv_hi:%d kv_head:%d", group_size, hi, kv_hi, kv_head);
           //fflush(stdout); 
           // e.g.,cur_len = 2, past_len=3
           // query:            t4 t5 
           // key:  t0 t1 t2 t3 t4 t5
           //output shape (2, 5)
           //[qk_t0 qk_t1 qk_t2 qk_t3 qk_t4 -10000.0]
           //[qk_t0 qk_t1 qk_t2 qk_t3 qk_t4 qk_t5   ]
           //fused div+add+softmax
           float p[cur_len][seq_len];
           auto mask_ptr_start = mask_ptr + bi * mask_token_stride;
           for(auto query_ti = 0; query_ti < cur_len; query_ti++){
                for(auto ti = 0; ti < seq_len; ti++){                           
                    //auto t_out_stride  =  out_stride + query_ti * seq_len;
                    //auto attn_w_pos = attn_w_ptr + t_out_stride + ti ;
                    auto q_ptr_start = q_ptr + (bi * cur_len + query_ti) * head_num * head_size  + hi * head_size;                    
                    auto k_ptr_start = k_ptr + (bi * cur_len + query_ti) * kv_head * head_size + kv_hi * head_size;   
                    p[query_ti][ti] = 0.0f;                 
                    if(ti > query_ti + offset){//only caculate the innerproduct for the past token and current token
                        p[query_ti][ti] = -100000.0;
                    }else if(ti == query_ti + offset){//caculate the innerproduct for the current token and store the key
                        auto kc_token_start = ti * kc_token_stride;
                        auto kc_t_beam_start = kc_token_start + bi * kv_head * head_size;
                        auto kc_head_start = k_cache_ptr + kc_t_beam_start + kv_hi * head_size;            
                        reduce_head<T>(q_ptr_start, k_ptr_start, &p[query_ti][ti], head_size, true, kc_head_start);
                    }else{//caculate the innerproduct for the past token
                        auto kc_token_start = ti * kc_token_stride;
                        auto kc_t_beam_start = kc_token_start + beam_idx[bi][ti] * kv_head * head_size;
                        auto kc_head_start = k_cache_ptr + kc_t_beam_start + kv_hi * head_size;                        
                        reduce_head<T>(q_ptr_start, kc_head_start, &p[query_ti][ti], head_size, false, nullptr);                
                    }                                    
                }                    
            }
            
            //div+add+softmax            
            #if defined(CPU_CAPABILITY_AVX512)
            for(auto qi = 0; qi < cur_len; qi++){
                auto max_val = -100000.0f;
                torch_ipex::cpu::kernel::_dil_div_add_reduce_max_fusion_kernel<float, T>(&p[qi][0], mask_ptr_start+qi*seq_len, scale_factor, seq_len, &p[qi][0], max_val);
                torch_ipex::cpu::kernel::_dil_exp_reduce_sum_fusion_kernel(&p[qi][0], seq_len, &p[qi][0], max_val);
                torch_ipex::cpu::kernel::_dil_normalization_kernel<float>(&p[qi][0], max_val, seq_len, &p[qi][0]);
            }
            #else
            assert(false && "AVX512 is required in ipex::scale_dot_product_for_indirect_access_kv_cache");
            #endif
            //calculate weighted value and store the result to attn_outs[bs, head_num, cur_len, head_size]   
            auto attn_out_head_stride = (bi * head_num + hi) * cur_len * head_size;         
            for(auto qi = 0; qi < cur_len; qi++){
                auto attn_out_start = attn_out_ptr + attn_out_head_stride + qi * head_size;
                for(auto i = 0; i < head_size; i++){
                    attn_out_start[i] = 0.0f;
                }
                for(auto vi = 0; vi < seq_len; vi++){
                    auto vc_token_start = vi * kc_token_stride;                    
                    if(vi == qi + offset){//caculate the attention values for the current token
                        auto vc_t_beam_start = vc_token_start + bi * kv_head * head_size;
                        auto v_cache_head_start = v_cache_ptr + vc_t_beam_start + kv_hi * head_size;                        
                        auto v_ptr_start = v_ptr + (bi * cur_len + qi) * kv_head * head_size + kv_hi * head_size;
                        mul_attenion_weights_and_value_of_head<T>(p[qi][vi], v_ptr_start, attn_out_start, head_size, true, v_cache_head_start);
                    }else{//caculate attention values for the past token                        
                        auto vc_t_beam_start = vc_token_start + beam_idx[bi][vi] * kv_head * head_size;
                        auto v_cache_head_start = v_cache_ptr + vc_t_beam_start + kv_hi * head_size;
                        mul_attenion_weights_and_value_of_head<T>(p[qi][vi], v_cache_head_start, attn_out_start, head_size, false, nullptr);
                    }                   
                }
            }
        }        
    }
    return attn_outs;
}

/* 
*The masked self attention for decoder layer with zero-copy of kv_cache
*@param  query Query embeeding with the of [beam_size*batch, cur_len, head_num, head_size]
*@param  key Key embeeding with the of [beam_size*batch, cur_len, head_num, head_size]
*@param  value Value embeeding with the of [beam_size*batch, cur_len, head_num, head_size] -> Todo may be perf is better with [beam_size*batch, cur_len, head_size, head_num]
*@param  key_cache Cache past key embeeding with the of [max_seq_len, beam_size*batch, cur_len, head_num, head_size], the past key state is (beam_size, 1, head_num, head_size) for every token
*@param  value_cache Cache past value embeeding with the of [max_seq_len, beam_size*batch, cur_len, head_num, head_size], the past value state is (beam_size, 1, head_num, head_size) for every token
*@param  beam_idx Cache past beam_idx with the of [max_positions, bs]
*@param  offset  The length of decoded(past) token. 
*@return attn_outs, attn_weights
*/
template <typename Q_T, typename V_T> 
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>  MaskedMHAKernel(
    at::Tensor query,
    at::Tensor key,
    at::Tensor value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    at::Tensor&  beam_idx,
    const int64_t offset, 
    const float scale_attn,
    const int64_t max_positions,
    const c10::optional<at::Tensor>& head_mask/* optional */,
    const c10::optional<at::Tensor>& attention_mask/* optional */) {
    //assert(query.size(1) == 1);
    //beam_idx: [beam_size, offset] for every decoded token, the beam_idx is the target beam idx for the past token
    //the target beam_idx for the input tokens are always 0
    //compute the offset info for the past token 
    //std::cout << "beam_idx:" << beam_idx << std::endl;
    auto bs = query.size(0);
    //the targe beam for the past token 
    auto new_beam_idx = std::vector<std::vector<long>>(bs, std::vector<long>(offset+query.size(1), 0));
    auto b_ptr = beam_idx.data_ptr<long>();
    for(auto i = 0; i < bs; i++){
        new_beam_idx[i][offset-1] = b_ptr[(offset-1) * bs + i];
        for(auto j = offset-2; j>=0; j--){//for the token of input, the target beam is alwarys 0 
            new_beam_idx[i][j] = b_ptr[j*bs+new_beam_idx[i][j+1]]; 
        }
    }
    //std::cout << "new_beam_idx:" << new_beam_idx << std::endl;
    auto mask = attention_mask.has_value() ? attention_mask.value():at::zeros({bs, 1, query.size(1), key.size(1)}, query.options());
    assert(head_mask.has_value() == false && "Head mask is not supported in ipex::scale_dot_product_for_indirect_access_kv_cache");
    auto attn_outs = scale_dot_product_for_indirect_access_kv_cache<Q_T>(query, key, new_beam_idx, key_cache, offset, scale_attn, mask, value, value_cache);
    return {attn_outs, attn_outs, key_cache, value_cache, beam_idx};   //ToDO just return attn_weights_origin for debug    
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
    const c10::optional<at::Tensor>& head_mask/* optional */,
    const c10::optional<at::Tensor>& attention_mask/* optional */) {
    
    //std::cout << "new_beam_idx:" << new_beam_idx << std::endl;
    assert(key.scalar_type()==at::kBFloat16 || key.scalar_type()==at::kFloat);
    if (key.scalar_type() == at::kFloat && value.scalar_type() == at::kFloat) {
        return MaskedMHAKernel<float, float>(query, key, value, key_cache, value_cache, beam_idx, offset, scale_attn, max_positions, head_mask, attention_mask);
    }else if(key.scalar_type() == at::kFloat && value.scalar_type() == at::kBFloat16){
        return MaskedMHAKernel<float, at::BFloat16>(query, key, value, key_cache, value_cache, beam_idx, offset, scale_attn, max_positions, head_mask, attention_mask);
    }else if(key.scalar_type() == at::kBFloat16 && value.scalar_type() == at::kFloat){
        return MaskedMHAKernel<at::BFloat16, float>(query, key, value, key_cache, value_cache, beam_idx, offset, scale_attn, max_positions, head_mask, attention_mask);
    }
    return MaskedMHAKernel<at::BFloat16, at::BFloat16>(query, key, value, key_cache, value_cache, beam_idx, offset, scale_attn, max_positions, head_mask, attention_mask);  
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> first_token_masked_mha(
    at::Tensor query,
    at::Tensor key,
    at::Tensor value,
    const int64_t batch_size,
    const double scale_attn,
    int64_t max_positions,
    const c10::optional<at::Tensor>& head_mask/* optional */,
    const c10::optional<at::Tensor>& attention_mask/* optional */
) {
    
    auto query_length = query.size(1);
    auto key_lenght = key.size(1);
    auto kv_head_num = key.size(2);
    auto head_size = key.size(3);
    auto expand_size = batch_size / query.size(0);
    auto casual_mask = at::full({query_length, key_lenght}, -1e6, query.options());
    casual_mask = at::triu(casual_mask, 1);    
    casual_mask = casual_mask.unsqueeze(0).unsqueeze(0);
    if(max_positions < query_length){
        max_positions = query_length + max_positions;
    }
    //allocate the kv cache buffer for the first token    
    auto key_cache = at::zeros({max_positions, batch_size, kv_head_num, head_size}, key.options());
    auto value_cache = at::zeros({max_positions, batch_size, kv_head_num, head_size}, value.options());    
    //key [batch_size, seq_len, kv_headm_num, head_size]
    for (auto i = 0; i < query.size(1); i++) {
        key_cache.select(0, i).copy_(key.select(1, i).repeat_interleave(expand_size, 0));
        value_cache.select(0, i).copy_(value.select(1, i).repeat_interleave(expand_size, 0));
    }
    //allocate beam_idx buffer for the first token
    auto beam_idx = at::zeros({max_positions, batch_size}, at::kLong);
    //ToDo surpport MGQ/MQA
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
    auto attn_weights_origin = attn_weights.clone();
    attn_weights = attn_weights.div(scale_attn);
    attn_weights = attn_weights + casual_mask;
    if (attention_mask.has_value()) {
        attn_weights = attn_weights + attention_mask.value();
    }
    attn_weights = attn_weights.softmax(-1);
    if (head_mask.has_value()) {
        attn_weights = attn_weights * head_mask.value();
    }
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
    auto bs = beam_idx.size(1);//need to prepare the fake beam_idx as (max_position, bs) for the first token      
    auto offset = seq_info.data_ptr<long>()[0];
    auto cache_size = key_cache.size(0);
    auto cur_len = query.size(1);
    if(offset > 0 && offset + cur_len > cache_size) {
        auto new_cache_size = cache_size * 2;
        auto new_key_cache = at::zeros({new_cache_size, bs, key.size(2), key.size(3)}, key.options());
        auto new_value_cache = at::zeros({new_cache_size, bs, value.size(2), value.size(3)}, value.options());
        auto new_beam_idx = at::zeros({new_cache_size, bs}, beam_idx.options());
        new_key_cache.slice(0, 0, cache_size).copy_(key_cache);
        new_value_cache.slice(0, 0, cache_size).copy_(value_cache);
        new_beam_idx.slice(0, 0, cache_size).copy_(beam_idx);
        key_cache = new_key_cache;
        value_cache = new_value_cache;
        beam_idx = new_beam_idx;
    }
    if(offset > 0){
        return zero_copy_kv_cache_masked_multihead_self_attention_kernel_impl(query, key, value, key_cache, value_cache, beam_idx, offset, scale_attn, max_positions, head_mask, attention_mask);
    }else{
        return first_token_masked_mha(query, key, value, bs, scale_attn, max_positions, head_mask, attention_mask);
    }
    
}


} // anonymous namespace

REGISTER_DISPATCH(masked_multihead_self_attention_kernel_stub, &masked_multihead_self_attention_kernel_impl);

} // namespace cpu
} // namespace torch_ipex