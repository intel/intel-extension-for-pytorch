#include <aten/RotaryPositionEmbedding.h>
#include <torch/csrc/autograd/function.h>
#include <torch/all.h>
#include <ATen/Tensor.h>
#include "vec/vec.h"

namespace torch_ipex {
namespace cpu {

namespace {

template <typename T_in, typename T_emb>
void apply_rope_along_head(
    T_in* in_ptr_start,
    T_emb* cos_start,
    T_emb* sin_start,
    int64_t rotary_ndims,   
    int64_t offset) {
  for (int h = 0; h < rotary_ndims/2; h++) {//used by lamma, ToDo:vectorization                
        float in0 = in_ptr_start[h];
        float in1 = in_ptr_start[h+offset];
        float sin = sin_start[h];
        float cos = cos_start[h];
        float out0 = in0 * cos - in1 * sin;
        float out1 = in1 * cos + in0 * sin;
        in_ptr_start[h] = out0;
        in_ptr_start[h+offset] = out1;
  }   
}

template <>
void apply_rope_along_head(
    at::BFloat16* in_ptr_start,
    float* cos_start,
    float* sin_start,
    int64_t rotary_ndims,   
    int64_t offset) {
    auto vec_size = 16;
    auto h = 0;
    #if defined(CPU_CAPABILITY_AVX512)
    for(h = 0; h <= rotary_ndims/2 - vec_size; h += vec_size) {
        auto in0 = torch_ipex::cpu::kernel::convert_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(in_ptr_start + h)));
        auto in1 = torch_ipex::cpu::kernel::convert_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(in_ptr_start + h + offset)));
        auto sin = _mm512_loadu_ps(sin_start + h);
        auto cos = _mm512_loadu_ps(cos_start + h);
        auto out0 = _mm512_sub_ps(_mm512_mul_ps(in0, cos), _mm512_mul_ps(in1, sin));
        auto out1 = _mm512_add_ps(_mm512_mul_ps(in1, cos), _mm512_mul_ps(in0, sin));
        _mm256_storeu_si256((__m256i*)(in_ptr_start + h), cvt_fp32_to_bf16(out0));
        _mm256_storeu_si256((__m256i*)(in_ptr_start + h + offset), cvt_fp32_to_bf16(out1));
    }
    for(; h < rotary_ndims/2; h++) {        
        float in0 = in_ptr_start[h];
        float in1 = in_ptr_start[h+offset];
        float sin = sin_start[h];
        float cos = cos_start[h];
        float out0 = in0 * cos - in1 * sin;
        float out1 = in1 * cos + in0 * sin;
        in_ptr_start[h] = out0;
        in_ptr_start[h+offset] = out1;
    }
    #else
    for(h=0; h < rotary_ndims/2; h++) {        
        float in0 = in_ptr_start[h];
        float in1 = in_ptr_start[h+offset];
        float sin = sin_start[h];
        float cos = cos_start[h];
        float out0 = in0 * cos - in1 * sin;
        float out1 = in1 * cos + in0 * sin;
        in_ptr_start[h] = out0;
        in_ptr_start[h+offset] = out1;
    }
    #endif
}

template <>
void apply_rope_along_head(
    at::BFloat16* in_ptr_start,
    at::BFloat16* cos_start,
    at::BFloat16* sin_start,
    int64_t rotary_ndims,   
    int64_t offset) {
    auto vec_size = 16;
    auto h = 0;
    #if defined(CPU_CAPABILITY_AVX512)
    for(h = 0; h <= rotary_ndims/2 - vec_size; h += vec_size) {
        auto in0 = torch_ipex::cpu::kernel::convert_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(in_ptr_start + h)));
        auto in1 = torch_ipex::cpu::kernel::convert_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(in_ptr_start + h + offset)));
        auto sin = torch_ipex::cpu::kernel::convert_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(sin_start + h)));
        auto cos = torch_ipex::cpu::kernel::convert_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(cos_start + h)));
        auto out0 = _mm512_sub_ps(_mm512_mul_ps(in0, cos), _mm512_mul_ps(in1, sin));
        auto out1 = _mm512_add_ps(_mm512_mul_ps(in1, cos), _mm512_mul_ps(in0, sin));
        _mm256_storeu_si256((__m256i*)(in_ptr_start + h), cvt_fp32_to_bf16(out0));
        _mm256_storeu_si256((__m256i*)(in_ptr_start + h + offset), cvt_fp32_to_bf16(out1));
    }
    for(; h < rotary_ndims/2; h++) {        
        float in0 = in_ptr_start[h];
        float in1 = in_ptr_start[h+offset];
        float sin = sin_start[h];
        float cos = cos_start[h];
        float out0 = in0 * cos - in1 * sin;
        float out1 = in1 * cos + in0 * sin;
        in_ptr_start[h] = out0;
        in_ptr_start[h+offset] = out1;
    }
    #else
    for(h=0; h < rotary_ndims/2; h++) {        
        float in0 = in_ptr_start[h];
        float in1 = in_ptr_start[h+offset];
        float sin = sin_start[h];
        float cos = cos_start[h];
        float out0 = in0 * cos - in1 * sin;
        float out1 = in1 * cos + in0 * sin;
        in_ptr_start[h] = out0;
        in_ptr_start[h+offset] = out1;
    }
    #endif
}


template <>
void apply_rope_along_head(
    float* in_ptr_start,
    float* cos_start,
    float* sin_start,
    int64_t rotary_ndims,   
    int64_t offset) {
    auto vec_size = 16;
    auto h = 0;
    #if defined(CPU_CAPABILITY_AVX512)
    for(h = 0; h <= rotary_ndims/2 - vec_size; h += vec_size) {
        auto in0 = _mm512_loadu_ps(in_ptr_start + h);
        auto in1 = _mm512_loadu_ps(in_ptr_start + h + offset);
        auto sin = _mm512_loadu_ps(sin_start + h);
        auto cos = _mm512_loadu_ps(cos_start + h);
        auto out0 = _mm512_sub_ps(_mm512_mul_ps(in0, cos), _mm512_mul_ps(in1, sin));
        auto out1 = _mm512_add_ps(_mm512_mul_ps(in1, cos), _mm512_mul_ps(in0, sin));
        _mm512_storeu_ps(in_ptr_start + h, out0);
        _mm512_storeu_ps(in_ptr_start + h + offset, out1);
    }
    for(; h < rotary_ndims/2; h++) {        
        float in0 = in_ptr_start[h];
        float in1 = in_ptr_start[h+offset];
        float sin = sin_start[h];
        float cos = cos_start[h];
        float out0 = in0 * cos - in1 * sin;
        float out1 = in1 * cos + in0 * sin;
        in_ptr_start[h] = out0;
        in_ptr_start[h+offset] = out1;
    }
    #else
    for(h=0; h < rotary_ndims/2; h++) {        
        float in0 = in_ptr_start[h];
        float in1 = in_ptr_start[h+offset];
        float sin = sin_start[h];
        float cos = cos_start[h];
        float out0 = in0 * cos - in1 * sin;
        float out1 = in1 * cos + in0 * sin;
        in_ptr_start[h] = out0;
        in_ptr_start[h+offset] = out1;
    }
    #endif
}
template <typename T>
void ApplyROPEKernel(
    at::Tensor& t_in,
    at::Tensor& t_emb_pos,
    at::Tensor& t_pos,
    int64_t N,//N: number of head, H: head size
    int64_t H,
    int64_t offset,
    int64_t rotary_ndims) {
  auto in_sizes = t_in.sizes(); // in[B][S][F]
  auto MP = t_emb_pos.size(0); // Max Pos
  auto HR = t_emb_pos.size(1); // rotary_dim
  auto B = in_sizes[0];
  auto S = in_sizes[1];
  auto COFF = HR / 2;
  t_in = t_in.contiguous();
  t_emb_pos = t_emb_pos.contiguous();
  t_pos = t_pos.contiguous();
  auto in_ptr = t_in.data_ptr<T>(); // [B][S][N][H]
  auto in_stride_b = S*N*H;
  auto in_stride_s = N*H;
  auto emb_pos_ptr = t_emb_pos.data_ptr<float>(); // [MP][HR]
  auto pos_ptr = t_pos.data_ptr<long>(); // [MB][S]
  {

#pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
      for (int s = 0; s < S; s++) {
        for (int n = 0; n < N; n++) {
          if(1 == offset) {//used by GPT-J 6B
            for (int h = 0, h2 = 0; h < HR; h += 2, h2++) {
                auto in_offset =b*in_stride_b+s*in_stride_s+n*H + h;
                float in0 = in_ptr[in_offset];
                float in1 = in_ptr[in_offset+1];
                long  p = pos_ptr[b*S + s];
                float sin = emb_pos_ptr[p*HR +h2];
                float cos = emb_pos_ptr[p*HR + COFF + h2];
                float out0 = in0 * cos - in1 * sin;
                float out1 = in1 * cos + in0 * sin;
                in_ptr[in_offset] = out0;
                in_ptr[in_offset+1] = out1;
            }
          }else{              
              auto in_ptr_start =in_ptr + b*in_stride_b+s*in_stride_s+n*H;
              long  p = pos_ptr[b*S + s];
              auto sin_start = emb_pos_ptr + p*HR;
              auto cos_start = emb_pos_ptr + p*HR + COFF;
              apply_rope_along_head<T, float>(in_ptr_start, cos_start, sin_start, rotary_ndims, offset);             
          }       
        }
      }
    }
  }
}

void rotary_position_embedding_kernel_impl(
    at::Tensor& t_in,
    at::Tensor& t_emb_pos,
    at::Tensor& t_pos,
    int64_t N,//N: number of head, H: head size
    int64_t H,
    int64_t offset,
    int64_t rotary_ndims) {
    if (t_in.scalar_type() == at::kFloat) {
        ApplyROPEKernel<float>(t_in, t_emb_pos, t_pos, N, H, offset, rotary_ndims);
    }else if (t_in.scalar_type() == at::kBFloat16) {
        ApplyROPEKernel<at::BFloat16>(t_in, t_emb_pos, t_pos, N, H, offset, rotary_ndims);  
    }else if (t_in.scalar_type() == at::kHalf){
	      ApplyROPEKernel<at::Half>(t_in, t_emb_pos, t_pos, N, H, offset, rotary_ndims);
    }else{
       assert(0);
    }


}

} // anonymous namespace

REGISTER_DISPATCH(rotary_position_embedding_kernel_stub, &rotary_position_embedding_kernel_impl);

} // namespace cpu
} // namespace torch_ipex
