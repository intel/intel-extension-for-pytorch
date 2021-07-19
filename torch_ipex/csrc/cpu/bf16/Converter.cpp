#include "Converter.h"
#include "vec/vec_type_cvt.h"
#include <ATen/Parallel.h>
#include <torch/extension.h>

#if defined(AVX512)
#define BF16_2_FP32(dst, src, len) cvt_bf16_to_fp32(dst, src, len)
#define FP32_2_BF16(dst, src, len) cvt_fp32_to_bf16(dst, src, len)
#else
#define BF16_2_FP32(dst, src, len)
#define FP32_2_BF16(dst, src, len)
#endif

namespace torch_ipex {
namespace cpu {
namespace bf16 {
namespace converter {

void bf16_to_fp32(void *dst, const void *src, int len) {
  BF16_2_FP32((float *)dst, (at::BFloat16 *)src, len);
}

void fp32_to_bf16(void *dst, const void *src, int len) {
  FP32_2_BF16((at::BFloat16 *)dst, (float *)src, len);
}

at::Tensor cat_bfloat16_float(const at::Tensor top_half, const at::Tensor bottom_half){
  TORCH_CHECK(top_half.scalar_type() == at::kBFloat16 && bottom_half.scalar_type() == at::kBFloat16, 
      "pack_bfloat16_float: expect both args to be at::BFloat16");
  at::Tensor output = at::empty(top_half.sizes(), top_half.options().dtype(at::kFloat));
  using bVec = at::vec::Vectorized<at::BFloat16>;
  using fVec = at::vec::Vectorized<float>;
  at::BFloat16* top_half_data = top_half.data_ptr<at::BFloat16>();
  at::BFloat16* bottom_half_data = bottom_half.data_ptr<at::BFloat16>();
  float* output_data = output.data_ptr<float>();
  int64_t grain_size = 512;
  at::parallel_for(0, top_half.numel(), grain_size, [&](int64_t begin, int64_t end) {
    // local pointers
    at::BFloat16* top_half_ptr = top_half_data + begin;
    at::BFloat16* bottom_half_ptr = bottom_half_data + begin;
    float* output_ptr = output_data + begin;
    const int64_t size = end - begin;
    int64_t d = 0;
    for (; d < size - (size % bVec::size()); d += bVec::size()) {
      bVec top_half_bvec = bVec::loadu(top_half_ptr + d);
      bVec bottom_half_bvec = bVec::loadu(bottom_half_ptr + d);
      fVec fvec,fvec2;
      std::tie(fvec, fvec2) = pack_bfloat16_float(top_half_bvec, bottom_half_bvec);
      fvec.store(output_ptr + d);
      fvec2.store(output_ptr + d + fVec::size());
    }
    for (; d < size; d++) {
      output_ptr[d] =  bf16::pack_bfloat16_float(top_half_ptr[d], bottom_half_ptr[d]);
    }
  });
  return output;
}

std::tuple<at::Tensor, at::Tensor> split_float_bfloat16(const at::Tensor tensor){
  TORCH_CHECK(tensor.scalar_type() == at::kFloat, 
      "pack_bfloat16_float: expect both tensor to be at::kFloat");
  at::Tensor top_half = at::empty(tensor.sizes(), tensor.options().dtype(at::kBFloat16));
  at::Tensor bottom_half = at::empty(tensor.sizes(), tensor.options().dtype(at::kBFloat16));
  using bVec = at::vec::Vectorized<at::BFloat16>;
  using fVec = at::vec::Vectorized<float>;
  at::BFloat16* top_half_data = top_half.data_ptr<at::BFloat16>();
  at::BFloat16* bottom_half_data = bottom_half.data_ptr<at::BFloat16>();
  float* tensor_data = tensor.data_ptr<float>();
  int64_t grain_size = 512;
  at::parallel_for(0, top_half.numel(), grain_size, [&](int64_t begin, int64_t end) {
    // local pointers
    at::BFloat16* top_half_ptr = top_half_data + begin;
    at::BFloat16* bottom_half_ptr = bottom_half_data + begin;
    float* tensor_ptr = tensor_data + begin;
    const int64_t size = end - begin;
    int64_t d = 0;
    for (; d < size - (size % bVec::size()); d += bVec::size()) {
      fVec fvec = fVec::loadu(tensor_ptr + d);
      fVec fvec2 = fVec::loadu(tensor_ptr + d + fVec::size());
      bVec top_half_bvec, bottom_half_bvec;
      std::tie(top_half_bvec, bottom_half_bvec) = unpack_float_bfloat16(fvec, fvec2);
      top_half_bvec.store(top_half_ptr + d);
      bottom_half_bvec.store(bottom_half_ptr + d);
    }
    for (; d < size; d++) {
      at::BFloat16 top_half_val;
      at::BFloat16 bottom_half_val;
      std::tie(top_half_val, bottom_half_val) = unpack_float_bfloat16(tensor_ptr[d]);
      top_half_ptr[d] = top_half_val;
      bottom_half_ptr[d] = bottom_half_val;
    }
  });
  return std::tie(top_half, bottom_half);
}

}  // namespace converter
}  // namespace bf16
}  // namespace cpu
}  // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def("split_float_bfloat16(Tensor tensor) -> (Tensor top, Tensor bot)", torch_ipex::cpu::bf16::converter::split_float_bfloat16);
  m.def("cat_bfloat16_float(Tensor top_half, Tensor bot_half) -> Tensor", torch_ipex::cpu::bf16::converter::cat_bfloat16_float);
}

}
