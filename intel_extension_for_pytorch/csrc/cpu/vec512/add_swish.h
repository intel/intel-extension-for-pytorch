#pragma once

#include <immintrin.h>

#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Parallel.h>
#include <c10/util/SmallVector.h>
#include <limits>
#include "add_softmax.h"
#include "utils.h"

namespace torch_ipex {
namespace cpu {
namespace kernel {
namespace vec {
namespace vec512 {

template <typename scalar_t>
inline void _dil_add_swish_fusion_kernel(
    scalar_t* a,
    const scalar_t* b,
    const int& size) {
  auto vec_ps_min = _mm512_set1_ps(std::numeric_limits<float>::min());
  auto vec_ps_1 = _mm512_set1_ps(1.0);
  __m512 vec_a, vec_b;
  __m512 vec_add_tmp, vec_addone_tmp;

  int i = 0;

  // load tensor<float> a & b
  // assum the same size , no need to broadcast
  for (; i <= size - 16; i += 16) {
    // a is first operand of add, b is bias
    vec_a = _loadu(a + i);
    vec_b = _loadu(b + i);

    // add bias
    vec_a = _mm512_add_ps(vec_a, vec_b);
    vec_add_tmp =
        vec_a; // keep the intermediate result for later use in the mul

    // caculate sigmoid e^x / (1 + e^x)
    vec_a = _dil_exp_kernel(vec_a);
    vec_addone_tmp = _mm512_add_ps(vec_a, vec_ps_1);
    vec_a = _mm512_div_ps(vec_a, vec_addone_tmp);
    vec_a = _mm512_mul_ps(vec_a, vec_add_tmp);

    _storeu(a + i, vec_a);
  }

  // 512 tail
  if (i < size) {
    // mask load
    __mmask16 mask = (1 << (size - i)) - 1;
    vec_a = _maskz_loadu(a + i, mask);
    vec_b = _maskz_loadu(b + i, mask);

    // add bias
    vec_a = _mm512_add_ps(vec_a, vec_b);
    vec_add_tmp =
        vec_a; // keep the intermediate result for later use in the second mul

    // caculate sigmoid e^x / (1 + e^x)
    vec_a = _dil_exp_kernel(vec_a);
    vec_addone_tmp = _mm512_add_ps(vec_a, vec_ps_1);
    vec_a = _mm512_div_ps(vec_a, vec_addone_tmp);

    vec_a = _mm512_mul_ps(vec_a, vec_add_tmp);

    // mask store
    _mask_storeu(a + i, vec_a, mask);
  }
}

template <typename scalar_t>
at::Tensor dil_add_swish(const at::Tensor& mm_output, const at::Tensor& bias) {
  scalar_t* mm_output_data_base = mm_output.data_ptr<scalar_t>();
  scalar_t* bias_data_base = bias.data_ptr<scalar_t>();

  auto infered_size = mm_output.sizes().vec();
  int64_t dim_size = infered_size[infered_size.size() - 1];
  int64_t outer_size = 1;
  // The last dim is the loop unit. We need to minus 2 to exclude the last dim.
  // infered_size.size() - 2 is the -2th dimension.
  for (int64_t i = infered_size.size() - 2; i >= 0; i--) {
    // Calculate outer loop number;
    outer_size *= infered_size[i];
  }

  int64_t grain_size = at::internal::GRAIN_SIZE / (16 * dim_size);
  if (grain_size < 1)
    grain_size = 1;

  at::parallel_for(0, outer_size, grain_size, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; i++) {
      _dil_add_swish_fusion_kernel<scalar_t>(
          mm_output_data_base + i * dim_size, bias_data_base, dim_size);
    }
  });

  return mm_output;
} // dil_add_swish

} // namespace vec512
} // namespace vec
} // namespace kernel
} // namespace cpu
} // namespace torch_ipex
