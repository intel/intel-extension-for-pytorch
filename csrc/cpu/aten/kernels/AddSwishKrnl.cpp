#include <aten/AddSwish.h>

#include "vec/vec.h"

namespace torch_ipex {
namespace cpu {

namespace {

#if defined(CPU_CAPABILITY_AVX512)

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
      kernel::_dil_add_swish_fusion_kernel<scalar_t>(
          mm_output_data_base + i * dim_size, bias_data_base, dim_size);
    }
  });

  return mm_output;
} // dil_add_swish
#endif

at::Tensor add_swish_kernel_impl(
    at::Tensor& x,
    at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c) {
#if defined(CPU_CAPABILITY_AVX512)
  if (a.scalar_type() == at::kFloat && c.scalar_type() == at::kFloat) {
    return dil_add_swish<float>(a, c);
  } else if (
      a.scalar_type() == at::kBFloat16 && c.scalar_type() == at::kBFloat16) {
    return dil_add_swish<at::BFloat16>(a, c);
  }
#endif
  auto lin_res = at::linear(x, b, c);
  auto sigmoid_res = at::sigmoid(lin_res);
  return at::mul(lin_res, sigmoid_res);
}

} // anonymous namespace

REGISTER_DISPATCH(add_swish_kernel_stub, &add_swish_kernel_impl);

} // namespace cpu
} // namespace torch_ipex
