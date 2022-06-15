#include <csrc/aten/cpu/AddSoftmax.h>
#include "csrc/cpu/vec/vec.h"

namespace torch_ipex {
namespace cpu {

namespace {

#if defined(CPU_CAPABILITY_AVX512)
using namespace torch_ipex::cpu::kernel;

inline int64_t _calc_element_offset(
    const int64_t& outer_loop_idx,
    const std::vector<int64_t>& outer_loop_size,
    const std::vector<int64_t>& outer_loop_strides) {
  int64_t __outer_loop_idx = outer_loop_idx;
  int64_t b_offset = 0;
  for (int j = 0; j < outer_loop_size.size(); j++) {
    auto idx = __outer_loop_idx / outer_loop_size[j];
    __outer_loop_idx -= idx * outer_loop_size[j];
    // The stride could be any number if the dim equals to 1
    b_offset += idx * outer_loop_strides[j];
  }
  return b_offset;
}

inline std::vector<int64_t> _adjust_strides(
    const at::Tensor& src,
    std::vector<int64_t>& infered_size) {
  // We does NOT support broadcasting last dim which mean last_dim = 1
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src.stride(src.ndimension() - 1) == 1);

  auto original_shape = src.sizes();
  auto original_stride = src.strides();
  auto offset = infered_size.size() - original_shape.size();

  std::vector<int64_t> adjusted_stride;
  if (offset > 0)
    adjusted_stride.resize(infered_size.size(), 0);
  else
    adjusted_stride.resize(infered_size.size());

  for (size_t i = 0; i < original_shape.size(); i++) {
    // see NOTE: [Computing output strides]
    if (original_shape[i] == 1 && infered_size[offset + i] != 1) {
      adjusted_stride[offset + i] = 0;
    } else {
      adjusted_stride[offset + i] = original_stride[i];
    }
  }

  return adjusted_stride;
}

/**
 * @brief Fuse the div (div scalar or mul 1/scalar) add operator and softmax
 * operator. softmax(alpah * a + b)
 *
 * @attention
 * There are some assumptions for this operator.
 * - The reduce dimension for softmax is the last dimension
 * - The reduce dimension for softmax is the leading dimension
 * - The elements number of the reduce dimension for softmax is n*16
 * - The input tensors are contiguous
 * - The number of the input tensor dimension should be >=2
 * - Only the second input tensor is brodcastable
 * - The datatype for inpusts(a,b) and output are same.
 *
 * @param[in] a a contiguous tensor to be added
 * @param[in] b a tensor to be added while it should be broadcastable
 * @return The tensor stores the result of @code softmax(a + b) @endcode
 */
template <typename scalar_t>
at::Tensor dil_div_add_softmax(
    const at::Tensor& a,
    const at::Tensor& b,
    const float& dim_per_head) {
  scalar_t* a_data_base = a.data_ptr<scalar_t>();
  scalar_t* b_data_base = b.data_ptr<scalar_t>();

  // Check if the tensor needs to be broadcasted
  auto infered_size = a.sizes().vec();
  auto need_broadcast = (infered_size != b.sizes());
  if (need_broadcast) {
    infered_size = at::infer_size(a.sizes(), b.sizes());
  }
  at::Tensor output = at::empty_like(a);
  // Create an new tensor to store the output
  scalar_t* output_data_base = output.data_ptr<scalar_t>();

  // Calculate the strides for the input tensor
  std::vector<int64_t> b_adjusted_strides = _adjust_strides(b, infered_size);

  std::vector<int64_t> outer_size_per_dim;
  int64_t dim_size = infered_size[infered_size.size() - 1];
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dim_size != 1);

  int64_t outer_size = 1;
  // The last dim is the loop unit. We need to minus 2 to exclude the last dim.
  // infered_size.size() - 2 is the -2th dimension.
  for (int64_t i = infered_size.size() - 2; i >= 0; i--) {
    // Record outer dimensions
    outer_size_per_dim.insert(outer_size_per_dim.begin(), outer_size);
    // Calculate outer loop number;
    outer_size *= infered_size[i];
  }

  int64_t grain_size = at::internal::GRAIN_SIZE / (16 * dim_size);
  if (grain_size < 1)
    grain_size = 1;

  int64_t outer_dims_num = outer_size_per_dim.size();
  at::parallel_for(0, outer_size, grain_size, [&](int64_t begin, int64_t end) {
    float val = 0.0;
    int64_t b_offset = 0;
    at::Tensor tmp_out = at::empty({dim_size});
    float* tmp_out_ptr = tmp_out.data_ptr<float>();
    for (int64_t i = begin; i < end; i++) {
      if (need_broadcast) {
        b_offset =
            _calc_element_offset(i, outer_size_per_dim, b_adjusted_strides);
      } else {
        b_offset = i * dim_size;
      }
      // Add a and b and get the maximum value:
      //    output_data = a + b
      //    val = max(output_data)
      _dil_div_add_reduce_max_fusion_kernel<scalar_t>(
          a_data_base + i * dim_size,
          b_data_base + b_offset,
          dim_per_head,
          dim_size,
          tmp_out_ptr,
          val);
      // Calculate the e^x and get the sum value:
      //    output_data = output_data - max(output_data)
      //    output_data = e^(output_data)
      //    val = sum(output_data)
      _dil_exp_reduce_sum_fusion_kernel(
          tmp_out_ptr, dim_size, tmp_out_ptr, val);
      // Calculat the normalization [e^x / sum(e^x)]:
      //    output_data = output_data / sum(output_data)
      _dil_normalization_kernel<scalar_t>(
          tmp_out_ptr, val, dim_size, output_data_base + i * dim_size);
    }
  });
  return output;
} // dil_add_softmax

/**
 * @brief Fuse the add operator and softmax
 * operator. softmax(a + b)
 *
 * @attention
 * There are some assumptions for this operator.
 * - The reduce dimension for softmax is the last dimension
 * - The reduce dimension for softmax is the leading dimension
 * - The elements number of the reduce dimension for softmax is n*16
 * - The input tensors are contiguous
 * - The number of the input tensor dimension should be >=2
 * - Only the second input tensor is broadcastable
 * - The datatype for inputs(a,b) are same.
 *
 * @param[in] a a contiguous tensor to be added
 * @param[in] b a tensor to be added while it should be broadcastable
 * @return The tensor stores the result of @code softmax(a + b) @endcode
 */
at::Tensor& dil_add_softmax_(at::Tensor& a, const at::Tensor& b) {
  float* a_data_base = a.data_ptr<float>();
  float* b_data_base = b.data_ptr<float>();

  // Check if the tensor needs to be broadcasted
  auto infered_size = a.sizes().vec();
  auto need_broadcast = (infered_size != b.sizes());
  if (need_broadcast) {
    infered_size = at::infer_size(a.sizes(), b.sizes());
  }

  // Calculate the strides for the input tensor
  std::vector<int64_t> b_adjusted_strides = _adjust_strides(b, infered_size);

  std::vector<int64_t> outer_size_per_dim;
  int64_t dim_size = infered_size[infered_size.size() - 1];
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dim_size != 1);

  int64_t outer_size = 1;
  // The last dim is the loop unit. We need to minus 2 to exclude the last dim.
  // infered_size.size() - 2 is the -2th dimension.
  for (int64_t i = infered_size.size() - 2; i >= 0; i--) {
    // Record outer dimensions
    outer_size_per_dim.insert(outer_size_per_dim.begin(), outer_size);
    // Calculate outer loop number;
    outer_size *= infered_size[i];
  }

  int64_t grain_size = at::internal::GRAIN_SIZE / (16 * dim_size);
  if (grain_size < 1)
    grain_size = 1;

  int64_t outer_dims_num = outer_size_per_dim.size();
  at::parallel_for(0, outer_size, grain_size, [&](int64_t begin, int64_t end) {
    float val = 0.0;
    int64_t b_offset = 0;
    for (int64_t i = begin; i < end; i++) {
      if (need_broadcast) {
        b_offset =
            _calc_element_offset(i, outer_size_per_dim, b_adjusted_strides);
      } else {
        b_offset = i * dim_size;
      }
      // Add a and b and get the maximum value:
      //    output_data = a + b
      //    val = max(output_data)
      _dil_add_reduce_max_fusion_kernel(
          a_data_base + i * dim_size,
          b_data_base + b_offset,
          dim_size,
          a_data_base + i * dim_size,
          val);
      // Calculate the e^x and get the sum value:
      //    output_data = output_data - max(output_data)
      //    output_data = e^(output_data)
      //    val = sum(output_data)

      _dil_exp_reduce_sum_fusion_kernel(
          a_data_base + i * dim_size,
          dim_size,
          a_data_base + i * dim_size,
          val);
      // Calculat the normalization [e^x / sum(e^x)]:
      //  output_data = output_data / sum(output_data)

      _dil_normalization_kernel<float>(
          a_data_base + i * dim_size,
          val,
          dim_size,
          a_data_base + i * dim_size);
    }
  });
  return a;
} // add_softmax_
#endif

at::Tensor div_add_softmax_kernel_impl(
    at::Tensor& a,
    const at::Tensor& b,
    const float& dim_per_head) {
#if defined(CPU_CAPABILITY_AVX512)
  if (a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat) {
    return dil_div_add_softmax<float>(a, b, dim_per_head);
  } else if (
      a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16) {
    return dil_div_add_softmax<at::BFloat16>(a, b, dim_per_head);
  }
#endif
  a = at::div(a, dim_per_head);
  return at::softmax(at::add(a, b, 1.0f), -1);
}

at::Tensor& add_softmax_inplace_kernel_impl(
    at::Tensor& a,
    const at::Tensor& b) {
#if defined(CPU_CAPABILITY_AVX512)
  if (a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat) {
    return dil_add_softmax_(a, b);
  }
#endif
  a.copy_(at::softmax(a.add_(b), -1));
  return a;
}

} // anonymous namespace

REGISTER_DISPATCH(div_add_softmax_kernel_stub, &div_add_softmax_kernel_impl);
REGISTER_DISPATCH(
    add_softmax_inplace_kernel_stub,
    &add_softmax_inplace_kernel_impl);

} // namespace cpu
} // namespace torch_ipex
