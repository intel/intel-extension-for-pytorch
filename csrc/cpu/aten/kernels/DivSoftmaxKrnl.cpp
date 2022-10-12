#include <aten/DivSoftmax.h>

#include "vec/vec.h"

namespace torch_ipex {
namespace cpu {

namespace {

#if defined(CPU_CAPABILITY_AVX512)
using namespace torch_ipex::cpu::kernel;
/**
 * @brief This function is caculating the loop unit offset for current loop idx
 * element, and the loop is for reading a tensor with its last dim as the loop
 * unit
 * @param[in] outer_loop_idx the loop idx
 * @param[in] outer_loop_size the list of the size for each loop dim
 * @param[in] outer_loop_strides the adjusted strides (broadcastable reading
 * strides) of the reading tensor
 * @return the offset for the loop idx element
 * @endcode
 */
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

/**
 * @brief This function is adjusting the strides of src tensor based on the
 * target infered_size, and make sure the strides can be used for element
 * offset calculation of broadcastable reading.
 * For example:
 * [56, 1, 128 ,128] broadcasting reading for [56, 12, 128, 128]
 * where the 2nd dim of src size is 1 and we will repeatedly read the same
 * base (0 element offset) when broadcasting 2nd dim from target size (12)
 * input:
 * src tensor size [56, 1, 128 ,128]
 * src tensor strides [16384, 16384, 128 ,1]
 * infered_size (target tensor size) [56, 12, 128, 128]
 * output:
 * [16384, 0, 128, 1]
 * @param[in] src original tensor that needs to be adjusted
 * @param[in] infered_size the target size to be broadcasted
 * @return adjusted strides
 * @endcode
 */
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
 * @brief Fuse the div (div scalar or mul 1/scalar), masked_fill operator and
 * softmax operator. softmax(mask? a/dim_per_head : fill value)
 *
 * @attention
 * There are some assumptions for this operator.
 * - The reduce dimension for softmax is the last dimension
 * - The reduce dimension for softmax is the leading dimension
 * - The input tensors are contiguous
 * - The number of the input tensor dimension should be >=2
 * - The mask b has the same dimension as a, or it can be expand_as a with (bs
 * :: seq_length), i.e., 2D tensor expands from mid dims
 * - The datatype for inpust a and output are same.
 *
 * @param[in] a a contiguous tensor to do div and softmax
 * @param[in] b a mask tensor to be masked_fill into tensor a after div and
 * before softmax
 * @return The tensor stores the result of @code softmax(mask? a/dim_per_head :
 * fill value) @endcode
 */
template <typename scalar_t>
at::Tensor dil_div_maskfill_softmax(
    const at::Tensor& a, // qk scores
    const at::Tensor& b, // mask
    const float& fill_value,
    const float& dim_per_head) {
  scalar_t* a_data_base = a.data_ptr<scalar_t>();
  float* b_data_base = b.data_ptr<float>();

  auto infered_size = a.sizes().vec();

  auto need_broadcast = a.dim() == b.dim() && (infered_size != b.sizes());
  auto need_expand_from_2d = b.dim() == 2;
  if (need_broadcast) {
    infered_size = at::infer_size(a.sizes(), b.sizes());
  }

  // Calculate the strides for the input tensor
  std::vector<int64_t> b_adjusted_strides = _adjust_strides(b, infered_size);

  // Create an new tensor to store the output
  at::Tensor output = at::empty_like(a);
  scalar_t* output_data_base = output.data_ptr<scalar_t>();

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

  auto mask_offset = outer_size / infered_size[0];

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
      } else if (need_expand_from_2d) {
        // b_offset needs to consider the mask offset because the mask is
        // expand_as a with the mid dims (bs :: seq_length)
        b_offset = i / mask_offset * dim_size;
      } else {
        b_offset = i * dim_size;
      }

      // mask fill and do div on a and get the maximum value:
      //    output_data = mask? a/dim_per_head : fill value
      //    val = max(output_data)
      _dil_maskedfill_div_max_fusion_kernel<scalar_t>(
          a_data_base + i * dim_size,
          b_data_base + b_offset,
          fill_value,
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
} // dil_div_maskfill_softmax
#endif

at::Tensor div_maskedfill_softmax_kernel_impl(
    at::Tensor& a,
    const at::Tensor& b,
    const at::IntArrayRef& mask_shape,
    const float& fill,
    const float& dim_per_head) {
#if defined(CPU_CAPABILITY_AVX512)
  if (a.scalar_type() == at::kFloat) {
    return dil_div_maskfill_softmax<float>(a, b, fill, dim_per_head);
  } else if (a.scalar_type() == at::kBFloat16) {
    return dil_div_maskfill_softmax<at::BFloat16>(a, b, fill, dim_per_head);
  }
#endif
  // convert the mask back to bool for fallback path
  auto _b = b.toType(at::kBool);
  a = at::div(a, dim_per_head);
  // 2D mask need to expand
  if (b.dim() == 2 && !mask_shape.empty()) {
    auto expand_mask = _b.view(mask_shape).expand_as(a);
    auto a_fill = a.masked_fill_(expand_mask, fill);
    return at::softmax(a_fill, -1);
  } else {
    auto a_fill = a.masked_fill_(_b, fill);
    return at::softmax(a_fill, -1);
  }
}

} // anonymous namespace

REGISTER_DISPATCH(
    div_maskedfill_softmax_kernel_stub,
    &div_maskedfill_softmax_kernel_impl);

} // namespace cpu
} // namespace torch_ipex
