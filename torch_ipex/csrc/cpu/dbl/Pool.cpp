#include "Pool.h"

#include "Common.h"

namespace torch_ipex {
namespace cpu {
namespace dbl {
namespace pool {

inline std::vector<int64_t> expand_param_if_needed(
    at::IntArrayRef list_param,
    const char* param_name,
    int64_t expected_dim) {
  if (list_param.size() == 1) {
    return std::vector<int64_t>(expected_dim, list_param[0]);
  } else if ((int64_t)list_param.size() != expected_dim) {
    std::ostringstream ss;
    ss << "expected " << param_name << " to be a single integer value or a "
       << "list of " << expected_dim << " values to match the convolution "
       << "dimensions, but got " << param_name << "=" << list_param;
    AT_ERROR(ss.str());
  } else {
    return list_param.vec();
  }
}

template<typename T>
static inline T div_rtn(T x, T y) {
    int q = x/y;
    int r = x%y;
    if ((r!=0) && ((r<0) != (y<0))) --q;
    return q;
}

template<typename T>
static inline T pooling_output_shape_pad_lr(
        T inputSize, T kernelSize, T pad_l, T pad_r, T stride, T dilation,
        bool ceil_mode) {
    T outputSize = div_rtn<T>(
        inputSize + pad_l + pad_r - dilation * (kernelSize - 1) - 1 +
        (ceil_mode ? stride - 1 : 0), stride) + 1;
    if (pad_l) {
        // ensure that the last pooling starts inside the image
        // needed to avoid problems in ceil mode
        if ((outputSize - 1) * stride >= inputSize + pad_l)
          --outputSize;
    }
    return outputSize;
}

std::vector<int64_t> pool_output_sizes(
    at::IntArrayRef input_size,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding_l,
    at::IntArrayRef padding_r,
    at::IntArrayRef dilation,
    bool ceil_mode) {
  std::vector<int64_t> output_size(input_size.size());
  // copy N and C
  output_size[0] = input_size[0];
  output_size[1] = input_size[1];

  for (size_t i = 2; i < input_size.size(); ++i) {
    output_size[i] = pooling_output_shape_pad_lr<int64_t>(
      input_size[i],
      kernel_size[i - 2],
      padding_l[i - 2],
      padding_r[i - 2],
      stride[i - 2],
      dilation[i - 2],
      ceil_mode
    );
  }

   return output_size;
}

at::Tensor _dil_pooling(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    dil::algorithm algo) {
  const int64_t dims = input.dim() - 2;
  auto kernel_size_vec = expand_param_if_needed(kernel_size, "kernel_size", dims);
  auto stride_vec = expand_param_if_needed(stride, "stride", dims);
  auto padding_vec = expand_param_if_needed(padding, "padding", dims);
  auto padding_vec_l = padding_vec;
  auto padding_vec_r = padding_vec;
  auto dilation_vec = expand_param_if_needed(dilation, "dilation", dims);

  const dil::tensor& x = dbl::comm::try_gen_dil_tensor(input);
  std::vector<int64_t> output_sizes;

  if (ceil_mode) {
    // MKLDNN does not support ceil mode, so we adjust padding
    // on the right side to match behavior. Adjust output size
    // accordingly.
    const std::vector<int64_t> output_sizes_ceil = pool_output_sizes(
        input.sizes(),
        kernel_size_vec,
        stride_vec,
        padding_vec_l,
        padding_vec_r,
        dilation_vec,
        true /* ceil_mode */);

    // adjust padding until output sizes agree
    bool all_equal = false;
    while (!all_equal) {
      output_sizes = pool_output_sizes(
          input.sizes(),
          kernel_size_vec,
          stride_vec,
          padding_vec_l,
          padding_vec_r,
          dilation_vec,
          false /*ceil_mode */);

      all_equal = true;
      for (size_t i = 2; i < input.sizes().size(); ++i) {
        if (output_sizes[i] < output_sizes_ceil[i]) {
           padding_vec_r[i - 2]++;
           all_equal = false;
        }
      }
    }
  } else {
    output_sizes = pool_output_sizes(
        input.sizes(),
        kernel_size_vec,
        stride_vec,
        padding_vec_l,
        padding_vec_r,
        dilation_vec,
        false /*ceil_mode */);
  }

  dil::tensor y;
  dil::pooling_forward::compute(
      x,
      {output_sizes.cbegin(), output_sizes.cend()},
      y,
      {stride_vec.cbegin(), stride_vec.cend()},
      {kernel_size_vec.cbegin(), kernel_size_vec.cend()},
      {padding_vec_l.cbegin(), padding_vec_l.cend()},
      {padding_vec_r.cbegin(), padding_vec_r.cend()},
      algo,
      dil::prop_kind::forward);

  return dbl::comm::gen_aten_tensor_by(std::move(y));
}

at::Tensor _dil_pooling_backward(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    dil::algorithm algo) {
  const int64_t dims = input.dim() - 2;
  auto kernel_size_vec = expand_param_if_needed(kernel_size, "kernel_size", dims);
  auto stride_vec = expand_param_if_needed(stride, "stride", dims);
  auto padding_vec = expand_param_if_needed(padding, "padding", dims);
  auto padding_vec_l = padding_vec;
  auto padding_vec_r = padding_vec;
  auto dilation_vec = expand_param_if_needed(dilation, "dilation", dims);

  if (ceil_mode) {
    // MKLDNN does not support ceil mode, so we adjust padding
    // on the right side to match behavior. Adjust output size
    // accordingly.
    const std::vector<int64_t> output_sizes_ceil = pool_output_sizes(
        input.sizes(),
        kernel_size_vec,
        stride_vec,
        padding_vec_l,
        padding_vec_r,
        dilation_vec,
        true /* ceil_mode */);

    // adjust padding until output sizes agree
    bool all_equal = false;
    std::vector<int64_t> output_sizes;
    while (!all_equal) {
      output_sizes = pool_output_sizes(
          input.sizes(),
          kernel_size_vec,
          stride_vec,
          padding_vec_l,
          padding_vec_r,
          dilation_vec,
          false /*ceil_mode */);

      all_equal = true;
      for (size_t i = 2; i < input.sizes().size(); ++i) {
        if (output_sizes[i] < output_sizes_ceil[i]) {
           padding_vec_r[i - 2]++;
           all_equal = false;
        }
      }
    }
  }

  const dil::tensor& grady = dbl::comm::try_gen_dil_tensor(grad_output);
  const dil::tensor& y = dbl::comm::try_gen_dil_tensor(output);
  const dil::tensor& x = dbl::comm::try_gen_dil_tensor(input);
  dil::tensor gradx;
  dil::pooling_backward::compute(
      grady,
      y,
      x,
      gradx,
      {stride_vec.cbegin(), stride_vec.cend()},
      {kernel_size_vec.cbegin(), kernel_size_vec.cend()},
      {padding_vec_l.cbegin(), padding_vec_l.cend()},
      {padding_vec_r.cbegin(), padding_vec_r.cend()},
      algo);

  return dbl::comm::gen_aten_tensor_by(std::move(gradx));
}

}  // namespace pool
}  // namespace dbl
}  // namespace cpu
}  // namespace torch_ipex
