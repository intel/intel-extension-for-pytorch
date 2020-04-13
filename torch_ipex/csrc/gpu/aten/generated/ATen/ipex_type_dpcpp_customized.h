#ifndef IPEX_TYPE_DPCPP_CUSTOMIZED_H
#define IPEX_TYPE_DPCPP_CUSTOMIZED_H

#include <ATen/ATen.h>

namespace at {
namespace AtenIpexTypeDPCPP {

at::Tensor convolution_sum(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, at::Tensor& accumu, at::Scalar alpha=0.f, at::Scalar beta=0.f, at::Scalar scale=1.0);

at::Tensor convolution_sum_relu(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, at::Tensor& accumu, at::Scalar alpha=0.f, at::Scalar beta=0.f, at::Scalar scale=1.0);

at::Tensor convolution_relu(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, at::Scalar alpha=0.f, at::Scalar beta=0.f, at::Scalar scale=1.0);

at::Tensor & fill_slice_with_index(at::Tensor & t, int dim);

at::Tensor & std_var_out(at::Tensor & result, const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim, bool take_sqrt);

std::tuple<Tensor&,Tensor&> std_var_mean_out(const char* fname, Tensor &result1, Tensor &result2, const Tensor &self, IntArrayRef dim, bool unbiased, bool keepdim, bool take_sqrt);

}
}

#endif
