#ifndef IPEX_TYPE_DPCPP_CUSTOMIZED_H
#define IPEX_TYPE_DPCPP_CUSTOMIZED_H

#include <ATen/ATen.h>

namespace at {
namespace AtenIpexTypeDPCPP {

at::Tensor & fill_slice_with_index(at::Tensor & t, int dim);
at::Tensor & std_var_out(at::Tensor & result, const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim, bool take_sqrt);
std::tuple<Tensor&,Tensor&> std_var_mean_out(const char* fname, Tensor &result1, Tensor &result2, const Tensor &self, IntArrayRef dim, bool unbiased, bool keepdim, bool take_sqrt);

}
}

#endif
