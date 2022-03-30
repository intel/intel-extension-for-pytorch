#pragma once

#include <c10/util/ArrayRef.h>
#include <vector>

namespace torch_ipex {
namespace cpu {

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

inline std::vector<int64_t> gen_dummy_input_size_for(
    at::IntArrayRef weight_sizes,
    int64_t groups) {
  // ported from csrc/cpu/ideep/ideep/opreators/conv.hpp::expected_weights_desc
  // Construct a dummy case, those shapes are from resnet50 model,
  // just make some simple tests to make sure it can get our expected
  // format(real case), may be changed in the future.
  auto input_dim =
      weight_sizes.size(); // weights_dims is 4 for conv2d and 5 for conv3d
  std::vector<int64_t> kernel_size;
  if (5 == input_dim) {
    kernel_size.push_back(weight_sizes[input_dim - 3]);
  }
  kernel_size.push_back(weight_sizes[input_dim - 2]);
  kernel_size.push_back(weight_sizes[input_dim - 1]);
  std::vector<int64_t> input_sizes;
  auto grouped = groups > 1;
  auto weights_dims_g = grouped
      ? ideep::utils::group_dims(weight_sizes.vec(), groups)
      : weight_sizes.vec();
  auto ic = groups * weights_dims_g[1 + grouped];
  input_sizes.push_back(32);
  input_sizes.push_back(ic);
  if (4 == input_dim) {
    input_sizes.push_back(14 * kernel_size[0]);
    input_sizes.push_back(14 * kernel_size[1]);
  } else {
    input_sizes.push_back(14 * kernel_size[0]);
    input_sizes.push_back(14 * kernel_size[1]);
    input_sizes.push_back(14 * kernel_size[2]);
  }
  return input_sizes;
}

} // namespace cpu
} // namespace torch_ipex
