#pragma once

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/NativeFunctions.h>
#include <c10/core/ScalarType.h>

#include <vector>


namespace torch_ipex {
namespace cpu {
namespace aten {
namespace embedding_bag {

at::Tensor embedding_bag_impl(const at::Tensor & weight, const at::Tensor & indices,
  const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse,
  const at::Tensor & per_sample_weights, bool include_last_offset);

at::Tensor embedding_bag_backward_impl(const at::Tensor & grad, const at::Tensor & indices,
  const at::Tensor & offsets, const at::Tensor & offset2bag, const at::Tensor & bag_size, const at::Tensor & maximum_indices,
  int64_t num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse,
  const at::Tensor & per_sample_weights);

at::Tensor embedding_bag_get_offset2bag(const at::Tensor indices, const at::Tensor & offsets, const at::Tensor & offset2bag);

bool embedding_bag_backward_fast_path_sum(const at::Tensor grad, const at::Tensor indices, const at::Tensor offset2bag, const at::Tensor per_sample_weights, bool scale_grad_by_freq, int64_t mode);

bool embedding_bag_fast_path_sum(const at::Tensor weight, const at::Tensor per_sample_weights, int64_t mode);

}  // namespace embedding_bag
}  // namespace aten
}  // namespace cpu
}  // namespace torch_ipex
