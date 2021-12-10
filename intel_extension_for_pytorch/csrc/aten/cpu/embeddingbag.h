#include <ATen/Tensor.h>
#include <torch/extension.h>

namespace torch_ipex {

at::Tensor embedding_bag(
    const at::Tensor& weight,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    bool sparse,
    bool include_last_offset);

bool embedding_bag_fast_path_sum(
    const at::Tensor weight,
    const c10::optional<at::Tensor> per_sample_weights,
    int64_t mode,
    const c10::optional<int64_t> padding_idx);

} // namespace torch_ipex
