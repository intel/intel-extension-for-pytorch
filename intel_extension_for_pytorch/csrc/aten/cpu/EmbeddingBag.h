#include <ATen/Tensor.h>
#include <intel_extension_for_pytorch/csrc/dyndisp/DispatchStub.h>
#include <torch/extension.h>

const int MODE_SUM = 0;
const int MODE_MEAN = 1;
const int MODE_MAX = 2;

namespace torch_ipex {

bool embedding_bag_fast_path_sum(
    const at::Tensor weight,
    const c10::optional<at::Tensor> per_sample_weights,
    int64_t mode,
    const c10::optional<int64_t> padding_idx);

at::Tensor embedding_bag(
    const at::Tensor& weight,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    bool sparse,
    bool include_last_offset);

} // namespace torch_ipex

namespace torch_ipex {
namespace cpu {

#if defined(DYN_DISP_BUILD)
namespace {
#endif

at::Tensor embedding_bag_kernel_impl(
    const at::Tensor& weight,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    bool include_last_offset);

at::Tensor embedding_bag_backward_kernel_impl(
    const at::Tensor& grad,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    int64_t num_weights,
    bool sparse);

at::Tensor embedding_bag_int8_kernel_impl(
    const at::Tensor& qweight,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    bool include_last_offset);

#if defined(DYN_DISP_BUILD)
}
#endif

using embedding_bag_kernel_fn = at::Tensor (*)(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    bool);
DECLARE_DISPATCH(embedding_bag_kernel_fn, embedding_bag_kernel_stub);

using embedding_bag_backward_kernel_fn = at::Tensor (*)(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    int64_t,
    bool);
DECLARE_DISPATCH(
    embedding_bag_backward_kernel_fn,
    embedding_bag_backward_kernel_stub);

using embedding_bag_int8_kernel_fn = at::Tensor (*)(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    bool);
DECLARE_DISPATCH(embedding_bag_int8_kernel_fn, embedding_bag_int8_kernel_stub);

} // namespace cpu
} // namespace torch_ipex
