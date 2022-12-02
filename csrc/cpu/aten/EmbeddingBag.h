#include <ATen/Tensor.h>
#include <dyndisp/DispatchStub.h>
#include <torch/all.h>

namespace torch_ipex {

at::Tensor embedding_bag(
    const at::Tensor& weight,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    bool sparse,
    bool include_last_offset);

} // namespace torch_ipex

namespace torch_ipex {
namespace cpu {

namespace {

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

} // namespace

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
