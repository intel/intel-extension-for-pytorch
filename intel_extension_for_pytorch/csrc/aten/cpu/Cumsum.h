#include <ATen/Tensor.h>
#include <torch/extension.h>

namespace torch_ipex {

at::Tensor cumsum_impl(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype);

} // namespace torch_ipex
