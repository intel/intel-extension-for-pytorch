#include <ATen/Tensor.h>
#include <csrc/dyndisp/DispatchStub.h>
#include <torch/extension.h>

namespace torch_ipex {
namespace cpu {

#if defined(DYN_DISP_BUILD)
namespace {
#endif

at::Tensor cumsum_kernel_impl(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype);

#if defined(DYN_DISP_BUILD)
}
#endif

using cumsum_kernel_fn = at::Tensor (*)(
    at::Tensor&,
    const at::Tensor&,
    int64_t,
    c10::optional<at::ScalarType>);
DECLARE_DISPATCH(cumsum_kernel_fn, cumsum_kernel_stub);

} // namespace cpu
} // namespace torch_ipex
