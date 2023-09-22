#include <ATen/Context.h>

namespace at {
namespace AtenIpexTypeXPU {

at::Tensor& std_var_out(
    Tensor& result,
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<Scalar>& correction_opt,
    bool keepdim,
    bool take_sqrt);
std::tuple<Tensor&, Tensor&> std_var_mean_out(
    const char* fname,
    Tensor& result1,
    Tensor& result2,
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<Scalar>& correction_opt,
    bool keepdim,
    bool take_sqrt);

} // namespace AtenIpexTypeXPU
} // namespace at
