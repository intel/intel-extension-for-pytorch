#include <core/Memory.h>
#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>
#ifdef USE_OVERRIDE_OP
#include "utils/CustomOperatorRegistration.h"
#endif
#include <utils/DPCPP.h>
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include "Sort.h"

using namespace torch_ipex::xpu::dpcpp::detail;
using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

std::tuple<Tensor&, Tensor&> sort_out(
    const Tensor& input,
    int64_t dim,
    bool order,
    Tensor& sorted,
    Tensor& indices) {
  return sort_out_stable(input, false, dim, order, sorted, indices);
}

std::tuple<at::Tensor, at::Tensor> sort(
    const at::Tensor& self,
    int64_t dim,
    bool descending) {
  Tensor sorted, indices;
  return sort_out_stable(self, false, dim, descending, sorted, indices);
}

std::tuple<Tensor, Tensor> sort(
    const Tensor& self,
    c10::optional<bool> stable,
    int64_t dim,
    bool descending) {
  Tensor sorted, indices;
  return sort_out_stable(self, stable, dim, descending, sorted, indices);
}

std::tuple<Tensor&, Tensor&> sort_out(
    const Tensor& self,
    c10::optional<bool> stable,
    int64_t dim,
    bool descending,
    Tensor& values,
    Tensor& indices) {
  return sort_out_stable(self, stable, dim, descending, values, indices);
}

Tensor argsort(const Tensor& self, bool stable, int64_t dim, bool descending) {
  Tensor sorted, indices;
  return std::get<1>(
      sort_out_stable(self, stable, dim, descending, sorted, indices));
}
#ifdef USE_OVERRIDE_OP
std::tuple<at::Tensor, at::Tensor> sort_ipex(
    const at::Tensor& self,
    int64_t dim,
    bool descending) {
  return at::AtenIpexTypeXPU::sort(self, dim, descending);
}

std::tuple<Tensor&, Tensor&> sort_values(
    const Tensor& input,
    int64_t dim,
    bool order,
    Tensor& sorted,
    Tensor& indices) {
  return at::AtenIpexTypeXPU::sort_out(input, dim, order, sorted, indices);
}

std::tuple<Tensor, Tensor> sort_stable(
    const Tensor& self,
    c10::optional<bool> stable,
    int64_t dim,
    bool descending) {
  return at::AtenIpexTypeXPU::sort(self, stable, dim, descending);
}

std::tuple<Tensor&, Tensor&> sort_values_stable(
    const Tensor& self,
    c10::optional<bool> stable,
    int64_t dim,
    bool descending,
    Tensor& values,
    Tensor& indices) {
  return at::AtenIpexTypeXPU::sort_out(
      self, stable, dim, descending, values, indices);
}
#endif
} // namespace AtenIpexTypeXPU
} // namespace at

#ifdef USE_OVERRIDE_OP
namespace {

IPEX_TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl("sort", TORCH_FN((&at::AtenIpexTypeXPU::sort_ipex)));
  m.impl("sort.stable", TORCH_FN((&at::AtenIpexTypeXPU::sort_stable)));
  m.impl("sort.values", TORCH_FN((&at::AtenIpexTypeXPU::sort_values)));
  m.impl(
      "sort.values_stable",
      TORCH_FN((&at::AtenIpexTypeXPU::sort_values_stable)));
}

} // namespace
#endif
