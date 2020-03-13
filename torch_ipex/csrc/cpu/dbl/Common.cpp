#include "Common.h"

#include <ATen/Tensor.h>
#include <c10/util/Exception.h>

#include "cpu/dil/dil.hpp"
#include "cpu/ShadeDataContext.h"
#include "torch_ipex/csrc/ipex_tensor_impl.h"

namespace torch_ipex {
namespace cpu {
namespace dbl {
namespace comm {

dil::tensor dil_tensor_from_dense(const at::Tensor& tensor) {
  AT_ASSERTM(
    tensor.device().type() == at::DeviceType::DPCPP,
    "dil_tensor_view_from_dense expects CPU tensor input");
  AT_ASSERTM(
    tensor.layout() == at::Layout::Strided,
    "dil_tensor_view_from_dense expects dense tensor input");
  AT_ASSERTM(
    !tensor.is_variable(),
    "dil_tensor_view_from_dense: should not be a variable");
  at::ScalarType cur_type = tensor.scalar_type();
  return {{{tensor.sizes().cbegin(), tensor.sizes().cend()}, get_dil_data_type(cur_type)},
          tensor.data_ptr()};
}

at::Tensor dil_tensor_to_dense(const dil::tensor& dil_tensor) {
  auto dims = dil_tensor.get_dims();
  // NOTE: int32_t dims from ideep::tensor but sizes needs int64_t
  at::Tensor cpu_tensor = at::empty(
    std::vector<int64_t>(dims.begin(), dims.end()),
    ipexTensor.options().device(c10::kCPU).layout(c10::kStrided));
  // make sure that it is not a in-place tensor
  TORCH_INTERNAL_ASSERT(ipexTensor.unsafeGetTensorImpl()->version_counter().current_version() == 1);
  dil_tensor.to_public(cpu_tensor.data_ptr(), dil_tensor.get_data_type());
  return cpu_tensor;
}

dil::tensor try_gen_dil_tensor(const at::Tensor &input) {
  if (cpu::ShadeDataContext::isDilTensor(input)) {
    return cpu::ShadeDataContext::getDilTensor(input);
  } else {
    TORCH_INTERNAL_ASSERT(input.is_contiguous());
    return dnnl::comm::dil_tensor_from_dense(input);
  }
}

at::Tensor gen_aten_tensor_by(const dil::tensor& dil_tensor) {
  // Generate new CPU Tensor and store dil tensor at its storage
  cpu::ShadeDataContext *shade_data_context = cpu::ShadeDataContext::allocShadeDataContext();
  shade_data_context->dil_tensor = dil_tensor;
  shade_data_context->data_type = cpu::SHADE_DATA_TYPE::DIL;
  c10::DataPtr shade_data_ptr(
    nullptr,
    shade_data_context,
    cpu::ShadeDataContext::freeShadeDataContext,
    at::DeviceType::DPCPP);
  auto dims = dil_tensor.get_dims();
  std::vector<int64_t> _tensor_sizes(dims.begin(), dims.end())
  auto at_data_type = get_at_data_type(dil_tensor.get_data_type());
  auto storage_impl = c10::make_intrusive<at::StorageImpl>(
    at::scalarTypeToTypeMeta(at_data_type),
    dil_tensor.get_nelems(),
    std::move(shade_data_ptr),
    nullptr,
    /*resizeable=*/false);
  auto _tensor = at::detail::make_tensor<torch_ipex::IPEXTensorImpl>(storage_impl, at::TensorTypeId::DPCPPTensorId);
  if (_tensor_sizes.size() != 1 || _tensor_sizes[0] != 0) {
    _tensor.unsafeGetTensorImpl()->set_sizes_contiguous(_tensor_sizes);
  }
  return _tensor;
}

}  // namespace comm
}  // namespace dbl
}  // namespace cpu
}  // namespace torch_ipex
