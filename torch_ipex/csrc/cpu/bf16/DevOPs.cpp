#include "torch_ipex/csrc/cpu/bf16/DevOPs.hpp"

#include "torch_ipex/csrc/aten_ipex_bridge.h"
#include "torch_ipex/csrc/ipex_tensor_impl.h"
#include "torch_ipex/csrc/utils.h"
#include "torch_ipex/csrc/cpu/dbl/Common.h"
#include "torch_ipex/csrc/cpu/ShadeDataContext.h"

namespace torch_ipex {
namespace cpu {
namespace bf16 {

at::Tensor index_select(const at::Tensor & self, int64_t dim, const at::Tensor & index) {
  dbl::comm::reorder_to_public(self, true/*keep data type*/);

  dil::tensor& self_dil_storage = ShadeDataContext::getDilStorage(self);
  c10::DataPtr shade_data_ptr(
    self_dil_storage.get_data_handle(),
    nullptr,
    &(c10::detail::deleteNothing),
    at::DeviceType::CPU);
  auto storage_impl = c10::make_intrusive<at::StorageImpl>(
    at::scalarTypeToTypeMeta(at::kBFloat16),
    self_dil_storage.get_nelems(),
    std::move(shade_data_ptr),
    nullptr,
    /*resizeable=*/false);
  auto _tensor = at::detail::make_tensor<torch_ipex::IPEXTensorImpl>(storage_impl, at::DispatchKey::CPUTensorId);
  IPEXTensorImpl* cur_ipex_impl = (IPEXTensorImpl *)_tensor.unsafeGetTensorImpl();
  cur_ipex_impl->copy_meta_info(self.unsafeGetTensorImpl(), true);
  auto&& _ipex_index = bridge::shallowFallbackToCPUTensor(index);
  auto&& _ipex_result = at::index_select(_tensor, dim, _ipex_index);

  auto *_ipex_result_tensor_impl = _ipex_result.unsafeGetTensorImpl();
  auto _ipex_result_tensor_storage = _ipex_result_tensor_impl->storage().unsafeGetStorageImpl();
  _ipex_result_tensor_storage->data_ptr().release_context();
  dil::tensor y = dbl::comm::dil_tensor_from_cpu_buffer(
    _ipex_result,
    _ipex_result_tensor_storage->data_ptr().get_deleter());
  // Generate new aten tensor
  auto res = dbl::comm::gen_aten_tensor_by(std::move(y));
  IPEXTensorImpl* res_ipex_impl = (IPEXTensorImpl *)res.unsafeGetTensorImpl();
  res_ipex_impl->copy_meta_info(_ipex_result.unsafeGetTensorImpl(), true);
  return res;
}

}  // namespace bf16
}  // namespace cpu
}  // namespace torch_ipex
