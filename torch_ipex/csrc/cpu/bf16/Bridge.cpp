#include "torch_ipex/csrc/cpu/bf16/Bridge.hpp"

#include "torch_ipex/csrc/aten_ipex_bridge.h"
#include "torch_ipex/csrc/ipex_tensor_impl.h"
#include "torch_ipex/csrc/utils.h"
#include "torch_ipex/csrc/cpu/dbl/Common.h"
#include "torch_ipex/csrc/cpu/ShadeDataContext.h"

namespace torch_ipex {
namespace cpu {
namespace bf16 {

at::Tensor gen_consistent_tensor(const at::Tensor & self) {
  // Reorder dil buffer to public because aten tensor does not support blocked format
  dbl::comm::reorder_to_public(self, /*keep data type*/true);

  dil::tensor& self_dil_storage = ShadeDataContext::getDilStorage(self);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self_dil_storage.get_data_type() == dil::data_type::bf16)
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
  return _tensor;
}

at::Tensor gen_mix_prec_tensor(const at::Tensor & self) {
  auto _self_tensor_storage = self.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
  dil::tensor y = dbl::comm::dil_tensor_from_cpu_buffer(
    self,
    _self_tensor_storage->data_ptr().get_deleter());
  _self_tensor_storage->data_ptr().release_context();
  auto res = dbl::comm::gen_aten_tensor_by(std::move(y));
  IPEXTensorImpl* res_ipex_impl = (IPEXTensorImpl *)res.unsafeGetTensorImpl();
  res_ipex_impl->copy_meta_info(self.unsafeGetTensorImpl(), /*keep data type*/true);
  return res;
}

}  // namespace bf16
}  // namespace cpu
}  // namespace torch_ipex
