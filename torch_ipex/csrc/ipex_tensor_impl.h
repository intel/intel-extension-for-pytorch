#pragma once

#include <ATen/Tensor.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>

#include <memory>

namespace torch_ipex {

class IPEXTensorImpl : public c10::TensorImpl {
 public:
  explicit IPEXTensorImpl(at::Tensor tensor, at::Storage storage, at::TensorTypeId type_id);
  explicit IPEXTensorImpl(at::Storage storage, at::TensorTypeId type_id);
  ~IPEXTensorImpl() {
    static_cast<void>(0);
  }

  void copy_meta_info(const c10::TensorImpl *);
  void set_dpcpp_tensor_id();

  static c10::Device GetCurrentAtenDevice();
  static c10::Device SetCurrentAtenDevice(c10::Device);
  static void CopyMetadata(c10::TensorImpl *, const c10::TensorImpl *);
  static void CopySizeStridesAndOffset(c10::TensorImpl *, const c10::TensorImpl *);

private:
  c10::optional<at::Tensor> m_data_tensor;
};

} // namespace torch_ipex
