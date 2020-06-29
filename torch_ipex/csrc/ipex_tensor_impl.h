#pragma once

#include <ATen/Tensor.h>
#include <ATen/TensorUtils.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>

#include <memory>

namespace torch_ipex {

class IPEXTensorImpl : public c10::TensorImpl {
 public:
  explicit IPEXTensorImpl(at::Tensor tensor, at::Storage storage, at::DispatchKey type_id);
  explicit IPEXTensorImpl(at::Storage storage, at::DispatchKey type_id);
  explicit IPEXTensorImpl(at::DispatchKeySet type_set, const caffe2::TypeMeta& data_type, c10::optional<c10::Device> device_opt);
  ~IPEXTensorImpl() {
    static_cast<void>(0);
  }

  void copy_auto_grad(c10::TensorImpl *);
  void copy_meta_info(const c10::TensorImpl *);
  void keep_source_data_tensor(at::Tensor);
  void set_storage_data_ptr(c10::DataPtr);
  void set_dpcpp_tensor_id();
  void set_strided(at::IntArrayRef size, at::IntArrayRef stride, int64_t storage_offset);
  void reset_data_type(at::ScalarType dst_type);

  c10::Storage& get_storage() {
    return this->storage_;
  }

  static c10::Device GetCurrentAtenDevice();
  static c10::Device SetCurrentAtenDevice(c10::Device);
  static void CopyMetadata(c10::TensorImpl *, const c10::TensorImpl *);
  static void CopySizeStridesAndOffset(c10::TensorImpl *, const c10::TensorImpl *);

private:
  // The main purpose of this tensor is to prevent the release of the source tensor. Because IPEXTensorImpl may share
  // same buffer of the source tensor.
  c10::optional<at::Tensor> m_src_data_tensor;
};

} // namespace torch_ipex
