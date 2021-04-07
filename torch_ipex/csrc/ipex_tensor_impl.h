#pragma once

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>

#include <memory>

namespace torch_ipex {

class IPEXTensorImpl : public c10::TensorImpl {
 public:
  explicit IPEXTensorImpl(at::Storage storage, at::DispatchKey type_id, at::ScalarType dtype);
  explicit IPEXTensorImpl(at::DispatchKeySet type_set, const caffe2::TypeMeta& data_type, c10::optional<c10::Device> device_opt);
  ~IPEXTensorImpl() {
    static_cast<void>(0);
  }

  void copy_auto_grad(c10::TensorImpl *);
  void copy_meta_info(const c10::TensorImpl *, bool keep_dtype = false);
  void set_storage_data_ptr(c10::DataPtr);
  void set_strided(at::IntArrayRef size, at::IntArrayRef stride, int64_t storage_offset, at::ScalarType dtype, int64_t padding_size = 0);
  void reset_data_type(at::ScalarType dst_type);

  c10::Storage& get_storage() {
    return this->storage_;
  }

  static c10::Device GetCurrentAtenDevice();
  static c10::Device SetCurrentAtenDevice(c10::Device);
  static void CopyMetadata(c10::TensorImpl *, const c10::TensorImpl *);
  static void CopySizeStridesAndOffset(c10::TensorImpl *, const c10::TensorImpl *);
};

} // namespace torch_ipex
