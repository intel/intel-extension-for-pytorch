#include "torch_ipex/csrc/ipex_tensor_impl.h"

#include <c10/core/ScalarType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>

#include "utils.h"

namespace torch_ipex {

namespace {

thread_local c10::Device g_current_device(at::DeviceType::DPCPP, 0);

struct IPEXGuardImpl : public c10::impl::DeviceGuardImplInterface {
  at::DeviceType type() const override { return at::DeviceType::DPCPP; }

  c10::Device exchangeDevice(c10::Device device) const override {
    std::swap(g_current_device, device);
    return device;
  }

  c10::Device getDevice() const override { return g_current_device; }

  void setDevice(c10::Device device) const override {
    g_current_device = device;
  }

  void uncheckedSetDevice(c10::Device device) const noexcept override {
    g_current_device = device;
  }

  c10::Stream getStream(c10::Device device) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, device);
  }

  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, g_current_device);
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    c10::DeviceIndex dev_count = -1;
    get_device_count(g_current_device, &dev_count);
    return dev_count;
  }
};

C10_REGISTER_GUARD_IMPL(DPCPP, IPEXGuardImpl);

}  // namespace


IPEXTensorImpl::IPEXTensorImpl(at::Tensor tensor, at::Storage storage, at::TensorTypeId type_id) :
    m_data_tensor(std::move(tensor)),
    c10::TensorImpl(std::move(storage), type_id) {}

IPEXTensorImpl::IPEXTensorImpl(at::Storage storage, at::TensorTypeId type_id) :
    c10::TensorImpl(std::move(storage), type_id) {}

void IPEXTensorImpl::set_dpcpp_tensor_id() {
  this->type_set_ = at::TensorTypeSet(at::TensorTypeId::DPCPPTensorId);
  this->type_set_.add(at::TensorTypeId::VariableTensorId);
}

void IPEXTensorImpl::copy_meta_info(const c10::TensorImpl *src_impl) {
  /*
  dest_impl->storage_ = src_impl->storage_;
  dest_impl->device_opt_ = src_impl->device_opt_;
  dest_impl->reserved_ = src_impl->reserved_;
  dest_impl->type_set_ = src_impl->type_set();
  */
  this->sizes_ = src_impl->sizes();
  this->strides_ = src_impl->strides();
  this->storage_offset_ = src_impl->storage_offset();
  this->data_type_ = src_impl->dtype();
  this->is_contiguous_ = src_impl->is_contiguous();
  this->is_channels_last_contiguous_ = src_impl->is_contiguous(at::MemoryFormat::ChannelsLast);
  this->is_channels_last_ = src_impl->is_strides_like_channels_last();
  this->is_non_overlapping_and_dense_ = src_impl->is_non_overlapping_and_dense();
  this->is_wrapped_number_ = src_impl->is_wrapped_number();
  this->set_version_counter(src_impl->version_counter().current_version());
  bool allow_tensor_metadata_change_ = src_impl->allow_tensor_metadata_change();
  this->set_allow_tensor_metadata_change(allow_tensor_metadata_change_);
  if (src_impl->named_tensor_meta() != nullptr) {
    this->set_named_tensor_meta(src_impl->named_tensor_meta()->clone());
  }
  this->refresh_numel();
}

c10::Device IPEXTensorImpl::GetCurrentAtenDevice() {
  return g_current_device;
}

c10::Device IPEXTensorImpl::SetCurrentAtenDevice(c10::Device device) {
  std::swap(g_current_device, device);
  return device;
}

void IPEXTensorImpl::CopySizeStridesAndOffset(c10::TensorImpl *dest_impl, const c10::TensorImpl *src_impl) {
  dest_impl->set_sizes_and_strides(src_impl->sizes(), src_impl->strides());
  dest_impl->set_storage_offset(src_impl->storage_offset());
}

void IPEXTensorImpl::CopyMetadata(c10::TensorImpl *dest_impl, const c10::TensorImpl *src_impl) {
  if (dest_impl->dim() == 0) {
    dest_impl->set_wrapped_number(src_impl->is_wrapped_number());
  }

  dest_impl->set_version_counter(src_impl->version_counter().current_version());

  bool allow_tensor_metadata_change_ = src_impl->allow_tensor_metadata_change();
  dest_impl->set_allow_tensor_metadata_change(allow_tensor_metadata_change_);

  if (src_impl->named_tensor_meta() != nullptr) {
    dest_impl->set_named_tensor_meta(src_impl->named_tensor_meta()->clone());
  }
}

}  // namespace torch_ipex
