#include "ipex_sparse_tensor_impl.h"

namespace torch_ipex {

IPEXSparseTensorImpl::IPEXSparseTensorImpl(at::TensorTypeSet type_set, const caffe2::TypeMeta& data_type) :
    at::SparseTensorImpl(type_set, data_type) {
}

IPEXSparseTensorImpl * IPEXSparseTensorImpl::get_ipex_sparse_impl(const at::Tensor& ipex_tensor) {
  TORCH_INTERNAL_ASSERT(ipex_tensor.layout() == c10::kSparse);
  // TORCH_INTERNAL_ASSERT(ipex_tensor.device().type() == at::DeviceType::DPCPP);
  return static_cast<IPEXSparseTensorImpl*>(ipex_tensor.unsafeGetTensorImpl());
}

void IPEXSparseTensorImpl::copy_meta_info(const at::SparseTensorImpl *src_impl) {
  // TensorImpl fields, align with IPEXTensorImpl::copy_meta_info
  /*
  this->storage_ = src_impl->storage_;
  this->device_opt_ = src_impl->device_opt_;
  this->reserved_ = src_impl->reserved_;
  this->type_set_ = src_impl->type_set();
  // SparseTensorImpl do not have
  this->strides_ = src_impl->strides();
  this->is_contiguous_ = src_impl->is_contiguous();
  this->storage_offset_ = src_impl->storage_offset();
  this->is_channels_last_contiguous_ = src_impl->is_contiguous(at::MemoryFormat::ChannelsLast);
  */

  this->sizes_ = src_impl->sizes();
  this->data_type_ = src_impl->dtype();
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

  // SparseImpl specific fields
  /*
  this->indices_ = src_impl->indices();
  this->values_ = src_impl->values();
  */
  this->sparse_dim_ = src_impl->sparse_dim();
  this->dense_dim_ = src_impl->dense_dim();
  this->coalesced_ = src_impl->coalesced();
}

void IPEXSparseTensorImpl::copy_indices_and_values(const at::Tensor& indices, const at::Tensor& values) {
  this->indices_ = indices;
  this->values_ = values;
}

} // namespace torch_ipex
