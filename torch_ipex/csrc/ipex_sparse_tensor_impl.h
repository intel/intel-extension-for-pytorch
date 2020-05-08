#pragma once
#include <ATen/ATen.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/SparseTensorUtils.h>

namespace torch_ipex {

class IPEXSparseTensorImpl : public at::SparseTensorImpl {
 public:
  explicit IPEXSparseTensorImpl(at::DispatchKeySet type_set, const caffe2::TypeMeta& data_type);
  void copy_meta_info(const at::SparseTensorImpl *);
  void copy_indices_and_values(const at::Tensor& indices, const at::Tensor& values);
  static IPEXSparseTensorImpl * get_ipex_sparse_impl(const at::Tensor &);
};

} // namespace torch_ipex
