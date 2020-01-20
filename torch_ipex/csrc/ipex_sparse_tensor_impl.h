#pragma once
#include <ATen/ATen.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/SparseTensorUtils.h>

namespace torch_ipex {

class IPEXSparseTensorImpl : public at::SparseTensorImpl {
 public:
  explicit IPEXSparseTensorImpl(at::TensorTypeSet type_set, const caffe2::TypeMeta& data_type);
  void copy_meta_info(const at::SparseTensorImpl *);
  static IPEXSparseTensorImpl * get_ipex_sparse_impl(const at::Tensor &);
};

} // namespace torch_ipex
