
#pragma once

#include <ATen/ATen.h>
#include <ATen/Config.h>

#include "ideep.hpp"

namespace torch_ipex {
namespace cpu {

// Mapping ScalarType to ideep tensor data_type
ideep::tensor::data_type get_mkldnn_dtype(at::ScalarType type);

// Construct an `ideep::tensor` "view" from dense tensor, note the
// ideep::tensor will share the underlying buffer
ideep::tensor itensor_view_from_dense(const at::Tensor& tensor);

// Construct an `ideep::tensor` "view" from dense tensor using given desc, note
// the ideep::tensor will share the underlying buffer
ideep::tensor itensor_view_from_dense(
    const at::Tensor& tensor,
    const ideep::tensor::desc& desc);

// Construct aten MKL-DNN tensor given an ideep tensor
at::Tensor new_with_itensor_mkldnn(
    ideep::tensor&& it,
    c10::optional<at::ScalarType> dtype,
    c10::optional<c10::Device> device);

at::Tensor mkldnn_to_dense(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype = c10::nullopt);

ideep::tensor itensor_from_tensor(const at::Tensor& tensor);

at::Tensor empty_aten_tensor_from_desc(
    const ideep::tensor::desc& desc,
    const at::TensorOptions& options);

} // namespace cpu
} // namespace torch_ipex
