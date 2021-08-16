#pragma once

#include <ATen/ATen.h>
#include <ATen/Config.h>

#include "ideep/ideep.hpp"

namespace torch_ipex { namespace cpu {

// Mapping ScalarType to ideep tensor data_type
ideep::tensor::data_type get_mkldnn_dtype(at::ScalarType type);

// Construct aten MKL-DNN tensor given an ideep tensor
at::Tensor new_with_itensor_mkldnn(ideep::tensor&& it, c10::optional<at::ScalarType> dtype, c10::optional<c10::Device> device);

// Retrieve `ideep::tensor` from MKL-DNN tensor
ideep::tensor& itensor_from_mkldnn(const at::Tensor& mkldnn_tensor);

// Construct an `ideep::tensor` "view" from dense tensor, note the
// ideep::tensor will share the underlying buffer
ideep::tensor itensor_view_from_dense(const at::Tensor& tensor);

// Helper function for getting an ideep tensor out of an aten Tensor or MKL-DNN tensor.
ideep::tensor itensor_from_tensor(const at::Tensor& tensor);

at::Tensor empty_aten_tensor_from_desc(const ideep::tensor::desc& desc, const at::TensorOptions& options);

int mkldnn_set_verbose(int level);
}}
