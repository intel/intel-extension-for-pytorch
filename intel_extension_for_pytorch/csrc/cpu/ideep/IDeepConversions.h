
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

// ##Background##
// This function returns the input tensor's stride with a workaround that checks
// (and fixes) the stride when the input tensor has dim size 1. Currently oneDNN
// is not expected the behavior that with dim size 1, a PyTorch tensor's stride
// is meanless and may not follow strict contiguous context, which may make
// oneDNN go into ref path (perf drop). For example: A tensor with shape [1,
// 768] and stride [1536, 1] is not expected to current oneDNN though PyTorch
// will think it is contiguous since dim0 is size 1. Such a Tensor can be
// constructed by slice [:,0,:] from another tensor with shape [1, 2, 768] and
// stride [1536, 768, 1], and it is a real case in Albert model pooler layer.
// ##Performance Impact##
// It takes ~0.05us on average for calling this function when creating a mkldnn
// tensor.
// ##TODO##
// Will remove this workaround after oneDNN's fix.
dnnl::memory::dims get_stride_with_size_1_fix(const at::Tensor& tensor);

} // namespace cpu
} // namespace torch_ipex
