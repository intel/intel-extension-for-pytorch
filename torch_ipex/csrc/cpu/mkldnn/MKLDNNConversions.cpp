#include "MKLDNNConversions.h"
#include "MKLDNNCommon.h"
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

namespace torch_ipex { namespace cpu {

at::Tensor mkldnn_to_dense(const at::Tensor& mkldnn_tensor, c10::optional<at::ScalarType> dtype) {
  TORCH_CHECK(mkldnn_tensor.scalar_type() == at::ScalarType::Float ||
              mkldnn_tensor.scalar_type() == at::ScalarType::BFloat16,
              "mkldnn_to_dense expects float or bfloat16 tensor input");
  ideep::tensor& stensor = itensor_from_mkldnn(mkldnn_tensor);
  auto dims = stensor.get_dims();
  auto data_type = dtype.has_value() ? dtype.value() : mkldnn_tensor.scalar_type();
  TORCH_CHECK(data_type == at::ScalarType::Float || data_type == at::ScalarType::BFloat16,
              "mkldnn tensor only can be converted to be a float or bfloat16 cpu tensor")
  // NOTE: int32_t dims from ideep::tensor but sizes needs int64_t
  at::Tensor cpu_tensor = at::empty(
    std::vector<int64_t>(dims.begin(), dims.end()),
    mkldnn_tensor.options().layout(c10::kStrided).dtype(data_type));
  if (stensor.is_empty()) return cpu_tensor;
  auto pub_tensor =
      data_type == at::ScalarType::Float
      ? stensor.to_public(cpu_tensor.template data_ptr<float>(),
                          ideep::tensor::data_type::f32)
      : stensor.to_public(cpu_tensor.template data_ptr<c10::BFloat16>(),
                         ideep::tensor::data_type::bf16);
  cpu_tensor.as_strided_(dims, pub_tensor.get_strides());
  return cpu_tensor;
}

}}
