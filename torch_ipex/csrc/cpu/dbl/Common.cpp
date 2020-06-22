#include "Common.h"

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <c10/util/Exception.h>

#include "cpu/dil/dil_pin_singletons.hpp"
#include "cpu/ShadeDataContext.h"
#include "torch_ipex/csrc/aten_ipex_bridge.h"
#include "torch_ipex/csrc/ipex_tensor_impl.h"
#include "torch_ipex/csrc/utils.h"
#include "torch_ipex/csrc/auto_opt_config.h"

namespace torch_ipex {
namespace cpu {
namespace dbl {
namespace comm {

dil::tensor dil_tensor_from_dense(const at::Tensor& tensor) {
  TORCH_CHECK(
    tensor.layout() == at::Layout::Strided,
    "dil_tensor_from_dense expects dense tensor input");
  TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Float
             "dil_tensor_from_dense expects  tensor input");
  TORCH_CHECK(tensor.dim() <= 5,
             "Can't convert cpu tensor with the number of dimensions > 5");
  at::ScalarType cur_type = tensor.scalar_type();
  return {tensor.sizes().vec(), get_dil_data_type(cur_type), tensor.strides().vec(), tensor.data_ptr()};
}

at::Tensor dil_tensor_to_dense(const at::Tensor& tensor) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(cpu::ShadeDataContext::isDilTensor(tensor));
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(tensor.unsafeGetTensorImpl()->version_counter().current_version() == 1);
  auto dil_tensor = cpu::ShadeDataContext::getDilTensor(tensor);
  at::Tensor cpu_tensor = at::empty(
    dil_tensor.get_dims(),
    tensor.options().device(c10::kCPU).layout(c10::kStrided));
  dil_tensor.to_public(cpu_tensor.data_ptr(), dil_tensor.get_data_type());
  return cpu_tensor;
}

void reorder_to_bf16_for_mix_prec(const at::Tensor& tensor) {
  if (check_auto_mix_bf16_fp32())
    bridge::reorderTensorToScalarTypeForDNNL(tensor, at::kBFloat16);
}

dil::tensor try_gen_dil_tensor(const at::Tensor &input) {
  if (cpu::ShadeDataContext::isDilTensor(input)) {
    auto dil_tensor = cpu::ShadeDataContext::getDilTensor(input);
    if ((!check_aten_dil_shape_info(input, dil_tensor)) && dil_tensor.is_public_format()) {
      dil_tensor.set_dims_and_strides(input.sizes().vec(), input.strides().vec());
    }
    // Does not support the case if the dil tensor is block format but it is just a part of tensor buffer
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dil_tensor.is_public_format() || check_tensor_own_whole_storage(input));
    return dil_tensor;
  } else {
    return dil_tensor_from_dense(input);
  }
}

at::Tensor gen_aten_tensor_by(dil::tensor&& dil_tensor) {
  // Generate new CPU Tensor and store dil tensor at its storage
  cpu::ShadeDataContext *shade_data_context = cpu::ShadeDataContext::allocShadeDataContext();
  auto dil_tensor_type = dil_tensor.get_data_type();
  shade_data_context->dil_tensor = std::forward<dil::tensor>(dil_tensor);
  shade_data_context->data_type = cpu::SHADE_DATA_TYPE::DIL;

  void *tensor_data = nullptr;
  auto at_data_type = get_at_data_type(dil_tensor_type);
  if (check_auto_mix_bf16_fp32() && dil_tensor_type == dil::data_type::bf16) {
    // If the user enables auto-mix-precision, then the aten tensor should always be float.
    // And even the dil tensor is plain format, it also cannot be shared with cpu buffer.
    shade_data_context->mix_prec_type = cpu::MIX_PREC_TYPE::MIX_BF_FP32;
    at_data_type = at::kFloat;
  } else {
    if (shade_data_context->dil_tensor->is_public_format()) {
      // The buffer of a tensor with public format is shared between CPU and DNNL
      tensor_data = shade_data_context->dil_tensor->get_data_handle();
      shade_data_context->cpu_raw_data = shade_data_context->dil_tensor->get_data_handle();
      shade_data_context->cpu_del_fun = &(c10::detail::deleteNothing);
    }
  }

  c10::DataPtr shade_data_ptr(
    tensor_data,
    shade_data_context,
    cpu::ShadeDataContext::freeShadeDataContext,
    at::DeviceType::DPCPP);
  auto storage_impl = c10::make_intrusive<at::StorageImpl>(
    at::scalarTypeToTypeMeta(at_data_type),
    shade_data_context->dil_tensor->get_nelems(),
    std::move(shade_data_ptr),
    nullptr,
    /*resizeable=*/false);
  auto _tensor = at::detail::make_tensor<torch_ipex::IPEXTensorImpl>(storage_impl, at::DispatchKey::DPCPPTensorId);
  sync_shape_from_dil_to_aten(_tensor, shade_data_context->dil_tensor.value());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(_tensor.layout() == c10::kStrided);
  return _tensor;
}

at::Tensor empty_dil_tensor(at::IntArrayRef sizes, const at::TensorOptions& options) {
  /*TORCH_CHECK(
     !optional_memory_format.has_value(),
     "'memory_format' argument is incompatible with mkldnn tensor");*/
  auto data_type = get_dil_data_type(at::typeMetaToScalarType(options.dtype()));
  dil::tensor it {sizes.vec(), data_type};
  return gen_aten_tensor_by(std::move(it));
}

void sync_shape_from_dil_to_aten(const at::Tensor& ipex_tensor, const dil::tensor &dil_tensor) {
  dil::dims sizes = dil_tensor.get_dims();
  if (dil_tensor.is_public_format()) {
    dil::dims strides = dil_tensor.get_strides();
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ipex_tensor.device().type() == at::DeviceType::DPCPP);
    auto* _tensor_impl = (IPEXTensorImpl *)ipex_tensor.unsafeGetTensorImpl();
    _tensor_impl->force_set_strided(sizes, strides);
  } else {
    // Blockformat does not inlcude stride information
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(sizes.size() != 1 || sizes[0] != 0);
    ipex_tensor.unsafeGetTensorImpl()->set_sizes_contiguous(sizes);
  }
}

std::vector<int64_t> expand_param_if_needed(
    at::IntArrayRef list_param,
    const char* param_name,
    int64_t expected_dim) {
  if (list_param.size() == 1) {
    return std::vector<int64_t>(expected_dim, list_param[0]);
  } else if ((int64_t)list_param.size() != expected_dim) {
    std::ostringstream ss;
    ss << "expected " << param_name << " to be a single integer value or a "
       << "list of " << expected_dim << " values to match the convolution "
       << "dimensions, but got " << param_name << "=" << list_param;
    AT_ERROR(ss.str());
  } else {
    return list_param.vec();
  }
}

}  // namespace comm
}  // namespace dbl
}  // namespace cpu
}  // namespace torch_ipex
