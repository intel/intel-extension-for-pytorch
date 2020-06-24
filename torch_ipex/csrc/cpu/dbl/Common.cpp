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
  AT_ASSERTM(
    tensor.layout() == at::Layout::Strided,
    "dil_tensor_view_from_dense expects dense tensor input");
  at::ScalarType cur_type = tensor.scalar_type();
  return {tensor.sizes().vec(), get_dil_data_type(cur_type), tensor.strides().vec(), tensor.data_ptr()};
}

void reorder_to_bf16_for_mix_prec(const at::Tensor& tensor) {
  if (!check_auto_mix_bf16_fp32())
    return;

  auto tensor_dtype = tensor.scalar_type();
  TORCH_CHECK(!(tensor_dtype == at::kBFloat16), "Please disable auto mix-precision if you want to enable BFloat16 manually");
  if (tensor_dtype != at::kFloat)
    return;

  reorder_to_dtype(tensor, at::kBFloat16);
}

void reorder_to_dtype(const at::Tensor& tensor, at::ScalarType dst_scalar_type) {
  auto src = try_gen_dil_tensor(tensor);
  if (get_at_data_type(src.get_data_type()) == dst_scalar_type) {
    // The data type of DIL tensor is same as the dst data type. DO NOTHING
    return;
  }
  auto dst_desc = src.get_desc().to_type(get_dil_data_type(dst_scalar_type));
  reorder_to_desc(tensor, dst_desc);
}

void reorder_to_public(const at::Tensor& tensor) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(cpu::ShadeDataContext::isDilTensor(tensor));
  auto& dil_tensor = cpu::ShadeDataContext::getDilTensor(tensor);
  auto dst_desc = dil_tensor.get_desc();
  auto aten_tensor_scalar_type = tensor.scalar_type();
  auto *shade_data_context = (cpu::ShadeDataContext*)tensor.unsafeGetTensorImpl()->storage().data_ptr().get_context();
  if (!dil_tensor.is_public_format()) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(tensor.storage().unsafeGetStorageImpl()->data_ptr().get_deleter() == &(cpu::ShadeDataContext::freeShadeDataContext));
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(shade_data_context->cpu_del_fun == nullptr);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(aten_tensor_scalar_type == at::kFloat || aten_tensor_scalar_type == at::kBFloat16);
    dst_desc = dst_desc.to_default_format();
  } else if (cpu::ShadeDataContext::isTensorMixPrecision(tensor)) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(tensor.storage().unsafeGetStorageImpl()->data_ptr().get() == nullptr);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(shade_data_context->cpu_raw_data == nullptr);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(shade_data_context->cpu_del_fun == nullptr);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(aten_tensor_scalar_type == at::kFloat);
  } else {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(shade_data_context->cpu_raw_data == shade_data_context->dil_tensor->get_data_handle());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(shade_data_context->cpu_del_fun != nullptr);
    return;
  }
  dst_desc = dst_desc.to_type(get_dil_data_type(aten_tensor_scalar_type));
  reorder_to_desc(tensor, dst_desc);
}

void reorder_to_desc(const at::Tensor& tensor, const dil::tensor::desc& expected_desc) {
  auto src = try_gen_dil_tensor(tensor);
  dil::tensor dst {expected_desc};
  dst.feed_from(src);
  equip_dil_buffer(tensor,  dst);
}

void equip_dil_buffer(const at::Tensor& tensor, dil::tensor dil_tensor_buffer) {
  TORCH_CHECK(
      tensor.device().is_dpcpp(),
      "dil buffer can only be equipped to dpcpp tensor");

  TORCH_CHECK(
      check_tensor_own_whole_storage(tensor),
      "dil buffer can only be equipped to tensors that own the whole storage, "
      "as dil buffer is going to replace the original storage");

  // Build new shade data context
  cpu::ShadeDataContext *new_shade_data_context = cpu::ShadeDataContext::allocShadeDataContext();
  new_shade_data_context->data_type = cpu::SHADE_DATA_TYPE::DIL;
  new_shade_data_context->dil_tensor = dil_tensor_buffer;

  void *tensor_data = nullptr;
  if (dil_tensor_buffer.get_data_type() != get_dil_data_type(tensor.scalar_type())) {
    new_shade_data_context->mix_prec_type = cpu::MIX_PREC_TYPE::MIX_BF16_FP32;
  } else if (dil_tensor_buffer.is_public_format()) {
    tensor_data = dil_tensor_buffer.get_data_handle();
    new_shade_data_context->cpu_raw_data = tensor_data;
    new_shade_data_context->cpu_del_fun = &(c10::detail::deleteNothing);
  }

  // Create a new DataPtr instances because the DataPtr class does not support set
  // its data or context directly
  c10::DataPtr shade_data_ptr(
    tensor_data,
    new_shade_data_context,
    &(cpu::ShadeDataContext::freeShadeDataContext),
    tensor.device().type());

  IPEXTensorImpl* ipex_tensor_impl = (IPEXTensorImpl *)tensor.unsafeGetTensorImpl();
  ipex_tensor_impl->storage().set_data_ptr(std::move(shade_data_ptr));
 
  // After equip_dil_buffer(), whole storage should be managed by dil tensor,
  // and thus storage metadata should be overwritten by dil tensor 
  // Note: Storage::set_numel() might be removed later
  ipex_tensor_impl->storage().set_numel(dil_tensor_buffer.get_nelems());
  cpu::dbl::comm::sync_shape_from_dil_to_aten(tensor, dil_tensor_buffer);
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
    shade_data_context->mix_prec_type = cpu::MIX_PREC_TYPE::MIX_BF16_FP32;
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
