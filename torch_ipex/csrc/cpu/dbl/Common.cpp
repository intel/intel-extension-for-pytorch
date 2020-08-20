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

dil::tensor dil_tensor_from_cpu_buffer(const at::Tensor& tensor) {
  IPEX_CHECK(tensor.layout() == at::Layout::Strided,
      "dil_tensor_from_cpu_buffer expects dense tensor input");
  IPEX_CHECK(tensor.sizes().size() <= 6,
      "dil_tensor_from_cpu_buffer only support rank <= 6");
  auto cur_type = tensor.scalar_type();
  return {tensor.sizes().vec(), get_dil_data_type(cur_type), tensor.strides().vec(), tensor.data_ptr()};
}

dil::tensor dil_tensor_from_dil_buffer(const at::Tensor& tensor) {
  auto dil_buffer = cpu::ShadeDataContext::getDilStorage(tensor);
  // for low prcision, x.reshape() + linear, has some issue, need return dil_buffer directly
  //return dil_buffer;

  if (dil_buffer.is_public_format()) {
    auto size = tensor.sizes().vec();
    auto stride = tensor.strides().vec();
    auto data_type = dil_buffer.get_data_type();
    auto data_ptr = static_cast<void *>(
        static_cast<char *>(dil_buffer.get_data_handle()) +
        dil_buffer.get_item_size() * tensor.storage_offset());

    // return a new tensor wrapper that may be part of the dil storage
    auto groups = dil_buffer.get_groups(); // copy group info
    auto desc = dil::tensor::desc({size, data_type, stride}, groups);
    dil::tensor result {desc, data_ptr};

    // copy workspace
    if (dil_buffer.has_workspace()) {
      result.copy_workspace(dil_buffer);
    }

    if (dil_buffer.has_scale()) {
      result.set_scale(dil_buffer.get_scale());
    }
    // TODO(xpz): copy scales and zero_points of qtensor (what if slicing?)

    return result;
  } else {
    // When dil storage is blocked format or low precision data , tensor itself
    // should own the whole storage and should not to be sliced by pytorch at all,
    // because that does not make any sense for pytorch to slice a blocked dil tensor
    // So we directly return the dil buffer here.
    TORCH_CHECK(check_tensor_own_whole_storage(tensor),
        "Blocked tensor should own the whole storage. Should not reach here.");
    return dil_buffer;
  }
}

dil::tensor try_gen_dil_tensor(const at::Tensor &input) {
  if (cpu::ShadeDataContext::isDilTensor(input)) {
    return dil_tensor_from_dil_buffer(input);
  } else {
    return dil_tensor_from_cpu_buffer(input);
  }
}

dil::tensor try_gen_dil_storage(const at::Tensor &input) {
  if (cpu::ShadeDataContext::isDilTensor(input)) {
    return cpu::ShadeDataContext::getDilStorage(input);
  } else {
    return dil_tensor_from_cpu_buffer(input);
  }
}

void reorder_to_bf16_for_mix_prec(const at::Tensor& tensor) {
  if (!check_auto_mix_bf16_fp32())
    return;

  auto tensor_dtype = tensor.scalar_type();
  TORCH_CHECK(tensor_dtype != at::kBFloat16, "Please disable auto mix-precision if you want to enable BFloat16 manually");
  if (tensor_dtype != at::kFloat)
    return;

  reorder_to_dtype(tensor, at::kBFloat16);
}

std::tuple<std::vector<float>, bool> get_int8_scales(const at::Tensor& input, bool uint8_used) {
  if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
    auto src_dil_type = try_gen_dil_tensor(input).get_data_type();
    bool input_uint8_used = (src_dil_type == dil::data_type::u8);
    return get_indicator_scales({input_uint8_used, uint8_used});
  } else {
    return std::make_tuple(std::vector<float>(), false);
  }
}

void reorder_to_int8_for_mix_prec(const at::Tensor& tensor, std::vector<float> scales, bool uint8_used) {
  if (!check_auto_mix_int8_fp32() || check_int8_calibration())
    return;

  auto tensor_dtype = tensor.scalar_type();
  TORCH_CHECK(!(tensor_dtype == at::kQInt8 || tensor_dtype == at::kQUInt8), "Please disable auto mix-precision if you want to enable int8/uint8 manually");
  if (tensor_dtype != at::kFloat)
    return;
 
  auto src_type = try_gen_dil_storage(tensor).get_data_type();
  if (!uint8_used && (src_type == dil::data_type::u8 || src_type == dil::data_type::s8))
    return;

  auto dst_scalar_type = uint8_used ? at::kQUInt8 : at::kQInt8;

  auto inner_scales = scales;
  if (scales.empty()) {
    // compute weight scales for per_channel
    for (auto i = 0; i < tensor.size(0); i++) {
      inner_scales.push_back(float(127.5) / tensor[i].abs().max().item<float>());
    }
  }
 
  reorder_to_dtype(tensor, dst_scalar_type, inner_scales);
}

void reorder_to_dtype(const at::Tensor& tensor, at::ScalarType dst_scalar_type, std::vector<float> scales) {
  auto src = try_gen_dil_storage(tensor);
  if (get_at_data_type(src.get_data_type()) == dst_scalar_type) {
    // The data type of DIL tensor is same as the dst data type. DO NOTHING
    return;
  }
  auto dst_desc = src.get_desc().to_type(get_dil_data_type(dst_scalar_type));
 // src may bf16 or fp32 tensor with block format,
 // there has issue for conv weight prepack if given a block format weight
  if (!src.is_public_format()) {
    dst_desc = dst_desc.to_default_format();
  }

  reorder_to_desc(tensor, dst_desc, scales);
}

void equip_dil_buffer_nosync_shape(const at::Tensor& tensor, dil::tensor dil_buffer) {
  TORCH_CHECK(
      tensor.device().is_dpcpp(),
      "dil buffer can only be equipped to dpcpp tensor");

  // Build new shade data context
  cpu::ShadeDataContext *new_shade_data_context = cpu::ShadeDataContext::allocShadeDataContext();
  new_shade_data_context->data_type = cpu::SHADE_DATA_TYPE::DIL;

  //dil_buffer.get_desc().print_desc();
  // TORCH_CHECK(dil_buffer.is_dense(true), "dil storage must be dense");
  
  new_shade_data_context->dil_tensor = dil_buffer;

  void *tensor_data = nullptr;
  if (dil_buffer.get_data_type() != get_dil_data_type(tensor.scalar_type())) {
    if (dil_buffer.get_data_type() == dil::data_type::bf16) {
      new_shade_data_context->mix_prec_type = cpu::MIX_PREC_TYPE::MIX_BF16_FP32;
    } else {
      new_shade_data_context->mix_prec_type = cpu::MIX_PREC_TYPE::MIX_INT8_FP32;
    }
  } else if (dil_buffer.is_public_format()) {
    tensor_data = dil_buffer.get_data_handle();
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
  ipex_tensor_impl->storage().set_numel(dil_buffer.get_nelems());
}

void equip_dil_buffer(const at::Tensor& tensor, dil::tensor dil_buffer) {
  equip_dil_buffer_nosync_shape(tensor, dil_buffer);

  IPEXTensorImpl* ipex_tensor_impl = (IPEXTensorImpl *)tensor.unsafeGetTensorImpl();
  if (dil_buffer.is_public_format()) {
    ipex_tensor_impl->set_strided(dil_buffer.get_dims(), dil_buffer.get_strides(), ipex_tensor_impl->storage_offset());
  } else {
    // ??? TORCH_INTERNAL_ASSERT_DEBUG_ONLY(sizes.size() != 1 || sizes[0] != 0);
    // Blockformat does not inlcude stride information
    ipex_tensor_impl->set_sizes_contiguous(dil_buffer.get_dims());
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
  } else if (check_auto_mix_int8_fp32() && (dil_tensor_type == dil::data_type::s8 || dil_tensor_type == dil::data_type::u8)) {
    // If the user enables auto-mix-precision, then the aten tensor should always be float.
    // And even the dil tensor is plain format, it also cannot be shared with cpu buffer.
    shade_data_context->mix_prec_type = cpu::MIX_PREC_TYPE::MIX_INT8_FP32;
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
    _tensor_impl->set_strided(sizes, strides, _tensor_impl->storage_offset());
  } else {
    // Blockformat does not inlcude stride information
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(sizes.size() != 1 || sizes[0] != 0);
    ipex_tensor.unsafeGetTensorImpl()->set_sizes_contiguous(sizes);
  }
}

void reorder_to_public(const at::Tensor& tensor, bool remain_dtype) {
  if (!cpu::ShadeDataContext::isDilTensor(tensor)) {
    // non DIL tensor is a public tensor by nature
    return;
  }

  auto& dil_buffer = cpu::ShadeDataContext::getDilStorage(tensor);
  auto dst_desc = dil_buffer.get_desc();
  auto aten_dtype = tensor.scalar_type();
  bool is_public_format = dil_buffer.is_public_format();
  bool is_public_dtype = !cpu::ShadeDataContext::isTensorMixPrecision(tensor);

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      is_public_dtype && (aten_dtype == at::kFloat || aten_dtype == at::kBFloat16) || !is_public_dtype && aten_dtype == at::kFloat)

  bool should_reorder_format = !is_public_format;
  bool should_reorder_dtype = !is_public_dtype && !remain_dtype;

  if (!should_reorder_format && !should_reorder_dtype) {
    return;
  }

  if (should_reorder_format) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(tensor.storage().unsafeGetStorageImpl()->data_ptr().get_deleter() == &(cpu::ShadeDataContext::freeShadeDataContext));
    dst_desc = dst_desc.to_default_format();
  }

  if (should_reorder_dtype) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(tensor.storage().unsafeGetStorageImpl()->data_ptr().get() == nullptr);
    dst_desc = dst_desc.to_type(get_dil_data_type(aten_dtype));
  }

  reorder_to_desc(tensor, dst_desc);
}

// Reorder *Storage* to expected_desc
void reorder_to_desc(const at::Tensor& tensor, const dil::tensor::desc& expected_desc, const std::vector<float> scales) {
  auto& mutex = cpu::ShadeDataContext::getMutex(tensor);
  std::lock_guard<std::mutex> lock(mutex); 
  auto src = try_gen_dil_storage(tensor);
  if (src.get_desc() == expected_desc)
    return;
  dil::tensor dst {expected_desc};
  if(!scales.empty()) {
    dst.set_scale(scales);
  }
  dst.feed_from(src);

  // If a max pool output is converting from bf16 back to fp32,
  // its workspace has also to be copied onto the new tensor
  if (src.has_workspace()) {
    dst.copy_workspace(src);
  }
  equip_dil_buffer_nosync_shape(tensor, dst);
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
