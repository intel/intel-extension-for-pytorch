#include "Common.h"

#include <ATen/ATen.h>
#include <ATen/ATen.h>
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

dil::tensor dil_tensor_from_cpu_buffer(const at::Tensor& tensor, const dil::tensor::desc& desc) {
  IPEX_CHECK(tensor.layout() == at::Layout::Strided,
      "dil_tensor_from_cpu_buffer expects dense tensor input");
  IPEX_CHECK(tensor.sizes().size() <= 6,
      "dil_tensor_from_cpu_buffer only support rank <= 6");
  return {desc, tensor.data_ptr()};
}

dil::tensor dil_tensor_from_cpu_buffer(const at::Tensor& tensor, const std::vector<int64_t> desc_size, const dil::format_tag dil_format_tag) {
  IPEX_CHECK(tensor.layout() == at::Layout::Strided,
      "dil_tensor_from_cpu_buffer expects dense tensor input");
  IPEX_CHECK(tensor.sizes().size() <= 6,
      "dil_tensor_from_cpu_buffer only support rank <= 6");
  auto cur_type = tensor.scalar_type();
  return {{desc_size, get_dil_data_type(cur_type), dil_format_tag}, tensor.data_ptr()};
}

dil::tensor dil_tensor_from_cpu_buffer(const at::Tensor& tensor, dil::deleter_ptr deleter_fn) {
  IPEX_CHECK(tensor.layout() == at::Layout::Strided,
      "dil_tensor_from_cpu_buffer expects dense tensor input");
  IPEX_CHECK(tensor.sizes().size() <= 6,
      "dil_tensor_from_cpu_buffer only support rank <= 6");
  auto cur_type = tensor.scalar_type();
  return {tensor.sizes().vec(), get_dil_data_type(cur_type), tensor.strides().vec(), tensor.data_ptr(), deleter_fn};
}

dil::tensor dil_tensor_from_dil_buffer(const at::Tensor& tensor) {
  auto dil_buffer = cpu::ShadeDataContext::getDilStorage(tensor);
  auto data_type = dil_buffer.get_data_type();
  if (dil_buffer.is_public_format() &&
      dil_buffer.get_groups() <= 1 &&
      !(data_type == dil::data_type::s8 || data_type == dil::data_type::u8)) {
    auto size = tensor.sizes().vec();
    auto stride = tensor.strides().vec();
    auto data_ptr = static_cast<void *>(
        static_cast<char *>(dil_buffer.get_data_handle()) +
        dil_buffer.get_item_size() * tensor.storage_offset());

    // return a new tensor wrapper that may be part of the dil storage
    dil::tensor result {size, data_type, stride, data_ptr};

    // copy workspace
    if (dil_buffer.has_workspace()) {
      result.copy_workspace(dil_buffer);
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

dil::tensor dil_tensor_from_dil_buffer(const at::Tensor& tensor, const dil::tensor::desc& desc) {
  auto dil_buffer = cpu::ShadeDataContext::getDilStorage(tensor);
  auto data_ptr = static_cast<void *>(
      static_cast<char *>(dil_buffer.get_data_handle()) +
      dil_buffer.get_item_size() * tensor.storage_offset());
  // return a new tensor wrapper that may be part of the dil storage
  dil::tensor result {desc, data_ptr};
  // copy workspace
  if (dil_buffer.has_workspace()) {
    result.copy_workspace(dil_buffer);
  }
  // TODO(xpz): copy scales and zero_points of qtensor (what if slicing?)
  return result;
}

dil::tensor dil_tensor_from_dil_buffer(const at::Tensor& tensor, const std::vector<int64_t> desc_size, const dil::format_tag dil_format_tag) {
  auto dil_buffer = cpu::ShadeDataContext::getDilStorage(tensor);
  auto data_type = dil_buffer.get_data_type();
  auto data_ptr = static_cast<void *>(
      static_cast<char *>(dil_buffer.get_data_handle()) +
      dil_buffer.get_item_size() * tensor.storage_offset());
  // return a new tensor wrapper that may be part of the dil storage
  dil::tensor result {{desc_size, data_type, dil_format_tag}, data_ptr};
  // copy workspace
  if (dil_buffer.has_workspace()) {
    result.copy_workspace(dil_buffer);
  }
  // TODO(xpz): copy scales and zero_points of qtensor (what if slicing?)
  return result;
}

dil::tensor try_gen_dil_tensor(const at::Tensor &input) {
  if (cpu::ShadeDataContext::isDilTensor(input)) {
    return dil_tensor_from_dil_buffer(input);
  } else {
    return dil_tensor_from_cpu_buffer(input);
  }
}

dil::tensor try_gen_dil_tensor(const at::Tensor &input, const dil::tensor::desc& desc) {
  if (cpu::ShadeDataContext::isDilTensor(input)) {
    return dil_tensor_from_dil_buffer(input, desc);
  } else {
    return dil_tensor_from_cpu_buffer(input, desc);
  }
}

dil::tensor try_gen_dil_tensor(const at::Tensor &input, const std::vector<int64_t> desc_size, const dil::format_tag dil_format_tag) {
  if (cpu::ShadeDataContext::isDilTensor(input)) {
    return dil_tensor_from_dil_buffer(input, desc_size, dil_format_tag);
  } else {
    return dil_tensor_from_cpu_buffer(input, desc_size, dil_format_tag);
  }
}

dil::tensor try_gen_dil_storage(const at::Tensor &input) {
  if (cpu::ShadeDataContext::isDilTensor(input)) {
    return cpu::ShadeDataContext::getDilStorage(input);
  } else {
    return dil_tensor_from_cpu_buffer(input);
  }
}

void reorder_to_bf16_for_mix_prec(const at::Tensor& tensor, bool not_reorder_for_training) {
  if (!check_auto_mix_bf16_fp32() || (check_auto_mix_bf16_fp32() && check_train() && not_reorder_for_training))
    return;

  auto tensor_dtype = tensor.scalar_type();
  TORCH_CHECK(tensor_dtype != at::kBFloat16, "Please disable auto mix-precision if you want to enable BFloat16 manually");
  if (tensor_dtype != at::kFloat)
    return;

  reorder_to_dtype(tensor, at::kBFloat16);
}

dil::tensor reorder_dil_tensor_to_dtype(const dil::tensor &dil_tensor, dil::data_type dtype) {
  if (!check_auto_mix_bf16_fp32() || dil_tensor.get_data_type() == dtype)
    return dil_tensor;
  auto expected_desc = dil_tensor.get_desc().to_type(dtype);
  dil::tensor dst {expected_desc};
  dst.feed_from(dil_tensor);

  // If a max pool output is converting from bf16 back to fp32,
  // its workspace has also to be copied onto the new tensor
  if (dil_tensor.has_workspace()) {
    dst.copy_workspace(dil_tensor);
  }
  return dst;
}

std::vector<std::vector<float>> get_int8_scales(const at::TensorList &inputs,
                                                bool uint8_used,
                                                const int64_t ops_id) {
  IPEX_CHECK(check_auto_mix_int8_fp32(),
             "Need enable auto mix_int8 _p32 to query int8 scales");
  IPEX_CHECK(!check_int8_calibration(),
             "Should query int8 scales after calibration");
  std::vector<bool> inputs_uint8_used;
  for (auto i = 0; i < inputs.size(); i++) {
    auto src_dil_type = try_gen_dil_tensor(inputs[i]).get_data_type();
    inputs_uint8_used.push_back(src_dil_type == dil::data_type::u8);
  }
  return get_indicator_scales(inputs_uint8_used, {uint8_used}, ops_id);
}

bool get_int8_quantized_status(const int64_t ops_id) {
  if (check_auto_mix_int8_fp32() && !check_int8_calibration())
    return get_indicator_quantized_status(ops_id);
  return false;
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
  // should fallback if not dil tensor and not own whole storage
  IPEX_CHECK(cpu::ShadeDataContext::isDilTensor(tensor) || check_tensor_own_whole_storage(tensor),  "Reorder only works while tensor owns the whole storage or tensor is a dil tensor");

  auto dst_desc = src.get_desc().to_type(get_dil_data_type(dst_scalar_type));
  reorder_to_desc(tensor, dst_desc, scales);
}

void equip_dil_buffer_nosync_shape(const at::Tensor& tensor, dil::tensor dil_buffer) {
  TORCH_CHECK(
      tensor.device().is_xpu(),
      "dil buffer can only be equipped to xpu tensor");

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
    tensor.device());

  IPEXTensorImpl* ipex_tensor_impl = (IPEXTensorImpl *)tensor.unsafeGetTensorImpl();
  ipex_tensor_impl->storage().set_data_ptr(std::move(shade_data_ptr));

  // After equip_dil_buffer(), whole storage should be managed by dil tensor,
  // and thus storage metadata should be overwritten by dil tensor
  ipex_tensor_impl->storage().set_nbytes(dil_buffer.get_nelems() * tensor.itemsize());
}

void equip_dil_buffer(const at::Tensor& tensor, dil::tensor dil_buffer, int64_t padding_size) {
  equip_dil_buffer_nosync_shape(tensor, dil_buffer);

  IPEXTensorImpl* ipex_tensor_impl = (IPEXTensorImpl *)tensor.unsafeGetTensorImpl();
  if (dil_buffer.is_public_format()) {
    ipex_tensor_impl->set_strided(
      dil_buffer.get_dims(),
      dil_buffer.get_strides(),
      ipex_tensor_impl->storage_offset(),
      tensor.scalar_type(),
      padding_size);
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
  } else if ((dil_tensor_type == dil::data_type::s8 || dil_tensor_type == dil::data_type::u8)) {
    // If the dil_tensor is int8 or unint8, then the aten tensor should always be float.
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
    c10::Device(at::DeviceType::XPU, 0));
  size_t nbytes = shade_data_context->dil_tensor->get_nelems() * c10::elementSize(at_data_type);
  auto storage_impl = c10::make_intrusive<at::StorageImpl>(
    at::StorageImpl::use_byte_size_t(),
    nbytes,
    std::move(shade_data_ptr),
    nullptr,
    /*resizeable=*/false);
  auto _tensor = at::detail::make_tensor<torch_ipex::IPEXTensorImpl>(storage_impl, at::DispatchKey::XPU, at_data_type);
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
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ipex_tensor.device().type() == at::DeviceType::XPU);
    auto* _tensor_impl = (IPEXTensorImpl *)ipex_tensor.unsafeGetTensorImpl();
    _tensor_impl->set_strided(sizes, strides, _tensor_impl->storage_offset(), ipex_tensor.scalar_type());
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

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(is_public_dtype || (!is_public_dtype && aten_dtype == at::kFloat));

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

// port from aten/src/ATen/native/Convolution.cpp
at::Tensor subtensor(at::Tensor& tensor, int dim, int groups, int g) {
  if (!tensor.defined()) {
    return at::Tensor();
  }
  int64_t n = tensor.sizes()[dim] / groups;
  return tensor.narrow(dim, n * g, n).contiguous();
}

}  // namespace comm
}  // namespace dbl
}  // namespace cpu
}  // namespace torch_ipex
