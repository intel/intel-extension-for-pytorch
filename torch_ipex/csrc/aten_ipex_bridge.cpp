#include "aten_ipex_bridge.h"

#include <map>
#include <string>
#include <vector>

#include <ATen/Tensor.h>
#include <c10/core/StorageImpl.h>
#include <c10/util/Exception.h>

#include "ipex_tensor_impl.h"

namespace torch_ipex {
namespace bridge {

#define CHECK_TENSOR(a, b) \
  TORCH_INTERNAL_ASSERT(a.numel() == b.numel()); \
  TORCH_INTERNAL_ASSERT(a.dtype() == b.dtype()); \
  TORCH_INTERNAL_ASSERT(a.unsafeGetTensorImpl()->sizes() == b.unsafeGetTensorImpl()->sizes()); \
  TORCH_INTERNAL_ASSERT(a.unsafeGetTensorImpl()->dtype() == b.unsafeGetTensorImpl()->dtype()); \
  TORCH_INTERNAL_ASSERT(a.unsafeGetTensorImpl()->is_contiguous() == b.unsafeGetTensorImpl()->is_contiguous()); \
  TORCH_INTERNAL_ASSERT(a.unsafeGetTensorImpl()->is_contiguous(at::MemoryFormat::ChannelsLast) == b.unsafeGetTensorImpl()->is_contiguous(at::MemoryFormat::ChannelsLast)); \
  TORCH_INTERNAL_ASSERT(a.unsafeGetTensorImpl()->is_strides_like_channels_last() == b.unsafeGetTensorImpl()->is_strides_like_channels_last()); \
  TORCH_INTERNAL_ASSERT(a.unsafeGetTensorImpl()->is_non_overlapping_and_dense() == b.unsafeGetTensorImpl()->is_non_overlapping_and_dense()); \
  TORCH_INTERNAL_ASSERT(a.unsafeGetTensorImpl()->is_wrapped_number() == b.unsafeGetTensorImpl()->is_wrapped_number()); \
  TORCH_INTERNAL_ASSERT(a.unsafeGetTensorImpl()->version_counter().current_version() == b.unsafeGetTensorImpl()->version_counter().current_version()); \
  TORCH_INTERNAL_ASSERT(a.unsafeGetTensorImpl()->allow_tensor_metadata_change() == b.unsafeGetTensorImpl()->allow_tensor_metadata_change())

#define CHECK_TENSOR_CRITICAL(a, b) \
  TORCH_INTERNAL_ASSERT(a.data_ptr() == b.data_ptr()); \
  TORCH_INTERNAL_ASSERT(a.unsafeGetTensorImpl()->strides() == b.unsafeGetTensorImpl()->strides()); \
  TORCH_INTERNAL_ASSERT(a.unsafeGetTensorImpl()->storage_offset() == b.unsafeGetTensorImpl()->storage_offset()); \
  CHECK_TENSOR(a, b)

// Fallback DPCPP tensor to CPU Tensor.
// It will allocate new memory buffer and then duplicate the DPCPP tensor buffer to create new CPU Tensor
at::Tensor fallbackToCPUTensor(const at::Tensor& ipexTensor) {
  TORCH_INTERNAL_ASSERT(ipexTensor.is_contiguous());
  TORCH_INTERNAL_ASSERT(ipexTensor.layout() == c10::kStrided);
  if (ipexTensor.device().is_cpu())
    return ipexTensor;

  if (ipexTensor.device().type() != at::DeviceType::DPCPP) {
    assert(false);
  }

  auto* allocator = c10::GetAllocator(c10::DeviceType::CPU);
  int64_t nelements = ipexTensor.numel();
  auto dtype = ipexTensor.dtype();
  int64_t data_size = nelements * dtype.itemsize();
  auto storage_impl = c10::make_intrusive<at::StorageImpl>(
    dtype,
    nelements,
    allocator->allocate(data_size),
    allocator,
    /*resizeable=*/true);
  memcpy(storage_impl->data(), ipexTensor.unsafeGetTensorImpl()->data(), data_size);

  auto _tensor =  at::detail::make_tensor<at::TensorImpl>(storage_impl, at::TensorTypeId::CPUTensorId);
  IPEXTensorImpl::CopyMetadata(_tensor.unsafeGetTensorImpl(), ipexTensor.unsafeGetTensorImpl());
  auto _tensor_sizes = ipexTensor.sizes();
  if (_tensor_sizes.size() != 1 || _tensor_sizes[0] != 0) {
    _tensor.unsafeGetTensorImpl()->set_sizes_contiguous(_tensor_sizes);
  }
  CHECK_TENSOR(_tensor, ipexTensor);
  return _tensor;
}


// Fallback CPU tensor to DPCPP Tensor with shallow copy
// It will create an new CPU tensor but shares DPCPP tensor buffer
at::Tensor shallowFallbackToCPUTensor(const at::Tensor& ipexTensor) {
  if (ipexTensor.device().is_cpu())
    return ipexTensor;

  TORCH_INTERNAL_ASSERT(ipexTensor.device().type() == at::DeviceType::DPCPP);
  auto* allocator = c10::GetAllocator(c10::DeviceType::CPU);
  void* tensor_raw_data = ipexTensor.unsafeGetTensorImpl()->storage().data();
  c10::DataPtr cpu_data_ptr(tensor_raw_data, at::DeviceType::CPU);
  auto storage_impl = c10::make_intrusive<at::StorageImpl>(
    ipexTensor.unsafeGetTensorImpl()->storage().dtype(),
    ipexTensor.unsafeGetTensorImpl()->storage().numel(),
    std::move(cpu_data_ptr),
    allocator,
    ipexTensor.unsafeGetTensorImpl()->storage().resizable()
  );

  auto _tensor =  at::detail::make_tensor<at::TensorImpl>(storage_impl, at::TensorTypeId::CPUTensorId);
  IPEXTensorImpl::CopySizeStridesAndOffset(_tensor.unsafeGetTensorImpl(), ipexTensor.unsafeGetTensorImpl());
  IPEXTensorImpl::CopyMetadata(_tensor.unsafeGetTensorImpl(), ipexTensor.unsafeGetTensorImpl());
  CHECK_TENSOR_CRITICAL(ipexTensor, _tensor);
  //TODO: Cannot reserved_ 
  //dest_impl->reserved_ = src_impl->reserved_;
  return _tensor;
}


// Upgrade CPU tensor to DPCPP Tensor.
// It will allocate new memory buffer and then duplicate the CPU tensor buffer to create new DPCPP Tensor
at::Tensor upgradeToDPCPPTensor(const at::Tensor& cpuTensor) {
  if (cpuTensor.device().type() == at::DeviceType::DPCPP) {
    return cpuTensor;
  }

  auto* allocator = c10::GetAllocator(c10::DeviceType::DPCPP);
  int64_t nelements = cpuTensor.numel();
  auto dtype = cpuTensor.dtype();
  int64_t data_size = nelements * dtype.itemsize();
  auto storage_impl = c10::make_intrusive<at::StorageImpl>(
    dtype,
    nelements,
    allocator->allocate(data_size),
    allocator,
    /*resizeable=*/true);
  memcpy(storage_impl->data(), cpuTensor.unsafeGetTensorImpl()->data(), data_size);
  auto&& _tensor = at::detail::make_tensor<at::TensorImpl>(storage_impl, at::TensorTypeId::DPCPPTensorId);
  auto _tensor_sizes = cpuTensor.sizes();
  if (_tensor_sizes.size() != 1 || _tensor_sizes[0] != 0) {
    _tensor.unsafeGetTensorImpl()->set_sizes_contiguous(_tensor_sizes);
  }
  IPEXTensorImpl::CopyMetadata(_tensor.unsafeGetTensorImpl(), cpuTensor.unsafeGetTensorImpl());
  CHECK_TENSOR(_tensor, cpuTensor);
  return _tensor;
}


// Upgrade CPU tensor to DPCPP Tensor with shallow copy
// It will create an new DPCPP tensor but shares CPU tensor buffer
at::Tensor shallowUpgradeToDPCPPTensor(const at::Tensor& cpuTensor) {
  if (cpuTensor.device().type() == at::DeviceType::DPCPP) {
    return cpuTensor;
  }

  TORCH_INTERNAL_ASSERT(cpuTensor.device().type() == at::DeviceType::CPU);
  auto* allocator = c10::GetAllocator(c10::DeviceType::DPCPP);
  void* tensor_raw_data = cpuTensor.unsafeGetTensorImpl()->storage().data();
  c10::DataPtr dpcpp_data_ptr(tensor_raw_data, at::DeviceType::DPCPP);
  auto storage_impl = c10::make_intrusive<at::StorageImpl>(
    cpuTensor.unsafeGetTensorImpl()->storage().dtype(),
    cpuTensor.unsafeGetTensorImpl()->storage().numel(),
    std::move(dpcpp_data_ptr),
    allocator,
    cpuTensor.unsafeGetTensorImpl()->storage().resizable()
  );

  auto _tensor =  at::detail::make_tensor<IPEXTensorImpl>(cpuTensor, storage_impl, at::TensorTypeId::DPCPPTensorId);
  IPEXTensorImpl* impex_impl = (IPEXTensorImpl *)_tensor.unsafeGetTensorImpl();
  impex_impl->copy_meta_info(cpuTensor.unsafeGetTensorImpl());
  CHECK_TENSOR_CRITICAL(_tensor, cpuTensor);
  //TODO: Cannot reserved_ 
  //dest_impl->reserved_ = src_impl->reserved_;
  return _tensor;
}


void copyTensor(at::Tensor& dstTensor, const at::Tensor& scrTensor, c10::DeviceType devType) {
  TORCH_INTERNAL_ASSERT(dstTensor.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT(scrTensor.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT(dstTensor.is_contiguous());
  TORCH_INTERNAL_ASSERT(scrTensor.is_contiguous());
  TORCH_INTERNAL_ASSERT(dstTensor.numel() == scrTensor.numel());
  TORCH_INTERNAL_ASSERT(dstTensor.dtype() == scrTensor.dtype());
  TORCH_INTERNAL_ASSERT(dstTensor.nbytes() == scrTensor.nbytes());
  TORCH_INTERNAL_ASSERT(dstTensor.layout() == scrTensor.layout());
  TORCH_INTERNAL_ASSERT((devType == c10::DeviceType::CPU) || (devType == c10::DeviceType::DPCPP));
  TORCH_INTERNAL_ASSERT(dstTensor.device().type() == devType);
  TORCH_INTERNAL_ASSERT((dstTensor.device().type() == c10::DeviceType::CPU) || (dstTensor.device().type() == c10::DeviceType::DPCPP));
  TORCH_INTERNAL_ASSERT((scrTensor.device().type() == c10::DeviceType::CPU) || (scrTensor.device().type() == c10::DeviceType::DPCPP));
  memcpy(dstTensor.unsafeGetTensorImpl()->data(), scrTensor.unsafeGetTensorImpl()->data(), dstTensor.nbytes());
  IPEXTensorImpl::CopySizeStridesAndOffset(dstTensor.unsafeGetTensorImpl(), scrTensor.unsafeGetTensorImpl());
  IPEXTensorImpl::CopyMetadata(dstTensor.unsafeGetTensorImpl(), scrTensor.unsafeGetTensorImpl());
}


std::vector<at::Tensor> fallbackToCPUTensorList(const at::TensorList& tensor_list) {
  std::vector<at::Tensor> dpcpp_tensor_vec(tensor_list.size());
  for (size_t i = 0; i < tensor_list.size(); ++i) {
    const at::Tensor& tensor = tensor_list[i];
    TORCH_INTERNAL_ASSERT(tensor.defined());
    if (tensor.defined()) {
      dpcpp_tensor_vec[i] = fallbackToCPUTensor(tensor);
    }
  }
  return dpcpp_tensor_vec;
}


std::vector<at::Tensor> shallowFallbackToCPUTensorList(const at::TensorList& tensor_list) {
  std::vector<at::Tensor> dpcpp_tensor_vec(tensor_list.size());
  for (size_t i = 0; i < tensor_list.size(); ++i) {
    const at::Tensor& tensor = tensor_list[i];
    TORCH_INTERNAL_ASSERT(tensor.defined());
    if (tensor.defined()) {
      dpcpp_tensor_vec[i] = shallowFallbackToCPUTensor(tensor);
    }
  }
  return dpcpp_tensor_vec;
}


std::vector<at::Tensor> upgradeToDPCPPTensorVec(const std::vector<at::Tensor> &tensor_vec) {
  std::vector<at::Tensor> ret_dpcpp_tensor_vec;
  for (size_t i = 0; i < tensor_vec.size(); i++) {
    auto&& cur_tensor = tensor_vec[i];
    TORCH_INTERNAL_ASSERT(cur_tensor.defined());
    TORCH_INTERNAL_ASSERT(cur_tensor.layout() == c10::kStrided);
    TORCH_INTERNAL_ASSERT(cur_tensor.is_contiguous());
    auto&& cur_dpcpp_tensor = upgradeToDPCPPTensor(cur_tensor);
    ret_dpcpp_tensor_vec.push_back(cur_dpcpp_tensor);
  }
  return ret_dpcpp_tensor_vec;
}


std::vector<at::Tensor> shallowUpgradeToDPCPPTensorVec(const std::vector<at::Tensor> &tensor_vec) {
  std::vector<at::Tensor> ret_dpcpp_tensor_vec;
  for (size_t i = 0; i < tensor_vec.size(); i++) {
    auto&& cur_tensor = tensor_vec[i];
    TORCH_INTERNAL_ASSERT(cur_tensor.defined());
    TORCH_INTERNAL_ASSERT(cur_tensor.layout() == c10::kStrided);
    TORCH_INTERNAL_ASSERT(cur_tensor.is_contiguous());
    auto&& cur_dpcpp_tensor = shallowUpgradeToDPCPPTensor(cur_tensor);
    ret_dpcpp_tensor_vec.push_back(cur_dpcpp_tensor);
  }
  return ret_dpcpp_tensor_vec;
}

}  // namespace bridge
}  // namespace torch_ipex
