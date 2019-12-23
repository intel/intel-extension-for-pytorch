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

// Fallback DPCPP tensor to CPU Tensor.
// It will allocate new memory buffer and then duplicate the DPCPP tensor buffer to create new CPU Tensor
at::Tensor fallbackToCPUTensor(const at::Tensor& ipexTensor) {
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
  auto _tensor_sizes = ipexTensor.sizes();
  if (_tensor_sizes.size() != 1 || _tensor_sizes[0] != 0) {
    _tensor.unsafeGetTensorImpl()->set_sizes_contiguous(_tensor_sizes);
  }

  IPEXTensorImpl::CopyMetadata(_tensor.unsafeGetTensorImpl(), ipexTensor.unsafeGetTensorImpl(), true);
  TORCH_INTERNAL_ASSERT(ipexTensor.dtype() == _tensor.dtype());
  return _tensor;
}


// Fallback CPU tensor to DPCPP Tensor with shallow copy
// It will create an new CPU tensor but shares DPCPP tensor buffer
at::Tensor fallbackToCPUTensor_(const at::Tensor& ipexTensor) {
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
    true
  );

  auto _tensor =  at::detail::make_tensor<at::TensorImpl>(storage_impl, at::TensorTypeId::CPUTensorId);
  IPEXTensorImpl::CopySizeStridesAndOffset(_tensor.unsafeGetTensorImpl(), ipexTensor.unsafeGetTensorImpl(), true);
  IPEXTensorImpl::CopyMetadata(_tensor.unsafeGetTensorImpl(), ipexTensor.unsafeGetTensorImpl(), true);
  TORCH_INTERNAL_ASSERT(ipexTensor.dtype() == _tensor.dtype());
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

  IPEXTensorImpl::CopyMetadata(_tensor.unsafeGetTensorImpl(), cpuTensor.unsafeGetTensorImpl(), true);
  TORCH_INTERNAL_ASSERT(cpuTensor.dtype() == _tensor.dtype());
  return _tensor;
}


// Upgrade CPU tensor to DPCPP Tensor with shallow copy
// It will create an new DPCPP tensor but shares CPU tensor buffer
at::Tensor upgradeToDPCPPTensor_(const at::Tensor& cpuTensor) {
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
    true
  );

  auto _tensor =  at::detail::make_tensor<at::TensorImpl>(storage_impl, at::TensorTypeId::DPCPPTensorId);
  IPEXTensorImpl::CopySizeStridesAndOffset(_tensor.unsafeGetTensorImpl(), cpuTensor.unsafeGetTensorImpl(), true);
  IPEXTensorImpl::CopyMetadata(_tensor.unsafeGetTensorImpl(), cpuTensor.unsafeGetTensorImpl(), true);
  TORCH_INTERNAL_ASSERT(cpuTensor.dtype() == _tensor.dtype());
  return _tensor;
}


void copyTensor(at::Tensor& dstTensor, const at::Tensor& scrTensor, c10::DeviceType devType) {
  TORCH_CHECK(dstTensor.layout() == c10::kStrided);
  TORCH_CHECK(scrTensor.layout() == c10::kStrided);
  TORCH_CHECK(dstTensor.is_contiguous());
  TORCH_CHECK(scrTensor.is_contiguous());
  TORCH_CHECK(dstTensor.numel() == scrTensor.numel());
  TORCH_CHECK(dstTensor.dtype() == scrTensor.dtype());
  TORCH_CHECK(dstTensor.nbytes() == scrTensor.nbytes());
  TORCH_CHECK(dstTensor.layout() == scrTensor.layout());
  TORCH_CHECK((devType == c10::DeviceType::CPU) || (devType == c10::DeviceType::DPCPP));
  TORCH_CHECK(dstTensor.device().type() == devType);
  TORCH_CHECK((dstTensor.device().type() == c10::DeviceType::CPU) || (dstTensor.device().type() == c10::DeviceType::DPCPP));
  TORCH_CHECK((scrTensor.device().type() == c10::DeviceType::CPU) || (scrTensor.device().type() == c10::DeviceType::DPCPP));
  memcpy(dstTensor.unsafeGetTensorImpl()->data(), scrTensor.unsafeGetTensorImpl()->data(), dstTensor.nbytes());
  IPEXTensorImpl::CopySizeStridesAndOffset(dstTensor.unsafeGetTensorImpl(), scrTensor.unsafeGetTensorImpl(), true);
  IPEXTensorImpl::CopyMetadata(dstTensor.unsafeGetTensorImpl(), scrTensor.unsafeGetTensorImpl(), true);
}


at::TensorList fallbackToCPUTensorList(const at::TensorList& tensor_list) {
  std::vector<at::Tensor> dpcpp_tensor_vec;
  for (const auto& tensor : tensor_list) {
    if (tensor.defined()) {
      dpcpp_tensor_vec.push_back(fallbackToCPUTensor(tensor));
    }
  }
  return at::TensorList(dpcpp_tensor_vec);
}


at::TensorList fallbackToCPUTensorList_(const at::TensorList& tensor_list) {
  std::vector<at::Tensor> dpcpp_tensor_vec;
  for (const auto& tensor : tensor_list) {
    if (tensor.defined()) {
      dpcpp_tensor_vec.push_back(fallbackToCPUTensor_(tensor));
    }
  }
  return at::TensorList(dpcpp_tensor_vec);
}


std::vector<at::Tensor> upgradeToDPCPPTensorVec(const std::vector<at::Tensor> &tensor_vec) {
  std::vector<at::Tensor> ret_dpcpp_tensor_vec;
  for (size_t i = 0; i < tensor_vec.size(); i++) {
    auto&& cur_tensor = tensor_vec[i];
    TORCH_CHECK(cur_tensor.layout() == c10::kStrided);
    TORCH_CHECK(cur_tensor.is_contiguous());
    auto&& cur_dpcpp_tensor = upgradeToDPCPPTensor(cur_tensor);
    ret_dpcpp_tensor_vec.push_back(cur_dpcpp_tensor);
  }
  return ret_dpcpp_tensor_vec;
}


std::vector<at::Tensor> upgradeToDPCPPTensorVec_(const std::vector<at::Tensor> &tensor_vec) {
  std::vector<at::Tensor> ret_dpcpp_tensor_vec;
  for (size_t i = 0; i < tensor_vec.size(); i++) {
    auto&& cur_tensor = tensor_vec[i];
    TORCH_CHECK(cur_tensor.layout() == c10::kStrided);
    TORCH_CHECK(cur_tensor.is_contiguous());
    auto&& cur_dpcpp_tensor = upgradeToDPCPPTensor_(cur_tensor);
    ret_dpcpp_tensor_vec.push_back(cur_dpcpp_tensor);
  }
  return ret_dpcpp_tensor_vec;
}

}  // namespace bridge
}  // namespace torch_ipex
