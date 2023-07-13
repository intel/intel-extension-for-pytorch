#pragma once

#include <memory>

#include <ATen/TensorUtils.h>
#include <c10/core/Device.h>
#include <core/DeviceInfo.h>
#include <utils/Macros.h>

using namespace at;

namespace xpu {
namespace dpcpp {

IPEX_API DeviceIndex prefetch_device_count() noexcept;

IPEX_API DeviceIndex device_count() noexcept;

IPEX_API DeviceIndex current_device();

IPEX_API void set_device(DeviceIndex device);

IPEX_API void* sycl_device(DeviceIndex device);

DeviceIndex get_device_index_from_ptr(void* ptr);

DeviceInfo* getCurrentDeviceInfo();

IPEX_API DeviceInfo* getDeviceInfo(DeviceIndex device);

IPEX_API std::vector<int> prefetchDeviceIdListForCard(int card_id);

IPEX_API std::vector<int>& getDeviceIdListForCard(int card_id);

static inline void isOnSameDevice(
    at::CheckedFrom c,
    const at::TensorArg& t1,
    const at::TensorArg& t2) {
  if ((t1->device().type() != at::kXPU) || (t2->device().type() != at::kXPU)) {
    std::ostringstream oss;
    if (t1->device().type() != at::kXPU) {
      oss << "Tensor for " << t1 << " is not on XPU, ";
    }
    if (t2->device().type() != at::kXPU) {
      oss << "Tensor for " << t2 << " is not on XPU, ";
    }
    oss << "but expected "
        << ((!(t1->device().type() == at::kXPU ||
               t2->device().type() == at::kXPU))
                ? "them"
                : "it")
        << " to be on XPU (while checking arguments for " << c << ")";
    TORCH_CHECK(0, oss.str());
  }
  TORCH_CHECK(
      t1->get_device() == t2->get_device(),
      "Expected tensor for ",
      t1,
      " to have the same device as tensor for ",
      t2,
      "; but device ",
      t1->get_device(),
      " does not equal ",
      t2->get_device(),
      " (while checking arguments for ",
      c,
      ")");
}

static inline bool isOnSameDevice(const at::TensorList& tensor_list) {
  if (tensor_list.empty()) {
    return true;
  }
  Device curDevice = Device(kXPU, current_device());
  for (const Tensor& t : tensor_list) {
    if (t.device() != curDevice)
      return false;
  }
  return true;
}
} // namespace dpcpp
} // namespace xpu
