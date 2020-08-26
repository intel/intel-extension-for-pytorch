#include <ATen/Config.h>
#include <ATen/Context.h>
#include <c10/util/Exception.h>

#include <core/DPCPPUtils.h>
#include <core/Device.h>
#include <core/Generator.h>
#include <core/detail/Hooks.h>
#include <utils/General.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <iostream>

namespace at {
namespace dpcpp {
namespace detail {

void DPCPPHooks::initDPCPP() const {
  std::cout<< "DPCPP hooks: " << __FUNCTION__ << std::endl;
  // TODO:
}

bool DPCPPHooks::hasDPCPP() const {
  std::cout<< "DPCPP hooks: " << __FUNCTION__ << std::endl;
  return true;
}

bool DPCPPHooks::hasOneMKL() const {
  std::cout<< "DPCPP hooks: " << __FUNCTION__ << std::endl;
#ifdef USE_ONEMKL
  return true;
#else
  return false;
#endif
}

bool DPCPPHooks::hasOneDNN() const {
  std::cout<< "DPCPP hooks: " << __FUNCTION__ << std::endl;
  return true;
}

std::string DPCPPHooks::showConfig() const {
  std::cout<< "DPCPP hooks: " << __FUNCTION__ << std::endl;
  return "DPCPP backend version: 1.0";
}

int64_t DPCPPHooks::getCurrentDevice() const {
  std::cout<< "DPCPP hooks: " << __FUNCTION__ << std::endl;
  c10::DeviceIndex device_index;
  dpcppGetDevice(&device_index);
  return device_index;
}

int DPCPPHooks::getDeviceCount() const {
  std::cout<< "DPCPP hooks: " << __FUNCTION__ << std::endl;
  int count;
  dpcppGetDeviceCount(&count);
  return count;
}

c10::Device DPCPPHooks::getDeviceFromPtr(void* data) const {
  std::cout<< "DPCPP hooks: " << __FUNCTION__ << std::endl;
  return getDeviceFromPtr(data);
}

bool DPCPPHooks::isPinnedPtr(void* data) const {
  std::cout<< "DPCPP hooks: " << __FUNCTION__ << std::endl;
#ifndef USE_USM
  // Not support pin memory if no USM support
  return false;
#else
  // TODO:
  return true;
#endif
}

c10::Allocator* DPCPPHooks::getPinnedMemoryAllocator() const {
  std::cout<< "DPCPP hooks: " << __FUNCTION__ << std::endl;
#ifndef USE_USM
  // Not support pin memory if no USM support
  throw std::runtime_error("The pin memory is not supported without USM support.");
  return nullptr;
#else
  // TODO:
  return nullptr;
#endif
}

at::Generator* DPCPPHooks::getDefaultDPCPPGenerator(DeviceIndex device_index = -1) const {
  std::cout<< "DPCPP hooks: " << __FUNCTION__ << std::endl;
  return at::dpcpp::detail::getDefaultDPCPPGenerator(device_index);
}

REGISTER_DPCPP_HOOKS(DPCPPHooks);

} // detail
} // dpcpp
} // namespace
