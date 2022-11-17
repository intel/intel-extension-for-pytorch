/* Build command:
$ dpcpp device_enum.cpp -o device_enum
*/

/* Example result:
$ ./device_enum
=====================================================================================
Available DPC++ Platforms / Devices
=====================================================================================
|Platform 0:
|Intel(R) OpenCL HD Graphics (opencl)
| |__|Device 0 :
|    |Intel(R) UHD Graphics 630 [0x3e9b] (gpu)
-------------------------------------------------------------------------------------
|Platform 1:
|Intel(R) Level-Zero (level_zero)
| |__|Device 0 :
|    |Intel(R) UHD Graphics 630 [0x3e9b] (gpu)
-------------------------------------------------------------------------------------
|Platform 2:
|SYCL host platform (host)
| |__|Device 0 :
|    |SYCL host device (host)
-------------------------------------------------------------------------------------
*/

#include <CL/sycl.hpp>
#include <stdlib.h>
#include <iomanip>
#include <vector>

#define DEL_WIDTH 85

using namespace sycl;

std::string getDeviceTypeName(const device& Device) {
  auto DeviceType = Device.get_info<info::device::device_type>();
  switch (DeviceType) {
    case info::device_type::cpu:
      return "cpu";
    case info::device_type::gpu:
      return "gpu";
    case info::device_type::host:
      return "host";
    case info::device_type::accelerator:
      return "accelerator";
    default:
      return "unknown";
  }
}

int main(int argc, char* argv[]) {
  // print header
  std::cout << std::setw(DEL_WIDTH) << std::setfill('=') << '=' << std::endl;
  std::cout << "Available DPC++ Platforms / Devices" << std::endl;
  std::cout << std::setw(DEL_WIDTH) << std::setfill('=') << '=' << std::endl;
  // enum Platforms
  std::vector<platform> Platforms = platform::get_platforms();
  for (size_t PlatformID = 0; PlatformID < Platforms.size(); PlatformID++) {
    auto PlatformName = Platforms[PlatformID].get_info<info::platform::name>();
    backend Backend = Platforms[PlatformID].get_backend();
    // print Platform Info
    std::cout << "|Platform " << PlatformID << ":" << std::endl
              << "|" << PlatformName << " (" << Backend << ")" << std::endl;
    // enum Devices
    std::vector<device> Devices = Platforms[PlatformID].get_devices();
    for (size_t DevicesID = 0; DevicesID < Devices.size(); DevicesID++) {
      auto DeviceName = Devices[DevicesID].get_info<info::device::name>();
      auto DeviceType =
          Devices[DevicesID].get_info<info::device::device_type>();
      std::string DeviceTypeName = getDeviceTypeName(Devices[DevicesID]);
      // print Device Info
      std::cout << "|\t|__|Device " << DevicesID << ":" << std::endl;
      if (DevicesID == Devices.size() - 1) {
        std::cout << "|\t   |" << DeviceName << " (" << DeviceTypeName << ")"
                  << std::endl;
      } else {
        std::cout << "|\t|  |" << DeviceName << " (" << DeviceTypeName << ")"
                  << std::endl;
      }
    }
    std::cout << std::setw(DEL_WIDTH) << std::setfill('-') << '-' << std::endl;
  }
}
