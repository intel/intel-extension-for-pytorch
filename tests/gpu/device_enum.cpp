/* Build command:
$ dpcpp device_enum.cpp -fsycl -o device_enum
*/

/* Example result:
$ ./device_enum
================================================================
Available DPC++ Platforms / Devices
================================================================
|Platform0 :
|Intel(R) OpenCL HD Graphics
|       |__|Device0 :
|          |Intel(R) UHD Graphics 630 [0x3e98] (GPU)
----------------------------------------------------------------
|Platform1 :
|Intel(R) Level-Zero
|       |__|Device0 :
|          |Intel(R) UHD Graphics 630 [0x3e98] (GPU)
----------------------------------------------------------------
|Platform2 :
|SYCL host platform
|       |__|Device0 :
|          |SYCL host device (NonGPU)
----------------------------------------------------------------
*/

#include <stdlib.h>
#include <vector>
#include <CL/sycl.hpp>

int main(int argc, char *argv[]) {
  std::cout<< "================================================================\n";
  std::cout<< "           Available DPC++ Platforms / Devices                  \n";
  std::cout<< "================================================================\n";

  std::vector<sycl::platform> platforms = sycl::platform::get_platforms();
  for (size_t pid = 0; pid < platforms.size(); pid++) {
    sycl::string_class pname = platforms[pid].get_info<sycl::info::platform::name>();
    std::cout << "|Platform" << pid << " :\n" << "|" << pname << std::endl;

    std::vector<sycl::device> devices = platforms[pid].get_devices(sycl::info::device_type::all);
    for (size_t device_id = 0; device_id < devices.size(); device_id++) {
      sycl::string_class dname = devices[device_id].get_info<sycl::info::device::name>();

      sycl::string_class dtype;
      if (devices[device_id].is_gpu()) {
        dtype = "GPU";
      } else {
        dtype = "NonGPU";
      }

      std::cout << "|\t|__|Device" << device_id << " :\n";
      if (device_id == devices.size() - 1) {
        std::cout << "|\t   |" << dname << " (" << dtype << ")" << std::endl;
      } else {
        std::cout << "|\t|  |" << dname << " (" << dtype << ")" << std::endl;
      }
    }

    std::cout<< "----------------------------------------------------------------\n";
  }
}

