#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif
#include <torch/extension.h>
#include <iostream>

// This routine is used to check if the pointer 'd_ptr' is a sycl device.
// NOTE: If the pointer is, this routine will return true. Otherwise, it will
// causes a segmentation fault that we can NOT catch.
extern "C" bool isSYCLDevice(void* d_ptr) {
  auto& sycl_device = *reinterpret_cast<sycl::device*>(d_ptr);
  auto dev_name = sycl_device.get_info<sycl::info::device::name>();
  return true;
}

PYBIND11_MODULE(mod_test_sycl_device, m) {
  m.def(
      "is_sycl_device",
      &isSYCLDevice,
      "check sycl device void pointer in a capsule");
}
