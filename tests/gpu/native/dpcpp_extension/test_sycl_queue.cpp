#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif
#include <torch/extension.h>
#include <iostream>

struct isSYCLQueueFunctor {
  void operator()() const {}
};

// This routine is used to check if the pointer 'q_ptr' is a sycl queue.
// NOTE: If the pointer is, this routine will return true. Otherwise, it will
// causes a segmentation fault that we can NOT catch.
extern "C" bool isSYCLQueue(void* q_ptr) {
  auto& sycl_queue = *reinterpret_cast<sycl::queue*>(q_ptr);
  auto src = at::ones({2, 2}, at::device(at::kXPU).dtype(at::kInt));
  auto dst = at::empty({2, 2}, at::dtype(at::kInt));

  // This code is used to test compiler linker.
  isSYCLQueueFunctor f;
  auto cgf = [&](sycl::handler& cgh) { cgh.single_task<decltype(f)>(f); };
  sycl_queue.submit(cgf);

  sycl_queue.memcpy(dst.data_ptr(), src.data_ptr(), src.nbytes()).wait();
  auto dst_accessor = dst.accessor<int32_t, 2>();

  if (dst_accessor[0][0] == 1 && dst_accessor[0][1] == 1 &&
      dst_accessor[1][0] == 1 && dst_accessor[1][1] == 1) {
    return true;
  }

  return false;
}

bool is_sycl_queue(py::object obj) {
  TORCH_CHECK(PyLong_Check(obj.ptr()), "invalid SYCL queue pointer");
  auto q_ptr = PyLong_AsVoidPtr(obj.ptr());
  if (q_ptr == nullptr && PyErr_Occurred()) {
    throw python_error();
  }
  return isSYCLQueue(q_ptr);
}

PYBIND11_MODULE(mod_test_sycl_queue, m) {
  m.def(
      "is_sycl_queue",
      &is_sycl_queue,
      "check sycl queue void pointer in a capsule");
}
