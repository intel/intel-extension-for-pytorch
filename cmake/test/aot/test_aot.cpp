#include <CL/sycl.hpp>

void simple_copy_kernel() {
  using namespace cl::sycl;
  queue myQueue;

  int inf_int = std::numeric_limits<int>::max();
  int data_int[1024];
  buffer<int, 1>resultBuf_int { data_int, range<1>{1024} };
  myQueue.submit( [&](handler& cgh) {
      auto writeResult_int = resultBuf_int.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class simple_test_int>(range<1>{1024}, [=](id<1> idx) {
          writeResult_int[idx] = inf_int;
      });
  });

  uint64_t inf_uint64_t = std::numeric_limits<uint64_t>::max();
  uint64_t data_uint64_t[1024];
  buffer<uint64_t, 1>resultBuf_uint64_t { data_uint64_t, range<1>{1024} };
  myQueue.submit( [&](handler& cgh) {
      auto writeResult_uint64_t = resultBuf_uint64_t.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class simple_test_uint64>(range<1>{1024}, [=](id<1> idx) {
          writeResult_uint64_t[idx] = inf_uint64_t;
      });
  });

  float inf_float = std::numeric_limits<float>::max();
  float data_float[1024];
  buffer<float, 1>resultBuf_float { data_float, range<1>{1024} };
  myQueue.submit( [&](handler& cgh) {
      auto writeResult_float = resultBuf_float.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class simple_test_float>(range<1>{1024}, [=](id<1> idx) {
          writeResult_float[idx] = inf_float;
      });
  });

  double inf_double = std::numeric_limits<double>::max();
  double data_double[1024];
  buffer<double, 1>resultBuf_double { data_double, range<1>{1024} };
  myQueue.submit( [&](handler& cgh) {
      auto writeResult_double = resultBuf_double.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class simple_test_double>(range<1>{1024}, [=](id<1> idx) {
          writeResult_double[idx] = inf_double;
      });
  });
}

int main() {
  simple_copy_kernel();
  return 0;
}
