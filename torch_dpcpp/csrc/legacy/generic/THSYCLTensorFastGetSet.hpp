#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "THDP/generic/THSYCLTensorFastGetSet.hpp"
#else
#include <THDP/THSYCLDeviceTensor.h>

static inline scalar_t THSYCLTensor_(fastGetLegacy1dNoScalars)(THSYCLDeviceTensor<1, scalar_t, cl::sycl::access::mode::read>  self, int64_t x0) {
  int64_t stride = self.dim() == 0 ? 1 : self.stride(0);
  return self.template data<scalar_t>()[x0*stride];
}

static inline scalar_t THSYCLTensor_(fastGet1d)(THSYCLDeviceTensor<1, scalar_t, cl::sycl::access::mode::read> self, int64_t x0) {
  return self.template data<scalar_t>()[x0*self.stride(0)];
}

static inline scalar_t THSYCLTensor_(fastGet2d)(THSYCLDeviceTensor<2, scalar_t, cl::sycl::access::mode::read> self, int64_t x0, int64_t x1) {
  return self.template data<scalar_t>()[x0*self.stride(0)+x1*self.stride(1)];
}

static inline scalar_t THSYCLTensor_(fastGet3d)(THSYCLDeviceTensor<3, scalar_t, cl::sycl::access::mode::read> self, int64_t x0, int64_t x1, int64_t x2) {
  return self.template data<scalar_t>()[x0*self.stride(0)+x1*self.stride(1)+x2*self.stride(2)];
}

static inline scalar_t THSYCLTensor_(fastGet4d)(THSYCLDeviceTensor<4, scalar_t, cl::sycl::access::mode::read> self, int64_t x0, int64_t x1, int64_t x2, int64_t x3) {
  return self.template data<scalar_t>()[x0*self.stride(0)+x1*self.stride(1)+x2*self.stride(2)+x3*self.stride(3)];
}

static inline scalar_t THSYCLTensor_(fastGet5d)(THSYCLDeviceTensor<5, scalar_t, cl::sycl::access::mode::read> self, int64_t x0, int64_t x1, int64_t x2, int64_t x3, int64_t x4) {
  return self.template data<scalar_t>()[x0*self.stride(0)+x1*self.stride(1)+x2*self.stride(2)+x3*self.stride(3)+(x4)*self.stride(4)];
}

static inline void THSYCLTensor_(fastSet1d)(THSYCLDeviceTensor<1, scalar_t, cl::sycl::access::mode::discard_write> self, int64_t x0, scalar_t value) {
  self.template data<scalar_t>()[x0*self.stride(0)] = value;
}

static inline void THSYCLTensor_(fastSet2d)(THSYCLDeviceTensor<2, scalar_t, cl::sycl::access::mode::discard_write> self, int64_t x0, int64_t x1, scalar_t value) {
  self.template data<scalar_t>()[x0*self.stride(0)+x1*self.stride(1)] = value;
}

static inline void THSYCLTensor_(fastSet3d)(THSYCLDeviceTensor<3, scalar_t, cl::sycl::access::mode::discard_write> self, int64_t x0, int64_t x1, int64_t x2, scalar_t value) {
  self.template data<scalar_t>()[x0*self.stride(0)+x1*self.stride(1)+x2*self.stride(2)] = value;
}

static inline void THSYCLTensor_(fastSet4d)(THSYCLDeviceTensor<4, scalar_t, cl::sycl::access::mode::discard_write> self, int64_t x0, int64_t x1, int64_t x2, int64_t x3, scalar_t value) {
  self.template data<scalar_t>()[x0*self.stride(0)+x1*self.stride(1)+x2*self.stride(2)+x3*self.stride(3)] = value;
}

static inline void THSYCLTensor_(fastSet5d)(THSYCLDeviceTensor<5, scalar_t, cl::sycl::access::mode::discard_write> self, int64_t x0, int64_t x1, int64_t x2, int64_t x3, int64_t x4, scalar_t value) {
  self.template data<scalar_t>()[x0*self.stride(0)+x1*self.stride(1)+x2*self.stride(2)+x3*self.stride(3)+(x4)*self.stride(4)] = value;
}

#endif
