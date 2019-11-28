#ifndef SYCL_STATE_INC
#define SYCL_STATE_INC

struct THSYCLState {
  struct THSYCLRNGState* rngState;
  int numDevices;
  c10::Allocator* syclHostAllocator;
  c10::Allocator* syclDeviceAllocator;
};

#endif
