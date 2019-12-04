#ifndef THSYCL_DEVICE_ALLOCATOR_INC
#define THSYCL_DEVICE_ALLOCATOR_INC

#ifdef __cplusplus
#include <c10/dpcpp/SYCLStream.h>
#include <ATen/dpcpp/ATenSYCLGeneral.h>
#endif

#if (__cplusplus >= 201103L) || (defined(__MEC_VER) && defined(__cplusplus))
#include <mutex>
#endif

#include <THDP/THSYCLGeneral.h>

THSYCL_API c10::Allocator* THSYCLAllocator_get(void);

#endif
