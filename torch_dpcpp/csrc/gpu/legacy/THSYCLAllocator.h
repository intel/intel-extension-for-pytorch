#ifndef THSYCL_DEVICE_ALLOCATOR_INC
#define THSYCL_DEVICE_ALLOCATOR_INC

#ifdef __cplusplus
#include <core/SYCLStream.h>
#include <core/ATenSYCLGeneral.h>
#endif

#if (__cplusplus >= 201103L) || (defined(__MEC_VER) && defined(__cplusplus))
#include <mutex>
#endif

c10::Allocator* THSYCLAllocator_get(void);

#endif
