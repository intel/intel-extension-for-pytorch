#ifndef THSYCL_GENERAL_INC
#define THSYCL_GENERAL_INC
#include <TH/THGeneral.h>
#include <TH/THAllocator.h>
#ifdef __cplusplus
#include <core/SYCLStream.h>
#include <c10/core/Allocator.h>
#endif

#ifdef __cplusplus
#define THSYCL_EXTERNC extern "C"
#else
#define THSYCL_EXTERNC extern
#endif


#ifdef _WIN32
#else
#define THSYCL_API THSYCL_EXTERNC CAFFE2_API
#define THSYCL_CLASS CAFFE2_API

#endif

#ifndef M_PI
# define M_PI 3.14159265358979323846
#endif

#define THSYCL_CONCAT_STRING_2(x,y) THSYCL_CONCAT_STRING_2_EXPAND(x,y)
#define THSYCL_CONCAT_STRING_2_EXPAND(x,y) #x #y

#define THSYCL_CONCAT_STRING_3(x,y,z) THSYCL_CONCAT_STRING_3_EXPAND(x,y,z)
#define THSYCL_CONCAT_STRING_3_EXPAND(x,y,z) #x #y #z

#define THSYCL_CONCAT_STRING_4(x,y,z,w) THSYCL_CONCAT_STRING_4_EXPAND(x,y,z,w)
#define THSYCL_CONCAT_STRING_4_EXPAND(x,y,z,w) #x #y #z #w

#define THSYCL_CONCAT_2(x,y) THSYCL_CONCAT_2_EXPAND(x,y)
#define THSYCL_CONCAT_2_EXPAND(x,y) x ## y

#define THSYCL_CONCAT_3(x,y,z) THSYCL_CONCAT_3_EXPAND(x,y,z)
#define THSYCL_CONCAT_3_EXPAND(x,y,z) x ## y ## z

#define THSYCL_CONCAT_4_EXPAND(x,y,z,w) x ## y ## z ## w
#define THSYCL_CONCAT_4(x,y,z,w) THSYCL_CONCAT_4_EXPAND(x,y,z,w)

/* Global state of THSYCL. */
struct THSYCLState {
  struct THSYCLRNGState* rngState;
  int numDevices;
  c10::Allocator* syclHostAllocator;
  c10::Allocator* syclDeviceAllocator;
};

typedef THSYCLState THSYCLState;
struct THSYCLState;

#define THSYCLCheck(err)  __THSYCLCheck(err, __FILE__, __LINE__)
THSYCL_API void __THSYCLCheck(int err, const char *file, const int line);

THSYCL_API THSYCLState* THSYCLState_alloc(void);
THSYCL_API void THSYCLState_free(THSYCLState* state);

THSYCL_API void THSyclInit(THSYCLState* state);

THSYCL_API struct THSYCLRNGState* THSYCLState_getRngState(THSYCLState* state);
THSYCL_API c10::Allocator* THSYCLState_getSYCLHostAllocator(THSYCLState* state);

#define THSYCLAssertSameGPU(expr) if (!expr) THError("arguments are located on different GPUs")

#endif

