#pragma once

#ifdef _WIN32
# if defined(ATen_sycl_EXPORTS) || defined(caffe2_sycl_EXPORTS) || defined(CAFFE2_SYCL_BUILD_MAIN_LIB)
#  define AT_SYCL_API __declspec(dllexport)
# else
#  define AT_SYCL_API __declspec(dllimport)
# endif
#elif defined(__GNUC__)
#if defined(ATen_sycl_EXPORTS) || defined(caffe2_sycl_EXPORTS)
#define AT_SYCL_API __attribute__((__visibility__("default")))
#else
#define AT_SYCL_API
#endif
#else
# define AT_SYCL_API
#endif

