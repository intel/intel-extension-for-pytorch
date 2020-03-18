#pragma once

#ifdef _WIN32
# if defined(ATen_dpcpp_EXPORTS) || defined(caffe2_dpcpp_EXPORTS) || defined(CAFFE2_DPCPP_BUILD_MAIN_LIB)
#  define AT_DPCPP_API __declspec(dllexport)
# else
#  define AT_DPCPP_API __declspec(dllimport)
# endif
#elif defined(__GNUC__)
#if defined(ATen_dpcpp_EXPORTS) || defined(caffe2_dpcpp_EXPORTS)
#define AT_DPCPP_API __attribute__((__visibility__("default")))
#else
#define AT_DPCPP_API
#endif
#else
# define AT_DPCPP_API
#endif

