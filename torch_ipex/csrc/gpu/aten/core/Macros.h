#pragma once

// See c10/macros/Export.h for a detailed explanation of what the function
// of these macros are.  We need one set of macros for every separate library
// we build.

#ifdef _WIN32
#if defined(AT_DPCPP_BUILD_SHARED_LIBS)
#define AT_DPCPP_EXPORT __declspec(dllexport)
#define AT_DPCPP_IMPORT __declspec(dllimport)
#else
#define AT_DPCPP_EXPORT
#define AT_DPCPP_IMPORT
#endif
#else // _WIN32
#if defined(__GNUC__)
#define AT_DPCPP_EXPORT __attribute__((__visibility__("default")))
#else // defined(__GNUC__)
#define AT_DPCPP_EXPORT
#endif // defined(__GNUC__)
#define AT_DPCPP_IMPORT AT_DPCPP_EXPORT
#endif // _WIN32

// This one is being used by libc10_cuda.so
#ifdef AT_DPCPP_BUILD_MAIN_LIB
#define AT_DPCPP_API AT_DPCPP_EXPORT
#else
#define AT_DPCPP_API AT_DPCPP_IMPORT
#endif

#define DPCPP_SUCCESS 0
