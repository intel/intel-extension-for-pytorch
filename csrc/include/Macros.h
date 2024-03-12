#pragma once

#ifdef _WIN32
//#if defined(IPEX_BUILD_SHARED_LIBS)
#define IPEX_EXPORT __declspec(dllexport)
#define IPEX_IMPORT __declspec(dllimport)
//#else
//#define IPEX_EXPORT
//#define IPEX_IMPORT
//#endif
#else // _WIN32
#if defined(__GNUC__)
#define IPEX_EXPORT __attribute__((__visibility__("default")))
#else // defined(__GNUC__)
#define IPEX_EXPORT
#endif // defined(__GNUC__)
#define IPEX_IMPORT IPEX_EXPORT
#endif // _WIN32

#ifdef BUILD_IPEX_MAIN_LIB
#define IPEX_API IPEX_EXPORT
#else
#define IPEX_API IPEX_IMPORT
#endif
