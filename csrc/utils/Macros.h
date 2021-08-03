#pragma once

#include <iostream>

#ifdef _WIN32
#if defined(IPEX_BUILD_SHARED_LIBS)
#define IPEX_EXPORT __declspec(dllexport)
#define IPEX_IMPORT __declspec(dllimport)
#else
#define IPEX_EXPORT
#define IPEX_IMPORT
#endif
#else // _WIN32
#if defined(__GNUC__)
#define IPEX_EXPORT __attribute__((__visibility__("default")))
#else // defined(__GNUC__)
#define IPEX_EXPORT
#endif // defined(__GNUC__)
#define IPEX_IMPORT IPEX_EXPORT
#endif // _WIN32

#ifndef BUILD_IPEX_MAIN_LIB
#define IPEX_API IPEX_EXPORT
#else
#define IPEX_API IPEX_IMPORT
#endif

#define IPEX_DISABLE_COPY_AND_ASSIGN(classname) \
  classname(const classname&) = delete;         \
  classname& operator=(const classname&) = delete

enum DPCPP_STATUS {
  DPCPP_SUCCESS = 0,
  DPCPP_FAILURE = 1,
};

// Host side print utils
template <typename T>
void ipex_host_print(T& t) {
  std::cout << t;
}

template <typename T, typename... Ts>
void ipex_host_print(T& t, Ts&&... args) {
  std::cout << t;
  ipex_host_print(std::forward<Ts>(args)...);
}

#define IPEX_H_PRINT(...)         \
  {                               \
    ipex_host_print(__VA_ARGS__); \
    std::cout << std::endl;       \
  }
