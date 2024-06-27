#pragma once

#include <sycl/sycl.hpp>
#include <functional>
#include <vector>

namespace torch_ipex::xpu::xetla {
// Type alias for a vector of CGFs
using cgf_t = std::function<void(sycl::handler&)>;
using cgfs_t = std::vector<cgf_t>;

#ifdef _WIN32
#define XETLA_KERNEL_EXPORT __declspec(dllexport)
#define XETLA_KERNEL_IMPORT __declspec(dllimport)
#else // _WIN32
#if defined(__GNUC__)
#define XETLA_KERNEL_EXPORT __attribute__((__visibility__("default")))
#else // defined(__GNUC__)
#define XETLA_KERNEL_EXPORT
#endif // defined(__GNUC__)
#define XETLA_KERNEL_IMPORT XETLA_KERNEL_EXPORT
#endif // _WIN32

#ifdef BUILD_XETLA_KERNEL_LIB
#define XETLA_KERNEL_API XETLA_KERNEL_EXPORT
#else
#define XETLA_KERNEL_API XETLA_KERNEL_IMPORT
#endif
} // namespace torch_ipex::xpu::xetla
