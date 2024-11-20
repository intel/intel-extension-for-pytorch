#pragma once

#include <functional>
#include <vector>

namespace esimd {
using cgf_t = std::function<void(sycl::handler&)>;
using cgfs_t = std::vector<cgf_t>;

#ifdef _WIN32
#define ESIMD_KERNEL_EXPORT __declspec(dllexport)
#define ESIMD_KERNEL_IMPORT __declspec(dllimport)
#else // _WIN32
#if defined(__GNUC__)
#define ESIMD_KERNEL_EXPORT __attribute__((__visibility__("default")))
#else // defined(__GNUC__)
#define ESIMD_KERNEL_EXPORT
#endif // defined(__GNUC__)
#define ESIMD_KERNEL_IMPORT ESIMD_KERNEL_EXPORT
#endif // _WIN32

#define ESIMD_KERNEL_API ESIMD_KERNEL_EXPORT

} // namespace esimd