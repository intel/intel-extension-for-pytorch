#pragma once 
#include <CL/sycl.hpp>
#include <c10/dpcpp/SYCLUtils.h>

namespace c10 {
namespace sycl {
void syclMemoryScale(void * dst, const void * src, size_t n_elements, float alpha);
void syclMemoryScale1(void * dst, const void * src, size_t n_elements, const double eps);
void syclMemoryScale2(void * dst, const void * src, size_t n_elements, const float alpha, const double eps);
}
} // c10::sycl
 
