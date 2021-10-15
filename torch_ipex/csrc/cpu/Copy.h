#pragma once

#include <ATen/ATen.h>

namespace torch_ipex {
namespace cpu {

template <typename T>
inline void transpose_kernel_8x8(
    const T* src,
    int64_t ld_src,
    T* dst,
    int64_t ld_dst);

template <>
inline void transpose_kernel_8x8<float>(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst);

template <>
inline void transpose_kernel_8x8<at::BFloat16>(
    const at::BFloat16* src,
    int64_t ld_src,
    at::BFloat16* dst,
    int64_t ld_dst);

static void copy_kernel(at::TensorIterator& iter, bool non_blocking);

template <typename scalar_t>
void transpose_copy_kernel_impl(at::Tensor& self, const at::Tensor& src);

static void transpose_copy_kernel(at::Tensor& self, const at::Tensor& src);

bool copy_transpose_valid(const at::Tensor& self, const at::Tensor& src);

void copy_same_type_transpose_(at::Tensor& self, const at::Tensor& src);

bool is_supported_device(at::Device device);

at::Tensor& quantized_copy_from_float_cpu_(
    at::Tensor& self,
    const at::Tensor& src);

static at::Tensor& copy_impl(
    at::Tensor& self,
    const at::Tensor& src,
    bool non_blocking);

at::Tensor& copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking);

} // namespace cpu
} // namespace torch_ipex