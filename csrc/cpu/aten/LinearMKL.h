#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <dyndisp/DispatchStub.h>
#include <ideep.hpp>
#include <vector>
#include "cpu/kernels/OpContext.h"
#include "mkl.h"

namespace torch_ipex {
namespace cpu {

at::Tensor mkl_sgemm_pack_weight(
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const at::Tensor& ori_weight);

void mkl_sgemm_kernel_output(
    const at::Tensor& self,
    const at::Tensor& ori_weight,
    const at::Tensor& bias,
    at::Tensor& output);

at::Tensor mkl_sgemm_kernel(
    const at::Tensor& self,
    const at::Tensor& ori_weight,
    const at::Tensor& bias);

void mkl_prepack_sgemm_kernel_output(
    const at::Tensor& self,
    const at::Tensor& mkl_weight,
    const at::Tensor& bias,
    const int64_t out_features,
    at::Tensor& output);

at::Tensor mkl_prepack_sgemm_kernel(
    const at::Tensor& self,
    const at::Tensor& mkl_weight,
    const at::Tensor& bias,
    const int64_t out_features);

at::Tensor mkl_sgemm_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& op_context,
    const c10::optional<int64_t> out_features);

namespace {

void _mkl_sgemm_packB_impl(
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const at::Tensor& ori_weight,
    at::Tensor& mkl_weight);

at::Tensor mkl_sgemm_packB_impl(
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const at::Tensor& ori_weight);

void mkl_sgemm_base_kernel_impl(
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const int64_t N,
    at::Tensor& output,
    bool pack);

void mkl_sgemm_kernel_impl(
    const at::Tensor& self,
    const at::Tensor& ori_weight,
    const at::Tensor& bias,
    at::Tensor& output);

void mkl_prepack_sgemm_kernel_impl(
    const at::Tensor& self,
    const at::Tensor& mkl_weight,
    const at::Tensor& bias,
    const int64_t out_features,
    at::Tensor& output);

} // namespace

using mkl_sgemm_packB_fn = at::Tensor (*)(
    const int64_t,
    const int64_t,
    const int64_t,
    const at::Tensor&);
using mkl_sgemm_kernel_fn = void (*)(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor&);
using mkl_prepack_sgemm_kernel_fn = void (*)(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const int64_t,
    at::Tensor&);
IPEX_DECLARE_DISPATCH(mkl_sgemm_packB_fn, mkl_sgemm_packB_stub);
IPEX_DECLARE_DISPATCH(mkl_sgemm_kernel_fn, mkl_sgemm_kernel_stub);
IPEX_DECLARE_DISPATCH(
    mkl_prepack_sgemm_kernel_fn,
    mkl_prepack_sgemm_kernel_stub);

} // namespace cpu
} // namespace torch_ipex
