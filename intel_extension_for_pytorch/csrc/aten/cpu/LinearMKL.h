#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <csrc/dyndisp/DispatchStub.h>
#include <vector>
#include "csrc/cpu/ideep/ideep.hpp"
#include "csrc/jit/cpu/kernels/OpContext.h"
#include "mkl.h"

namespace torch_ipex {
namespace cpu {

void mkl_sgemm_repack_weight(
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const at::Tensor& ori_weight,
    at::Tensor& mkl_weight);

at::Tensor mkl_sgemm_pack_weight(
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const at::Tensor& ori_weight);

void mkl_sgemm_kernel_output(
    const at::Tensor& self,
    const at::Tensor& mkl_weight,
    const at::Tensor& bias,
    const int64_t out_features,
    at::Tensor& output);

at::Tensor mkl_sgemm_kernel(
    const at::Tensor& self,
    const at::Tensor& mkl_weight,
    const at::Tensor& bias,
    const int64_t out_features);

at::Tensor mkl_sgemm_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& op_context);

namespace {

void _mkl_sgemm_packB_impl(
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const at::Tensor& ori_weight,
    at::Tensor& mkl_weight);

void mkl_sgemm_repackB_impl(
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

void mkl_sgemm_kernel_impl(
    const at::Tensor& self,
    const at::Tensor& mkl_weight,
    const at::Tensor& bias,
    const int64_t out_features,
    at::Tensor& output);

} // namespace

using mkl_sgemm_repackB_fn = void (*)(
    const int64_t,
    const int64_t,
    const int64_t,
    const at::Tensor&,
    at::Tensor&);
using mkl_sgemm_packB_fn = at::Tensor (*)(
    const int64_t,
    const int64_t,
    const int64_t,
    const at::Tensor&);
using mkl_sgemm_kernel_fn = void (*)(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const int64_t,
    at::Tensor&);
DECLARE_DISPATCH(mkl_sgemm_repackB_fn, mkl_sgemm_repackB_stub);
DECLARE_DISPATCH(mkl_sgemm_packB_fn, mkl_sgemm_packB_stub);
DECLARE_DISPATCH(mkl_sgemm_kernel_fn, mkl_sgemm_kernel_stub);

} // namespace cpu
} // namespace torch_ipex
